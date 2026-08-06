"""Multi-PEFT-method disease two-tower HERO (benchmark-native, self-contained).

Sibling of hero_peft.py. SAME architecture / contract / invariance / fallback --
the ONLY difference is that the disease-encoder adapter is chosen by env
``PEFT_METHOD`` in {lora, ia3, vera, prompt} (default lora). Motivation: prior
both-cold experiments show LoRA can HURT C2 on the small variant (overfitting the
cold-start block). Lower-parameter PEFT methods (IA3, VeRA, prompt-tuning) train
far fewer parameters, so this hero lets us test whether reducing adapter capacity
reduces that overfitting.

ARCHITECTURE  (identical to hero_peft.py)
  lncRNA tower  : e = _Tower(Clnc)                 # MLP 702->256->m, frozen features
  disease enc   : BERT(S-BioBert-snli-multinli-stsb) + chosen PEFT adapter (base FROZEN),
                  attention-masked mean-pool -> 768 -> Linear(768->m) projection (trainable).
                  Disease TEXTS come from env (node-ordered doids + doid->text map), NOT Cdis.
  score         : logits = e_lnc @ e_dis.T ; predict = sigmoid over ALL (n_l, n_d).

ADAPTERS (env PEFT_METHOD)
  lora  : LoraConfig(r=PEFT_R=8, lora_alpha=PEFT_ALPHA=16, target_modules=[query,value],
          lora_dropout=0.05, bias=none)                         -- same as hero_peft.py
  ia3   : IA3Config(target_modules=[key,value,output.dense],
          feedforward_modules=[output.dense]) -- learns only per-feature rescale vectors
  vera  : VeraConfig(r=VERA_R=256, target_modules=[query,value]) -- shared frozen random
          low-rank matrices across layers, trains only small scaling vectors
  prompt: PromptTuningConfig(task_type=FEATURE_EXTRACTION, num_virtual_tokens=PROMPT_TOKENS=16)
          -- PREPENDS num_virtual_tokens learned embeddings to the sequence; the pooling
          mask is extended by num_virtual_tokens ones so the virtual tokens are pooled too.

FALLBACK (plumbing / smoke): if disease-text env vars are missing/invalid, the disease
side degrades to a trainable Linear(Cdis.shape[1] -> m) on the passed Cdis (no BERT).
self.peft_active reflects which path ran.

CONTRACT / SUB-BLOCK INVARIANCE: identical to hero_peft.py. fit reads ONLY
M[np.ix_(train_lnc, train_dis)]; predict -> (n_l, n_d) float32 finite; disease
embeddings are text-only (M-independent) so scrambling M outside the train sub-block
leaves predict() unchanged. Seed fixed (SEED).

ENV KNOBS
  PEFT_METHOD   {lora,ia3,vera,prompt}        (default lora)
  PEFT_R        LoRA rank                      (default 8)
  PEFT_ALPHA    LoRA alpha                     (default 16)
  VERA_R        VeRA rank                      (default 256)
  PROMPT_TOKENS num virtual tokens (prompt)    (default 16)
  PEFT_LR       Adam lr for tower + proj       (default 1e-3)
  PEFT_ENC_LR   Adam lr for adapter params     (default 5e-4)
  PEFT_WD       Adam weight_decay              (default 1e-4)
  PEFT_EPOCHS   epochs (else TT_EPOCHS)        (default 100)
  PEFT_M        latent width m (else TT_M)     (default 128)
  PEFT_DROPOUT  lncRNA tower dropout           (default 0.0)
  PEFT_MAXLEN   disease-text token cap         (default 64)
  PEFT_L2       L2-normalize disease proj      (default 0 = off)
  CCDIFF_DIS_DOIDS   path to disease_doids.txt (node order)   [required for PEFT path]
  CCDIFF_DIS_TEXTS   path to disease_texts.json (doid->text)  [required for PEFT path]
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device      # snapshot_src (on bench path)
from ccdiff_common import SEED

NAME = "TwoTower-PEFT-method (disease)"

_SBERT_ID = "pritamdeka/S-BioBert-snli-multinli-stsb"


class _Tower(nn.Module):
    """Same MLP as TwoTowerContent._Tower: Linear(in,256)->ReLU->Dropout->Linear(256,m)."""

    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))

    def forward(self, x):
        return self.net(x)


class _PEFTDiseaseEncoder(nn.Module):
    """PEFT-adapted BERT -> attention-masked mean-pool -> Linear(768->m).

    ``prepend`` is the number of virtual tokens the adapter PREPENDS to the encoder
    output (0 for lora/ia3/vera; num_virtual_tokens for prompt-tuning). When >0 the
    pooling mask is extended with that many leading ones so the prepended virtual
    tokens are pooled together with the real tokens.
    """

    def __init__(self, peft_bert, hidden, m, l2=False, prepend=0):
        super().__init__()
        self.bert = peft_bert                 # base FROZEN, only adapter trainable
        self.proj = nn.Linear(hidden, m)      # trainable
        self.l2 = l2
        self.prepend = int(prepend)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state                          # (n, prepend+L, hidden)
        mask = attention_mask
        if self.prepend > 0:
            # prompt-tuning prepends `prepend` virtual-token embeddings -> extend mask
            ones = attention_mask.new_ones((attention_mask.shape[0], self.prepend))
            mask = torch.cat([ones, attention_mask], dim=1)   # (n, prepend+L)
        mask = mask.unsqueeze(-1).to(tok.dtype)              # (n, prepend+L, 1)
        pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        e = self.proj(pooled)
        if self.l2:
            e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        return e


class _LinearDiseaseEncoder(nn.Module):
    """Fallback (no BERT): trainable Linear(Cdis.shape[1] -> m) on the passed Cdis."""

    def __init__(self, in_dim, m):
        super().__init__()
        self.proj = nn.Linear(in_dim, m)

    def forward(self, Xd):
        return self.proj(Xd)


def _load_disease_texts(n_d):
    """Return list[str] of length n_d in node order, or None if env missing/invalid."""
    doids_path = os.environ.get("CCDIFF_DIS_DOIDS")
    texts_path = os.environ.get("CCDIFF_DIS_TEXTS")
    if not doids_path or not texts_path:
        return None
    if not (os.path.isfile(doids_path) and os.path.isfile(texts_path)):
        return None
    try:
        with open(doids_path) as f:
            doids = [ln.strip() for ln in f if ln.strip()]
        with open(texts_path) as f:
            tmap = json.load(f)
    except Exception:
        return None
    if len(doids) != n_d:
        return None
    return [str(tmap.get(d, d)) for d in doids]


class PEFTMethodHero:
    def __init__(self, method=None, r=None, alpha=None, vera_r=None, prompt_tokens=None,
                 epochs=None, lr=None, enc_lr=None, wd=None, m=None, dropout=None,
                 seed=SEED, device=None):
        self.method = str(os.environ.get("PEFT_METHOD",
                                         method if method is not None else "lora")).lower()
        self.r = int(os.environ.get("PEFT_R", r if r is not None else 8))
        self.alpha = int(os.environ.get("PEFT_ALPHA", alpha if alpha is not None else 16))
        self.vera_r = int(os.environ.get("VERA_R", vera_r if vera_r is not None else 256))
        self.prompt_tokens = int(os.environ.get("PROMPT_TOKENS",
                                                prompt_tokens if prompt_tokens is not None else 16))
        self.epochs = int(os.environ.get("PEFT_EPOCHS",
                                         os.environ.get("TT_EPOCHS", epochs if epochs is not None else 100)))
        self.lr = float(os.environ.get("PEFT_LR", lr if lr is not None else 1e-3))
        self.enc_lr = float(os.environ.get("PEFT_ENC_LR", enc_lr if enc_lr is not None else 5e-4))
        self.wd = float(os.environ.get("PEFT_WD", wd if wd is not None else 1e-4))
        self.m = int(os.environ.get("PEFT_M", os.environ.get("TT_M", m if m is not None else 128)))
        self.dropout = float(os.environ.get("PEFT_DROPOUT", dropout if dropout is not None else 0.0))
        self.maxlen = int(os.environ.get("PEFT_MAXLEN", "64"))
        self.l2 = os.environ.get("PEFT_L2", "0") not in ("0", "", "false", "False")
        self.seed = seed
        self.device = device or get_device()
        self.peft_active = False
        self.adapter_trainable_params = 0

    # ---- adapter config selector -----------------------------------------
    def _adapter_config(self):
        """Build the peft config for self.method. Returns (config, prepend_tokens)."""
        from peft import (LoraConfig, IA3Config, VeraConfig, PromptTuningConfig,
                          TaskType)
        method = self.method
        if method == "lora":
            cfg = LoraConfig(r=self.r, lora_alpha=self.alpha,
                             target_modules=["query", "value"], lora_dropout=0.05,
                             bias="none", task_type=TaskType.FEATURE_EXTRACTION)
            return cfg, 0
        if method == "ia3":
            # IA3 learns only per-feature rescaling vectors (very few params).
            cfg = IA3Config(target_modules=["key", "value", "output.dense"],
                            feedforward_modules=["output.dense"],
                            task_type=TaskType.FEATURE_EXTRACTION)
            return cfg, 0
        if method == "vera":
            # VeRA shares frozen random low-rank matrices across layers, trains only
            # small per-layer scaling vectors. query/value are 768x768 in BERT so the
            # shared-matrix same-shape constraint is satisfied.
            cfg = VeraConfig(r=self.vera_r, target_modules=["query", "value"],
                             task_type=TaskType.FEATURE_EXTRACTION)
            return cfg, 0
        if method == "prompt":
            cfg = PromptTuningConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                     num_virtual_tokens=self.prompt_tokens)
            return cfg, self.prompt_tokens
        raise ValueError(f"unknown PEFT_METHOD={method!r} (expected lora|ia3|vera|prompt)")

    # ---- disease-side builders -------------------------------------------
    def _build_peft_encoder(self, texts, dev):
        """Load S-BioBERT, wrap BERT with the chosen adapter, tokenize texts once."""
        from sentence_transformers import SentenceTransformer
        from peft import get_peft_model

        st = SentenceTransformer(_SBERT_ID, device="cpu")
        tmod = st[0]                                   # Transformer module
        base_bert = tmod.auto_model                    # BertModel, hidden 768
        tokenizer = tmod.tokenizer
        hidden = base_bert.config.hidden_size

        cfg, prepend = self._adapter_config()
        peft_bert = get_peft_model(base_bert, cfg)     # base frozen, only adapter trainable

        tok = tokenizer(texts, padding=True, truncation=True,
                        max_length=self.maxlen, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(dev)
        self.attn_mask = tok["attention_mask"].to(dev)
        enc = _PEFTDiseaseEncoder(peft_bert, hidden, self.m, l2=self.l2,
                                  prepend=prepend).to(dev)
        return enc

    def _e_dis(self, enc, Xd):
        return enc(self.input_ids, self.attn_mask) if self.peft_active else enc(Xd)

    # ---- contract --------------------------------------------------------
    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)              # adapter init + all params deterministic
        dev = self.device
        n_l, n_d = M.shape

        Xl = torch.tensor(Clnc, device=dev)
        Xd = torch.tensor(Cdis, device=dev)

        # lncRNA tower (frozen features, trainable MLP)
        self.tow_l = _Tower(Clnc.shape[1], self.m, self.dropout).to(dev)

        # disease side: PEFT-BERT from text, else Linear(Cdis) fallback
        texts = _load_disease_texts(n_d)
        if texts is not None:
            self.enc = self._build_peft_encoder(texts, dev)
            self.peft_active = True
            adapter_params = [p for p in self.enc.bert.parameters() if p.requires_grad]
            tower_proj_params = list(self.tow_l.parameters()) + list(self.enc.proj.parameters())
        else:
            self.enc = _LinearDiseaseEncoder(Cdis.shape[1], self.m).to(dev)
            self.peft_active = False
            adapter_params = []
            tower_proj_params = list(self.tow_l.parameters()) + list(self.enc.parameters())

        # report adapter capacity (compare across methods)
        self.adapter_trainable_params = int(sum(p.numel() for p in adapter_params))
        print(f"[PEFT] method={self.method} adapter_trainable_params={self.adapter_trainable_params}")

        # param groups: tower+proj at self.lr, adapter at smaller self.enc_lr
        groups = [{"params": tower_proj_params, "lr": self.lr}]
        if adapter_params:
            groups.append({"params": adapter_params, "lr": self.enc_lr})
        opt = torch.optim.Adam(groups, weight_decay=self.wd)

        tl = torch.tensor(train_lnc, device=dev, dtype=torch.long)
        td = torch.tensor(train_dis, device=dev, dtype=torch.long)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)], device=dev)
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self.tow_l.train(); self.enc.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            e_lnc = self.tow_l(Xl)                    # (n_l, m)
            e_dis = self._e_dis(self.enc, Xd)         # (n_d, m) -- all diseases, one batch
            logits = e_lnc[tl] @ e_dis[td].T          # (|train_lnc|, |train_dis|)
            loss = bce(logits, Yblk)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())

        # final full matrix from the tuned encoder (cold diseases via their text)
        self.tow_l.eval(); self.enc.eval()
        with torch.no_grad():
            e_lnc = self.tow_l(Xl)
            e_dis = self._e_dis(self.enc, Xd)
            self._S = torch.sigmoid(e_lnc @ e_dis.T).cpu().numpy().astype(np.float32)
        return self

    def predict(self):
        return self._S


def build(device):
    return PEFTMethodHero(device=device)
