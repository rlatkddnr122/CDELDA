"""PEFT-disease two-tower HERO (benchmark-native, self-contained).

Motivation: prior both-cold experiments show the ceiling is set by CONTENT quality,
not the interaction head. The principled way to try to raise it is to TASK-ADAPT the
disease text encoder itself. This hero LoRA-fine-tunes the disease encoder
(S-BioBERT) end-to-end inside the two-tower, on the train association sub-block only.
The lncRNA side stays a frozen-feature MLP tower (RNA-FM content is expensive/opaque;
we do not touch it).

ARCHITECTURE
  lncRNA tower  : e = _Tower(Clnc)                 # same MLP as TwoTowerContent, 702->256->m
                  (Clnc frozen features, MLP trainable)
  disease enc   : BERT(S-BioBert-snli-multinli-stsb) with LoRA adapters (base FROZEN),
                  attention-masked mean-pool -> 768 -> Linear(768->m) projection (trainable).
                  Disease TEXTS come from env (node-ordered doids + doid->text map), NOT Cdis.
  score         : logits = e_lnc @ e_dis.T ; predict = sigmoid over ALL (n_l, n_d).

FALLBACK (plumbing / smoke): if the disease-text env vars are missing/invalid, the
disease side degrades to a trainable Linear(Cdis.shape[1] -> m) on the passed Cdis
(no BERT). self.peft_active reflects which path ran.

CONTRACT
  fit reads ONLY M[np.ix_(train_lnc, train_dis)] as labels. predict -> (n_l, n_d)
  float32, all finite. Cold diseases get embeddings from their TEXT (inductive) so they
  do NOT collapse to a floor -- that is the point.

SUB-BLOCK INVARIANCE
  Disease embeddings depend only on text (M-independent); the lncRNA tower + LoRA are
  trained purely on the train-block labels; the seed is fixed. Scrambling M outside
  M[np.ix_(train_lnc, train_dis)] therefore leaves predict() unchanged (verified ~0).

ENV KNOBS
  PEFT_R        LoRA rank                    (default 8)
  PEFT_ALPHA    LoRA alpha                   (default 16)
  PEFT_LR       Adam lr for tower + proj     (default 1e-3)
  PEFT_ENC_LR   Adam lr for LoRA adapters    (default 5e-4)
  PEFT_WD       Adam weight_decay            (default 1e-4)
  PEFT_EPOCHS   epochs (else TT_EPOCHS)      (default 100)
  PEFT_M        latent width m (else TT_M)   (default 128)
  PEFT_DROPOUT  lncRNA tower dropout         (default 0.0)
  PEFT_MAXLEN   disease-text token cap       (default 64)
  PEFT_L2       L2-normalize disease proj    (default 0 = off)
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

NAME = "TwoTower-PEFT-disease (LoRA S-BioBERT)"

_SBERT_ID = "pritamdeka/S-BioBert-snli-multinli-stsb"


class _Tower(nn.Module):
    """Same MLP as TwoTowerContent._Tower: Linear(in,256)->ReLU->Dropout->Linear(256,m)."""

    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))

    def forward(self, x):
        return self.net(x)


class _LoRADiseaseEncoder(nn.Module):
    """LoRA-adapted BERT -> attention-masked mean-pool -> Linear(768->m)."""

    def __init__(self, peft_bert, hidden, m, l2=False):
        super().__init__()
        self.bert = peft_bert                 # base FROZEN, LoRA adapters trainable
        self.proj = nn.Linear(hidden, m)      # trainable
        self.l2 = l2

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state                          # (n, L, hidden)
        mask = attention_mask.unsqueeze(-1).to(tok.dtype)    # (n, L, 1)
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


class PEFTDiseaseHero:
    def __init__(self, r=None, alpha=None, epochs=None, lr=None, enc_lr=None, wd=None,
                 m=None, dropout=None, seed=SEED, device=None, content_l=True, content_d=True):
        # content_l/content_d toggle each tower's content (False -> free per-index
        # embedding that cannot generalize to cold nodes) -- used for axis ablation.
        self.content_l = content_l
        self.content_d = content_d
        self.r = int(os.environ.get("PEFT_R", r if r is not None else 8))
        self.alpha = int(os.environ.get("PEFT_ALPHA", alpha if alpha is not None else 16))
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

    # ---- disease-side builders -------------------------------------------
    def _build_peft_encoder(self, texts, dev):
        """Load S-BioBERT, wrap BERT with LoRA, tokenize texts once. Returns encoder module."""
        from sentence_transformers import SentenceTransformer
        from peft import LoraConfig, get_peft_model, TaskType

        st = SentenceTransformer(_SBERT_ID, device="cpu")
        tmod = st[0]                                   # Transformer module
        base_bert = tmod.auto_model                    # BertModel, hidden 768
        tokenizer = tmod.tokenizer
        hidden = base_bert.config.hidden_size

        lcfg = LoraConfig(r=self.r, lora_alpha=self.alpha,
                          target_modules=["query", "value"], lora_dropout=0.05,
                          bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        peft_bert = get_peft_model(base_bert, lcfg)    # base frozen, only LoRA trainable

        tok = tokenizer(texts, padding=True, truncation=True,
                        max_length=self.maxlen, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(dev)
        self.attn_mask = tok["attention_mask"].to(dev)
        enc = _LoRADiseaseEncoder(peft_bert, hidden, self.m, l2=self.l2).to(dev)
        return enc

    def _e_lnc(self, Xl):
        return self.tow_l(Xl) if self.content_l else self.emb_l.weight

    def _e_dis(self, enc, Xd):
        if not self.content_d:
            return self.emb_d.weight
        return enc(self.input_ids, self.attn_mask) if self.peft_active else enc(Xd)

    # ---- contract --------------------------------------------------------
    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)              # LoRA init + all params deterministic
        dev = self.device
        n_l, n_d = M.shape

        Xl = torch.tensor(Clnc, device=dev)
        Xd = torch.tensor(Cdis, device=dev)

        # lncRNA side: content tower, or free per-index embedding (axis ablation)
        if self.content_l:
            self.tow_l = _Tower(Clnc.shape[1], self.m, self.dropout).to(dev)
            l_params = list(self.tow_l.parameters())
        else:
            self.emb_l = nn.Embedding(n_l, self.m).to(dev); self.tow_l = None
            l_params = list(self.emb_l.parameters())

        # disease side: PEFT-BERT text encoder / Linear fallback, or free embedding (axis ablation)
        lora_params = []
        if self.content_d:
            texts = _load_disease_texts(n_d)
            if texts is not None:
                self.enc = self._build_peft_encoder(texts, dev)
                self.peft_active = True
                lora_params = [p for p in self.enc.bert.parameters() if p.requires_grad]
                d_params = list(self.enc.proj.parameters())
            else:
                self.enc = _LinearDiseaseEncoder(Cdis.shape[1], self.m).to(dev)
                self.peft_active = False
                d_params = list(self.enc.parameters())
        else:
            self.emb_d = nn.Embedding(n_d, self.m).to(dev); self.enc = None
            self.peft_active = False
            d_params = list(self.emb_d.parameters())

        tower_proj_params = l_params + d_params

        # param groups: tower+proj at self.lr, LoRA adapters at smaller self.enc_lr
        groups = [{"params": tower_proj_params, "lr": self.lr}]
        if lora_params:
            groups.append({"params": lora_params, "lr": self.enc_lr})
        opt = torch.optim.Adam(groups, weight_decay=self.wd)

        tl = torch.tensor(train_lnc, device=dev, dtype=torch.long)
        td = torch.tensor(train_dis, device=dev, dtype=torch.long)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)], device=dev)
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # --- training-time negative-sampling scheme (ablation knob) --------------
        # PEFT_NEG_MODE: "weighted" (default; all negatives, pos_weight-reweighted) |
        #   "uniform" (k random negatives per positive) | "degree" (negatives sampled
        #   ∝ node popularity) | "hard" (top-scoring negatives, re-mined each epoch).
        neg_mode = os.environ.get("PEFT_NEG_MODE", "weighted")
        neg_ratio = int(os.environ.get("PEFT_NEG_RATIO", "5"))
        yf = Yblk.flatten()
        pos_ix = (yf > 0).nonzero(as_tuple=True)[0]
        neg_ix = (yf == 0).nonzero(as_tuple=True)[0]
        n_keep = min(neg_ratio * int(pos_ix.numel()), int(neg_ix.numel()))
        gen = torch.Generator(device=dev).manual_seed(SEED)
        if neg_mode == "degree":
            rdeg = Yblk.sum(1); cdeg = Yblk.sum(0)
            nb = Yblk.shape[1]
            negw = (rdeg[neg_ix // nb] + 1.0) * (cdeg[neg_ix % nb] + 1.0)   # popularity-matched
            negw = negw / negw.sum()

        mods = [m for m in (self.tow_l, self.enc,
                            getattr(self, "emb_l", None), getattr(self, "emb_d", None)) if m is not None]
        for m in mods:
            m.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            e_lnc = self._e_lnc(Xl)                   # (n_l, m)
            e_dis = self._e_dis(self.enc, Xd)         # (n_d, m) -- all diseases, one batch
            logits = e_lnc[tl] @ e_dis[td].T          # (|train_lnc|, |train_dis|)
            if neg_mode == "weighted":
                loss = bce(logits, Yblk)
            elif neg_mode == "pu":
                # non-negative PU risk (Kiryo et al. 2017) with the logistic surrogate:
                # unobserved pairs are treated as UNLABELED (mixture of hidden positives at
                # class-prior PU_PI and true negatives), not as certain negatives.
                pi = float(os.environ.get("PU_PI", "0.05"))
                lf = logits.flatten()
                sp = nn.functional.softplus
                Rp_pos = sp(-lf[pos_ix]).mean()          # positive risk of positives
                Rp_neg = sp(lf[pos_ix]).mean()           # negative risk of positives
                Ru_neg = sp(lf[neg_ix]).mean()           # negative risk of unlabeled
                loss = pi * Rp_pos + torch.clamp(Ru_neg - pi * Rp_neg, min=0.0)
            else:
                lf = logits.flatten()
                if neg_mode == "uniform":
                    sel = neg_ix[torch.randint(int(neg_ix.numel()), (n_keep,), device=dev, generator=gen)]
                elif neg_mode == "degree":
                    sel = neg_ix[torch.multinomial(negw, n_keep, replacement=True, generator=gen)]
                elif neg_mode == "hard":
                    sel = neg_ix[torch.topk(lf[neg_ix], n_keep).indices]
                else:
                    raise ValueError(f"unknown PEFT_NEG_MODE={neg_mode}")
                idx = torch.cat([pos_ix, sel])
                loss = nn.functional.binary_cross_entropy_with_logits(lf[idx], yf[idx])
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())

        # final full matrix from the tuned encoder (cold diseases via their text)
        for m in mods:
            m.eval()
        with torch.no_grad():
            e_lnc = self._e_lnc(Xl)
            e_dis = self._e_dis(self.enc, Xd)
            self._S = torch.sigmoid(e_lnc @ e_dis.T).cpu().numpy().astype(np.float32)
        return self

    def predict(self):
        return self._S


def build(device):
    return PEFTDiseaseHero(device=device)
