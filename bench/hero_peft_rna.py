"""PEFT-lncRNA two-tower HERO (benchmark-native, self-contained).

FEASIBILITY PILOT. The sibling `hero_peft.py` LoRA-tunes the *disease* text encoder.
This hero instead LoRA-fine-tunes the *lncRNA* sequence encoder (RNA-FM) end-to-end
inside the two-tower, on the train association sub-block only. RNA-FM forward is
expensive (~4 min for one full pass over 4236 seqs), so training here is STRICTLY
step-bounded: mini-batch SGD over train lncRNAs, a hard cap on total optimizer
steps, and a hard cap on windows/seq. The disease side stays a frozen-feature MLP
tower over the precomputed Cdis (cheap, trainable projection).

ARCHITECTURE
  disease tower : q = _Tower(Cdis)  (Cdis 768d frozen features, MLP 768->256->m
                  trainable). Recomputed cheaply each step.
  lncRNA enc    : RNA-FM (multimolecule/rnafm, RnaFmModel + RnaTokenizer) with LoRA
                  adapters (base FROZEN; r=RNAFM_R, alpha=16, target=["query","value"],
                  lora_dropout=0.05, bias=none). Per lncRNA: clean (T->U) + split into
                  <=1022-nt windows (cap RNAFM_MAXWIN), tokenize each window, forward
                  RNA-FM+LoRA, masked-mean-pool tokens per window then mean across
                  windows -> 640, then Linear(640->m) projection (trainable).
  score         : logits = e_lnc @ q_dis.T ; predict = sigmoid over ALL (n_l, n_d).

FALLBACK (plumbing / smoke): if the sequence env vars are missing/invalid, the lncRNA
side degrades to a trainable Linear(Clnc.shape[1] -> m) on the passed Clnc (no RNA-FM).
self.peft_active reflects which path ran.

CONTRACT
  fit reads ONLY M[np.ix_(train_lnc, train_dis)] as labels. predict -> (n_l, n_d)
  float32, all finite. Cold lncRNAs get embeddings from their SEQUENCE (inductive) so
  they do NOT collapse to a floor -- that is the point.

SUB-BLOCK INVARIANCE
  lncRNA embeddings depend only on sequence (M-independent); the disease tower + LoRA
  are trained purely on the train-block labels; the seed (incl. the mini-batch sampler)
  is fixed. Scrambling M outside M[np.ix_(train_lnc, train_dis)] therefore leaves
  predict() unchanged.

ENV KNOBS
  RNAFM_R        LoRA rank                         (default 8)
  RNAFM_MAXWIN   windows/seq cap (compute bound)   (default 8)
  RNAFM_BATCH    train lncRNAs sampled per step    (default 24)
  RNAFM_MAXSTEPS TOTAL optimizer steps (hard cap)  (default 300)
  RNAFM_M        latent width m                    (default 128)
  RNAFM_WINBATCH windows tokenized per forward     (default 8)
  RNAFM_WINBUDGET capped windows per backward group (memory bound, default 16)
  CCDIFF_LNC_NAMES  path to lnc_names.txt (node order)   [required for PEFT path]
  CCDIFF_LNC_SEQ    path to lncrna_seq.json (name->{seq,...}) [required for PEFT path]
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device      # snapshot_src (on bench path)
from ccdiff_common import SEED

NAME = "TwoTower-PEFT-lnc (LoRA RNA-FM)"

_RNAFM_ID = "multimolecule/rnafm"
WIN = 1022                                 # RNA-FM context window (nt), like the frozen pipeline
_ALPHA = 16
_TOWER_LR = 1e-3
_LORA_LR = 5e-4
_WD = 1e-4


class _Tower(nn.Module):
    """Same MLP as TwoTowerContent._Tower: Linear(in,256)->ReLU->Dropout->Linear(256,m)."""

    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))

    def forward(self, x):
        return self.net(x)


def _clean(seq):
    s = seq.upper().replace("T", "U")
    return "".join(c if c in "ACGUN" else "N" for c in s)


def _windows(s, max_win):
    chunks = [s[i:i + WIN] for i in range(0, len(s), WIN)]
    return chunks[:max_win]


def _load_lnc_seqs(n_l):
    """Return list[str|None] of length n_l in node order, or None if env missing/invalid."""
    names_path = os.environ.get("CCDIFF_LNC_NAMES")
    seq_path = os.environ.get("CCDIFF_LNC_SEQ")
    if not names_path or not seq_path:
        return None
    if not (os.path.isfile(names_path) and os.path.isfile(seq_path)):
        return None
    try:
        with open(names_path) as f:
            names = [ln.rstrip("\n") for ln in f]
        # drop a single trailing blank line if present
        while names and names[-1] == "":
            names.pop()
        with open(seq_path) as f:
            smap = json.load(f)
    except Exception:
        return None
    if len(names) != n_l:
        return None
    out = []
    for nm in names:
        entry = smap.get(nm)
        if isinstance(entry, dict):
            s = entry.get("seq")
        elif isinstance(entry, str):
            s = entry
        else:
            s = None
        out.append(s if s else None)
    return out


class PEFTRnaHero:
    def __init__(self, r=None, maxwin=None, batch=None, maxsteps=None, m=None,
                 seed=SEED, device=None):
        self.r = int(os.environ.get("RNAFM_R", r if r is not None else 8))
        self.maxwin = int(os.environ.get("RNAFM_MAXWIN", maxwin if maxwin is not None else 8))
        self.batch = int(os.environ.get("RNAFM_BATCH", batch if batch is not None else 24))
        self.maxsteps = int(os.environ.get("RNAFM_MAXSTEPS", maxsteps if maxsteps is not None else 300))
        self.m = int(os.environ.get("RNAFM_M", os.environ.get("TT_M", m if m is not None else 128)))
        self.winbatch = int(os.environ.get("RNAFM_WINBATCH", "8"))
        self.winbudget = int(os.environ.get("RNAFM_WINBUDGET", "16"))
        self.seed = seed
        self.device = device or get_device()
        self.peft_active = False

    # ---- lncRNA sequence encoder (RNA-FM + LoRA) -------------------------
    def _build_rna_encoder(self, dev):
        """Load RNA-FM + tokenizer, wrap with LoRA, build the 640->m projection."""
        from multimolecule import RnaTokenizer, RnaFmModel
        from peft import LoraConfig, get_peft_model, TaskType

        self.tok = RnaTokenizer.from_pretrained(_RNAFM_ID)
        base = RnaFmModel.from_pretrained(_RNAFM_ID)
        self.hidden = base.config.hidden_size          # 640

        lcfg = LoraConfig(r=self.r, lora_alpha=_ALPHA,
                          target_modules=["query", "value"], lora_dropout=0.05,
                          bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        self.rna = get_peft_model(base, lcfg).to(dev)   # base frozen, only LoRA trainable
        self.proj = nn.Linear(self.hidden, self.m).to(dev)

    def _nwin(self, li):
        """Capped window count for a lncRNA (>=1 so grouping never divides by zero)."""
        if not self.peft_active:
            return 1
        s = self.seqs[li]
        if not s:
            return 1
        return max(1, len(_windows(_clean(s), self.maxwin)))

    def _embed_group(self, idxs, dev):
        """Batched RNA-FM+LoRA embedding for a set of lncRNAs -> (len(idxs), m), grad-tracked.

        All windows across the group are tokenized/forwarded together (padded, in winbatch
        sub-batches), masked-mean-pooled per window, then mean-pooled per source lncRNA
        (index_add). Missing sequences contribute zero windows -> zero 640-d feature (like
        the frozen pipeline). Batching across seqs is what makes bounded training tractable."""
        n = len(idxs)
        chunks, owner = [], []
        for pos, li in enumerate(idxs):
            s = self.seqs[li]
            for c in (_windows(_clean(s), self.maxwin) if s else []):
                chunks.append(c); owner.append(pos)
        feats = torch.zeros(n, self.hidden, device=dev)
        if chunks:
            use_amp = str(dev).startswith("cuda")
            win = []
            for b in range(0, len(chunks), self.winbatch):
                enc = self.tok(chunks[b:b + self.winbatch], return_tensors="pt",
                               padding=True).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_amp):
                    out = self.rna(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"]).last_hidden_state  # (B,L,H)
                out = out.float()
                mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
                win.append((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9))            # (B,H)
            W = torch.cat(win, 0)                                                        # (nwin,H)
            own = torch.tensor(owner, device=dev, dtype=torch.long)
            cnt = torch.zeros(n, 1, device=dev).index_add_(
                0, own, torch.ones(len(owner), 1, device=dev))
            feats = feats.index_add(0, own, W) / cnt.clamp(min=1.0)                      # mean over windows
        return self.proj(feats)

    def _embed(self, idxs, dev):
        """(len(idxs), m) grad-tracked embedding: RNA-FM path or Linear(Clnc) fallback."""
        if self.peft_active:
            return self._embed_group(idxs, dev)
        return self.lin_l(torch.tensor(self._Clnc[np.asarray(idxs)], device=dev))

    # ---- contract --------------------------------------------------------
    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)              # LoRA init + tower init + sampler deterministic
        dev = self.device
        n_l, n_d = M.shape

        self._Clnc = np.asarray(Clnc, np.float32)
        Xl = torch.tensor(self._Clnc, device=dev)
        Xd = torch.tensor(np.asarray(Cdis, np.float32), device=dev)

        # disease tower (frozen Cdis features, trainable MLP)
        self.tow_d = _Tower(Cdis.shape[1], self.m).to(dev)

        # lncRNA side: RNA-FM+LoRA from sequences, else Linear(Clnc) fallback
        self.seqs = _load_lnc_seqs(n_l)
        if self.seqs is not None:
            self._build_rna_encoder(dev)
            self.peft_active = True
            self.rna.train()
            lora_params = [p for p in self.rna.parameters() if p.requires_grad]
            tower_proj_params = list(self.tow_d.parameters()) + list(self.proj.parameters())
        else:
            self.lin_l = nn.Linear(Clnc.shape[1], self.m).to(dev)
            self.peft_active = False
            lora_params = []
            tower_proj_params = list(self.tow_d.parameters()) + list(self.lin_l.parameters())

        groups = [{"params": tower_proj_params, "lr": _TOWER_LR}]
        if lora_params:
            groups.append({"params": lora_params, "lr": _LORA_LR})
        opt = torch.optim.Adam(groups, weight_decay=_WD)

        tl = np.asarray(train_lnc)
        td = torch.tensor(np.asarray(train_dis), device=dev, dtype=torch.long)
        n_tl = len(tl)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)].astype(np.float32), device=dev)  # (n_tl, n_td)
        n_td = Yblk.shape[1]
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce_sum = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="sum")
        bce_mean = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self.tow_d.train()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        bsz = min(self.batch, n_tl)
        denom = float(bsz * n_td)
        self.final_loss = float("nan")

        # STRICTLY BOUNDED mini-batch SGD. Each step: sample RNAFM_BATCH train lncRNAs,
        # split them into GROUPS whose total (capped) windows fit RNAFM_WINBUDGET, and
        # backward once per group -- so peak memory ~ one window-budget while windows are
        # still batched across seqs for GPU throughput. The disease side gets its gradient
        # in a second detached pass; both use the identical mean-BCE objective, so the
        # summed per-parameter gradients equal a full-batch step.
        for _ in range(self.maxsteps):
            bpos = torch.randperm(n_tl, generator=gen)[:bsz].numpy()      # positions within train_lnc
            batch_lnc = tl[bpos]
            y_batch = Yblk[torch.tensor(bpos, device=dev, dtype=torch.long)]   # (bsz, n_td)

            # deterministic window-budget grouping of the minibatch positions
            groups, cur, curw = [], [], 0
            for j in range(bsz):
                w = self._nwin(int(batch_lnc[j]))
                if cur and curw + w > self.winbudget:
                    groups.append(cur); cur, curw = [], 0
                cur.append(j); curw += w
            if cur:
                groups.append(cur)

            opt.zero_grad()
            q_det = self.tow_d(Xd)[td].detach()                          # (n_td, m), frozen for lnc side
            E_det = torch.empty(bsz, self.m, device=dev)
            step_loss = 0.0
            for grp in groups:
                gi = torch.tensor(grp, device=dev, dtype=torch.long)
                e_g = self._embed(batch_lnc[grp], dev)                   # (|grp|, m), RNA-FM graph
                E_det[gi] = e_g.detach()
                loss_g = bce_sum(e_g @ q_det.T, y_batch[gi]) / denom
                loss_g.backward()                                        # frees this group's graph
                step_loss += float(loss_g.item())

            q = self.tow_d(Xd)[td]                                       # (n_td, m), cheap graph
            loss_q = bce_mean(E_det @ q.T, y_batch)                      # grads only into disease tower
            loss_q.backward()
            opt.step()
            self.final_loss = step_loss

        # ---- one full pass over ALL lncRNAs through the tuned encoder (the ~4min pass) ----
        if self.peft_active:
            self.rna.eval()
        self.tow_d.eval()
        with torch.no_grad():
            q = self.tow_d(Xd)                                           # (n_d, m)
            if self.peft_active:
                rows = [self._embed_group(np.arange(b, min(b + self.batch, n_l)), dev)
                        for b in range(0, n_l, self.batch)]
                e = torch.cat(rows, 0)                                   # (n_l, m)
            else:
                e = self.lin_l(Xl)                                       # (n_l, m)
            self._S = torch.sigmoid(e @ q.T).cpu().numpy().astype(np.float32)
        return self

    def predict(self):
        return self._S


def build(device):
    return PEFTRnaHero(device=device)
