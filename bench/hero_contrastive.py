"""Contrastive (InfoNCE) two-tower HERO for cold-start lncRNA-disease association.

Same content-tower backbone as TwoTowerContent (snapshot_src/twoside_models.py):
each side is a small MLP _Tower(d -> 256 -> ReLU -> Dropout -> m). Embeddings are
L2-normalized, so <e_i, q_j> is a cosine similarity. Instead of the pointwise
weighted-BCE of TwoTowerContent, we train a SYMMETRIC multi-positive InfoNCE
(contrastive) objective over the TRAIN sub-block only, which learns a metric
content space where a lncRNA sits close to ITS diseases and far from every other
train disease (and vice-versa). Because both towers are purely content-driven and
inductive, cold (held-out) rows/cols still receive real content-based scores at
predict time -- no popularity floor collapse.

Contract (bench/interface.py): fit reads ONLY M[np.ix_(train_lnc, train_dis)] as
labels; content Clnc/Cdis is intrinsic to all nodes. predict -> (n_l, n_d) float32,
all finite. Sub-block invariant: nothing in fit depends on off-block M (labels are
sub-block-only, content is M-independent, torch seeded) -> off-block scramble
leaves predict unchanged (max|delta| ~ 0).

InfoNCE formulation
-------------------
Let e = l2norm(tower_l(Clnc)) (n_l x m), q = l2norm(tower_d(Cdis)) (n_d x m).
On the train block, S = e[tl] @ q[td].T / temp  (nTl x nTd), Y = M[ix(tl,td)].
For the lncRNA->disease direction, with row-wise log-softmax over ALL train
diseases (every non-positive train disease is a negative):
    L_l = mean_i  -(1/|P(i)|) * sum_{j in P(i)} log_softmax(S[i, :])[j]
i.e. each lncRNA's positive diseases are pulled up against all train diseases.
Symmetrically L_d uses S.T / Y.T (each disease's positive lncRNAs vs all train
lncRNAs). Total = 0.5*(L_l + L_d). Rows/cols with no positive are skipped.
An optional weighted-BCE auxiliary (CH_BCE, default 0 = off) can be added.

Tunable via env: CH_M (128), CH_TEMP (0.1), CH_LR (1e-3), CH_WD (1e-4),
CH_EPOCHS (500, TT_EPOCHS honored as fallback so smoke can bound it),
CH_DROPOUT (0.2), CH_BCE (0.0). Seeded with SEED for determinism.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccdiff_models import get_device      # snapshot_src (on bench path)
from ccdiff_common import SEED

NAME = "TwoTower-Contrastive (content)"


class _Tower(nn.Module):
    """Same MLP as TwoTowerContent._Tower: Linear(d->256)->ReLU->Dropout->Linear(256->m)."""

    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))

    def forward(self, x):
        return self.net(x)


def _multipos_infonce(S, Y):
    """Row-wise multi-positive InfoNCE for logits S (nR x nC) and binary Y (nR x nC).

    log-softmax over columns (all candidates are negatives except the row's
    positives); each valid row contributes the mean -log p over its positives.
    Rows with no positive are skipped. Returns a scalar tensor tied to the graph
    (0 if no valid row, so backward never fails)."""
    logp = F.log_softmax(S, dim=1)                       # (nR, nC)
    pos = Y.sum(1)                                        # (nR,) positives per row
    per_row = -(Y * logp).sum(1) / pos.clamp(min=1.0)    # mean -log p over positives
    valid = pos > 0
    if valid.any():
        return per_row[valid].mean()
    return 0.0 * S.sum()                                 # keep a grad path, value 0


class ContrastiveHero:
    def __init__(self, m=None, temp=None, lr=None, wd=None, epochs=None, dropout=None,
                 bce=None, seed=SEED, device=None):
        self.m = int(os.environ.get("CH_M", m if m is not None else 128))
        self.temp = float(os.environ.get("CH_TEMP", temp if temp is not None else 0.1))
        self.lr = float(os.environ.get("CH_LR", lr if lr is not None else 1e-3))
        self.wd = float(os.environ.get("CH_WD", wd if wd is not None else 1e-4))
        # CH_EPOCHS is primary; TT_EPOCHS is honored as a fallback (smoke bounds it).
        _ep = os.environ.get("CH_EPOCHS", os.environ.get("TT_EPOCHS",
                             str(epochs if epochs is not None else 500)))
        self.epochs = int(_ep)
        self.dropout = float(os.environ.get("CH_DROPOUT", dropout if dropout is not None else 0.2))
        self.bce = float(os.environ.get("CH_BCE", bce if bce is not None else 0.0))
        self.seed = seed
        self.device = device or get_device()

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)
        dev = self.device
        Xl = torch.tensor(np.asarray(Clnc, np.float32), device=dev)
        Xd = torch.tensor(np.asarray(Cdis, np.float32), device=dev)
        tl = torch.tensor(np.asarray(train_lnc), device=dev, dtype=torch.long)
        td = torch.tensor(np.asarray(train_dis), device=dev, dtype=torch.long)
        Yblk = torch.tensor(np.asarray(M, np.float32)[np.ix_(np.asarray(train_lnc),
                            np.asarray(train_dis))], device=dev)          # ONLY supervision

        self.tow_l = _Tower(Xl.shape[1], self.m, self.dropout).to(dev)
        self.tow_d = _Tower(Xd.shape[1], self.m, self.dropout).to(dev)
        params = list(self.tow_l.parameters()) + list(self.tow_d.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        bce = None
        if self.bce > 0:
            pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self.tow_l.train(); self.tow_d.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            e = F.normalize(self.tow_l(Xl), dim=1)       # (n_l, m)
            q = F.normalize(self.tow_d(Xd), dim=1)       # (n_d, m)
            S = (e[tl] @ q[td].T) / self.temp            # (nTl, nTd) cosine/temp logits
            loss = 0.5 * (_multipos_infonce(S, Yblk) + _multipos_infonce(S.T, Yblk.T))
            if bce is not None:
                loss = loss + self.bce * bce(S, Yblk)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())

        self.tow_l.eval(); self.tow_d.eval()             # dropout off for final embeddings
        with torch.no_grad():
            self.e = F.normalize(self.tow_l(Xl), dim=1).detach()
            self.q = F.normalize(self.tow_d(Xd), dim=1).detach()
        return self

    @torch.no_grad()
    def predict(self):
        S = torch.sigmoid((self.e @ self.q.T) / self.temp)
        return S.cpu().numpy().astype(np.float32)


def build(device):
    return ContrastiveHero(device=device)
