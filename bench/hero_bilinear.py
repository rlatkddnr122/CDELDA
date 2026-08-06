"""Bilinear-interaction two-tower HERO (benchmark-native, self-contained).

Novelty vs. the dot-product / dual-attention heroes: the two content towers still
produce inductive embeddings e = tower_l(Clnc) (n_l x m) and q = tower_d(Cdis)
(n_d x m) via the SAME _Tower MLP as TwoTowerContent, but the association score is
a LEARNED BILINEAR FORM plus a dot-product residual:

    score_ij = e_i^T W q_j + <e_i, q_j>

Vectorized over all pairs:

    logits = E @ W @ Q^T + E @ Q^T          # (n_l, n_d)

W is a learned (m x m) bilinear interaction matrix. It lets the model discover an
anisotropic, cross-dimension coupling between the lncRNA and disease latent spaces
that a bare dot product (W = I) cannot express, while the additive <e,q> residual
keeps the plain dot-product hero as a strict special case (so it can only help).

LOW-RANK option (BIL_RANK = r > 0): W = U V^T with U,V of shape (m, r). This
regularizes the interaction and makes the map E @ W @ Q^T = (E @ U) @ (Q @ V)^T
factor through an r-dim bottleneck. BIL_RANK=0 uses a full dense W.

Contract: fit reads ONLY M[np.ix_(train_lnc, train_dis)] as labels; content
Clnc/Cdis is intrinsic (all nodes). predict -> (n_l, n_d) float32 finite. Cold
nodes get content-based embeddings (inductive -> no floor collapse). Sub-block
invariant: nothing depends on off-block M, so scrambling the held-out region
leaves predict() unchanged (verified ~0).

Env knobs (for the wandb sweep):
    BIL_M       latent width m           (default 128)
    BIL_RANK    low-rank r; 0 = full W   (default 0)
    BIL_LR      Adam lr                  (default 1e-3)
    BIL_WD      Adam weight_decay        (default 1e-4)
    BIL_EPOCHS  epochs (else TT_EPOCHS)  (default 500)
    BIL_DROPOUT tower dropout            (default 0.2)
"""
import os

import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device      # snapshot_src (on bench path)
from ccdiff_common import SEED

NAME = "TwoTower-Bilinear (content)"


class _Tower(nn.Module):
    """Same MLP as TwoTowerContent._Tower: Linear(in,256)->ReLU->Dropout->Linear(256,m)."""

    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))

    def forward(self, x):
        return self.net(x)


class _Bilinear(nn.Module):
    """Learned bilinear interaction E @ W @ Q^T with dot-product residual E @ Q^T.

    rank == 0 -> full dense W (m x m), initialized at the identity so training
    starts from the pure dot-product hero and departs only if it helps.
    rank  > 0 -> W = U V^T low-rank; U,V small-random so E @ W @ Q^T starts ~0
    and the dot-product residual again dominates at init.
    """

    def __init__(self, m, rank=0):
        super().__init__()
        self.m, self.rank = m, rank
        if rank and rank > 0:
            self.U = nn.Parameter(torch.randn(m, rank) * (1.0 / (m ** 0.5)))
            self.V = nn.Parameter(torch.randn(m, rank) * (1.0 / (m ** 0.5)))
            self.W = None
        else:
            self.W = nn.Parameter(torch.eye(m))
            self.U = self.V = None

    def forward(self, E, Q):
        if self.W is not None:
            bil = (E @ self.W) @ Q.T                 # (n_l, n_d)
        else:
            bil = (E @ self.U) @ (Q @ self.V).T      # low-rank: factor through r
        return bil + E @ Q.T                          # + dot-product residual


class BilinearHero:
    def __init__(self, m=None, rank=None, epochs=None, lr=None, wd=None, dropout=None,
                 seed=SEED, device=None):
        self.m = int(os.environ.get("BIL_M", m if m is not None else 128))
        self.rank = int(os.environ.get("BIL_RANK", rank if rank is not None else 0))
        self.epochs = int(os.environ.get("BIL_EPOCHS",
                                         os.environ.get("TT_EPOCHS", epochs if epochs is not None else 500)))
        self.lr = float(os.environ.get("BIL_LR", lr if lr is not None else 1e-3))
        self.wd = float(os.environ.get("BIL_WD", wd if wd is not None else 1e-4))
        self.dropout = float(os.environ.get("BIL_DROPOUT", dropout if dropout is not None else 0.2))
        self.seed = seed
        self.device = device or get_device()

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)
        dev = self.device
        Xl = torch.tensor(Clnc, device=dev)
        Xd = torch.tensor(Cdis, device=dev)
        self.tow_l = _Tower(Clnc.shape[1], self.m, self.dropout).to(dev)
        self.tow_d = _Tower(Cdis.shape[1], self.m, self.dropout).to(dev)
        self.inter = _Bilinear(self.m, self.rank).to(dev)

        tl = torch.tensor(train_lnc, device=dev, dtype=torch.long)
        td = torch.tensor(train_dis, device=dev, dtype=torch.long)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)], device=dev)
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        params = list(self.tow_l.parameters()) + list(self.tow_d.parameters()) \
            + list(self.inter.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        for _ in range(self.epochs):
            opt.zero_grad()
            E = self.tow_l(Xl); Q = self.tow_d(Xd)
            logits = self.inter(E[tl], Q[td])              # (|train_lnc|, |train_dis|)
            loss = bce(logits, Yblk)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())

        self.tow_l.eval(); self.tow_d.eval()               # dropout off for final embeddings
        with torch.no_grad():
            E = self.tow_l(Xl); Q = self.tow_d(Xd)
            self._S = torch.sigmoid(self.inter(E, Q)).cpu().numpy().astype(np.float32)
        return self

    def predict(self):
        return self._S


def build(device):
    return BilinearHero(device=device)
