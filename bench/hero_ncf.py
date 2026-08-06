"""Neural-interaction two-tower HERO (NCF / NeuMF-style, benchmark-native, self-contained).

Two content towers produce inductive embeddings e = tower_l(Clnc) (n_l x m) and
q = tower_d(Cdis) (n_d x m) -- the SAME _Tower MLP as TwoTowerContent. The novelty
is the scoring head: instead of a bare dot product <e_i, q_j> (dual-tower) or a
bilinear attention fusion (dattn), each pair (i, j) is scored by a small NEURAL
interaction MLP over an explicit pairwise feature:

        feat(i, j) = [ e_i , q_j , e_i * q_j , |e_i - q_j| ]   (4m-dim)
        logit(i, j) = head(feat)  =  Linear(4m -> m) -> ReLU -> Dropout -> Linear(m -> 1)

The elementwise product e_i*q_j is the classic NCF "GMF" term; the concat + abs-diff
give the MLP room to learn a non-linear, non-symmetric matching function that a dot
product cannot represent. Cold lncRNA/disease nodes get embeddings straight from their
content (inductive) -> the head still scores them (no popularity floor).

TRAINING: BCE on the train sub-block only. Each epoch we take ALL positives of
M[ix_(train_lnc, train_dis)] plus an equal number of randomly sampled train-zero
negatives, run their pair features through the head, and minimise BCEWithLogitsLoss
(pos_weight for residual imbalance). Supervision touches ONLY the train sub-block.

PREDICT: every (n_l, n_d) pair is scored through the head, but ROW-CHUNKED so the
(chunk x n_d x 4m) feature tensor never blows up memory (runs 5102x245 comfortably).
Returns a finite (n_l, n_d) float32 sigmoid-probability matrix.

Contract: fit reads labels ONLY from M[ix_(train_lnc, train_dis)]; content is intrinsic
(all nodes). Sub-block invariant: nothing depends on off-block M (verified ~0).

Tunable via env: NCF_M (128), NCF_LR (1e-3), NCF_WD (1e-4),
NCF_EPOCHS (500, fallback TT_EPOCHS), NCF_DROPOUT (0.2), NCF_CHUNK (256). Seeded with SEED.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device        # snapshot_src (on bench path)
from ccdiff_common import SEED
from twoside_models import _Tower           # SAME MLP tower as TwoTowerContent

NAME = "TwoTower-NCF (content)"


class _InteractionHead(nn.Module):
    """NeuMF-style neural interaction over [e, q, e*q, |e-q|] -> scalar logit."""

    def __init__(self, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * m, m), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(m, 1),
        )

    def forward(self, e, q):
        # e, q broadcastable to (..., m). Returns (...,) logits.
        prod = e * q                                  # broadcasts to common (..., m)
        diff = torch.abs(e - q)
        e, q = torch.broadcast_tensors(e, q)          # match concat dims for e, q too
        feat = torch.cat([e, q, prod, diff], dim=-1)
        return self.net(feat).squeeze(-1)


class NCFHero:
    def __init__(self, m=None, epochs=None, lr=None, wd=None, dropout=None,
                 chunk=None, seed=SEED, device=None):
        self.m = int(os.environ.get("NCF_M", m if m is not None else 128))
        self.epochs = int(os.environ.get("NCF_EPOCHS",
                          os.environ.get("TT_EPOCHS", epochs if epochs is not None else 500)))
        self.lr = float(os.environ.get("NCF_LR", lr if lr is not None else 1e-3))
        self.wd = float(os.environ.get("NCF_WD", wd if wd is not None else 1e-4))
        self.dropout = float(os.environ.get("NCF_DROPOUT", dropout if dropout is not None else 0.2))
        self.chunk = int(os.environ.get("NCF_CHUNK", chunk if chunk is not None else 256))
        self.seed = int(seed)
        self.device = device or get_device()

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)
        dev = self.device
        Xl = torch.tensor(np.asarray(Clnc, np.float32), device=dev)
        Xd = torch.tensor(np.asarray(Cdis, np.float32), device=dev)
        tl = torch.tensor(np.asarray(train_lnc), device=dev, dtype=torch.long)
        td = torch.tensor(np.asarray(train_dis), device=dev, dtype=torch.long)

        self.tow_l = _Tower(Xl.shape[1], self.m, self.dropout).to(dev)
        self.tow_d = _Tower(Xd.shape[1], self.m, self.dropout).to(dev)
        self.head = _InteractionHead(self.m, self.dropout).to(dev)
        params = list(self.tow_l.parameters()) + list(self.tow_d.parameters()) \
            + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        # --- supervision: ONLY the train sub-block ---------------------------
        Yblk = np.asarray(M)[np.ix_(np.asarray(train_lnc), np.asarray(train_dis))]
        pos_a, pos_b = np.nonzero(Yblk > 0.5)            # local (row, col) of positives
        zero_a, zero_b = np.nonzero(Yblk <= 0.5)         # local (row, col) of train zeros
        n_pos = pos_a.shape[0]
        pos_a_t = torch.tensor(pos_a, device=dev, dtype=torch.long)
        pos_b_t = torch.tensor(pos_b, device=dev, dtype=torch.long)
        n_zero = zero_a.shape[0]
        # residual imbalance safety (balanced sampling makes this ~1)
        pos_w = torch.tensor([max(n_zero, 1) / max(n_pos, 1)], device=dev, dtype=torch.float32)
        pos_w = torch.clamp(pos_w, max=1.0)              # balanced batch -> keep it mild
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        rng = np.random.default_rng(self.seed)

        # degenerate guard: no positives -> nothing to learn, leave nets at init
        if n_pos == 0 or n_zero == 0:
            self.tow_l.eval(); self.tow_d.eval(); self.head.eval()
            return self

        for _ in range(self.epochs):
            opt.zero_grad()
            e = self.tow_l(Xl)[tl]                        # (nTl, m) train-lnc embeddings
            q = self.tow_d(Xd)[td]                        # (nTd, m) train-dis embeddings
            # equal number of sampled train-zero negatives
            neg_sel = rng.integers(0, n_zero, size=n_pos)
            neg_a = torch.tensor(zero_a[neg_sel], device=dev, dtype=torch.long)
            neg_b = torch.tensor(zero_b[neg_sel], device=dev, dtype=torch.long)
            e_pair = torch.cat([e[pos_a_t], e[neg_a]], 0)
            q_pair = torch.cat([q[pos_b_t], q[neg_b]], 0)
            logits = self.head(e_pair, q_pair)
            labels = torch.cat([torch.ones(n_pos, device=dev),
                                torch.zeros(n_pos, device=dev)], 0)
            loss = bce(logits, labels)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())

        self.tow_l.eval(); self.tow_d.eval(); self.head.eval()
        with torch.no_grad():
            self.e_all = self.tow_l(Xl).detach()          # (n_l, m)
            self.q_all = self.tow_d(Xd).detach()          # (n_d, m)
        return self

    @torch.no_grad()
    def predict(self):
        dev = self.device
        e = self.e_all; q = self.q_all                    # (n_l,m), (n_d,m)
        n_l, n_d = e.shape[0], q.shape[0]
        S = np.empty((n_l, n_d), np.float32)
        q_exp = q.unsqueeze(0)                            # (1, n_d, m)
        for i0 in range(0, n_l, self.chunk):
            i1 = min(i0 + self.chunk, n_l)
            e_chunk = e[i0:i1].unsqueeze(1)               # (c, 1, m) -> broadcasts to (c, n_d, m)
            logits = self.head(e_chunk, q_exp)            # (c, n_d)
            S[i0:i1] = torch.sigmoid(logits).float().cpu().numpy()
        return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build(device):
    return NCFHero(device=device)
