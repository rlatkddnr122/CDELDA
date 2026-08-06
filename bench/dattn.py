"""Dual-attention two-tower HERO (benchmark-native, self-contained).

Bidirectional cross-attention: disease embeddings attend to (train) lncRNA embeddings and
lncRNA embeddings attend to (train) disease embeddings; keys/values restricted to TRAIN entities
so a cold entity attends only to seen entities of the other modality (clean inductive, content-only,
no label leakage). Fused via concat + LayerNorm-MLP; score = sigmoid(Fl @ Fd^T).

Contract: fit reads ONLY M[ix_(train_lnc,train_dis)] as labels; content Clnc/Cdis is intrinsic
(all nodes). predict -> (n_l,n_d) float32 finite. Cold nodes get content-based embeddings (no floor
collapse -- that's the point). Sub-block invariant (verified 0.0): nothing depends on off-block M.

Tunable via env (for the wandb sweep): DATTN_H, DATTN_HEADS, DATTN_TEMP, TT_EPOCHS, TT_LR, TT_WD.
Mirrors ccdiff/src/twoside_attn_models.py:DualAttnScorer, copied here to stay on the bench path.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device      # snapshot_src (on bench path)
from ccdiff_common import SEED


class _DualAttnFusion(nn.Module):
    def __init__(self, dl, dd, h=128, heads=4):
        super().__init__()
        self.h, self.heads, self.dh = h, heads, h // heads
        self.pl = nn.Linear(dl, h); self.pd = nn.Linear(dd, h)
        self.Wq_d = nn.Linear(h, h); self.Wk_l = nn.Linear(h, h); self.Wv_l = nn.Linear(h, h)
        self.Wq_l = nn.Linear(h, h); self.Wk_d = nn.Linear(h, h); self.Wv_d = nn.Linear(h, h)
        self.fuse_d = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.fuse_l = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))

    def _mha(self, Q, K, V):
        nq, nk = Q.shape[0], K.shape[0]
        q = Q.view(nq, self.heads, self.dh).transpose(0, 1)
        k = K.view(nk, self.heads, self.dh).transpose(0, 1)
        v = V.view(nk, self.heads, self.dh).transpose(0, 1)
        a = torch.softmax(q @ k.transpose(1, 2) / (self.dh ** 0.5), dim=-1)
        return (a @ v).transpose(0, 1).reshape(nq, self.h)

    def forward(self, A, B, ref_l, ref_d):
        Ap, Bp = torch.relu(self.pl(A)), torch.relu(self.pd(B))
        attn_d = self._mha(self.Wq_d(Bp), self.Wk_l(Ap[ref_l]), self.Wv_l(Ap[ref_l]))
        attn_l = self._mha(self.Wq_l(Ap), self.Wk_d(Bp[ref_d]), self.Wv_d(Bp[ref_d]))
        Fd = self.fuse_d(torch.cat([Bp, attn_d], 1))
        Fl = self.fuse_l(torch.cat([Ap, attn_l], 1))
        return Fl, Fd


class DualAttnHero:
    def __init__(self, h=None, heads=None, epochs=None, lr=None, wd=None, temp=None,
                 seed=SEED, device=None):
        self.h = int(os.environ.get("DATTN_H", h if h is not None else 128))
        self.heads = int(os.environ.get("DATTN_HEADS", heads if heads is not None else 4))
        self.epochs = int(os.environ.get("TT_EPOCHS", epochs if epochs is not None else 500))
        self.lr = float(os.environ.get("TT_LR", lr if lr is not None else 1e-3))
        self.wd = float(os.environ.get("TT_WD", wd if wd is not None else 0.0))
        self.temp = float(os.environ.get("DATTN_TEMP", temp if temp is not None else 1.0))
        self.seed, self.device = seed, device or get_device()

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed); dev = self.device
        A = torch.tensor(Clnc, device=dev); B = torch.tensor(Cdis, device=dev)
        rl = torch.tensor(train_lnc, device=dev, dtype=torch.long)
        rd = torch.tensor(train_dis, device=dev, dtype=torch.long)
        self.net = _DualAttnFusion(Clnc.shape[1], Cdis.shape[1], self.h, self.heads).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)], device=dev)
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(self.epochs):
            opt.zero_grad()
            Fl, Fd = self.net(A, B, rl, rd)
            logits = (Fl[rl] @ Fd[rd].T) / self.temp
            loss = bce(logits, Yblk)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())
        with torch.no_grad():
            Fl, Fd = self.net(A, B, rl, rd)
            self._S = torch.sigmoid((Fl @ Fd.T) / self.temp).cpu().numpy().astype(np.float32)
        return self

    def predict(self):
        return self._S
