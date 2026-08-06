"""Two-sided content models. Each fit(M, Clnc, Cdis, train_lnc, train_dis) trains ONLY on the
train_lnc x train_dis association sub-block, then predict() returns a full (240,412) score matrix
(content towers generalize to held-out rows/cols).

  Popularity2          : outer(lnc_freq, dis_freq) from train block (content-free; held-out node -> 0)
  TwoTowerContent      : e=MLP(Clnc), q=MLP(Cdis), score=<e,q>; weighted BCE. content_l/content_d
                         toggle each side (False -> free per-index embedding that CANNOT generalize to cold)
"""
import os
import numpy as np
import torch
import torch.nn as nn

from ccdiff_models import get_device
from ccdiff_common import SEED


class Popularity2:
    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        n_l, n_d = M.shape
        lf = np.zeros(n_l, np.float32); df = np.zeros(n_d, np.float32)
        blk = M[np.ix_(train_lnc, train_dis)]
        lf[train_lnc] = blk.sum(1); df[train_dis] = blk.sum(0)
        self.S = np.outer(lf, df).astype(np.float32)        # held-out node freq stays 0
        return self
    def predict(self): return self.S


class _Tower(nn.Module):
    def __init__(self, in_dim, m, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, m))
    def forward(self, x): return self.net(x)


class TwoTowerContent:
    def __init__(self, m=32, epochs=500, lr=1e-3, content_l=True, content_d=True,
                 seed=SEED, device=None):
        self.m = int(os.environ.get("TT_M", m))
        self.epochs = int(os.environ.get("TT_EPOCHS", epochs))
        self.lr = float(os.environ.get("TT_LR", lr))
        self.wd = float(os.environ.get("TT_WD", "0"))
        self.dropout = float(os.environ.get("TT_DROPOUT", "0"))
        self.content_l, self.content_d = content_l, content_d
        self.seed = seed; self.device = device or get_device()

    def _build(self, Clnc, Cdis):
        dev = self.device
        n_l, n_d = Clnc.shape[0], Cdis.shape[0]
        if self.content_l:
            self.tow_l = _Tower(Clnc.shape[1], self.m, self.dropout).to(dev); self.emb_l = None
        else:
            self.emb_l = nn.Embedding(n_l, self.m).to(dev); self.tow_l = None
        if self.content_d:
            self.tow_d = _Tower(Cdis.shape[1], self.m, self.dropout).to(dev); self.emb_d = None
        else:
            self.emb_d = nn.Embedding(n_d, self.m).to(dev); self.tow_d = None

    def _e(self, Xl):  return self.tow_l(Xl) if self.content_l else self.emb_l.weight
    def _q(self, Xd):  return self.tow_d(Xd) if self.content_d else self.emb_d.weight

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)
        dev = self.device
        self.Clnc, self.Cdis = Clnc, Cdis
        Xl = torch.tensor(Clnc, device=dev); Xd = torch.tensor(Cdis, device=dev)
        self._build(Clnc, Cdis)
        params = [p for mod in [self.tow_l, self.tow_d, self.emb_l, self.emb_d] if mod is not None
                  for p in mod.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        tl = torch.tensor(train_lnc, device=dev, dtype=torch.long)
        td = torch.tensor(train_dis, device=dev, dtype=torch.long)
        Yblk = torch.tensor(M[np.ix_(train_lnc, train_dis)], device=dev)
        pos_w = torch.tensor([(Yblk == 0).sum() / (Yblk.sum() + 1)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(self.epochs):
            opt.zero_grad()
            e = self._e(Xl); q = self._q(Xd)
            logits = e[tl] @ q[td].T                       # (|train_lnc|,|train_dis|)
            loss = bce(logits, Yblk)
            loss.backward(); opt.step()
        self.final_loss = float(loss.item())
        for mod in [self.tow_l, self.tow_d]:
            if mod is not None:
                mod.eval()                                  # dropout off for final embeddings
        with torch.no_grad():
            self.e = self._e(Xl).detach(); self.q = self._q(Xd).detach()
        return self

    @torch.no_grad()
    def predict(self):
        return torch.sigmoid(self.e @ self.q.T).cpu().numpy().astype(np.float32)
