"""Device util + simple disease-side reference predictors for true cold-start. Each produces a
full (n_lnc, n_dis) score matrix; only the cold disease columns are evaluated.

Interface:  m.fit(M_train, Cdis, train_idx, cold_idx);  S = m.predict()   # (n_lnc, n_dis)

M_train has cold columns already zeroed (truly cold). Cdis (n_dis,768) is intrinsic content,
available for ALL diseases incl. cold.

  RandomPredictor      - seeded random (floor)
  PopularityPredictor  - lncRNA frequency over training diseases (content-free CEILING = the wall)
  KNNContentPredictor  - cold disease profile = content-NN average of training profiles (content, non-generative)
"""
import numpy as np
import torch
import torch.nn as nn

SEED = 2026


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class RandomPredictor:
    def __init__(self, seed=SEED): self.seed = seed
    def fit(self, M_train, Cdis, train_idx, cold_idx):
        self.shape = M_train.shape; return self
    def predict(self):
        rng = np.random.default_rng(self.seed)
        return rng.random(self.shape).astype(np.float32)


class PopularityPredictor:
    """Content-free ceiling: every disease gets the same profile = lncRNA association frequency
    among TRAINING diseases. personalization == 0 by construction."""
    def fit(self, M_train, Cdis, train_idx, cold_idx):
        lnc_freq = M_train[:, train_idx].sum(axis=1)            # (n_lnc,)
        self.S = np.tile(lnc_freq[:, None], (1, M_train.shape[1])).astype(np.float32)
        return self
    def predict(self): return self.S


class KNNContentPredictor:
    """Cold disease profile = average of its k content-nearest TRAINING diseases' profiles."""
    def __init__(self, k=10): self.k = k
    def fit(self, M_train, Cdis, train_idx, cold_idx):
        self.M_train, self.Cdis = M_train, Cdis
        self.train_idx, self.cold_idx = train_idx, cold_idx
        Ctr = Cdis[train_idx]                                   # (n_train, d) assumed L2-normalized
        self.Ctr = Ctr / (np.linalg.norm(Ctr, axis=1, keepdims=True) + 1e-8)
        return self
    def predict(self):
        n_lnc, n_dis = self.M_train.shape
        S = np.tile(self.M_train[:, self.train_idx].sum(1, keepdims=True), (1, n_dis)).astype(np.float32)
        for d in self.cold_idx:
            c = self.Cdis[d] / (np.linalg.norm(self.Cdis[d]) + 1e-8)
            sim = self.Ctr @ c                                  # (n_train,)
            nn_local = np.argsort(-sim)[:self.k]
            w = np.maximum(sim[nn_local], 0) + 1e-8
            cols = self.train_idx[nn_local]
            prof = (self.M_train[:, cols] * w[None, :]).sum(1) / w.sum()
            S[:, d] = prof.astype(np.float32)
        return S
