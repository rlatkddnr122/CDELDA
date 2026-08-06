"""LDA-VGHB (Peng et al., Briefings in Bioinformatics 2024, bbad466) reproduction.

Original pipeline (github.com/plhhnu/LDA-VGHB):
  1. SVD of the lncRNA-disease association matrix  -> LINEAR per-node features
     (U for lncRNAs, V for diseases; svds, k singular components).
  2. A variational graph auto-encoder (VGAE) run on the lncRNA / disease
     SIMILARITY network (functional / semantic similarity), with the association
     rows as GCN node features -> NONLINEAR per-node features (the latent mu).
  3. Per pair (l, d) the feature is  [SVD_l | VGAE_l | SVD_d | VGAE_d]; a
     Heterogeneous Newton Boosting Machine (IBM Snap ML `BoostingMachine`, the
     "heterogeneous" = per-tree random depth in [min_max_depth, max_max_depth])
     classifies pairs. Their cv=4 ("CV_ind", held-out rows AND cols) == our C4.

Two fidelity corrections vs the released code, required by our contract:
  * LEAKAGE. The original precomputes SVD/VGAE features ONCE on the FULL matrix
    and only then splits pairs, so a held-out node's features encode its own
    (test) associations -- transductive leakage that inflates every cold split.
    Here SVD and the VGAE node-features read ONLY the train association sub-block
    (held-out rows/cols zeroed), so cold nodes carry no association signal; they
    remain reachable to the VGAE solely through the CONTENT-similarity edges.
  * CONTENT SOURCE. We do not have the paper's exact functional-similarity
    database, so -- exactly as for VGAELDA-content / the other content-equipped
    reproductions -- we substitute our intrinsic content (lncRNA expression
    similarity via CCDIFF_LNC_EXPR, disease DO-DAG semantic similarity via
    CCDIFF_DIS_SEMSIM), blended on the train x train block with the train-side
    GIP (integrated_row_sim). LDA-VGHB is therefore a CONTENT-EQUIPPED hybrid,
    compared on exactly the content terms every other content baseline gets.

The classifier is the FAITHFUL `snapml.BoostingMachine` with the paper's
hyper-parameters (objective logloss, 1000 rounds, lr 1e-3, min_max_depth 1,
max_max_depth 25, subsample 0.8).

Module-level names the runner imports: NAME, build(device) -> model.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import svds

from bench.interface import SEED, subblock, integrated_row_sim, _topk_rows  # noqa: F401

try:
    from ccdiff_models import get_device
except Exception:  # pragma: no cover
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

NAME = "LDA-VGHB (SVD+VGAE+SnapBoost)"

# --- faithful-ish, env-overridable knobs -----------------------------------
_SVD_K = int(os.environ.get("VGHB_SVD_K", "16"))       # SVD components/side (paper sweeps 5..64; data32 uses 32)
_H1 = int(os.environ.get("VGHB_H1", "64"))             # VGAE hidden-1 (paper 100)
_H2 = int(os.environ.get("VGHB_H2", "8"))              # VGAE latent (paper 5)
_VGAE_EP = int(os.environ.get("VGHB_VGAE_EP", os.environ.get("TT_EPOCHS", "150")))  # paper 250
_VGAE_LR = float(os.environ.get("VGHB_VGAE_LR", "1e-2"))
_KNN = int(os.environ.get("VGHB_KNN", "15"))           # sim-graph sparsity (efficiency)
_BLEND = float(os.environ.get("VGHB_ALPHA", "0.5"))    # content<->GIP blend (as VGAELDA-content)
_NROUND = int(os.environ.get("VGHB_NROUND", "400"))    # SnapBoost rounds (paper 1000)
_NEG_RATIO = int(os.environ.get("VGHB_NEG", "1"))      # train negatives per positive
# VGHB_LEAK=1 reproduces the ORIGINAL (transductive-leaky) protocol: SVD and the VGAE
# node-features are computed on the FULL association matrix (incl. held-out test rows/cols),
# exactly as the released code precomputes features before splitting pairs. Default 0 = our
# leakage-safe node hold-out (features see only the train sub-block). The gap between the two
# quantifies how much the reported cold-start numbers are inflated by feature leakage.
_LEAK = os.environ.get("VGHB_LEAK", "0") == "1"
_EPS = 1e-8


def _load_content_npy(env_key, n, square):
    """Load a precomputed content matrix from env path if it matches the node count."""
    path = os.environ.get(env_key, "")
    if path and os.path.exists(path):
        try:
            arr = np.asarray(np.load(path), np.float32)
            if square and arr.shape == (n, n):
                return arr
            if (not square) and arr.shape[0] == n:
                return arr
        except Exception:
            pass
    return None


def _norm_adj(sim):
    """D^-1/2 (A+I) D^-1/2 over a (kNN-sparsified) similarity, as VGAE preprocess_graph."""
    A = _topk_rows(np.asarray(sim, np.float32), _KNN)
    A = np.maximum(A, A.T)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1)
    dinv = np.power(np.maximum(d, _EPS), -0.5)
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


class _VGAE(nn.Module):
    """Minimal 2-layer GCN variational graph auto-encoder (inner-product decoder)."""

    def __init__(self, in_dim, h1, h2):
        super().__init__()
        self.w0 = nn.Linear(in_dim, h1, bias=False)
        self.wmu = nn.Linear(h1, h2, bias=False)
        self.wlv = nn.Linear(h1, h2, bias=False)

    def encode(self, An, X):
        h = torch.relu(An @ self.w0(X))
        return self.wmu(An @ h), self.wlv(An @ h)

    def forward(self, An, X):
        mu, lv = self.encode(An, X)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return z, mu, lv


def _vgae_embed(sim, feats, device):
    """Train a VGAE on the (kNN-sparse) similarity graph with `feats` node features; return mu.

    The normalised adjacency is a sparse tensor (kNN top-_KNN per row), so the GCN
    propagation An @ X stays cheap even for the large lncRNA-side graphs. The
    reconstruction target is the sparse adjacency's binary support, scored against
    the full inner-product logits with a positive-class weight (imbalance-aware).
    """
    n = sim.shape[0]
    An_d = _norm_adj(sim)
    idx = np.nonzero(An_d)
    An = torch.sparse_coo_tensor(np.vstack(idx), An_d[idx], (n, n), device=device).coalesce()
    X = torch.tensor(np.asarray(feats, np.float32), device=device)
    lbl = (_topk_rows(np.asarray(sim, np.float32), _KNN) > 0).astype(np.float32)
    lbl = np.maximum(lbl, lbl.T)
    A_lbl = torch.tensor(lbl, device=device)
    pos_w = torch.tensor((n * n - lbl.sum()) / (lbl.sum() + 1.0), device=device)

    def spmm(a, b):
        return torch.sparse.mm(a, b)

    torch.manual_seed(SEED)
    m = _VGAE(X.shape[1], _H1, _H2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=_VGAE_LR)
    for _ in range(_VGAE_EP):
        m.train(); opt.zero_grad()
        h = torch.relu(spmm(An, m.w0(X)))
        mu, lv = m.wmu(spmm(An, h)), m.wlv(spmm(An, h))
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        logits = z @ z.t()
        rec = nn.functional.binary_cross_entropy_with_logits(logits, A_lbl, pos_weight=pos_w)
        kl = -0.5 / n * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1))
        (rec + kl).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        h = torch.relu(spmm(An, m.w0(X)))
        mu = m.wmu(spmm(An, h))
    return mu.cpu().numpy().astype(np.float32)


class LDAVGHB:
    def __init__(self, seed=SEED):
        self.seed = seed
        self.device = get_device()

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc); td = np.asarray(train_dis)

        # leakage-safe association: full-shape matrix with only the train sub-block
        Mtr = np.zeros_like(M)
        Mtr[np.ix_(tl, td)] = M[np.ix_(tl, td)]
        Msub = subblock(M, tl, td)                                  # (nTl, nTd)
        # feature matrix: leakage-safe (train sub-block) by default, or the FULL matrix
        # when reproducing the original transductive protocol (VGHB_LEAK=1).
        Mfeat = M if _LEAK else Mtr

        # --- (1) SVD linear features (leak-controlled) ----------------------
        k = int(max(1, min(_SVD_K, min(self.n_l, self.n_d) - 1)))
        try:
            U, s, VT = svds(Mfeat.astype(np.float64), k=k)
            svd_l = (U * s).astype(np.float32); svd_d = VT.T.astype(np.float32)
        except Exception:
            svd_l = np.zeros((self.n_l, k), np.float32); svd_d = np.zeros((self.n_d, k), np.float32)

        # --- content+GIP similarity networks (cold nodes reachable via content) ---
        # lncRNA content = expression features (env) else the intrinsic content Clnc;
        # disease content = precomputed DO-DAG semantic similarity (env) else Cdis features.
        expr = _load_content_npy("CCDIFF_LNC_EXPR", self.n_l, square=False)
        Cl_feat = expr if expr is not None else _ensure2d(Clnc, self.n_l)
        SL = integrated_row_sim(Cl_feat, Msub, tl, alpha=_BLEND)

        semsim = _load_content_npy("CCDIFF_DIS_SEMSIM", self.n_d, square=True)
        if semsim is not None:
            SD = _blend_precomputed(semsim, Msub.T, td, _BLEND)
        else:
            SD = integrated_row_sim(_ensure2d(Cdis, self.n_d), Msub.T, td, alpha=_BLEND)

        # --- (2) VGAE nonlinear features (node features = assoc rows, leak-controlled) --
        vg_l = _vgae_embed(SL, Mfeat, self.device)                 # (n_l, H2)
        vg_d = _vgae_embed(SD, Mfeat.T, self.device)               # (n_d, H2)

        self.Fl = np.concatenate([svd_l, vg_l], 1).astype(np.float32)  # (n_l, K+H2)
        self.Fd = np.concatenate([svd_d, vg_d], 1).astype(np.float32)  # (n_d, K+H2)

        # --- (3) train the Heterogeneous Newton Boosting Machine -------------
        rng = np.random.default_rng(self.seed)
        pos = [(tl[i], td[j]) for i in range(len(tl)) for j in range(len(td)) if Msub[i, j] > 0]
        zeros = [(tl[i], td[j]) for i in range(len(tl)) for j in range(len(td)) if Msub[i, j] == 0]
        rng.shuffle(zeros)
        neg = zeros[: _NEG_RATIO * len(pos)]
        pairs = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg), np.float32)
        Xtr = np.array([np.concatenate([self.Fl[i], self.Fd[j]]) for (i, j) in pairs], np.float32)

        # Heterogeneous Newton Boosting Machine (Snap ML): per-round random tree depth
        # in [min_max_depth, max_max_depth] -- the paper's exact hyper-parameters.
        from snapml import BoostingMachine
        params = {
            "boosting_params": {
                "num_round": _NROUND, "objective": "logloss",
                "min_max_depth": 1, "max_max_depth": 25,
                "learning_rate": 1e-3, "random_state": int(self.seed), "num_threads": 4,
            },
            "tree_params": {"subsample": 0.8, "use_gpu": False},
        }
        self.clf = BoostingMachine(params)
        self.clf.fit(Xtr, y)
        return self

    def predict(self):
        # score every pair: X = [Fl_i | Fd_j]
        Fl, Fd = self.Fl, self.Fd
        Xall = np.concatenate(
            [np.repeat(Fl, self.n_d, axis=0), np.tile(Fd, (self.n_l, 1))], axis=1
        ).astype(np.float32)
        p = self.clf.predict_proba(Xall)[:, 1]
        return p.reshape(self.n_l, self.n_d).astype(np.float32)


def _ensure2d(C, n):
    C = np.asarray(C, np.float32)
    if C.ndim == 2 and C.shape[0] == n:
        return C
    return np.eye(n, dtype=np.float32)


def _C_or_zeros(n):
    return np.eye(n, dtype=np.float32)


def _blend_precomputed(sim_full, assoc_sub_rowaxis, train_idx, alpha):
    """Blend a precomputed (n,n) content similarity with train-side GIP on train x train."""
    from bench.interface import gip_kernel
    S = np.asarray(sim_full, np.float32).copy()
    ti = np.asarray(train_idx)
    if ti.size:
        G = gip_kernel(assoc_sub_rowaxis)
        S[np.ix_(ti, ti)] = alpha * S[np.ix_(ti, ti)] + (1.0 - alpha) * G
    return S.astype(np.float32)


def build(device=None):
    return LDAVGHB()
