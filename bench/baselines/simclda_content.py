"""SIMCLDA -- CONTENT-EQUIPPED variant (semantic + expression similarity + GIP).

This is the native `bench/baselines/simclda.py` reproduction copied EXACTLY, with
ONE change: the SIMILARITY SOURCE the pca_energy eigen-features are computed from.
The native reproduction is content-blind and derives its primary linear features
from GIP(Msub) (lncRNA side) and GIP(Msub.T) (disease side) because the paper's
functional / semantic / expression databases were declared off-limits -- so cold
(held-out) nodes get an empty GIP profile -> zero feature -> S = 0 (the honest
cold collapse). This variant DELIBERATELY restores the paper's LITERAL intrinsic
content -- lncRNA EXPRESSION similarity and disease DO-DAG SEMANTIC similarity --
blended with the train-subblock GIP, so we can test whether the both-cold (C4)
collapse persists when the baseline is NOT content-starved.

Everything ELSE is IDENTICAL to the native reproduction: the pca_energy truncated
eigen-features (0.8 lncRNA / 0.6 disease energy thresholds, right singular vectors
== leading eigenvectors, no sigma scaling), the inductive matrix completion
X = A Z B^T solved by matrix-free ridge conjugate gradient (lambda=1), and the
FULL-TRAINING-FOLD observation mask (known 0-associations are the training
negatives; NOT reverted to positives-only).

BLEND (the ONLY deviation from the native GIP-only reproduction):
    SL = alpha * SL_con + (1 - alpha) * SL_gip        # lncRNA side
    SD = alpha * SD_con + (1 - alpha) * SD_gip        # disease side
    alpha = float(os.environ.get("SIMCLDA_ALPHA", "0.5"))
where
    SL_gip = gip_kernel(Msub)      (train association sub-block ONLY, as native)
    SD_gip = gip_kernel(Msub.T)    (train association sub-block ONLY, as native)
    SL_con = content_cosine(expr)  full (n_l, n_l), with
             expr = np.load(os.environ["CCDIFF_LNC_EXPR"]) if that env path is a
             valid (n_l, k) .npy, else the intrinsic content matrix Clnc.
    SD_con = semsim  full (n_d, n_d), with
             semsim = np.load(os.environ["CCDIFF_DIS_SEMSIM"]) if that env path is
             a valid (n_d, n_d) .npy (a precomputed DO-DAG semantic-similarity
             matrix), else content_cosine(Cdis).

Because SL_con / SD_con are M-INDEPENDENT and defined for EVERY node, the blended
similarity is FULL-NODE (n_l x n_l / n_d x n_d): the GIP term is added only on the
train x train block, content everywhere else. The pca_energy eigen-features are
then computed from these FULL matrices, so COLD rows/cols carry nonzero
content-driven features -> nonzero scores. That is the INTENDED effect of
equipping the baseline with content.

Sub-block invariance is PRESERVED: SL_con / SD_con are M-independent, and the GIP
terms read strictly the TRAIN association sub-block -- so scrambling any entry
outside M[np.ix_(train_lnc, train_dis)] leaves Msub, the GIP kernels, the blended
similarities, the eigen-features and the CG solve all unchanged -> predict() is
invariant (max|delta| ~ 0).
"""
import os

import numpy as np

try:                                    # top-r eigenpairs directly when available
    from scipy.linalg import eigh as _sp_eigh
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover - numpy fallback
    _HAVE_SCIPY = False

# Shared, leakage-safe helpers. Canonical import form (paper dir is on sys.path).
# Unlike the native (content-blind) SIMCLDA, content_cosine IS imported here on
# purpose: this variant is content-EQUIPPED.
from bench.interface import SEED, subblock, gip_kernel, content_cosine   # noqa: F401

NAME = "SIMCLDA-content (semsim+expr)"

_LAM = 1.0        # IMC ridge regularisation on Z (SIMC default lambda=1)
_ENERGY_L = 0.8   # pca_energy threshold for lncRNA similarity  (SIMCLDA_demo.m)
_ENERGY_D = 0.6   # pca_energy threshold for disease similarity (SIMCLDA_demo.m)
_CG_TOL = 1e-8    # CG relative-residual stop
_CG_MAXIT = 500   # CG iteration cap


def _load_content_npy(env_key, n, square):
    """Load an M-INDEPENDENT content/similarity matrix from an env-pointed .npy.

    Returns a validated float32 array, or None (caller falls back to intrinsic
    content). For the lncRNA side (square=False) a valid file is a (n, k)
    expression matrix; for the disease side (square=True) a valid file is a
    precomputed (n, n) semantic-similarity matrix.
    """
    path = os.environ.get(env_key)
    if not path or not os.path.isfile(path):
        return None
    try:
        arr = np.asarray(np.load(path), np.float32)
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    if square:
        if arr.shape != (n, n):
            return None
    else:
        if arr.shape[0] != n or arr.shape[1] < 1:
            return None
    return arr


def _pca_energy_features(K_full, energy, train_idx):
    """Primary linear features = pca_energy(K_full, energy) of a FULL similarity.

    Faithful to the original MATLAB pca_energy: truncated SVD keeping the fewest
    leading singular components whose cumulative energy reaches `energy`*total,
    returning the RIGHT SINGULAR VECTORS (no sigma scaling). For a symmetric
    similarity the SVD coincides (up to sign) with the eigen-decomposition, so we
    take the top-k eigenvectors and select k by the energy criterion.

    UNLIKE the native (content-blind) SIMCLDA -- whose kernel was the train-side
    GIP of size (nT, nT), giving cold rows a zero feature -- here K_full is the
    FULL-node blended similarity (n, n). The train rows are sliced out for the IMC
    solve; the full matrix Xful carries nonzero (content-driven) features for cold
    rows too.

    K_full     : (n, n) full-node blended similarity (rows == all nodes).
    energy     : cumulative singular-value energy fraction in (0, 1].
    train_idx  : positions of the train nodes into the full-axis feature matrix.
    Returns
      Xtr  : (nT, k) right singular vectors of the train rows (unscaled).
      Xful : (n,  k) full matrix; every row (train AND cold) is populated.
    """
    n = K_full.shape[0]
    Ksym = 0.5 * (np.asarray(K_full, np.float64) + np.asarray(K_full, np.float64).T)  # enforce symmetry
    if _HAVE_SCIPY:
        vals, vecs = _sp_eigh(Ksym)                      # ascending eigenpairs
    else:                                                # pragma: no cover
        vals, vecs = np.linalg.eigh(Ksym)
    order = np.argsort(vals)[::-1]                        # descending (== SVD sigma order)
    vals = np.clip(vals[order], 0.0, None)               # SPD -> sigma == eigenvalue
    vecs = vecs[:, order]
    total = float(vals.sum()) + 1e-30
    csum = np.cumsum(vals)
    # fewest leading components reaching the energy fraction (>=1, <= n)
    k = int(np.searchsorted(csum, energy * total) + 1)
    k = max(1, min(k, n))
    Xful = np.asarray(vecs[:, :k], np.float64)           # (n, k) full-node features (cold != 0)
    Xtr = np.asarray(Xful[np.asarray(train_idx)], np.float64)  # (nT, k) train rows
    return Xtr, Xful


def _cg_imc(X, Y, Msub, lam, tol, maxit):
    """Iterative IMC solve of  min_W sum_{train fold} (Msub - X W Y^T)^2 + lam||W||^2.

    Observed set = the whole train fold (known 0-associations are the training
    negatives; a positives-only objective is ill-posed and inverts the ranking).
    Solved by matrix-free conjugate gradient on the SPD normal-equation operator
        L(W) = X^T ( X W Y^T ) Y + lam W .
    Returns (W (p,q), n_iters, final_relative_residual). IDENTICAL to native.
    """
    p = X.shape[1]
    q = Y.shape[1]
    # Observed set in a fold-CV benchmark = the WHOLE train fold: the known
    # 0-associations are the training NEGATIVES (a positives-only P_Omega is
    # ill-posed here -> inverted ranking). CG still solves it ITERATIVELY.
    Mask = np.ones_like(Msub, np.float64)                # full training-fold support
    b = X.T @ (Mask * Msub) @ Y                          # A^T m over the observed fold

    def L(W):
        recon = X @ W @ Y.T                              # (nTl, nTd) dense reconstruction
        return X.T @ (Mask * recon) @ Y + lam * W        # masked -> P_Omega objective

    W = np.zeros((p, q), np.float64)
    R = b - L(W)                                         # == b (W0 = 0)
    P = R.copy()
    rs = float(np.sum(R * R))
    b_norm = float(np.sqrt(np.sum(b * b))) + 1e-30
    it = 0
    final_res = np.sqrt(rs) / b_norm
    for it in range(1, int(maxit) + 1):
        LP = L(P)
        denom = float(np.sum(P * LP)) + 1e-30
        alpha = rs / denom
        W = W + alpha * P
        R = R - alpha * LP
        rs_new = float(np.sum(R * R))
        final_res = np.sqrt(rs_new) / b_norm
        if final_res <= tol:
            break
        P = R + (rs_new / (rs + 1e-30)) * P
        rs = rs_new
    return W, it, final_res


class _SimCLDAContent:
    """Inductive matrix completion on eigen-features of a (content + train-GIP) blend."""

    def __init__(self, device="cpu"):
        self.device = device
        self.rng = np.random.default_rng(SEED)   # form only; result is deterministic
        # diagnostics populated by fit() (used by the isolated self-test)
        self.info = {}

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc)
        td = np.asarray(train_dis)

        Msub = subblock(M, tl, td).astype(np.float64)    # (nTl, nTd) ONLY supervision
        nTl, nTd = Msub.shape

        # Empty-train (or all-zero) guard -> honest all-zero floor.
        if nTl == 0 or nTd == 0 or not np.any(Msub):
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            self.info = {"nTl": nTl, "nTd": nTd, "p": 0, "q": 0,
                         "n_obs": 0, "cg_iters": 0, "cg_res": 0.0}
            return self

        # --- (1) GIP kernels (train association sub-block ONLY; as native) -----
        SL_gip = gip_kernel(Msub).astype(np.float64)     # (nTl, nTl)
        SD_gip = gip_kernel(Msub.T).astype(np.float64)   # (nTd, nTd)

        # --- CONTENT similarities (M-INDEPENDENT; literal intrinsic content) ---
        # lncRNA side: expression matrix -> cosine; disease side: DO-DAG semsim.
        expr = _load_content_npy("CCDIFF_LNC_EXPR", self.n_l, square=False)
        SL_con = np.asarray(
            content_cosine(expr) if expr is not None else content_cosine(Clnc), np.float64
        )                                                # (n_l, n_l)
        ss = _load_content_npy("CCDIFF_DIS_SEMSIM", self.n_d, square=True)
        SD_con = np.asarray(
            ss if ss is not None else content_cosine(Cdis), np.float64
        )                                                # (n_d, n_d)

        # --- (2) BLEND (the ONLY deviation from native GIP-only) ---------------
        # FULL-node similarity: content everywhere, GIP added on the train block.
        # SL = alpha*SL_con + (1-alpha)*SL_gip on train x train; content elsewhere.
        alpha = float(os.environ.get("SIMCLDA_ALPHA", "0.5"))
        SL = SL_con.copy()
        SL[np.ix_(tl, tl)] = alpha * SL_con[np.ix_(tl, tl)] + (1.0 - alpha) * SL_gip
        SD = SD_con.copy()
        SD[np.ix_(td, td)] = alpha * SD_con[np.ix_(td, td)] + (1.0 - alpha) * SD_gip

        # --- pca_energy primary linear features from the FULL blended similarity
        Xtr, Xful = _pca_energy_features(SL, _ENERGY_L, tl)   # (nTl,p),(n_l,p)
        Ytr, Yful = _pca_energy_features(SD, _ENERGY_D, td)   # (nTd,q),(n_d,q)

        # --- (3) iterative IMC over OBSERVED entries (masked CG; IDENTICAL) -----
        W, cg_it, cg_res = _cg_imc(Xtr, Ytr, Msub, _LAM, _CG_TOL, _CG_MAXIT)

        # --- (4) full-node scores; COLD rows/cols carry content features -> != 0
        S = Xful @ W @ Yful.T                            # (n_l, n_d)
        self.S = np.nan_to_num(
            S, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        self.info = {"nTl": nTl, "nTd": nTd, "p": Xtr.shape[1], "q": Ytr.shape[1],
                     "n_obs": int(np.count_nonzero(Msub)),
                     "n_block": int(Msub.size),
                     "cg_iters": int(cg_it), "cg_res": float(cg_res)}
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _SimCLDAContent(device)
