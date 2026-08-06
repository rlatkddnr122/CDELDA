"""DSCMF -- CONTENT-EQUIPPED variant (semantic + expression similarity + GIP).

This is the native `bench/baselines/dscmf.py` reproduction copied EXACTLY, with
ONE change: the collaborative-similarity SOURCE fed to the DSCMF factorization.
The native reproduction is content-blind: it forces the paper's integration
weight alpha = 0, so the collaborative kernels degrade to K_l = GIP_l and
K_d = GIP_d built strictly from the TRAIN association sub-block (Clnc / Cdis are
received per the contract but NEVER read), and every cold row/col stays at the 0
floor -- an honest collapse.

This variant DELIBERATELY restores the paper's LITERAL intrinsic content
(DSCMF Eq. 8-9):

    K_l = alpha * S_l + (1 - alpha) * GIP_l ,   S_l = lncRNA EXPRESSION similarity
    K_d = alpha * S_d + (1 - alpha) * GIP_d ,   S_d = disease DO-DAG SEMANTIC sim.

so we can test whether the both-cold (C4) collapse persists when the baseline is
NOT content-starved. Everything else -- WKNKN preimputation (K=5, p=0.7), the
SVD initialization (Eq.13), the dual L2,1-reweighted Tikhonov + CMF ALS update
rules (Eq.14 / Eq.15, 100 fixed sweeps) -- is IDENTICAL to the native
reproduction.

BLEND (the ONLY deviation from the native GIP-only reproduction):
    SL = alpha * SL_con + (1 - alpha) * SL_gip        # lncRNA side  (n_l x n_l)
    SD = alpha * SD_con + (1 - alpha) * SD_gip         # disease side (n_d x n_d)
    alpha = float(os.environ.get("DSCMF_ALPHA", "0.5"))
where
    SL_gip : (n_l, n_l), gip_kernel(Msub) placed on the train x train block, ZEROS
             elsewhere -- i.e. the native train-sub-block GIP, embedded full-size.
    SD_gip : (n_d, n_d), gip_kernel(Msub.T) on the train x train block, zeros else.
    SL_con : (n_l, n_l), M-INDEPENDENT, defined for ALL nodes (incl. cold):
             content_cosine(expr) with expr = np.load(os.environ["CCDIFF_LNC_EXPR"])
             if that env path is a valid (n_l, k) .npy, else content_cosine(Clnc).
    SD_con : (n_d, n_d), M-INDEPENDENT, defined for ALL nodes (incl. cold):
             semsim = np.load(os.environ["CCDIFF_DIS_SEMSIM"]) if that env path is a
             valid (n_d, n_d) .npy (a precomputed DO-DAG semantic-similarity
             matrix), else content_cosine(Cdis).

Because SL / SD are FULL (n_l x n_l) / (n_d x n_d) and are fed as the KL / KD
collaborative-similarity matrices into the SAME DSCMF ALS, the factorization is
full-size (A is n_l x k, B is n_d x k). A cold row of A carries no observed
association (its Y row is 0) but is pulled by the CMF term lam_l * (KL @ A) toward
the content-similar TRAIN rows, so cold rows/cols now receive NONZERO scores --
that is the INTENDED effect of equipping the baseline with content.

Sub-block invariance is PRESERVED (max|Delta| <= 1e-4): SL_con / SD_con are
M-INDEPENDENT; SL_gip / SD_gip read strictly the TRAIN association sub-block and
are zero elsewhere; the WKNKN-imputed Y is zero off the train x train block. So
scrambling ANY entry outside M[np.ix_(train_lnc, train_dis)] cannot change any of
SL, SD, Y -> predict() is invariant to off-sub-block content.
"""
import os

import numpy as np

# Leakage-safe shared helpers + the global seed. `subblock` reads the train
# association sub-block; gip_kernel builds the train-side GIP; content_cosine is
# the intrinsic-content similarity this variant is allowed to use.
from bench.interface import subblock, gip_kernel, content_cosine, SEED   # noqa: F401

NAME = "DSCMF-content (semsim+expr)"

# Fixed constants -- IDENTICAL to the native reproduction (deterministic solve).
_K_WKNKN = 5        # WKNKN neighbourhood size (paper: K=5)
_P = 0.7            # WKNKN rank decay (paper: p=0.7)
_LAM_H = 1.0        # Tikhonov + L2,1 weight lam_h (paper 2^0=1 default)
_LAM_L = 0.1        # lncRNA-side collaborative weight lam_l
_LAM_D = 0.1        # disease-side collaborative weight lam_d
_N_ITER = 100       # fixed ALS sweeps (paper: max 100 iterations)
_RANK_CAP = 50      # rank k cap, clamped to min(nTl, nTd) for tiny-data safety
_EPS = 1e-8         # denominator / division guard


def _load_content_npy(env_key, n, square):
    """Load an M-INDEPENDENT content/similarity matrix from an env-pointed .npy.

    Returns a validated float32 array, or None (caller falls back to intrinsic
    content). For the lncRNA side (square=False) a valid file is an (n, k)
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


def _wknkn(Y, KL, KD, K=_K_WKNKN, p=_P):
    """WKNKN preimputation of a {0,1} sub-block Y (rows=lncRNA, cols=disease).

    IDENTICAL to the native reproduction. KL (row-side) and KD (col-side) are the
    similarity kernels. Returns Y softened to [0,1]: the mean of the row-based,
    column-based, and combined weighted-nearest-known-neighbour estimates, with
    observed 1s kept (max with Y). Uses only Y and its kernels -> leakage-safe.
    """
    Y = np.asarray(Y, np.float64)
    nl, nd = Y.shape
    if nl == 0 or nd == 0:
        return Y.copy()

    row_known = Y.sum(1) > 0            # lncRNAs with >=1 observed association
    col_known = Y.sum(0) > 0           # diseases with >=1 observed association

    def _weights(S, known):
        """(n,n) sparse weight rows: K nearest KNOWN neighbours, decayed & normalised."""
        n = S.shape[0]
        W = np.zeros((n, n), np.float64)
        kidx = np.where(known)[0]
        if kidx.size == 0:
            return W
        for i in range(n):
            sims = np.asarray(S[i], np.float64).copy()
            sims[i] = -np.inf                      # exclude self
            cand = kidx[kidx != i]
            if cand.size == 0:
                continue
            cs = sims[cand]
            valid = cs > 0
            if not valid.any():
                continue
            cand, cs = cand[valid], cs[valid]
            order = np.argsort(-cs, kind="stable")[:K]     # K nearest known
            nb, ss = cand[order], cs[order]
            w = (p ** np.arange(nb.size)) * ss             # rank decay * similarity
            Z = ss.sum() + _EPS                            # normaliser = sum of sims
            W[i, nb] = w / Z
        return W

    Wd = _weights(KL, row_known)        # (nl, nl) row-neighbour weights
    Wt = _weights(KD, col_known)        # (nd, nd) col-neighbour weights

    Yd = Wd @ Y                         # row-based estimate
    Yt = Y @ Wt.T                       # column-based estimate
    Ydt = Wd @ Y @ Wt.T                 # combined (both-sides) estimate
    est = (Yd + Yt + Ydt) / 3.0
    return np.maximum(Y, est)           # keep observed 1s


class _DSCMFContentModel:
    """Content-EQUIPPED, faithful DSCMF (see module docstring)."""

    def __init__(self, device="cpu"):
        self.device = device  # CPU/numpy only; device accepted for the contract, ignored.

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc, dtype=np.int64)
        td = np.asarray(train_dis, dtype=np.int64)
        self._tl, self._td = tl, td

        # Full-shape output. Unlike the native GIP-only model (cold rows/cols at
        # the 0 floor), the content-equipped factorization is full-size, so every
        # entry -- including cold rows/cols -- is reconstructed below.
        S_full = np.zeros((self.n_l, self.n_d), np.float32)

        nTl, nTd = tl.size, td.size
        # Adaptive rank k: clamp so tiny smoke data (a few train nodes) cannot crash.
        k = int(min(_RANK_CAP, nTl, nTd))
        if k < 1:
            self.S = S_full
            return self

        # --- ONLY supervision touched: the train association sub-block. ---------
        X = subblock(M, tl, td).astype(np.float64)       # (nTl, nTd), {0,1}

        # --- CONTENT similarities (M-INDEPENDENT; literal intrinsic content),
        #     defined for ALL nodes incl. cold. lncRNA side: expression matrix ->
        #     cosine; disease side: DO-DAG semantic-similarity matrix. -----------
        expr = _load_content_npy("CCDIFF_LNC_EXPR", self.n_l, square=False)
        SL_con = (content_cosine(expr) if expr is not None
                  else content_cosine(Clnc)).astype(np.float64)      # (n_l, n_l)
        ss = _load_content_npy("CCDIFF_DIS_SEMSIM", self.n_d, square=True)
        SD_con = (ss if ss is not None
                  else content_cosine(Cdis)).astype(np.float64)      # (n_d, n_d)

        # --- GIP similarities (train association sub-block ONLY; as native),
        #     embedded FULL-SIZE with zeros outside the train x train block. -----
        SL_gip = np.zeros((self.n_l, self.n_l), np.float64)
        SD_gip = np.zeros((self.n_d, self.n_d), np.float64)
        SL_gip[np.ix_(tl, tl)] = gip_kernel(X).astype(np.float64)    # (nTl,nTl) lncRNA GIP
        SD_gip[np.ix_(td, td)] = gip_kernel(X.T).astype(np.float64)  # (nTd,nTd) disease GIP

        # --- BLEND (Eq.8-9): K_l = alpha*S_l + (1-alpha)*GIP_l (the ONLY change
        #     vs the native alpha=0 content-blind concession). SL/SD are the KL/KD
        #     collaborative-similarity matrices fed to the SAME DSCMF ALS. -------
        alpha = float(os.environ.get("DSCMF_ALPHA", "0.5"))
        KL = alpha * SL_con + (1.0 - alpha) * SL_gip     # (n_l, n_l)
        KD = alpha * SD_con + (1.0 - alpha) * SD_gip     # (n_d, n_d)

        # --- (1) WKNKN preimputation: soften {0,1} -> [0,1] BEFORE factorising,
        #     using the (train-restricted) blended kernels -- same source as ALS.
        #     Embed full-size: off the train x train block Y stays 0 (leakage-safe
        #     AND M-scramble invariant). -----------------------------------------
        Y_sub = _wknkn(X, KL[np.ix_(tl, tl)], KD[np.ix_(td, td)])    # (nTl, nTd) in [0,1]
        Y = np.zeros((self.n_l, self.n_d), np.float64)
        Y[np.ix_(tl, td)] = Y_sub

        # --- (4) SVD initialization (Eq.13): A = U S^{1/2}, B = V S^{1/2}. -------
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        sr = np.sqrt(np.maximum(s[:k], 0.0))
        A = U[:, :k] * sr[None, :]                       # (n_l, k), signed
        B = (Vt[:k].T) * sr[None, :]                     # (n_d, k), signed

        lam_h, lam_l, lam_d = _LAM_H, _LAM_L, _LAM_D
        Ik = np.eye(k)

        # --- (2)+(3) Unconstrained ALS: Eq.14 / Eq.15, single shared k x k inverse.
        #     Quadratic terms (A^TA, B^TB) and the L2,1 reweight (D1,D2) are frozen
        #     at the previous iterate, exactly as in the paper's linearisation.
        #     UNCHANGED from native except KL/KD are now the content+GIP blend. ---
        for _ in range(_N_ITER):
            # D1: k x k diagonal, d_jj = 1 / (2 ||A^j||_2)  (COLUMN norms of A).
            D1 = np.diag(1.0 / (2.0 * np.linalg.norm(A, axis=0) + _EPS))
            #   A = (Y B + lam_l K_l A) (B^T B + lam_h I + lam_l A^T A + lam_h D1)^-1
            num_A = Y @ B + lam_l * (KL @ A)
            den_A = B.T @ B + lam_h * Ik + lam_l * (A.T @ A) + lam_h * D1
            A = num_A @ np.linalg.solve(den_A, Ik)       # (n_l, k)

            # D2: k x k diagonal, d_jj = 1 / (2 ||B^j||_2)  (COLUMN norms of B).
            D2 = np.diag(1.0 / (2.0 * np.linalg.norm(B, axis=0) + _EPS))
            #   B = (Y^T A + lam_d K_d B)(A^T A + lam_h I + lam_d B^T B + lam_h D2)^-1
            num_B = Y.T @ A + lam_d * (KD @ B)
            den_B = A.T @ A + lam_h * Ik + lam_d * (B.T @ B) + lam_h * D2
            B = num_B @ np.linalg.solve(den_B, Ik)       # (n_d, k)

        # --- Reconstruct the FULL matrix; cold rows/cols now carry content-driven
        #     (nonzero) scores through the CMF collaborative term. ---------------
        pred = (A @ B.T).astype(np.float32)              # (n_l, n_d), signed scores
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        self.S = pred.astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _DSCMFContentModel(device)
