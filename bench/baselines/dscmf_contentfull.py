"""DSCMF -- COLD-EQUIPPED content variant (content fold-in of latent factors).

Sibling of `bench/baselines/dscmf_content.py`. That -content variant blends the
paper's literal intrinsic content (lncRNA expression cosine + disease DO-DAG
semantic similarity) with the train-subblock GIP and feeds the FULL (n_l x n_l) /
(n_d x n_d) blend as the CMF collaborative kernels K_l / K_d into a full-size
DSCMF ALS -- so cold rows/cols acquire NONZERO scores from the CMF term. BUT on
sparse, popularity-skewed association matrices those cold scores are
SYSTEMATICALLY INVERTED: at both-cold (C4) they land BELOW the 0.500 chance floor
(~0.42-0.44 on every RNADisease variant) -- worse than random.

ROOT CAUSE (why -content is sub-floor at C4)
--------------------------------------------
A cold lncRNA has an all-zero row in the WKNKN-imputed Y, so its factor row is
pulled purely by the CMF term via the ALS numerator  lam_l * (K_l @ A). At the
fixed point a cold row obeys

    A[cold] (BᵀB + lam_h I + lam_l AᵀA + lam_h D1) = lam_l (K_l A)[cold]
    => A[cold] ≈ lam_l * (Σ_j content(cold,j) * A[j]) * den⁻¹ ,  den = BᵀB + ...

but each TRAIN factor already satisfies A[j] ≈ Y[j,:] B den⁻¹, so the cold score
becomes

    pred[cold,:] = A[cold] Bᵀ ≈ lam_l * v * (B den⁻² Bᵀ) ,   v = Σ_j content·Y[j,:]

i.e. the content-weighted neighbour associations `v` (the CORRECT, positively
correlated signal) are passed through B den⁻² Bᵀ -- an EXTRA copy of the k x k
inverse beyond the warm reconstruction operator B den⁻¹ Bᵀ ≈ I. That extra
ill-conditioned twist reweights each latent direction by 1/λ², so when the
association matrix's dominant (disease-popularity) singular direction is not the
content-similar direction, the effective ranking flips and the cold AUC drops
below chance. Empirically reproduced on sparse + popularity-skewed synthetics
(C4 AUC ~0.35-0.49), matching the real RNADisease sub-floor; it does NOT show up
on dense block-structured toy data (where the twist is harmless), which is why it
is a genuine data-dependent pathology, not a coding sign error.

THE FIX (this variant)
----------------------
Keep the DSCMF ALS machinery EXACTLY, but run it TRANSDUCTIVELY on the TRAIN
sub-block only (native factorization: A_tr (nTl x k), B_tr (nTd x k)). Then equip
cold nodes by an EXPLICIT content FOLD-IN of the latent factors -- the standard,
correct cold-start extension for matrix factorization:

    A[i]  = Σ_a w^L(i,a) A_tr[a]   over train lncRNAs a   (train rows kept = A_tr)
    B[j]  = Σ_b w^D(j,b) B_tr[b]   over train diseases b   (train cols kept = B_tr)
    w^L(i,·) = row-normalised top-k NONNEGATIVE content cosine( lnc_i , train lncs )
    w^D(j,·) = row-normalised top-k NONNEGATIVE content cosine( dis_j , train dis )
    S = A @ Bᵀ

Because the weights are NONNEGATIVE and the train factors are used AS-IS,

    pred[cold_i, d] = Σ_a w^L(i,a) (A_tr[a]·B[d]) = Σ_a w^L(i,a) pred_train[a, d]

is a nonnegative-weighted average of the predictions of the cold node's
content-nearest TRAIN lncRNAs -- POSITIVELY correlated with content similarity to
the positives BY CONSTRUCTION, with no den⁻² twist. The train x train block is
the native DSCMF reconstruction A_tr B_trᵀ unchanged.

BLEND (train CMF kernel, same form as -content):
    KL_tr = alpha*SL_con[train,train] + (1-alpha)*GIP_l   (nTl x nTl)
    KD_tr = alpha*SD_con[train,train] + (1-alpha)*GIP_d   (nTd x nTd)
    alpha = float(os.environ.get("DSCMF_ALPHA", "0.5"))
  SL_con/SD_con: M-independent literal content (CCDIFF_LNC_EXPR expression cosine,
  CCDIFF_DIS_SEMSIM DO-DAG semsim; else content_cosine(Clnc)/(Cdis)).
  SL_gip/SD_gip: gip_kernel of the TRAIN association sub-block only (as native).
Fold-in weights use the M-independent content cosine only (no GIP).

CONTRACT + INVARIANCE. Labels/GIP come ONLY from M[np.ix_(train_lnc, train_dis)]
(subblock); content and fold-in weights are M-INDEPENDENT. Scrambling any entry
outside the train sub-block leaves X, GIP, the train ALS, the content weights and
every score identical -> sub-block invariance (max|Delta| <= 1e-4). Cold nodes
now score ABOVE floor from content -- the intended fair-test effect.

Module-level names the runner imports: NAME, build(device) -> model.
"""
import os

import numpy as np

# Leakage-safe shared helpers + the global seed.
from bench.interface import subblock, gip_kernel, content_cosine, SEED   # noqa: F401

NAME = "DSCMF-contentfull (cold-equipped)"

# Fixed constants -- IDENTICAL to the native DSCMF reproduction.
_K_WKNKN = 5        # WKNKN neighbourhood size (paper: K=5)
_P = 0.7            # WKNKN rank decay (paper: p=0.7)
_LAM_H = 1.0        # Tikhonov + L2,1 weight lam_h
_LAM_L = 0.1        # lncRNA-side collaborative weight lam_l
_LAM_D = 0.1        # disease-side collaborative weight lam_d
_N_ITER = 100       # fixed ALS sweeps (paper: max 100 iterations)
_RANK_CAP = 50      # rank k cap, clamped to min(nTl, nTd)
_EPS = 1e-8         # denominator / division guard
# Cold fold-in neighbourhood (content nearest train nodes). Env-overridable.
_FOLDIN_K = int(os.environ.get("DSCMF_FOLDIN_K", "10"))


def _load_content_npy(env_key, n, square):
    """Load an M-INDEPENDENT content/similarity matrix from an env-pointed .npy.

    square=False -> a valid file is an (n, k) expression matrix; square=True -> a
    valid (n, n) semantic-similarity matrix. Returns validated float32 or None
    (caller falls back to intrinsic content).
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
    if not np.isfinite(arr).all():
        return None
    return arr


def _wknkn(Y, KL, KD, K=_K_WKNKN, p=_P):
    """WKNKN preimputation of a {0,1} sub-block Y. IDENTICAL to native DSCMF."""
    Y = np.asarray(Y, np.float64)
    nl, nd = Y.shape
    if nl == 0 or nd == 0:
        return Y.copy()

    row_known = Y.sum(1) > 0
    col_known = Y.sum(0) > 0

    def _weights(S, known):
        n = S.shape[0]
        W = np.zeros((n, n), np.float64)
        kidx = np.where(known)[0]
        if kidx.size == 0:
            return W
        for i in range(n):
            sims = np.asarray(S[i], np.float64).copy()
            sims[i] = -np.inf
            cand = kidx[kidx != i]
            if cand.size == 0:
                continue
            cs = sims[cand]
            valid = cs > 0
            if not valid.any():
                continue
            cand, cs = cand[valid], cs[valid]
            order = np.argsort(-cs, kind="stable")[:K]
            nb, ss = cand[order], cs[order]
            w = (p ** np.arange(nb.size)) * ss
            Z = ss.sum() + _EPS
            W[i, nb] = w / Z
        return W

    Wd = _weights(KL, row_known)
    Wt = _weights(KD, col_known)
    Yd = Wd @ Y
    Yt = Y @ Wt.T
    Ydt = Wd @ Y @ Wt.T
    est = (Yd + Yt + Ydt) / 3.0
    return np.maximum(Y, est)


def _foldin_weights(sim_to_train, k):
    """Row-normalised top-k NONNEGATIVE fold-in weights.

    sim_to_train : (n, nT) content cosine of every node to the TRAIN nodes.
    Returns (n, nT) weights; each row = the node's normalised nonneg similarity to
    its k content-nearest train nodes (rest zeroed). Train rows recover ~their own
    one-hot (self-cosine = 1 is the top neighbour), but callers overwrite train
    rows with the exact trained factors anyway.
    """
    W = np.maximum(np.asarray(sim_to_train, np.float64), 0.0)
    nT = W.shape[1]
    if 0 < k < nT:
        idx = np.argpartition(-W, k - 1, axis=1)[:, :k]
        mask = np.zeros_like(W, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        W = np.where(mask, W, 0.0)
    W = W / (W.sum(1, keepdims=True) + _EPS)
    return W


class _DSCMFContentFull:
    """Transductive DSCMF core + content fold-in of latent factors to cold nodes."""

    def __init__(self, device="cpu"):
        self.device = device  # CPU/numpy only; accepted for the contract, ignored.

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc, dtype=np.int64)
        td = np.asarray(train_dis, dtype=np.int64)
        self._tl, self._td = tl, td

        S_full = np.zeros((self.n_l, self.n_d), np.float32)
        nTl, nTd = tl.size, td.size
        k = int(min(_RANK_CAP, nTl, nTd))
        if k < 1:
            self.S = S_full
            return self

        # --- ONLY supervision touched: the train association sub-block. ---------
        X = subblock(M, tl, td).astype(np.float64)              # (nTl, nTd) {0,1}

        # --- CONTENT similarities (M-INDEPENDENT; literal intrinsic content). ---
        expr = _load_content_npy("CCDIFF_LNC_EXPR", self.n_l, square=False)
        SL_con = np.asarray(content_cosine(expr) if expr is not None
                            else content_cosine(Clnc), np.float64)   # (n_l, n_l)
        ss = _load_content_npy("CCDIFF_DIS_SEMSIM", self.n_d, square=True)
        SD_con = np.asarray(ss if ss is not None
                            else content_cosine(Cdis), np.float64)   # (n_d, n_d)

        # --- GIP (train association sub-block ONLY; as native). -----------------
        SL_gip = gip_kernel(X).astype(np.float64)               # (nTl, nTl)
        SD_gip = gip_kernel(X.T).astype(np.float64)             # (nTd, nTd)

        # --- TRAIN CMF kernels: blend content(train,train) with GIP (Eq.8-9). ---
        alpha = float(os.environ.get("DSCMF_ALPHA", "0.5"))
        KL = alpha * SL_con[np.ix_(tl, tl)] + (1.0 - alpha) * SL_gip   # (nTl, nTl)
        KD = alpha * SD_con[np.ix_(td, td)] + (1.0 - alpha) * SD_gip   # (nTd, nTd)

        # --- (1) WKNKN preimputation on the TRAIN sub-block. --------------------
        Y = _wknkn(X, KL, KD)                                   # (nTl, nTd) in [0,1]

        # --- (4) SVD initialization (Eq.13): A = U S^{1/2}, B = V S^{1/2}. ------
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        sr = np.sqrt(np.maximum(s[:k], 0.0))
        A = U[:, :k] * sr[None, :]                              # (nTl, k)
        B = (Vt[:k].T) * sr[None, :]                            # (nTd, k)

        lam_h, lam_l, lam_d = _LAM_H, _LAM_L, _LAM_D
        Ik = np.eye(k)

        # --- (2)+(3) DSCMF ALS (Eq.14/15) on the TRAIN sub-block -- UNCHANGED. --
        for _ in range(_N_ITER):
            D1 = np.diag(1.0 / (2.0 * np.linalg.norm(A, axis=0) + _EPS))
            num_A = Y @ B + lam_l * (KL @ A)
            den_A = B.T @ B + lam_h * Ik + lam_l * (A.T @ A) + lam_h * D1
            A = num_A @ np.linalg.solve(den_A, Ik)             # (nTl, k)

            D2 = np.diag(1.0 / (2.0 * np.linalg.norm(B, axis=0) + _EPS))
            num_B = Y.T @ A + lam_d * (KD @ B)
            den_B = A.T @ A + lam_h * Ik + lam_d * (B.T @ B) + lam_h * D2
            B = num_B @ np.linalg.solve(den_B, Ik)             # (nTd, k)

        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

        # --- COLD FOLD-IN: content-weighted lift of the trained factors. --------
        # Every node's factor = nonneg-content-weighted average of TRAIN factors;
        # train nodes keep their exact trained rows (so the train x train block is
        # the native DSCMF reconstruction). Cold scores are then nonneg-weighted
        # averages of content-near train predictions -> positively correlated.
        WL = _foldin_weights(SL_con[:, tl], _FOLDIN_K)          # (n_l, nTl)
        WD = _foldin_weights(SD_con[:, td], _FOLDIN_K)          # (n_d, nTd)
        A_full = WL @ A                                         # (n_l, k)
        B_full = WD @ B                                         # (n_d, k)
        A_full[tl] = A                                          # keep exact train factors
        B_full[td] = B

        pred = (A_full @ B_full.T).astype(np.float32)          # (n_l, n_d)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        self.S = pred.astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _DSCMFContentFull(device)
