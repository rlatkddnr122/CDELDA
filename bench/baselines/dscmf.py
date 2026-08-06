"""DSCMF -- Dual-Sparse Collaborative Matrix Factorization (CONTENT-BLIND baseline).

FAITHFUL reproduction of DSCMF:

    Liu et al., "DSCMF: prediction of LncRNA-disease associations based on dual
    sparse collaborative matrix factorization", BMC Bioinformatics 22 (2021),
    DOI 10.1186/s12859-020-03868-w. (PMC8114493.)
    Precursor conference version: Springer LNCS, DOI 10.1007/978-3-030-26766-7_29.

We could not locate an official public reference implementation from the authors
(no GitHub link is given in the paper); this reproduction is coded directly from
the paper's equations (Eq. 8-15). The algorithm is Zheng's Collaborative Matrix
Factorization (CMF) with four DSCMF additions, all reproduced here:

  min_{A,B}  || Y - A B^T ||_F^2                                   (association MF)
           + lam_h ( ||A||_F^2 + ||B||_F^2 )                       (Tikhonov, Eq.12)
           + lam_h ||A||_{2,1} + lam_h ||B||_{2,1}                 (DUAL L2,1 sparsity)
           + lam_l ||K_l - A A^T||_F^2 + lam_d ||K_d - B B^T||_F^2 (CMF collaborative)

  1. WKNKN preimputation of Y (the train sub-block) BEFORE factorization
     (K=5, decay p=0.7): softens the {0,1} block to [0,1] using weighted
     K-nearest-KNOWN neighbours by similarity (rows via K_l, cols via K_d),
     as the average of the row-based, column-based, and combined estimates, with
     observed 1s kept (max with Y). The paper omits the exact WKNKN formula, so
     we use the standard Ezzat et al. (2017) WKNKN that DSCMF cites. Leakage-safe:
     train sub-block only.
  2. DUAL L2,1-norm sparsity on the k latent COLUMNS of BOTH factors, solved by
     iteratively-reweighted diagonal weights D1 = diag(1 / (2 ||A^j||_2)),
     D2 = diag(1 / (2 ||B^j||_2)); both are k x k (paper: d_jj = 1/(2||A_j||_2)).
  3. CMF collaborative similarity-RECONSTRUCTION terms ||K_l - A A^T||^2 and
     ||K_d - B B^T||^2 (K_l, K_d = similarity kernels), NOT a graph-Laplacian.
  4. UNCONSTRAINED ALS with SVD initialization (Eq.13: A=U S^{1/2}, B=V S^{1/2};
     factors may be negative -- NOT an NMF), fixed 100 iterations.

Update rules reproduced EXACTLY from Eq.14 / Eq.15 (single shared k x k inverse,
all quadratic / reweighting terms frozen at the previous iterate):

    A = (Y B + lam_l K_l A) (B^T B + lam_h I_k + lam_l A^T A + lam_h D1)^{-1}
    B = (Y^T A + lam_d K_d B)(A^T A + lam_h I_k + lam_d B^T B + lam_h D2)^{-1}

FORCED CONTENT-BLIND DEVIATION (the ONLY policy concession)
-----------------------------------------------------------
The paper integrates functional similarity with the GIP kernel:
    K_l = alpha*S_l + (1-alpha)*GIP_l ,  K_d = alpha*S_d + (1-alpha)*GIP_d   (Eq.8-9)
where S_l is lncRNA expression (Spearman) similarity and S_d is disease semantic
(DAG) similarity. Under the ratified CONTENT-BLIND policy a baseline may use ONLY
GIP kernels of the TRAIN association sub-block and must NEVER read Clnc / Cdis.
We therefore set alpha = 0, i.e. K_l = GIP_l and K_d = GIP_d built strictly from
the train sub-block. Clnc / Cdis are received per the contract but NEVER read.

Output
------
  S_full = zeros(n_l, n_d) ; S_full[ix_(train_lnc, train_dis)] = A @ B^T
COLD (held-out) rows/cols never enter the train sub-block, carry an empty GIP
profile, and stay at the 0 floor -- an honest collapse. float32, all finite.
Sub-block invariance holds: predict() reads ONLY M[ix_(train_lnc, train_dis)].
"""
import numpy as np

# Leakage-safe shared helpers + the global seed. `subblock` is the ONLY reader of
# supervision; gip_kernel builds the train-side GIP kernels. Clnc/Cdis are ignored.
from bench.interface import subblock, gip_kernel, SEED   # noqa: F401

NAME = "DSCMF (dual-sparse collab MF, content-blind)"

# Fixed constants (deterministic fixed-iteration solve).
_K_WKNKN = 5        # WKNKN neighbourhood size (paper: K=5)
_P = 0.7            # WKNKN rank decay (paper: p=0.7)
_LAM_H = 1.0        # Tikhonov + L2,1 weight lam_h (paper grid {2^-2..2^1}; 2^0=1 default)
_LAM_L = 0.1        # lncRNA-side collaborative weight lam_l (paper grid {0,1e-4..1e-1})
_LAM_D = 0.1        # disease-side collaborative weight lam_d (paper grid {0,1e-4..1e-1})
_N_ITER = 100       # fixed ALS sweeps (paper: max 100 iterations)
_RANK_CAP = 50      # rank k cap, clamped to min(nTl, nTd) for tiny-data safety
_EPS = 1e-8         # denominator / division guard


def _wknkn(Y, KL, KD, K=_K_WKNKN, p=_P):
    """WKNKN preimputation of a {0,1} sub-block Y (rows=lncRNA, cols=disease).

    KL (row-side) and KD (col-side) are the similarity kernels. Returns Y softened
    to [0,1]: the mean of the row-based, column-based, and combined weighted-
    nearest-known-neighbour estimates, with observed 1s kept (max with Y). Uses
    only Y and its kernels -> leakage-safe. Standard Ezzat-et-al. WKNKN (cited by
    DSCMF); the DSCMF paper omits the explicit formula.
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


class _DSCMFModel:
    """Content-blind, faithful DSCMF (see module docstring)."""

    def __init__(self, device="cpu"):
        self.device = device  # CPU/numpy only; device accepted for the contract, ignored.

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # --- Clnc / Cdis are intentionally UNUSED (content-blind policy). --------
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc, dtype=np.int64)
        td = np.asarray(train_dis, dtype=np.int64)
        self._tl, self._td = tl, td

        # Full-shape output pre-filled with the honest cold floor (0). Only the
        # train x train sub-block is ever overwritten below.
        S_full = np.zeros((self.n_l, self.n_d), np.float32)

        nTl, nTd = tl.size, td.size
        # Adaptive rank k: clamp so tiny smoke data (a few train nodes) cannot crash.
        k = int(min(_RANK_CAP, nTl, nTd))
        if k < 1:
            self.S = S_full
            return self

        # --- ONLY supervision touched: the train association sub-block. ---------
        X = subblock(M, tl, td).astype(np.float64)       # (nTl, nTd), {0,1}

        # --- Similarity kernels. Paper: K_l = a*S_l + (1-a)*GIP_l (Eq.8-9); the
        #     CONTENT-BLIND policy forces alpha=0 => K_l = GIP_l, K_d = GIP_d,
        #     both from the train sub-block only. Clnc/Cdis are NOT read. ---------
        KL = gip_kernel(X).astype(np.float64)            # (nTl, nTl) lncRNA-side GIP
        KD = gip_kernel(X.T).astype(np.float64)          # (nTd, nTd) disease-side GIP

        # --- (1) WKNKN preimputation: soften {0,1} -> [0,1] BEFORE factorising. --
        Y = _wknkn(X, KL, KD)                            # (nTl, nTd) in [0,1]

        # --- (4) SVD initialization (Eq.13): A = U S^{1/2}, B = V S^{1/2}. -------
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        sr = np.sqrt(np.maximum(s[:k], 0.0))
        A = U[:, :k] * sr[None, :]                       # (nTl, k), signed
        B = (Vt[:k].T) * sr[None, :]                     # (nTd, k), signed

        lam_h, lam_l, lam_d = _LAM_H, _LAM_L, _LAM_D
        Ik = np.eye(k)

        # --- (2)+(3) Unconstrained ALS: Eq.14 / Eq.15, single shared k x k inverse.
        #     Quadratic terms (A^TA, B^TB) and the L2,1 reweight (D1,D2) are frozen
        #     at the previous iterate, exactly as in the paper's linearisation. --
        for _ in range(_N_ITER):
            # D1: k x k diagonal, d_jj = 1 / (2 ||A^j||_2)  (COLUMN norms of A).
            D1 = np.diag(1.0 / (2.0 * np.linalg.norm(A, axis=0) + _EPS))
            #   A = (Y B + lam_l K_l A) (B^T B + lam_h I + lam_l A^T A + lam_h D1)^-1
            num_A = Y @ B + lam_l * (KL @ A)
            den_A = B.T @ B + lam_h * Ik + lam_l * (A.T @ A) + lam_h * D1
            A = num_A @ np.linalg.solve(den_A, Ik)       # (nTl, k)

            # D2: k x k diagonal, d_jj = 1 / (2 ||B^j||_2)  (COLUMN norms of B).
            D2 = np.diag(1.0 / (2.0 * np.linalg.norm(B, axis=0) + _EPS))
            #   B = (Y^T A + lam_d K_d B)(A^T A + lam_h I + lam_d B^T B + lam_h D2)^-1
            num_B = Y.T @ A + lam_d * (KD @ B)
            den_B = A.T @ A + lam_h * Ik + lam_d * (B.T @ B) + lam_h * D2
            B = num_B @ np.linalg.solve(den_B, Ik)       # (nTd, k)

        # --- Reconstruct the train sub-block; cold rows/cols remain 0. ----------
        pred = (A @ B.T).astype(np.float32)              # (nTl, nTd), signed scores
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        S_full[np.ix_(tl, td)] = pred
        self.S = S_full.astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _DSCMFModel(device)
