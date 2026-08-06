"""SIMCLDA -- inductive matrix completion (IMC) on eigen-features of the train GIP.

Faithful (structure-restoring) reproduction of

    Lu, Yang, Zeng, Zhou, Xiong, Chen, Liao, Xie, Li, Wei, Ge, Chen,
    "Prediction of lncRNA-disease associations based on inductive matrix
    completion (SIMCLDA)", Bioinformatics 34(19):3357-3364, 2018.
    Original code (MATLAB): https://github.com/bioinfomaticsCSU/SIMCLDA

--------------------------------------------------------------------------------
CONTENT-BLIND (RATIFIED POLICY, non-negotiable)
--------------------------------------------------------------------------------
This baseline MUST NOT read the content matrices Clnc / Cdis (RNA-FM /
S-BioBERT); they are received only to honour THE CONTRACT signature and are
deliberately ignored. Every similarity / feature derives ONLY from the Gaussian
Interaction Profile (GIP) kernel of the TRAIN association sub-block:

    Msub = subblock(M, train_lnc, train_dis)     # ONLY supervision touched
    SL   = gip_kernel(Msub)     -> (nTl, nTl)     # lncRNA train-side GIP
    SD   = gip_kernel(Msub.T)   -> (nTd, nTd)     # disease train-side GIP

The ONE policy concession vs. the paper: the original integrates GIP with a
disease-ontology *semantic* similarity and an lncRNA *functional* similarity,
and uses that side-information matrix (which is defined even for cold nodes) to
run the "inductive" trick -- substituting the GIP profile of the most-similar
KNOWN node for an empty cold node. That side-information similarity is exactly
what the content-blind policy forbids, so it is dropped: the algorithm STRUCTURE
below is restored, but fed GIP wherever the paper used integrated similarity,
and cold (held-out) nodes keep an empty GIP profile -> zero feature -> S = 0.
That cold collapse is by-construction and honest, not fabricated.

--------------------------------------------------------------------------------
ALGORITHM STRUCTURE (restored to match the original MATLAB)
--------------------------------------------------------------------------------
The original demo (SIMCLDA_demo.m) is literally:
    lnc_feature = pca_energy(LL,     0.8)   # 80% energy of the lncRNA GIP
    dis_feature = pca_energy(dissim, 0.6)   # 60% energy of the disease similarity
    Omega       = find(LD==1)               # observed = positives (see mask note)
    M_recover   = SIMC(LD, Omega, lnc_feature, dis_feature)   # lambda=1 APG solve
where pca_energy(S,p) = truncated SVD of S keeping the SMALLEST number of leading
singular components whose cumulative energy sum(sigma_1..k) >= p*sum(sigma), and
RETURNS THE RIGHT SINGULAR VECTORS V_k (no eigenvalue/sigma scaling). SIMC then
solves inductive matrix completion X = A Z B^T with a nuclear/ridge-regularised
low-rank factor Z (default lambda=1), so the recovered rank is r_a x r_b with
r_a = #cols(A), r_b = #cols(B) (the energy-driven component counts).

Our faithful restoration:
 1. GIP kernels SL, SD from the train sub-block (above). [content-blind, see note]
 2. PRIMARY LINEAR FEATURES = pca_energy: for each SPD kernel the SVD == the
    eigen-decomposition, so we take the top-k RIGHT SINGULAR VECTORS (== leading
    eigenvectors) with NO sigma scaling, k chosen adaptively by the ENERGY
    THRESHOLD (0.8 for lncRNA, 0.6 for disease), clamped to the train size so
    tiny smoke data cannot crash:
        X = V_k(SL)   (nTl, p)     # lncRNA features, p = energy-count @0.8
        Y = V_k(SD)   (nTd, q)     # disease features, q = energy-count @0.6
 3. INDUCTIVE MATRIX COMPLETION of the recovered matrix X = A Z B^T. We solve the
    numerically-equivalent RIDGE form of SIMC (lambda=1) by matrix-free CONJUGATE
    GRADIENT on the SPD normal-equation operator  L(Z) = X^T (X Z Y^T) Y + lam Z
    (an APG on a nuclear-norm surrogate collapses to this once the support is
    fixed; CG is deterministic and avoids a MEX solver). See the mask note for
    the observed set.
 4. PREDICT S = Xfull Z Yfull^T over ALL nodes; train rows/cols carry their
    pca_energy features, COLD rows/cols carry a zero feature vector -> S = 0.

--------------------------------------------------------------------------------
OBSERVED-SET (MASK) NOTE -- deliberate deviation from Omega=find(LD==1)
--------------------------------------------------------------------------------
The original sets Omega to the POSITIVES only. In this fold-CV cold-start
benchmark a positives-only P_Omega is ILL-POSED for the binary sub-block and
INVERTS the ranking (a prior fix diagnosed this). We therefore keep the observed
set = the WHOLE train sub-block (Mask = ones over the train fold): the known
0-associations are the training NEGATIVES. This mask fix is preserved on purpose
and must NOT be reverted to positives-only.

Determinism: SVD/eigh + zero-initialised CG are deterministic; no RNG draws
affect the result (seeded for form). CPU only (device ignored). float32, finite.

Module-level names the runner imports: NAME, build(device) -> model.
"""
import numpy as np

try:                                    # top-r eigenpairs directly when available
    from scipy.linalg import eigh as _sp_eigh
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover - numpy fallback
    _HAVE_SCIPY = False

# Leakage-safe helpers. Clnc / Cdis-derived helpers (content_cosine, cosine_sim,
# integrated_row_sim) are intentionally NOT imported: this baseline is content-blind.
from bench.interface import SEED, subblock, gip_kernel

NAME = "SIMCLDA (GIP-IMC eigen, content-blind)"

_LAM = 1.0        # IMC ridge regularisation on Z (SIMC default lambda=1)
_ENERGY_L = 0.8   # pca_energy threshold for lncRNA GIP  (SIMCLDA_demo.m)
_ENERGY_D = 0.6   # pca_energy threshold for disease similarity (SIMCLDA_demo.m)
_CG_TOL = 1e-8    # CG relative-residual stop
_CG_MAXIT = 500   # CG iteration cap


def _pca_energy_features(K, energy, n_full, train_idx):
    """Primary linear features = pca_energy(K, energy) of a (symmetric) GIP kernel.

    Faithful to the original MATLAB pca_energy: truncated SVD keeping the fewest
    leading singular components whose cumulative energy reaches `energy`*total,
    returning the RIGHT SINGULAR VECTORS (no sigma scaling). For an SPD kernel the
    SVD coincides with the eigen-decomposition, so we take the top-k eigenvectors
    (== right singular vectors) and select k by the energy criterion.

    K          : (nT, nT) train-side GIP kernel (rows == train nodes).
    energy     : cumulative singular-value energy fraction in (0, 1].
    n_full     : total nodes on this axis (n_l or n_d).
    train_idx  : positions of the train nodes into the full-axis feature matrix.
    Returns
      Xtr  : (nT, k) right singular vectors V_k (unscaled), k = energy count.
      Xful : (n_full, k) full matrix; train rows = Xtr, COLD rows = 0.
    """
    nT = K.shape[0]
    Ksym = 0.5 * (np.asarray(K, np.float64) + np.asarray(K, np.float64).T)  # enforce symmetry
    if _HAVE_SCIPY:
        vals, vecs = _sp_eigh(Ksym)                      # ascending eigenpairs
    else:                                                # pragma: no cover
        vals, vecs = np.linalg.eigh(Ksym)
    order = np.argsort(vals)[::-1]                        # descending (== SVD sigma order)
    vals = np.clip(vals[order], 0.0, None)               # SPD -> sigma == eigenvalue
    vecs = vecs[:, order]
    total = float(vals.sum()) + 1e-30
    csum = np.cumsum(vals)
    # fewest leading components reaching the energy fraction (>=1, <= nT)
    k = int(np.searchsorted(csum, energy * total) + 1)
    k = max(1, min(k, nT))
    Xtr = np.asarray(vecs[:, :k], np.float64)            # (nT, k) right singular vectors, unscaled
    Xful = np.zeros((n_full, k), np.float64)             # cold rows -> 0
    Xful[np.asarray(train_idx)] = Xtr
    return Xtr, Xful


def _cg_imc(X, Y, Msub, lam, tol, maxit):
    """Iterative IMC solve of  min_W sum_{train fold} (Msub - X W Y^T)^2 + lam||W||^2.

    Observed set = the whole train fold (known 0-associations are the training
    negatives; a positives-only objective is ill-posed and inverts the ranking).
    Solved by matrix-free conjugate gradient on the SPD normal-equation operator
        L(W) = X^T ( X W Y^T ) Y + lam W .
    Returns (W (p,q), n_iters, final_relative_residual).
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


class _SimCLDA:
    """Inductive matrix completion on eigen-features of the train GIP (content-blind)."""

    def __init__(self, device="cpu"):
        self.device = device
        self.rng = np.random.default_rng(SEED)   # form only; result is deterministic
        # diagnostics populated by fit() (used by the isolated self-test)
        self.info = {}

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # Clnc / Cdis are received to satisfy THE CONTRACT but are DELIBERATELY
        # IGNORED (content-blind). Supervision = the train sub-block ONLY.
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

        # --- (1-2) GIP kernels -> pca_energy primary linear features -----------
        SL = gip_kernel(Msub)          # (nTl, nTl) lncRNA train-side GIP
        SD = gip_kernel(Msub.T)        # (nTd, nTd) disease train-side GIP
        Xtr, Xful = _pca_energy_features(SL, _ENERGY_L, self.n_l, tl)   # (nTl,p),(n_l,p)
        Ytr, Yful = _pca_energy_features(SD, _ENERGY_D, self.n_d, td)   # (nTd,q),(n_d,q)

        # --- (3) iterative IMC over OBSERVED entries (masked CG) ---------------
        W, cg_it, cg_res = _cg_imc(Xtr, Ytr, Msub, _LAM, _CG_TOL, _CG_MAXIT)

        # --- (4) full-node scores; cold rows/cols carry zero features -> 0 -----
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
    return _SimCLDA(device)
