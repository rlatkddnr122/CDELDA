"""KATZLDA -- CONTENT-EQUIPPED variant (semantic + expression similarity + GIP).

This is the native `bench/baselines/katzlda.py` reproduction with ONE change: the
heterogeneous-network SIMILARITY SOURCE. The native reproduction is content-blind
and substitutes LS := GIP(Msub), DS := GIP(Msub.T) because the functional /
semantic / expression databases were declared off-limits. This variant
DELIBERATELY restores the paper's LITERAL intrinsic content -- lncRNA EXPRESSION
similarity and disease DO-DAG SEMANTIC similarity -- blended with the
train-subblock GIP, exactly as Chen (2015) integrates them:

    lncRNA LS = expression/functional similarity + GIP KL
    disease DS = MeSH/DO semantic similarity   + GIP KD

so we can test whether the both-cold (C4) collapse persists when the baseline is
NOT content-starved. Everything else -- the symmetric 2x2 block adjacency
A* = [[SL, Msub_full], [Msub_fullᵀ, SD]], the closed-form Katz inverse
(I - beta*A*)^{-1} - I, the in-fold beta sweep with the beta < 1/rho(A*)
convergence bound -- is the IDENTICAL Katz machinery.

WHY THE NETWORK SPANS ALL NODES. The native network is restricted to TRAIN nodes
only, because a cold node has an empty GIP profile -> no similarity edges ->
unreachable -> honest floor of 0. Here CONTENT gives every node (train OR cold)
similarity edges, so the heterogeneous adjacency naturally spans ALL n_l + n_d
nodes; cold nodes are reached through their content-similarity edges and receive
nonzero scores (the INTENDED effect of equipping the baseline with content). The
association block `Msub_full` is the FULL (n_l, n_d) matrix with ONLY the train
sub-block populated (everything else masked to 0) -- so supervision is read
strictly from M[np.ix_(train_lnc, train_dis)], exactly as the native GIP.

BLEND (the ONLY deviation from the native GIP-only reproduction):
    SL = alpha * SL_con + (1 - alpha) * SL_gip        # lncRNA side  (n_l x n_l)
    SD = alpha * SD_con + (1 - alpha) * SD_gip        # disease side (n_d x n_d)
    alpha = float(os.environ.get("KATZLDA_ALPHA", "0.5"))
where
    SL_gip : gip_kernel(Msub) placed on the train x train block (train-subblock
             ONLY, exactly as native), zeros elsewhere.
    SD_gip : gip_kernel(Msub.T) placed on the train x train block, zeros else.
    SL_con : content_cosine(expr), full (n_l, n_l), with
             expr = np.load(os.environ["CCDIFF_LNC_EXPR"]) if that env path is a
             valid (n_l, k) .npy, else the intrinsic content matrix Clnc.
    SD_con : full (n_d, n_d), with
             semsim = np.load(os.environ["CCDIFF_DIS_SEMSIM"]) if that env path is
             a valid (n_d, n_d) .npy (a precomputed DO-DAG semantic-similarity
             matrix), else content_cosine(Cdis).

Sub-block invariance is PRESERVED: SL_con / SD_con are M-INDEPENDENT (expression /
semantic content, not associations); SL_gip / SD_gip and Msub_full read strictly
the TRAIN association sub-block. Scrambling any entry outside
M[np.ix_(train_lnc, train_dis)] cannot change predict().
"""
import os

import numpy as np

# Shared, leakage-safe helpers. Canonical import form (paper dir is on sys.path).
from bench.interface import subblock, gip_kernel, content_cosine, SEED   # noqa: F401

NAME = "KATZLDA-content (semsim+expr)"

# In-fold beta grid, expressed as fractions of 1/rho(A). Every fraction < 1 so
# beta * rho(A) < 1 and (I - beta*A) stays invertible with a convergent series.
_BETA_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)

# Fraction of observed train associations held out for the in-fold beta sweep.
_HELD_FRAC = 0.2


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


class _KatzLDAContent:
    """Closed-form Katz on a (content + train-GIP) heterogeneous network."""

    def __init__(self, device="cpu"):
        self.device = device            # CPU only; ignored.
        self.rng = np.random.default_rng(SEED)   # determinism handle (internal hold-out)
        self.S = None
        self.beta = None                # chosen in-fold damping (diagnostic)

    # -- spectral radius of the symmetric adjacency (to bound the safe beta) --
    @staticmethod
    def _spectral_radius(A):
        n = A.shape[0]
        if n == 0:
            return 0.0
        # A is symmetric by construction -> eigvalsh is exact and deterministic.
        try:
            w = np.linalg.eigvalsh(A)
            r = float(np.max(np.abs(w)))
        except np.linalg.LinAlgError:
            # Fallback: deterministic power iteration.
            v = np.ones(n, dtype=np.float64) / np.sqrt(n)
            r = 0.0
            for _ in range(200):
                w = A @ v
                nrm = float(np.linalg.norm(w))
                if nrm <= 1e-12:
                    r = 0.0
                    break
                v = w / nrm
                r = nrm
        return r if np.isfinite(r) else 0.0

    @staticmethod
    def _assemble(SL, SD, Mfull, nL, nD):
        """Symmetric heterogeneous adjacency A = [[SL, Mfull], [Mfull.T, SD]]."""
        N = nL + nD
        A = np.zeros((N, N), np.float64)
        A[:nL, :nL] = SL
        A[:nL, nL:] = Mfull
        A[nL:, :nL] = Mfull.T
        A[nL:, nL:] = SD
        return A

    @staticmethod
    def _katz_block(A, beta, nL, nD):
        """Closed-form Katz lnc x dis block: [(I - beta*A)^{-1} - I][:nL, nL:].

        Only the disease columns of the inverse are solved for (RHS = unit
        vectors for the disease nodes), which is all the off-diagonal block
        needs. The subtracted identity has no support on this off-diagonal
        block, so it is omitted.
        """
        N = nL + nD
        Mmat = np.eye(N, dtype=np.float64) - beta * A       # (I - beta*A)
        rhs = np.eye(N, dtype=np.float64)[:, nL:]            # disease unit columns (N, nD)
        inv_cols = np.linalg.solve(Mmat, rhs)               # (N, nD) = inverse[:, disease]
        block = inv_cols[:nL, :]                             # (nL, nD) lnc x dis
        return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _auc(scores, pos_mask, neg_mask):
        """Deterministic Mann-Whitney AUC of scores over pos vs neg entries."""
        pos = scores[pos_mask]
        neg = scores[neg_mask]
        if pos.size == 0 or neg.size == 0:
            return 0.5
        order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
        ranks = np.empty(order.size, np.float64)
        ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
        # average ranks for ties (mergesort keeps ordering stable/deterministic)
        allv = np.concatenate([neg, pos])
        s = allv[order]
        i = 0
        while i < s.size:
            j = i + 1
            while j < s.size and s[j] == s[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = ranks[order[i:j]].mean()
            i = j
        pos_ranks = ranks[neg.size:]
        n_pos, n_neg = pos.size, neg.size
        return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def _select_beta(self, Msub, betas, tl, td, nL, nD, SL_con, SD_con, alpha):
        """In-fold beta selection via an internal masked-association hold-out.

        A deterministic subset of the observed TRAIN associations (the 1s in
        Msub) is masked to 0; the GIP blocks are recomputed on the MASKED
        sub-block and re-blended with the (M-independent) content similarities;
        the full-node adjacency is reassembled; each beta is scored by how well
        it ranks the masked (held-out) TRAIN positives above the never-observed
        TRAIN pairs (AUC). Evaluation is confined to the train sub-block, so no
        test pair and no cold node is ever consulted.
        """
        nTl, nTd = tl.size, td.size
        pos_r, pos_c = np.where(Msub > 0)                   # local train-block coords
        n_pos = pos_r.size
        n_hold = int(round(_HELD_FRAC * n_pos))
        # Need at least one held-out positive and one remaining positive to keep
        # the network non-empty; otherwise skip the sweep (fall back to caller).
        if n_pos < 2 or n_hold < 1 or n_hold >= n_pos:
            return None
        sel = self.rng.choice(n_pos, size=n_hold, replace=False)
        held_r, held_c = pos_r[sel], pos_c[sel]

        Mtr = Msub.copy()
        Mtr[held_r, held_c] = 0.0                           # hide the held-out positives

        # Rebuild full-node adjacency with the MASKED train sub-block.
        Mfull = np.zeros((nL, nD), np.float64)
        Mfull[np.ix_(tl, td)] = Mtr
        SLm = alpha * SL_con
        SDm = alpha * SD_con
        SLm[np.ix_(tl, tl)] += (1.0 - alpha) * gip_kernel(Mtr).astype(np.float64)
        SDm[np.ix_(td, td)] += (1.0 - alpha) * gip_kernel(Mtr.T).astype(np.float64)
        Am = self._assemble(SLm, SDm, Mfull, nL, nD)

        # Evaluation masks (in GLOBAL n_l x n_d coords, restricted to the train
        # sub-block): positives = held-out train entries; negatives = train pairs
        # that are 0 in the ORIGINAL Msub (never observed). Remaining-observed
        # train pairs and every cold/off-train pair are excluded.
        pos_mask = np.zeros((nL, nD), bool)
        pos_mask[tl[held_r], td[held_c]] = True
        neg_mask = np.zeros((nL, nD), bool)
        neg_mask[np.ix_(tl, td)] = (Msub <= 0)

        best_beta, best_auc = betas[0], -np.inf
        for b in betas:
            blk = self._katz_block(Am, b, nL, nD)
            auc = self._auc(blk, pos_mask, neg_mask)
            if auc > best_auc:
                best_auc, best_beta = auc, b
        return best_beta

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        nL, nD = self.n_l, self.n_d
        tl = np.asarray(train_lnc)
        td = np.asarray(train_dis)
        nTl, nTd = tl.size, td.size

        S_full = np.zeros((nL, nD), np.float32)
        if nTl == 0 or nTd == 0:
            # No usable train supervision -> everything is at the cold floor.
            self.S = S_full
            self.beta = 0.0
            return self

        Msub = subblock(M, tl, td).astype(np.float64)     # (nTl, nTd) association supervision

        # -- association block: FULL (n_l, n_d), train sub-block ONLY --
        Mfull = np.zeros((nL, nD), np.float64)
        Mfull[np.ix_(tl, td)] = Msub

        # -- CONTENT similarities (M-INDEPENDENT; literal intrinsic content) --
        # lncRNA side: expression matrix -> cosine; disease side: DO-DAG semsim.
        expr = _load_content_npy("CCDIFF_LNC_EXPR", nL, square=False)
        SL_con = np.asarray(content_cosine(expr) if expr is not None
                            else content_cosine(Clnc), np.float64)     # (n_l, n_l)
        ss = _load_content_npy("CCDIFF_DIS_SEMSIM", nD, square=True)
        SD_con = np.asarray(ss if ss is not None
                            else content_cosine(Cdis), np.float64)      # (n_d, n_d)

        # -- BLEND: SL = a*SL_con + (1-a)*SL_gip  (GIP only on train x train) --
        # GIP similarities (train association sub-block ONLY; exactly as native).
        alpha = float(os.environ.get("KATZLDA_ALPHA", "0.5"))
        SL = alpha * SL_con.copy()
        SD = alpha * SD_con.copy()
        SL[np.ix_(tl, tl)] += (1.0 - alpha) * gip_kernel(Msub).astype(np.float64)
        SD[np.ix_(td, td)] += (1.0 - alpha) * gip_kernel(Msub.T).astype(np.float64)

        A = self._assemble(SL, SD, Mfull, nL, nD)

        # rho of the FULL network bounds every candidate beta so beta*rho < 1 and
        # (I - beta*A) stays invertible with a convergent Katz series.
        rho = self._spectral_radius(A)
        if rho <= 1e-12:
            # (near-)nilpotent / empty network: Katz block is ~0 regardless.
            self.beta = 0.0
            self.S = S_full
            return self

        betas = [f / rho for f in _BETA_FRACS]            # documented in-fold grid
        chosen = self._select_beta(Msub, betas, tl, td, nL, nD, SL_con, SD_con, alpha)
        if chosen is None:
            # Too few train positives to run the sweep -> take a safe mid grid pt.
            chosen = betas[len(betas) // 2]               # 0.5 / rho
        self.beta = float(chosen)

        block = self._katz_block(A, self.beta, nL, nD)    # (n_l, n_d) closed-form Katz
        self.S = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _KatzLDAContent(device)
