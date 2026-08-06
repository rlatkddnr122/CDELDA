"""KATZLDA -- closed-form Katz index, CONTENT-BLIND reproduction.

Faithful reproduction of the published Katz measure for lncRNA-disease
association prediction:

    Chen, X. "KATZLDA: KATZ measure for the lncRNA-disease association
    prediction." Scientific Reports 5, 16840 (2015).
    https://doi.org/10.1038/srep16840
    Full text (open access): https://pmc.ncbi.nlm.nih.gov/articles/PMC4649494/
    (No official author code repository was released for KATZLDA; the method
    is a closed-form linear-algebra recipe reproduced here from the paper.)

ORIGINAL METHOD (Chen 2015). KATZLDA counts damped walks of every length on a
heterogeneous lncRNA-disease network. Let (A*)^l count the walks of length l
between two nodes; the Katz score sums them with a geometrically-decaying
damping sequence beta^l:

    S* = sum_{l>=1} beta^l (A*)^l  =  (I - beta * A*)^{-1} - I        (beta < 1/rho(A*))

The paper uses the CLOSED FORM on the right (an exact matrix inverse), NOT a
truncated finite walk sum -- so there is no fixed walk-length K; walks of all
lengths contribute, geometrically damped. The heterogeneous adjacency A* is the
symmetric 2x2 block matrix combining the integrated lncRNA-similarity network
(LS), the integrated disease-similarity network (DS), and the known association
matrix A:

    A* = [[ LS,   A  ],
          [ A^T,  DS ]]

The lncRNA x disease block of S* gives the predicted association scores. Because
A* is fixed and the score is a pure linear-algebra transform of it, KATZLDA
"could be applied to new diseases and lncRNAs without any known associations"
(Chen 2015) -- cold nodes are reached only through their similarity edges.

THIS REPRODUCTION (content-blind, leakage-safe). Everything above is preserved.
The heterogeneous adjacency is built ONLY from the train association sub-block:

    Msub = M[np.ix_(train_lnc, train_dis)]           # (nTl, nTd) -- the ONLY supervision
    SL   = gip_kernel(Msub)                           # (nTl, nTl) lncRNA-side similarity
    SD   = gip_kernel(Msub.T)                          # (nTd, nTd) disease-side similarity
    A*   = [[SL,      Msub],
            [Msub.T,  SD  ]]                           # (nTl+nTd) x (nTl+nTd), symmetric
    S*   = (I - beta * A*)^{-1} - I                    # solved exactly by np.linalg.solve

The lnc x dis block of S* (rows = train lnc, cols = train dis) is scattered back
into an (n_l, n_d) matrix at the train indices. COLD (held-out) rows/cols carry
no observed associations, so they get an empty GIP profile -> no similarity
edges -> unreachable in A* -> honest floor of 0. (In the original, cold nodes
would still be reachable through their functional/semantic similarity edges; we
cannot supply those here -- see the forced substitution below.)

FORCED CONTENT-BLIND SUBSTITUTION (ratified, non-negotiable). The original
INTEGRATED similarities are:
  * lncRNA LS = expression similarity ES + functional similarity FS + GIP KL;
  * disease DS = MeSH semantic similarity + GIP KD.
The functional / semantic / expression databases are unavailable for these three
benchmark datasets, and the content matrices Clnc / Cdis are OFF-LIMITS by the
content-blind policy. We therefore substitute LS := GIP(Msub) and DS := GIP(Msub.T),
i.e. Van Laarhoven Gaussian Interaction Profile kernels computed strictly from
the TRAIN association sub-block. This is the ONLY deviation from the paper; the
block layout, the closed-form Katz inverse, and the beta damping are unchanged.

BETA (damping). The paper introduces "nonnegative coefficient sequence beta^l ...
to dampen the contributions from longer walks" but never prints a numeric value.
The closed form converges iff beta < 1/rho(A*). We therefore select beta by an
IN-FOLD sweep over {0.1, 0.3, 0.5, 0.7, 0.9} / rho(A*) (every candidate satisfies
beta * rho(A*) < 1, so the inverse converges). The winner maximizes a purely
TRAIN-INTERNAL ranking criterion: a deterministic internal hold-out of observed
train associations is masked out, Katz is recomputed on the masked sub-block, and
each beta is scored by how well it ranks the held-out positives above the
never-observed train pairs (AUC). No test pair, and no held-out (cold) node, is
ever consulted.

CONTENT-BLIND (ratified, non-negotiable): Clnc / Cdis are received by fit() but
NEVER used. Similarity comes solely from the GIP kernel of the train sub-block.
"""
import numpy as np

# Shared, leakage-safe helpers. Canonical import form (paper dir is on sys.path).
from bench.interface import subblock, gip_kernel, SEED   # noqa: F401

NAME = "KATZLDA (Chen 2015; closed-form Katz inverse, in-fold beta, GIP-only)"

# In-fold beta grid, expressed as fractions of 1/rho(A). Every fraction < 1 so
# beta * rho(A) < 1 and (I - beta*A) stays invertible with a convergent series.
_BETA_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)

# Fraction of observed train associations held out for the in-fold beta sweep.
_HELD_FRAC = 0.2


class _KatzLDA:
    """Closed-form Katz on a train-only GIP heterogeneous network. Content-blind."""

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
    def _assemble(SL, SD, Msub, nTl, nTd):
        """Symmetric heterogeneous adjacency A = [[SL, Msub], [Msub.T, SD]]."""
        N = nTl + nTd
        A = np.zeros((N, N), np.float64)
        A[:nTl, :nTl] = SL
        A[:nTl, nTl:] = Msub
        A[nTl:, :nTl] = Msub.T
        A[nTl:, nTl:] = SD
        return A

    @staticmethod
    def _katz_block(A, beta, nTl, nTd):
        """Closed-form Katz lnc x dis block: [(I - beta*A)^{-1} - I][:nTl, nTl:].

        Only the disease columns of the inverse are solved for (RHS = unit
        vectors for the disease nodes), which is all the off-diagonal block
        needs. The subtracted identity has no support on this off-diagonal
        block, so it is omitted.
        """
        N = nTl + nTd
        Mmat = np.eye(N, dtype=np.float64) - beta * A       # (I - beta*A)
        rhs = np.eye(N, dtype=np.float64)[:, nTl:]           # disease unit columns (N, nTd)
        inv_cols = np.linalg.solve(Mmat, rhs)               # (N, nTd) = inverse[:, disease]
        block = inv_cols[:nTl, :]                            # (nTl, nTd) lnc x dis
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

    def _select_beta(self, Msub, betas, nTl, nTd):
        """In-fold beta selection via an internal masked-association hold-out.

        A deterministic subset of the observed train associations (the 1s in
        Msub) is masked to 0; the GIP blocks and Katz are recomputed on the
        MASKED sub-block; each beta is scored by how well it ranks the masked
        (held-out) positives above the never-observed train pairs (AUC). The
        beta with the best AUC wins. Purely train-internal -- no test access.
        """
        pos_r, pos_c = np.where(Msub > 0)
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
        SLm = gip_kernel(Mtr).astype(np.float64)
        SDm = gip_kernel(Mtr.T).astype(np.float64)
        Am = self._assemble(SLm, SDm, Mtr, nTl, nTd)

        # Evaluation masks: positives = held-out entries; negatives = pairs that
        # are 0 in the ORIGINAL Msub (never observed). Remaining-observed pairs
        # are excluded from the negative pool (they are known positives).
        pos_mask = np.zeros((nTl, nTd), bool)
        pos_mask[held_r, held_c] = True
        neg_mask = (Msub <= 0)

        best_beta, best_auc = betas[0], -np.inf
        for b in betas:
            blk = self._katz_block(Am, b, nTl, nTd)
            auc = self._auc(blk, pos_mask, neg_mask)
            if auc > best_auc:
                best_auc, best_beta = auc, b
        return best_beta

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # Clnc / Cdis deliberately IGNORED (content-blind policy).
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc)
        td = np.asarray(train_dis)
        nTl, nTd = tl.size, td.size

        S_full = np.zeros((self.n_l, self.n_d), np.float32)
        N = nTl + nTd
        if N == 0 or nTl == 0 or nTd == 0:
            # No usable train network -> everything is at the cold floor.
            self.S = S_full
            self.beta = 0.0
            return self

        Msub = subblock(M, tl, td).astype(np.float64)     # (nTl, nTd) ONLY supervision
        SL = gip_kernel(Msub).astype(np.float64)          # (nTl, nTl)
        SD = gip_kernel(Msub.T).astype(np.float64)        # (nTd, nTd)
        A = self._assemble(SL, SD, Msub, nTl, nTd)

        # rho of the FULL train network bounds every candidate beta. Because a
        # masked sub-network has rho' <= rho, frac/rho keeps beta*rho' < 1 there
        # too, so the in-fold inverses are all convergent.
        rho = self._spectral_radius(A)
        if rho <= 1e-12:
            # (near-)nilpotent / empty network: Katz block is ~0 regardless.
            self.beta = 0.0
            self.S = S_full
            return self

        betas = [f / rho for f in _BETA_FRACS]            # documented in-fold grid
        chosen = self._select_beta(Msub, betas, nTl, nTd)
        if chosen is None:
            # Too few train positives to run the sweep -> take a safe mid grid pt.
            chosen = betas[len(betas) // 2]               # 0.5 / rho
        self.beta = float(chosen)

        block = self._katz_block(A, self.beta, nTl, nTd)  # (nTl, nTd) closed-form Katz
        S_full[np.ix_(tl, td)] = block.astype(np.float32)

        self.S = np.nan_to_num(S_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _KatzLDA(device)
