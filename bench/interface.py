"""Model-interface contract + shared helpers + simple reference models.

=============================  THE CONTRACT  =================================
Every model (reference or reproduced baseline) is an object with two methods:

    fit(self, M, Clnc, Cdis, train_lnc, train_dis) -> self
        MUST use ONLY the association sub-block  M[np.ix_(train_lnc, train_dis)]
        as supervision. Any similarity / GIP kernel / neighbour profile MUST be
        computed strictly inside that sub-block (train indices) to avoid leakage.
    predict(self) -> S
        numpy array of shape (n_l, n_d), dtype float32, ALL entries finite
        (no NaN / Inf). Content-free / collaborative methods MAY degrade to
        0 / popularity for held-out (cold) nodes — that honest degradation is
        itself the evidence, so do NOT fabricate scores for cold nodes.

Inputs
  M     : (n_l, n_d) {0,1} association matrix. In the WARM protocol this is
          M_train (held-out positives already masked to 0); the whole matrix is
          the "train sub-block" because train_lnc / train_dis are all nodes.
          In the COLD protocol train_lnc / train_dis are node SUBSETS, so
          held-out rows/cols carry zero observed associations.
  Clnc  : (n_l, 702) lncRNA content  (lnc_ortho.npy = RNA-FM (+) structure (+) expression)
  Cdis  : (n_d, 768) disease content (S-BioBERT over Disease Ontology definitions)

Similarity convention (for methods that need one): lncRNA similarity = cosine of
Clnc, disease similarity = cosine of Cdis; GIP kernel from the association
sub-block only. Use the helpers below so every model shares the same definitions.
=============================================================================
"""
import os
import sys

import numpy as np

# --- path bootstrap so snapshot_src modules (twoside_models etc.) import -----
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../paper/bench
_PAPER = os.path.dirname(_HERE)                             # .../paper
_SNAP = os.path.join(_PAPER, "snapshot_src")
for _p in (_PAPER, _SNAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export the two snapshot reference models so callers import everything from
# one place. Popularity2 / TwoTowerContent already obey the contract.
from twoside_models import Popularity2, TwoTowerContent           # noqa: E402
from ccdiff_common import SEED                                    # noqa: E402

__all__ = [
    "SEED",
    "content_cosine", "cosine_sim", "gip_kernel", "subblock", "integrated_row_sim",
    "RandomScorer", "KNNContent",
    "Popularity2", "TwoTowerContent",
]


# ---------------------------------------------------------------------------
# Shared helpers (leakage-safe by construction: they only touch what you pass)
# ---------------------------------------------------------------------------
def _l2norm(C):
    C = np.asarray(C, np.float32)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)


def cosine_sim(A, B):
    """Row-wise cosine similarity between two content matrices -> (len(A), len(B))."""
    return (_l2norm(A) @ _l2norm(B).T).astype(np.float32)


def content_cosine(C):
    """Cosine self-similarity of a content matrix -> (n, n), float32."""
    return cosine_sim(C, C)


def gip_kernel(assoc_block):
    """Gaussian Interaction Profile kernel over the ROWS of an association sub-block.

    assoc_block : (n_rows, n_cols) {0,1} sub-block (already restricted to train
                  indices on BOTH axes -> no leakage). Row i's profile is
                  assoc_block[i]; K[i,j] = exp(-gamma * ||x_i - x_j||^2) with
                  gamma = 1 / mean_i ||x_i||^2 (the standard Van Laarhoven norm).
    Returns (n_rows, n_rows) float32 kernel. Callers that want the column-side
    GIP simply pass assoc_block.T.
    """
    X = np.asarray(assoc_block, np.float32)
    if X.shape[0] == 0:
        return np.zeros((0, 0), np.float32)
    sq = (X * X).sum(1)                                  # ||x_i||^2
    gamma = X.shape[0] / (sq.sum() + 1e-8)               # 1 / mean(||x||^2)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return np.exp(-gamma * D).astype(np.float32)


def subblock(M, train_lnc, train_dis):
    """The ONLY supervision a fit() is allowed to read: M[np.ix_(train_lnc, train_dis)]."""
    return np.asarray(M)[np.ix_(np.asarray(train_lnc), np.asarray(train_dis))]


def integrated_row_sim(C, assoc_subblock_rowaxis, train_idx, alpha=0.5):
    """Content-aware, leakage-safe node-node similarity for the reproduced baselines.

    Returns a FULL (n, n) similarity (n = len(C)) built as:
        * content cosine of C EVERYWHERE (intrinsic; available for cold nodes too), and
        * on the train x train block ONLY, a convex blend with the Gaussian
          Interaction Profile kernel computed strictly from the association
          sub-block:  S_blk = alpha * cos + (1-alpha) * GIP.
    This mirrors the classic "integrated similarity" of the LDA-family papers
    (GIP where associations are observed, functional/content similarity elsewhere),
    but substitutes our intrinsic content (RNA-FM/S-BioBERT) for the original
    functional-similarity databases we do not have for these 3 datasets.

    Leakage safety: the GIP term sees ONLY `assoc_subblock_rowaxis`, whose ROWS
    must correspond to `train_idx` (pass Msub for the lncRNA side, Msub.T for the
    disease side). Cold rows/cols (not in train_idx) keep pure content cosine, so
    a collaborative/topological baseline still degrades honestly on cold nodes --
    but it can NEVER be accused of being denied the content TwoTower uses.
    """
    S = content_cosine(C).copy()                                   # (n, n) intrinsic
    ti = np.asarray(train_idx)
    if ti.size:
        G = gip_kernel(assoc_subblock_rowaxis)                     # (nT, nT) train-row GIP
        S[np.ix_(ti, ti)] = (alpha * S[np.ix_(ti, ti)] + (1.0 - alpha) * G)
    return S.astype(np.float32)


def _topk_rows(sim, k):
    """Keep the top-k largest entries per row, zero the rest (kNN sparsification)."""
    if k is None or k <= 0 or k >= sim.shape[1]:
        return sim
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    out = np.zeros_like(sim)
    np.put_along_axis(out, idx, np.take_along_axis(sim, idx, axis=1), axis=1)
    return out


# ---------------------------------------------------------------------------
# Reference models (simple, dependency-free)
# ---------------------------------------------------------------------------
class RandomScorer:
    """Seeded uniform-random scores. Chance-level lower reference (ignores everything)."""

    def __init__(self, seed=SEED):
        self.seed = int(seed)

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        self.shape = np.asarray(M).shape
        return self

    def predict(self):
        return np.random.default_rng(self.seed).random(size=self.shape, dtype=np.float32)


class KNNContent:
    """Content-cosine label propagation:  S = Lsim @ M_sub @ Dsim.

    Lsim[i, a] = cos(Clnc_i, Clnc_{train_lnc[a]})   (all lnc  x  train lnc)
    Dsim[b, j] = cos(Cdis_{train_dis[b]}, Cdis_j)   (train dis x  all dis)
    M_sub      = M[np.ix_(train_lnc, train_dis)]     (ONLY supervision touched)

    Similarities are clipped to >=0, sparsified to the top-k content neighbours,
    and row/column normalised so each score is a neighbour-weighted average of
    observed train associations. Because the signal flows through CONTENT, cold
    (held-out) nodes still receive non-trivial scores -- this is the reference
    that is *supposed* to generalize, unlike collaborative/topological baselines.
    """

    def __init__(self, k=10, seed=SEED):
        self.k = int(os.environ.get("BENCH_KNN_K", k))
        self.seed = int(seed)

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        tl = np.asarray(train_lnc)
        td = np.asarray(train_dis)
        Msub = M[np.ix_(tl, td)]                                   # (nTl, nTd) supervision only
        Lsim = np.maximum(cosine_sim(Clnc, Clnc[tl]), 0.0)         # (n_l, nTl)
        Dsim = np.maximum(cosine_sim(Cdis[td], Cdis), 0.0)         # (nTd, n_d)
        Lsim = _topk_rows(Lsim, self.k)                            # top-k train-lnc neighbours per row
        Dsim = _topk_rows(Dsim.T, self.k).T                        # top-k train-dis neighbours per candidate col
        Lw = Lsim / (Lsim.sum(1, keepdims=True) + 1e-8)           # weighted avg over train lnc
        Dw = Dsim / (Dsim.sum(0, keepdims=True) + 1e-8)           # weighted avg over train dis
        S = Lw @ Msub @ Dw                                         # (n_l, n_d)
        self.S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self

    def predict(self):
        return self.S
