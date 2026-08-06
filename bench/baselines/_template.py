"""TEMPLATE for a reproduced-SOTA baseline (auto-discovery spec + worked example).

HOW TO ADD A BASELINE
  1. Copy this file to  bench/baselines/<method>.py   (name must NOT start with '_';
     files starting with '_' are never auto-discovered).
  2. Set NAME to a unique display name.
  3. Implement build(device) -> model, where model obeys THE CONTRACT below.
  4. Nothing else -- runner.py finds it, adds it to the registry, and checkpoints it.

THE CONTRACT (identical for every reference and baseline)
  model.fit(M, Clnc, Cdis, train_lnc, train_dis) -> self
      * M    : (n_l, n_d) {0,1}. In WARM it is M_train (held-out positives masked
               to 0) with train_lnc/train_dis = all nodes. In COLD, train_lnc/
               train_dis are SUBSETS and held-out rows/cols have zero associations.
      * Clnc : (n_l, 702) lncRNA content (lnc_ortho). Cdis: (n_d, 768) disease content.
      * MUST use ONLY M[np.ix_(train_lnc, train_dis)] as supervision. Any GIP /
        similarity / neighbour profile MUST be computed inside that sub-block.
        (The smoke test enforces this: it scrambles all off-sub-block entries and
        asserts predict() is unchanged.)
  model.predict() -> np.ndarray, shape (n_l, n_d), float32, ALL finite (no NaN/Inf).
      Collaborative / topological methods MAY degrade to 0 / popularity on cold
      nodes -- keep that honest; do NOT fabricate cold-node scores.

Module-level names the runner imports:
  NAME  : str
  build(device) -> model
"""
import numpy as np

# Shared, leakage-safe helpers. (Import works because runner puts the paper dir
# on sys.path; a bare `from bench.interface import ...` is the canonical form.)
from bench.interface import subblock, content_cosine, gip_kernel   # noqa: F401

NAME = "TEMPLATE (skipped: filename starts with _)"


class _TemplateModel:
    """Minimal, contract-correct example: content-weighted train-block popularity.

    Replace the body with the real method (e.g. DSCMF / KATZLDA / IPCARF /
    SIMCLDA). This example uses ONLY the train sub-block plus content, so it
    passes the sub-block-invariance smoke check.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl, td = np.asarray(train_lnc), np.asarray(train_dis)
        Msub = subblock(M, tl, td)                        # ONLY allowed supervision

        # Example signal: lncRNA content-kNN propagated over train-disease popularity.
        lf = np.zeros(self.n_l, np.float32)
        lf[tl] = Msub.sum(1)                             # per-train-lnc degree
        df = np.zeros(self.n_d, np.float32)
        df[td] = Msub.sum(0)                             # per-train-dis degree
        Lsim = content_cosine(Clnc)                      # (n_l, n_l) content similarity
        lnc_score = Lsim @ lf                            # cold lnc inherits neighbours' degree
        self.S = np.outer(lnc_score, df).astype(np.float32)
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _TemplateModel(device)
