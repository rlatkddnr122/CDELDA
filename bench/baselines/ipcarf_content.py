"""IPCARF-content: the reproduced IPCARF baseline, CONTENT-EQUIPPED.

This is an EXACT copy of the native content-blind IPCARF baseline
(bench/baselines/ipcarf.py -- joint IncrementalPCA over concatenated pair
features -> seeded RandomForest, 1:1 negatives), changing ONE thing only: the
similarity source that fills each node's feature row.

Motivation. The native reproduced IPCARF is content-blind by ratified policy:
its ONLY feature source is the Gaussian Interaction Profile (GIP) kernel of the
TRAIN association sub-block, so a held-out (cold) node has an all-zero profile
and collapses to a constant floor in the both-cold (C4) regime. That collapse is
honest, but a sceptic can object it is an artefact of *content starvation* rather
than of the algorithm. This variant removes that objection by handing IPCARF the
LITERAL intrinsic content its own paper used:

  * lncRNA side : EXPRESSION-profile similarity  (Zhu et al. use lncRNA
                  expression-similarity integrated with GIP);
  * disease side : Disease-Ontology DAG SEMANTIC similarity (Wang et al. DO-DAG
                  semantic similarity integrated with GIP).

Both are M-INDEPENDENT (they are computed from external content, never from the
association matrix), so sub-block invariance is preserved exactly: nothing
outside the train sub-block is ever read. GIP stays train-sub-block-only. The RF
stays seeded. The ONLY behavioural change is that cold nodes now receive a
DIFFERENTIATED (nonzero) content similarity instead of the zero GIP profile --
which is precisely the intended experiment: does the C4 collapse persist when the
baseline is NOT content-starved?

Similarity source (the sole deviation from native)
  Native:  Lsim = GIP(Msub)       Dsim = GIP(Msub.T)
  Here  :  a per-side convex BLEND of literal content and train-sub-block GIP
             Lsim = alpha * Lcon + (1 - alpha) * Lgip
             Dsim = alpha * Dcon + (1 - alpha) * Dgip
           alpha = float(os.environ.get("IPCARF_ALPHA", "0.5"))
    Lgip / Dgip  : GIP kernels of the TRAIN sub-block, EXACTLY as native
                   (cold nodes -> zero GIP row).
    Lcon (M-independent, defined for cold nodes too):
        content_cosine( np.load($CCDIFF_LNC_EXPR) )  if that file is a valid
        (n_l, k) .npy  (lncRNA EXPRESSION profiles)  else  content_cosine(Clnc).
    Dcon (M-independent):
        np.load($CCDIFF_DIS_SEMSIM)  if that file is a valid (n_d, n_d) .npy
        (a precomputed DO-DAG semantic-similarity matrix, used AS-IS) else
        content_cosine(Cdis).
  The blended per-node similarity rows are fed into the SAME
  IncrementalPCA(128) + RandomForest pipeline; EVERYTHING else is unchanged.

Contract: fit(M, Clnc, Cdis, train_lnc, train_dis) -> self ; predict() -> (n_l,
n_d) float32, all finite. Degenerate cases (no positives / one class) -> honest
zeros. Deterministic (seed=SEED). Sub-block invariant (content M-independent, GIP
train-sub-block-only, RF seeded): scrambling off-block M leaves predictions
unchanged (max|delta| <= 1e-4).
"""
import os

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier

# Shared, leakage-safe helpers + the global seed. Unlike the native baseline we
# DO import content_cosine: this variant is content-equipped by design.
from bench.interface import SEED, subblock, gip_kernel, content_cosine

NAME = "IPCARF-content (semsim+expr)"


def _load_valid_npy(env_key, expected_rows=None, square=False):
    """np.load($env_key) iff it is a valid 2-D .npy matching the shape guard.

    M-INDEPENDENT by construction (reads only an external file / env var, never
    the association matrix). Returns None on any failure so the caller can fall
    back to the intrinsic content matrix -- never raises.
      expected_rows : required shape[0] (n_l or n_d).
      square        : if True, require shape == (expected_rows, expected_rows).
    """
    path = os.environ.get(env_key, "")
    if not path or not os.path.isfile(path):
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return None
    if expected_rows is not None and arr.shape[0] != expected_rows:
        return None
    if square and arr.shape[1] != expected_rows:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr.astype(np.float32)


class _IPCARFContent:
    def __init__(self, device="cpu"):
        self.device = device          # stored for contract symmetry; sklearn is CPU
        self.S = None

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc).ravel().astype(int)
        td = np.asarray(train_dis).ravel().astype(int)
        nTl, nTd = tl.size, td.size

        # ---- GIP features from the TRAIN sub-block ONLY (no leakage) ----------
        Msub = subblock(M, tl, td)                       # (nTl, nTd) sole supervision
        n_pos = int((Msub == 1).sum())

        # Guards: nothing to learn from -> honest all-zero degrade (native).
        if nTl < 2 or nTd < 2 or n_pos == 0:
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            return self

        Lgip = gip_kernel(Msub)                          # (nTl, nTl) train-lnc profiles
        Dgip = gip_kernel(Msub.T)                        # (nTd, nTd) train-dis profiles

        # ---- FULL per-node GIP feature tables; cold (held-out) nodes -> zero ---
        # (native layout: feature axis = TRAIN nodes; cold node -> all-zero row)
        Lgip_full = np.zeros((self.n_l, nTl), np.float32)
        Dgip_full = np.zeros((self.n_d, nTd), np.float32)
        Lgip_full[tl] = Lgip
        Dgip_full[td] = Dgip

        # ---- LITERAL CONTENT similarity (M-INDEPENDENT; defined for cold too) --
        # lncRNA: expression-profile cosine  (or intrinsic content cosine).
        Lexpr = _load_valid_npy("CCDIFF_LNC_EXPR", expected_rows=self.n_l, square=False)
        Lcon_full = content_cosine(Lexpr if Lexpr is not None else Clnc)   # (n_l, n_l)
        # disease: precomputed DO-DAG semantic-similarity matrix, used AS-IS
        # (or intrinsic content cosine).
        Dsem = _load_valid_npy("CCDIFF_DIS_SEMSIM", expected_rows=self.n_d, square=True)
        Dcon_full = Dsem if Dsem is not None else content_cosine(Cdis)     # (n_d, n_d)

        # Restrict the content similarity to the TRAIN feature axis so the pair
        # feature dimension (nTl + nTd) is IDENTICAL to native; every ROW (incl.
        # cold nodes) still gets a differentiated, nonzero content profile.
        Lcon_cols = np.asarray(Lcon_full, np.float32)[:, tl]              # (n_l, nTl)
        Dcon_cols = np.asarray(Dcon_full, np.float32)[:, td]             # (n_d, nTd)

        # ---- BLEND: per-side convex combination of content + train-GIP --------
        alpha = float(os.environ.get("IPCARF_ALPHA", "0.5"))
        Lfeat_full = (alpha * Lcon_cols + (1.0 - alpha) * Lgip_full).astype(np.float32)
        Dfeat_full = (alpha * Dcon_cols + (1.0 - alpha) * Dgip_full).astype(np.float32)

        # ---- build the balanced PAIR training set: positives + 1:1 negatives --
        rng = np.random.default_rng(SEED)
        pos_i, pos_j = np.where(Msub == 1)               # train-local indices
        neg_ri, neg_rj = np.where(Msub == 0)
        n_neg_avail = neg_ri.size
        n_neg = min(n_pos, n_neg_avail)
        if n_neg > 0:
            sel = rng.choice(n_neg_avail, size=n_neg, replace=False)
            neg_i, neg_j = neg_ri[sel], neg_rj[sel]
        else:
            neg_i = np.empty(0, dtype=int)
            neg_j = np.empty(0, dtype=int)

        rows_i = np.concatenate([pos_i, neg_i])          # train-local lnc idx
        rows_j = np.concatenate([pos_j, neg_j])          # train-local dis idx
        y = np.concatenate([np.ones(n_pos, np.int64),
                            np.zeros(n_neg, np.int64)])

        # Only one class present -> cannot train a classifier -> honest zeros.
        if np.unique(y).size < 2:
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            return self

        # PAIR feature-vectors for the training pairs (native feature layout, now
        # over the BLENDED similarity rows of the TRAIN nodes):
        #   x = concat[ Lfeat[train i] , Dfeat[train j] ]  -> dim nTl+nTd
        X_pairs = np.concatenate([Lfeat_full[tl][rows_i],
                                  Dfeat_full[td][rows_j]], axis=1)          # (N, nTl+nTd)
        N, F = X_pairs.shape                                                # N pairs, F feats

        # ---- JOINT IncrementalPCA over the stacked pair-feature rows ----------
        # (unchanged from native: ONE model over the concatenated (lnc||dis)
        # feature space; fixed n_components=128 capped to IPCA's validity bound.)
        p = int(max(1, min(128, F, N)))
        ipca = IncrementalPCA(n_components=p)
        ipca.fit(X_pairs)                                # deterministic (no RNG)
        self.ipca = ipca
        Z_pairs = np.asarray(ipca.transform(X_pairs), np.float32)          # (N, p)

        # ---- RandomForest on the jointly-reduced pair vectors (unchanged) -----
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
        rf.fit(Z_pairs, y)
        pos_idx = int(np.where(rf.classes_ == 1)[0][0])

        # ---- score the FULL grid, row-chunked to bound memory (unchanged) -----
        # Each pair (i,j) -> concat[Lfeat_full[i], Dfeat_full[j]] -> ipca -> RF.
        # Cold pairs now carry a NONZERO content profile (intended), so they get
        # a differentiated score instead of the native constant floor.
        S = np.zeros((self.n_l, self.n_d), np.float32)
        rows_per_chunk = max(1, 2_000_000 // max(1, self.n_d * F))
        for start in range(0, self.n_l, rows_per_chunk):
            end = min(start + rows_per_chunk, self.n_l)
            nrows = end - start
            L_part = np.repeat(Lfeat_full[start:end], self.n_d, axis=0)  # (nrows*n_d, nTl)
            D_part = np.tile(Dfeat_full, (nrows, 1))                     # (nrows*n_d, nTd)
            X_part = np.concatenate([L_part, D_part], axis=1)           # (nrows*n_d, F)
            Z_part = ipca.transform(X_part)                            # joint reduce
            proba = rf.predict_proba(Z_part)[:, pos_idx]
            S[start:end] = proba.reshape(nrows, self.n_d)

        self.S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self

    def predict(self):
        if self.S is None:
            raise RuntimeError("IPCARF-content.predict() called before fit().")
        return self.S


def build(device="cpu"):
    return _IPCARFContent(device)
