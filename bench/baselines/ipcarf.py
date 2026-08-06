"""IPCARF baseline: JOINT IncrementalPCA over pair features + RandomForest (CONTENT-BLIND).

Faithful, as-published reproduction of IPCARF (Zhu R., Wang Y., Liu J.-X., et al.,
"IPCARF: improving lncRNA-disease association prediction using incremental principal
component analysis feature selection and a random forest classifier", BMC
Bioinformatics 2021;22(1):175. DOI 10.1186/s12859-021-04104-9. PMID 33794766).
Public code: https://github.com/zhurong1942/IPCARF_zr1 (IPCARF_zr/RF_IPCARF.py).
IPCARF is a purely collaborative / topological method whose named novelty is
*jointly* reducing the concatenated pair feature-vector with IncrementalPCA, then
classifying pairs with a Random Forest.

Verbatim from the reference RF_IPCARF.py: the balanced SampleFeature CSV holds
positives (first half, label 1) then an equal number of negatives (second half,
label 0) -> 1:1; the reduction is `IncrementalPCA(n_components=128).fit_transform(
SampleFeature)` over the ENTIRE stacked pos+neg pair-feature matrix (the joint
IPCA); the classifier is `RandomForestClassifier()` (sklearn defaults: 100 trees,
criterion='gini') and scoring uses `predict_proba(...)[:, 1]`. The paper's grid
search (RF_xuancan.py) reports best n_estimators=1500, but the released runnable
IPCARF pipeline uses the default 100 -- we follow the released code (100 trees).

Content-blind policy concession (ratified, records/DECISIONS.md 2026-07-01): the
original builds each node's feature from an INTEGRATED similarity (lncRNA
functional similarity + GIP ; disease Disease-Ontology semantic similarity + GIP).
We are forbidden the functional / semantic databases, so the ONLY feature source
is the Gaussian Interaction Profile (GIP) kernel of the TRAIN association
sub-block. Clnc / Cdis are received to satisfy the contract and are deliberately
NEVER read. That single substitution (integrated-sim -> GIP) is the only deviation
from the paper's structure; the ALGORITHM STRUCTURE is restored faithfully:

Pipeline (restored)
  Msub     = subblock(M, train_lnc, train_dis)                   # ONLY supervision
  Lgip     = gip_kernel(Msub)     -> (nTl, nTl) train-lnc GIP profiles
  Dgip     = gip_kernel(Msub.T)   -> (nTd, nTd) train-dis GIP profiles
  # PAIR feature (the paper's feature layout): for pair (i,j)
  #     x_{ij} = concat[ Lfeat[i] (dim nTl) , Dfeat[j] (dim nTd) ]  -> dim nTl+nTd
  # where Lfeat[cold lnc]=0, Dfeat[cold dis]=0 (empty GIP profile).
  # -- JOINT IncrementalPCA (the "IPCA" novelty), ONE model over the stacked
  #    TRAIN-PAIR feature rows, so cross-block (lnc<->dis) covariance is retained:
  ipca     = IncrementalPCA(p).fit( X_pairs_train )    # rows = pos + 1:1 neg pairs
  # -- RandomForest on the jointly-reduced pair vector:
  RF       = RandomForestClassifier(100, random_state=SEED)
             fit on ipca.transform(X_pairs_train), y in {1 (Msub==1), 0 (sampled)}
  predict  : RF positive-class proba over the FULL grid, each pair fed as
             ipca.transform(concat[Lfeat[i], Dfeat[j]]), row-chunked to bound RAM.
             Cold pairs -> zero GIP feature -> a single constant floor proba
             (honest degrade; cold nodes are NOT fabricated a differentiated score).

Contract: fit(M, Clnc, Cdis, train_lnc, train_dis) -> self ; predict() -> (n_l, n_d)
float32, all finite. Degenerate cases (no positives / one class) -> honest zeros.
Deterministic (seed=SEED). Sub-block invariant: nothing outside the train
sub-block is ever read, so scrambling off-block M leaves predictions unchanged.
"""
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier

# Shared, leakage-safe helpers + the global seed. Content helpers are deliberately
# NOT imported: this baseline is content-blind by ratified policy.
from bench.interface import SEED, subblock, gip_kernel

NAME = "IPCARF (joint-IncrementalPCA + RF, GIP-only)"


class _IPCARF:
    def __init__(self, device="cpu"):
        self.device = device          # stored for contract symmetry; sklearn is CPU
        self.S = None

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # Clnc / Cdis intentionally ignored (content-blind).
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc).ravel().astype(int)
        td = np.asarray(train_dis).ravel().astype(int)
        nTl, nTd = tl.size, td.size

        # ---- GIP features from the TRAIN sub-block ONLY (no leakage) ----------
        Msub = subblock(M, tl, td)                       # (nTl, nTd) sole supervision
        n_pos = int((Msub == 1).sum())

        # Guards: nothing to learn from -> honest all-zero degrade.
        if nTl < 2 or nTd < 2 or n_pos == 0:
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            return self

        Lgip = gip_kernel(Msub)                          # (nTl, nTl) train-lnc profiles
        Dgip = gip_kernel(Msub.T)                        # (nTd, nTd) train-dis profiles

        # ---- FULL per-node GIP feature tables; cold (held-out) nodes -> zero ---
        # Lfeat[i] is lnc i's GIP-profile row (dim nTl); a cold lnc has an empty
        # association profile -> the all-zero feature vector (honest no-info).
        Lfeat_full = np.zeros((self.n_l, nTl), np.float32)
        Dfeat_full = np.zeros((self.n_d, nTd), np.float32)
        Lfeat_full[tl] = Lgip
        Dfeat_full[td] = Dgip

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

        # PAIR feature-vectors for the training pairs (paper's feature layout):
        #   x = concat[ Lgip[train-local i] , Dgip[train-local j] ]  -> dim nTl+nTd
        X_pairs = np.concatenate([Lgip[rows_i], Dgip[rows_j]], axis=1)  # (N, nTl+nTd)
        N, F = X_pairs.shape                                            # N pairs, F feats

        # ---- JOINT IncrementalPCA over the stacked pair-feature rows ----------
        # ONE model over the concatenated (lnc||dis) feature space so cross-block
        # covariance is retained (the paper's IPCA step, restored).
        # Reference uses a FIXED n_components=128 (RF_IPCARF.py). We keep 128 but
        # cap it to IncrementalPCA's validity bound 1 <= p <= min(n_features,
        # n_train_pairs); on our small train sub-blocks p usually falls to F or N.
        p = int(max(1, min(128, F, N)))
        ipca = IncrementalPCA(n_components=p)
        ipca.fit(X_pairs)                                # deterministic (no RNG)
        self.ipca = ipca
        Z_pairs = np.asarray(ipca.transform(X_pairs), np.float32)      # (N, p)

        # ---- RandomForest on the jointly-reduced pair vectors -----------------
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
        rf.fit(Z_pairs, y)
        pos_idx = int(np.where(rf.classes_ == 1)[0][0])

        # ---- score the FULL grid, row-chunked to bound memory -----------------
        # Each pair (i,j) -> concat[Lfeat_full[i], Dfeat_full[j]] -> ipca -> RF.
        # Cold pairs feed the zero vector -> ipca maps it to the (constant) mean
        # offset -> a single constant floor proba (honest degrade).
        S = np.zeros((self.n_l, self.n_d), np.float32)
        # keep intermediate (rows*n_d, F) roughly <= 2e6 elements
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
            raise RuntimeError("IPCARF.predict() called before fit().")
        return self.S


def build(device="cpu"):
    return _IPCARF(device)
