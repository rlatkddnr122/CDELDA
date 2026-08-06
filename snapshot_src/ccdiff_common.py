"""CC-DiffLDA — common utilities for the DISEASE-SIDE true cold-start (C4) experiment.

Profiles are per-DISEASE: x_d = M[:, d] in {0,1}^240 (which lncRNAs associate with disease d).
Content c_d = disease text embedding (S-BioBERT, intrinsic from Disease Ontology) in R^768.
TRUE cold-start: hold out ENTIRE diseases (columns) -> a cold disease has ZERO observed
associations, so the only signal is its content embedding. 5-fold over the 412 diseases.

All metrics are computed ONLY over the held-out (cold) diseases.
"""
import os, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = int(os.environ.get("CCDIFF_SEED", "2026"))
DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def load():
    M = np.load(os.path.join(DATA, "M.npy")).astype(np.float32)          # (n_lnc, n_dis)
    Cdis = np.load(os.path.join(DATA, "disease_emb.npy")).astype(np.float32)  # (n_dis, d)
    assert Cdis.shape[0] == M.shape[1], \
        f"disease content rows {Cdis.shape} != n_dis {M.shape[1]}"
    return M, Cdis


def disease_folds(n_dis=412, k=5, seed=SEED):
    """Node-level (column) hold-out: each fold is a disjoint set of cold diseases."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_dis)
    return [np.sort(f) for f in np.array_split(order, k)]


# ----- metrics, all restricted to cold diseases -----------------------------
def recall_at_k(scores, labels, ks=(10, 20, 50)):
    """scores,labels: (n_cold_dis, n_lnc). Per-disease Recall@k over the FULL 240-lncRNA ranking,
    macro-averaged across cold diseases that have >=1 positive."""
    out = {}
    for k in ks:
        rs = []
        for s, y in zip(scores, labels):
            npos = int(y.sum())
            if npos == 0:
                continue
            topk = np.argsort(-s)[:k]
            rs.append(float(y[topk].sum()) / npos)
        out[f"Recall@{k}"] = float(np.mean(rs)) if rs else float("nan")
    return out


def auc_aupr_global(scores, labels):
    """Global AUC/AUPR over all (cold disease, lncRNA) pairs (all negatives)."""
    y = labels.reshape(-1)
    s = scores.reshape(-1)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def auc_aupr_sampled(scores, labels, ratio=1, n_draws=10, seed=SEED):
    """Balanced 1:1 negative-sampled AUC/AUPR (literature-comparison protocol), averaged over draws."""
    rng = np.random.default_rng(seed)
    pos = np.argwhere(labels > 0)
    neg = np.argwhere(labels == 0)
    if len(pos) == 0:
        return float("nan"), float("nan")
    pos_s = scores[pos[:, 0], pos[:, 1]]
    n_neg = min(ratio * len(pos), len(neg))
    aucs, auprs = [], []
    for _ in range(n_draws):
        sel = rng.choice(len(neg), size=n_neg, replace=False)
        neg_s = scores[neg[sel, 0], neg[sel, 1]]
        y = np.concatenate([np.ones(len(pos_s)), np.zeros(n_neg)])
        s = np.concatenate([pos_s, neg_s])
        aucs.append(roc_auc_score(y, s)); auprs.append(average_precision_score(y, s))
    return float(np.mean(aucs)), float(np.mean(auprs))


def personalization(scores):
    """Mean over lncRNAs of the std across cold diseases. ~0 => every cold disease gets
    the SAME predicted profile (no personalization = the popularity collapse)."""
    return float(np.mean(np.std(scores, axis=0)))


def evaluate_cold(score_full, M_full, cold_idx, seed=SEED):
    """score_full,(scores for) cold diseases extracted; returns metric dict.
    score_full: (n_lnc, n_dis) predicted scores for cold disease columns (others ignored).
    Evaluate columns in cold_idx, transposed to (n_cold_dis, n_lnc)."""
    S = score_full[:, cold_idx].T          # (n_cold, n_lnc)
    Y = M_full[:, cold_idx].T              # (n_cold, n_lnc) ground truth
    auc_all, aupr_all = auc_aupr_global(S, Y)
    auc_11, aupr_11 = auc_aupr_sampled(S, Y, ratio=1, seed=seed)
    res = {"AUC_all": auc_all, "AUPR_all": aupr_all, "AUC_1to1": auc_11, "AUPR_1to1": aupr_11,
           "personalization": personalization(S)}
    res.update(recall_at_k(S, Y))
    return res
