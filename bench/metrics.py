"""Evaluation metrics for the two protocols.

WARM (transductive positive-pair hold-out): warm_eval() scores the held-out
positives against the true-zero unknowns of the FULL matrix. Primary = AUPR on
a balanced 1:1 negative sample averaged over 10 draws; plus AUROC_1to1, the
full-negative AUPR_all, and Recall@20 with disease as the query.

COLD (strict node hold-out): cold_eval() is a thin wrapper over the frozen
snapshot_src eval_block(), so the cold protocol reuses the exact same evaluator
as the existing twoside harness.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_SNAP = os.path.join(_PAPER, "snapshot_src")
for _p in (_PAPER, _SNAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccdiff_common import SEED                      # noqa: E402
from twoside_common import eval_block               # noqa: E402

__all__ = ["warm_eval", "cold_eval", "SEED"]


def _recall_at_k_disease_query(S, M_full, test_pos, k=20):
    """Recall@k with DISEASE as query: per disease rank all lncRNAs, mask TRAIN
    positives to -inf, take top-k, count recovered held-out positives. Macro
    average over diseases that own >=1 held-out positive."""
    n_l, n_d = M_full.shape
    heldout = np.zeros((n_l, n_d), dtype=bool)
    heldout[test_pos[:, 0], test_pos[:, 1]] = True
    train_pos = (M_full > 0) & (~heldout)                  # observed training positives
    Smask = np.asarray(S, np.float64).copy()
    Smask[train_pos] = -np.inf                             # never re-rank a known train edge
    recalls = []
    kk = min(k, n_l)
    for j in range(n_d):
        hp = heldout[:, j]
        npos = int(hp.sum())
        if npos == 0:
            continue
        order = np.argsort(-Smask[:, j])[:kk]
        recalls.append(float(hp[order].sum()) / npos)
    return float(np.mean(recalls)) if recalls else float("nan")


def warm_eval(S, M_full, test_pos, seed=SEED, n_draws=10, k=20):
    """Warm-start metrics.

    S        : (n_l, n_d) predicted scores.
    M_full   : (n_l, n_d) the FULL true matrix (held-out positives == 1 in it).
    test_pos : (n_test, 2) held-out positive (lnc, dis) indices.
    Returns {AUPR_1to1 (primary), AUROC_1to1, AUPR_all, Recall@20}.
    """
    S = np.asarray(S)
    M_full = np.asarray(M_full)
    zr, zc = np.where(M_full == 0)                          # true-zero unknowns (train positives excluded)
    pos_s = S[test_pos[:, 0], test_pos[:, 1]]
    n_pos = len(pos_s)
    if n_pos == 0 or len(zr) == 0:
        return {"AUPR_1to1": float("nan"), "AUROC_1to1": float("nan"),
                "AUPR_all": float("nan"), "Recall@20": float("nan")}

    # balanced 1:1 sampled AUROC / AUPR (primary), averaged over n_draws
    rng = np.random.default_rng(seed)
    n_neg = min(n_pos, len(zr))
    aurocs, auprs = [], []
    for _ in range(n_draws):
        sel = rng.choice(len(zr), size=n_neg, replace=False)
        neg_s = S[zr[sel], zc[sel]]
        y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        s = np.concatenate([pos_s, neg_s])
        aurocs.append(roc_auc_score(y, s))
        auprs.append(average_precision_score(y, s))

    # full-negative AUPR (held-out positives vs ALL true zeros)
    y_all = np.concatenate([np.ones(n_pos), np.zeros(len(zr))])
    s_all = np.concatenate([pos_s, S[zr, zc]])
    aupr_all = float(average_precision_score(y_all, s_all))

    return {
        "AUPR_1to1": float(np.mean(auprs)),                # PRIMARY
        "AUROC_1to1": float(np.mean(aurocs)),
        "AUPR_all": aupr_all,
        "Recall@20": _recall_at_k_disease_query(S, M_full, test_pos, k=k),
    }


def cold_eval(S, M_full, eval_lnc, eval_dis, query_axis, seed=SEED):
    """Cold-start metrics -- thin wrapper over snapshot_src eval_block().

    Returns the full eval_block dict, which includes AUPR_1to1 (primary),
    AUC_1to1, AUPR_all, Recall@{10,20,50}, personalization and block sizes.
    query_axis: 'disease' -> per cold disease rank lncRNAs; 'lncRNA' -> vice-versa.
    """
    return eval_block(S, M_full, eval_lnc, eval_dis, query_axis, seed=seed)
