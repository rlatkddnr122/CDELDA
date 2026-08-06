"""Popularity-controlled re-analysis of the C-dis case study.

Motivation. At C-dis every lncRNA is still warm, so a method can score well at
top-20 simply by listing the globally best-studied lncRNAs. A trivial baseline
that ranks lncRNAs by their TRAINING degree alone (no content, no disease
information whatsoever) recovers 20/20 for every one of the five cancers, and
its top-5 is the same list each time. Any hits@20 figure evaluated against a
uniform hypergeometric null is therefore uninformative.

This script re-scores the saved rankings under three popularity controls:

  degree-baseline   the null itself: rank by training degree, reported as a row
  non-hub pool      restrict candidates to lncRNAs with training degree <= HUB,
                    so promiscuous loci cannot carry the ranking
  specific positives  restrict the positives to lncRNAs curated for this cancer
                    but for none of the other four targets, i.e. the part of the
                    signal that is actually cancer-specific

A method only demonstrates disease-specific cold-start ability if it beats the
degree baseline under these controls.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os
import sys
import json
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = f"{_DATA}/data_rd_l2d5"
RES = os.path.join(_PAPER, "results_case_study")
SEED = 2026
HUB = 5          # "non-hub" = curated for at most this many training diseases

NAMES = {"DOID:684": "Hepatocellular carcinoma", "DOID:10534": "Gastric cancer",
         "DOID:1612": "Breast cancer", "DOID:9256": "Colorectal cancer",
         "DOID:3908": "Non-small cell lung cancer"}
DISPLAY = {
    "TwoTower-PEFT-disease (LoRA S-BioBERT)": "Dual-encoder main (LoRA)",
    "TwoTower (content)": "Dual-encoder (dot)",
    "KATZLDA-content (semsim+expr)": "KATZLDA + content",
}


def hg(k, N, K, n):
    if K == 0 or n == 0:
        return 1.0
    return float(sum(comb(K, i) * comb(N - K, n - i)
                     for i in range(k, min(K, n) + 1)) / comb(N, n))


def topk_hits(scores, truth, pool, k, rng):
    """hits@k for a ranking restricted to `pool` (boolean mask), random tie-break."""
    idx = np.flatnonzero(pool)
    tie = rng.permutation(len(idx))
    order = idx[np.lexsort((tie, -scores[idx]))]
    return int(truth[order[:k]].sum()), order[:k]


def main():
    from twoside_common import folds
    M = np.load(os.path.join(DATA, "M.npy"))
    names = [l.rstrip("\n") for l in open(os.path.join(DATA, "lnc_names.txt"))]
    doids = [l.strip() for l in open(os.path.join(DATA, "disease_doids.txt"))]
    Z = np.load(os.path.join(RES, "case_cdis_scores.npz"))
    n_l, n_d = M.shape

    dfolds = folds(n_d, 5, seed=SEED)
    fold_of = {int(j): fi for fi, f in enumerate(dfolds) for j in f}
    tcols = {do: doids.index(do) for do in NAMES}

    report = {"hub_threshold": HUB, "cancers": {}}
    print(f"# Popularity-controlled C-dis case study (non-hub = training degree <= {HUB})\n")

    for do, disp in NAMES.items():
        j = tcols[do]
        fi = fold_of[j]
        tr_d = np.array([c for c in range(n_d) if fold_of[c] != fi])
        deg = M[:, tr_d].sum(1)                       # training degree: leakage-safe
        truth = M[:, j] > 0
        # cancer-specific positives: curated for this cancer and no other target
        others = [tcols[o] for o in NAMES if o != do]
        specific = truth & ~(M[:, others] > 0).any(1)
        nonhub = deg <= HUB

        rows = {}
        rng = lambda: np.random.default_rng(SEED)
        cands = [("degree baseline (no content)", deg.astype(np.float64))]
        for key, lab in DISPLAY.items():
            k = f"{key}||{do}"
            if k in Z:
                cands.append((lab, Z[k].astype(np.float64)))

        def spearman(a, b):
            ra = np.argsort(np.argsort(a)).astype(np.float64)
            rb = np.argsort(np.argsort(b)).astype(np.float64)
            ra -= ra.mean(); rb -= rb.mean()
            return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

        print(f"## {disp} [{do}]  positives={int(truth.sum())}  "
              f"specific-only={int(specific.sum())}  non-hub candidates={int(nonhub.sum())}")
        print(f"   {'method':<30}{'hits@20 all':>12}{'hits@20 non-hub':>17}"
              f"{'hits@20 specific':>18}{'rho(degree)':>13}")
        for lab, sc in cands:
            h_all, _ = topk_hits(sc, truth, np.ones(n_l, bool), 20, rng())
            h_nh, top_nh = topk_hits(sc, truth, nonhub, 20, rng())
            h_sp, _ = topk_hits(sc, specific, np.ones(n_l, bool), 20, rng())
            p_nh = hg(h_nh, int(nonhub.sum()), int(truth[nonhub].sum()), 20)
            rho = spearman(sc, deg)
            rows[lab] = {"hits20_all": h_all, "hits20_nonhub": h_nh,
                         "p_nonhub": p_nh, "hits20_specific": h_sp,
                         "spearman_with_degree": round(rho, 3),
                         "top20_nonhub": [names[i] for i in top_nh]}
            print(f"   {lab:<30}{h_all:>12}{h_nh:>10} (p={p_nh:.1e}){h_sp:>10}{rho:>13.3f}")
        report["cancers"][disp] = {
            "doid": do, "n_pos": int(truth.sum()), "n_specific": int(specific.sum()),
            "n_nonhub": int(nonhub.sum()),
            "n_pos_nonhub": int(truth[nonhub].sum()), "methods": rows}
        print()

    out = os.path.join(RES, "popularity_controlled.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
