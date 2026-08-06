"""Panel B case study: both-cold (C-both) ranking for the five representative cancers.

This is the paper's headline protocol. For a target cancer the whole disease
column is removed from training AND the paired lncRNA fold is removed, so the
candidates are lncRNAs the model has never seen either. Ranking is restricted to
that held-out lncRNA pool, exactly as scenario_indices("both", ...) defines the
C-both evaluation block.

Why this panel exists. Panel A (C-dis) turned out to be popularity-confounded:
every lncRNA is still warm there, so a baseline ranking by training degree alone
scored 20/20 on all five cancers. At C-both that confound is structurally absent
-- a held-out lncRNA has no training degree at all, because its whole row was
removed before any kernel or embedding was computed. The model must work from
the lncRNA's sequence, structure and expression and from the cancer's Disease
Ontology text, and from nothing else.

Two controls are reported next to the model. Neither is a competing method:

  oracle popularity   ranks the held-out lncRNAs by their degree in the FULL
                      matrix. No method has access to this at C-both; it is
                      reported to quantify how much of the achievable ranking is
                      explainable by "this lncRNA is well studied".
  chance              the exact hypergeometric expectation for the pool.

Outputs results_case_study/case_cboth_l2d5.json (+ scores npz).
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os
import sys
import json
import time
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = os.environ.get("CCDIFF_CASE_DATA",
                      f"{_DATA}/data_rd_l2d5")
OUT = os.path.join(_PAPER, "results_case_study")
SEED = 2026

TARGETS = {
    "DOID:684":   "Hepatocellular carcinoma",
    "DOID:10534": "Gastric cancer",
    "DOID:1612":  "Breast cancer",
    "DOID:9256":  "Colorectal cancer",
    "DOID:3908":  "Non-small cell lung cancer",
}
MODEL = "TwoTower-PEFT-disease (LoRA S-BioBERT)"


def hg(k, N, K, n):
    if K == 0 or n == 0 or n > N:
        return 1.0
    return float(sum(comb(K, i) * comb(N - K, n - i)
                     for i in range(k, min(K, n) + 1)) / comb(N, n))


def auroc_mw(scores, pos_mask, neg_mask):
    """AUROC via rank sum, plus the normal-approximation two-sided p-value."""
    s_pos, s_neg = scores[pos_mask], scores[neg_mask]
    n1, n2 = len(s_pos), len(s_neg)
    if n1 == 0 or n2 == 0:
        return None, None
    allv = np.concatenate([s_pos, s_neg])
    r = np.argsort(np.argsort(allv)).astype(np.float64) + 1.0
    # average ranks for ties
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    auc = u / (n1 * n2)
    mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    from math import erfc, sqrt
    p = erfc(abs(u - mu) / (sd * sqrt(2))) if sd > 0 else 1.0
    return float(auc), float(p)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.environ["CCDIFF_LNC_EXPR"] = os.path.join(DATA, "lnc_expr.npy")
    os.environ["CCDIFF_DIS_SEMSIM"] = os.path.join(DATA, "disease_semsim.npy")
    os.environ["CCDIFF_DIS_DOIDS"] = os.path.join(DATA, "disease_doids.txt")
    os.environ["CCDIFF_DIS_TEXTS"] = os.path.join(DATA, "disease_texts.json")

    from bench.runner import build_registry
    from ccdiff_models import get_device
    from twoside_common import folds, scenario_indices

    M = np.load(os.path.join(DATA, "M.npy")).astype(np.float32)
    Clnc = np.load(os.path.join(DATA, "lnc_ortho.npy")).astype(np.float32)
    Cdis = np.load(os.path.join(DATA, "disease_emb.npy")).astype(np.float32)
    lnc_names = [l.rstrip("\n") for l in open(os.path.join(DATA, "lnc_names.txt"))]
    doids = [l.strip() for l in open(os.path.join(DATA, "disease_doids.txt"))]
    n_l, n_d = M.shape

    lfolds = folds(n_l, 5, seed=SEED)
    dfolds = folds(n_d, 5, seed=SEED)
    fold_of = {int(j): fi for fi, f in enumerate(dfolds) for j in f}
    col = {do: doids.index(do) for do in TARGETS}
    need = sorted({fold_of[col[do]] for do in col})

    full_deg = M.sum(1)          # oracle only; never given to the model
    dev = get_device()
    ctor = dict(build_registry(dev))[MODEL]

    out = {"protocol": "C-both (both-cold, headline)", "variant": "l2d5", "seed": SEED,
           "model": MODEL, "M_shape": [n_l, n_d], "device": str(dev),
           "note": ("candidates are the held-out lncRNA fold, which has no training "
                    "degree by construction, so the Panel A popularity confound "
                    "cannot operate"),
           "cancers": {}}
    dump = {}

    for fi in need:
        t0 = time.time()
        tr_l, tr_d, ev_l, ev_d = scenario_indices("both", lfolds[fi], dfolds[fi],
                                                  n_lnc=n_l, n_dis=n_d)
        mdl = ctor().fit(M, Clnc, Cdis, tr_l, tr_d)
        S = mdl.predict()
        assert np.isfinite(S).all(), "non-finite scores"
        print(f"[fold {fi}] fitted in {time.time()-t0:.1f}s  "
              f"train {len(tr_l)}x{len(tr_d)}  eval pool {len(ev_l)} lncRNAs", flush=True)

        pool = np.asarray(ev_l)
        for do, disp in TARGETS.items():
            j = col[do]
            if fold_of[j] != fi:
                continue
            sc = np.asarray(S[pool, j], dtype=np.float64)
            truth = np.asarray(M[pool, j] > 0)
            others = [col[o] for o in TARGETS if o != do]
            specific = truth & ~(M[np.ix_(pool, others)] > 0).any(1)
            deg = full_deg[pool]
            n_c, n_pos = len(pool), int(truth.sum())

            rng = np.random.default_rng(SEED)
            tie = rng.permutation(n_c)
            rec = {"n_cand": n_c, "n_pos": n_pos, "n_specific": int(specific.sum()),
                   "prevalence": n_pos / n_c, "methods": {}}
            for lab, v in (("dual-encoder (C-both)", sc),
                           ("oracle popularity (not available to any method)", deg)):
                order = np.lexsort((tie, -v))
                h20 = int(truth[order[:20]].sum())
                h50 = int(truth[order[:50]].sum())
                a_all, p_all = auroc_mw(v, truth, ~truth)
                a_sp, p_sp = auroc_mw(v, specific, ~truth)
                rec["methods"][lab] = {
                    "hits@20": h20, "expected@20": 20 * n_pos / n_c,
                    "p@20": hg(h20, n_c, n_pos, 20),
                    "recall@50": h50 / n_pos if n_pos else None,
                    "auroc_all": a_all, "p_all": p_all,
                    "auroc_specific": a_sp, "p_specific": p_sp,
                    "spearman_with_true_degree": float(np.corrcoef(
                        np.argsort(np.argsort(v)), np.argsort(np.argsort(deg)))[0, 1]),
                    "top20": [{"rank": r + 1, "lncRNA": lnc_names[pool[i]],
                               "held_out_positive": bool(truth[i]),
                               "cancer_specific": bool(specific[i]),
                               "true_degree": int(deg[i])}
                              for r, i in enumerate(order[:20])],
                }
            out["cancers"][disp] = rec
            dump[f"{do}"] = sc.astype(np.float32)
            m = rec["methods"]["dual-encoder (C-both)"]
            o = rec["methods"]["oracle popularity (not available to any method)"]
            print(f"    {disp:<30} pool={n_c} pos={n_pos} spec={int(specific.sum())}\n"
                  f"        model  hits@20={m['hits@20']:2d} (exp {m['expected@20']:.1f}, "
                  f"p={m['p@20']:.1e})  AUROC={m['auroc_all']:.3f}  "
                  f"spec-AUROC={m['auroc_specific']:.3f} (p={m['p_specific']:.1e})  "
                  f"rho(deg)={m['spearman_with_true_degree']:.2f}\n"
                  f"        oracle hits@20={o['hits@20']:2d}  AUROC={o['auroc_all']:.3f}  "
                  f"spec-AUROC={o['auroc_specific']:.3f}", flush=True)
        del mdl, S

    jp = os.path.join(OUT, "case_cboth_l2d5.json")
    with open(jp + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(jp + ".tmp", jp)
    np.savez_compressed(os.path.join(OUT, "case_cboth_scores.tmp.npz"), **dump)
    os.replace(os.path.join(OUT, "case_cboth_scores.tmp.npz"),
               os.path.join(OUT, "case_cboth_scores.npz"))
    print(f"\nwrote {jp}", flush=True)


if __name__ == "__main__":
    main()
