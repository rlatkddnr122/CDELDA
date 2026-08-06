"""Panel A case study: disease-cold (C-dis) ranking for five representative cancers.

Protocol is the paper's own C-dis protocol, unchanged: the 5-fold disease split
from folds(n_d, 5, seed=2026), each target cancer evaluated inside the fold it
already belongs to (so ~49 diseases are held out simultaneously, which is harder
than a leave-one-disease-out and keeps the case study identical to the run that
produced Table 2). For a held-out cancer the model sees no association at all in
its column and must rank all 5102 lncRNAs from the disease's Disease Ontology
text plus each lncRNA's intrinsic content.

Targets were fixed before looking at any model output, by four rules:
  (1) a cancer (Disease Ontology subtree of DOID:162),
  (2) present in the headline variant l2d5,
  (3) >= 350 curated lncRNA associations,
  (4) lncRNA-per-PMID < 1.0 in RNADisease v4.0, i.e. evidence accumulated over
      many independent studies rather than one high-throughput screen.

Outputs results_case_study/case_cdis_l2d5.json
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os
import sys
import json
import time

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
TOPK = 50

# DOID -> display name. Fixed in advance by the four selection rules above.
TARGETS = {
    "DOID:684":   "Hepatocellular carcinoma",
    "DOID:10534": "Gastric cancer",
    "DOID:1612":  "Breast cancer",
    "DOID:9256":  "Colorectal cancer",
    "DOID:3908":  "Non-small cell lung cancer",
}

# The four methods that make the paper's contrast visible at name level.
MODELS = [
    "TwoTower-PEFT-disease (LoRA S-BioBERT)",
    "TwoTower (content)",
    "KATZLDA-content (semsim+expr)",
    "KATZLDA (Chen 2015; closed-form Katz inverse, in-fold beta, GIP-only)",
    "Popularity2",
]


def hypergeom_sf(k, N, K, n):
    """P(X >= k) for X ~ Hypergeometric(N population, K successes, n draws)."""
    from math import comb
    total = comb(N, n)
    return float(sum(comb(K, i) * comb(N - K, n - i)
                     for i in range(k, min(K, n) + 1)) / total)


def _save(json_path, npz_path, out, score_dump):
    """Atomic checkpoint: write to a temp file, then rename, so an interrupted
    write cannot leave a half-written result behind."""
    tmp_json = json_path + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp_json, json_path)
    # np.savez_compressed appends .npz unless the name already ends in it
    tmp_npz = npz_path.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp_npz, **score_dump)
    os.replace(tmp_npz, npz_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    # env the content-equipped baselines and the LoRA encoder read
    os.environ["CCDIFF_LNC_EXPR"] = os.path.join(DATA, "lnc_expr.npy")
    os.environ["CCDIFF_DIS_SEMSIM"] = os.path.join(DATA, "disease_semsim.npy")
    os.environ["CCDIFF_DIS_DOIDS"] = os.path.join(DATA, "disease_doids.txt")
    os.environ["CCDIFF_DIS_TEXTS"] = os.path.join(DATA, "disease_texts.json")

    from bench.runner import build_registry                 # noqa: E402
    from ccdiff_models import get_device                    # noqa: E402
    from twoside_common import folds, scenario_indices      # noqa: E402

    M = np.load(os.path.join(DATA, "M.npy")).astype(np.float32)
    Clnc = np.load(os.path.join(DATA, "lnc_ortho.npy")).astype(np.float32)
    Cdis = np.load(os.path.join(DATA, "disease_emb.npy")).astype(np.float32)
    lnc_names = [l.rstrip("\n") for l in open(os.path.join(DATA, "lnc_names.txt"))]
    doids = [l.strip() for l in open(os.path.join(DATA, "disease_doids.txt"))]
    dis_texts = json.load(open(os.path.join(DATA, "disease_texts.json")))
    n_l, n_d = M.shape
    assert len(lnc_names) == n_l and len(doids) == n_d

    dfolds = folds(n_d, 5, seed=SEED)
    col = {do: doids.index(do) for do in TARGETS if do in doids}
    missing = [do for do in TARGETS if do not in col]
    if missing:
        raise SystemExit(f"targets absent from l2d5: {missing}")
    fold_of = {}
    for fi, f in enumerate(dfolds):
        for j in f:
            fold_of[int(j)] = fi
    need_folds = sorted({fold_of[col[do]] for do in col})

    # --- leakage check specific to this case study: the disease text must not
    #     name any lncRNA in the corpus (that would hand the answer to the model).
    sym = {s.upper() for s in lnc_names if len(s) >= 4}
    text_leak = {}
    for do in col:
        words = {w.strip(".,;:()[]").upper() for w in str(dis_texts.get(do, "")).split()}
        hit = sorted(words & sym)
        if hit:
            text_leak[do] = hit
    print(f"[leak-check] lncRNA symbols found in target disease texts: "
          f"{text_leak if text_leak else 'none'}", flush=True)

    dev = get_device()
    registry = dict(build_registry(dev))
    for m in MODELS:
        if m not in registry:
            raise SystemExit(f"model not in registry: {m}\navailable: {list(registry)}")

    # --- resume support: a model already present in the checkpoint is skipped, so
    #     an interrupted run (or a host reboot) costs only the model in flight.
    json_path = os.path.join(OUT, "case_cdis_l2d5.json")
    npz_path = os.path.join(OUT, "case_cdis_scores.npz")
    score_dump = {}   # (model, doid) -> full 5102-vector, so popularity-controlled
                      # re-analysis does not need another fit
    done = set()
    if os.path.exists(json_path) and os.path.exists(npz_path):
        prev = json.load(open(json_path))
        with np.load(npz_path) as z:
            score_dump = {k: z[k] for k in z.files}
        done = {m for m in prev.get("models", {})
                if all(f"{m}||{do}" in score_dump for do in TARGETS)}
        if done:
            print(f"[resume] already complete: {sorted(done)}", flush=True)

    out = {"protocol": "C-dis (disease-cold)", "variant": "l2d5", "seed": SEED,
           "M_shape": [n_l, n_d], "n_pos": int((M > 0).sum()),
           "selection_rules": ["cancer (DOID:162 subtree)", "present in l2d5",
                               ">=350 curated lncRNA associations",
                               "lncRNA-per-PMID < 1.0 in RNADisease v4.0"],
           "text_leak_check": text_leak, "device": str(dev), "models": {}}

    if done:
        out["models"].update({m: prev["models"][m] for m in done})

    for name in MODELS:
        if name in done:
            print(f"[skip] {name} (already checkpointed)", flush=True)
            continue
        ctor = registry[name]
        out["models"][name] = {}
        for fi in need_folds:
            t0 = time.time()
            tr_l, tr_d, ev_l, ev_d = scenario_indices("disease", np.array([], int), dfolds[fi],
                                                      n_lnc=n_l, n_dis=n_d)
            mdl = ctor().fit(M, Clnc, Cdis, tr_l, tr_d)
            S = mdl.predict()
            assert np.isfinite(S).all(), f"{name}: non-finite scores"
            print(f"[{name}] fold {fi} fitted in {time.time()-t0:.1f}s", flush=True)

            for do, disp in TARGETS.items():
                j = col[do]
                if fold_of[j] != fi:
                    continue
                scores = np.asarray(S[:, j], dtype=np.float64)
                truth = np.asarray(M[:, j] > 0)
                n_pos = int(truth.sum())
                # A content-blind method gives every lncRNA the same score for a
                # cold disease, so the top-k is decided entirely by the tie-break.
                # Index order is NOT neutral (lnc_names.txt is sorted, and the
                # early entries are enriched for well-studied loci), so ties are
                # broken by a seeded random permutation instead. The number of
                # distinct score values is recorded so degenerate rankings show.
                tie = np.random.default_rng(SEED).permutation(n_l)
                order = np.lexsort((tie, -scores))
                n_distinct = int(len(np.unique(scores)))
                rec = {}
                for k in (20, 50):
                    hits = int(truth[order[:k]].sum())
                    rec[f"hits@{k}"] = hits
                    rec[f"recall@{k}"] = hits / n_pos if n_pos else None
                    rec[f"expected_hits@{k}"] = k * n_pos / n_l
                    rec[f"hypergeom_p@{k}"] = hypergeom_sf(hits, n_l, n_pos, k)
                rec["precision@20"] = rec["hits@20"] / 20
                rec["n_pos"] = n_pos
                rec["n_cand"] = n_l
                rec["n_distinct_scores"] = n_distinct
                rec["degenerate_ranking"] = n_distinct <= 1
                rec["top50"] = [
                    {"rank": r + 1, "lncRNA": lnc_names[i], "index": int(i),
                     "score": float(scores[i]), "held_out_positive": bool(truth[i])}
                    for r, i in enumerate(order[:TOPK])
                ]
                out["models"][name][do] = rec
                score_dump[f"{name}||{do}"] = scores.astype(np.float32)
                print(f"    {disp:<30} n_pos={n_pos:4d}  hits@20={rec['hits@20']:2d} "
                      f"(exp {rec['expected_hits@20']:.1f}, p={rec['hypergeom_p@20']:.2e})  "
                      f"R@50={rec['recall@50']:.3f}  distinct={n_distinct}", flush=True)
            del mdl, S
        # checkpoint after every model so a crash or reboot costs at most one fit
        _save(json_path, npz_path, out, score_dump)
        print(f"[checkpoint] {name} saved", flush=True)

    path = os.path.join(OUT, "case_cdis_l2d5.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "case_cdis_scores.npz"), **score_dump)
    print(f"\nwrote {path} and case_cdis_scores.npz", flush=True)


if __name__ == "__main__":
    main()
