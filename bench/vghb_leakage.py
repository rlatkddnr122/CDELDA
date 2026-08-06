"""LDA-VGHB under its original transductive setting versus our leakage-safe node
hold-out (Table 5).

The released LDA-VGHB precomputes its SVD factors and VGAE node features on the
FULL association matrix before pairs are split, so a held-out row or column has
already shaped the features it is later asked to predict. bench/baselines/lda_vghb.py
reproduces that path under VGHB_LEAK=1 and our leakage-safe path under VGHB_LEAK=0;
everything else -- folds, model, seed, schedule -- is identical, so the gap between
the two is attributable to the feature-computation choice alone.

Both settings share a shortened 60-epoch VGAE so the comparison is not confounded
by schedule; that is why the leakage-safe column here differs slightly from the
LDA-VGHB column of Table 3, which uses the full schedule.

VGHB_LEAK is read at import time, so each setting needs its own process.

Run:  python -m bench.vghb_leakage --leak 0
      python -m bench.vghb_leakage --leak 1
Out:  results_vghb_leakage/vghb_{safe,original}.json
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_ROOT = _DATA
OUT = os.path.join(_PAPER, "results_vghb_leakage")
VARIANTS = ["l5d5", "l4d4", "l5d3"]
SEED = 2026
SCN = [("C-dis", "disease", "disease"),
       ("C-lnc", "lncRNA", "lncRNA"),
       ("C-both", "both", "disease")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak", type=int, choices=[0, 1], required=True)
    a = ap.parse_args()

    os.environ["VGHB_LEAK"] = str(a.leak)
    os.environ["VGHB_VGAE_EP"] = "60"          # one schedule for both settings
    os.environ.setdefault("TT_M", "128")

    from twoside_common import folds, scenario_indices, eval_block
    from ccdiff_models import get_device
    from bench.baselines.lda_vghb import build as vghb_build
    from bench.baselines import lda_vghb

    assert lda_vghb._LEAK == bool(a.leak), "VGHB_LEAK did not take effect at import"
    assert lda_vghb._VGAE_EP == 60

    dev = get_device()
    os.makedirs(OUT, exist_ok=True)
    tag = "original" if a.leak else "safe"
    path = os.path.join(OUT, "vghb_%s.json" % tag)
    out = {"setting": "original (features on full matrix)" if a.leak
           else "leakage-safe (features on train sub-block)",
           "vgae_epochs": 60, "seed": SEED, "metric": "AUPR_1to1", "variants": {}}
    if os.path.exists(path):
        out = json.load(open(path))

    for v in VARIANTS:
        if v in out["variants"]:
            print("[skip] %s" % v, flush=True)
            continue
        dr = os.path.join(DATA_ROOT, "data_rd_%s" % v)
        os.environ["CCDIFF_LNC_EXPR"] = os.path.join(dr, "lnc_expr.npy")
        os.environ["CCDIFF_DIS_SEMSIM"] = os.path.join(dr, "disease_semsim.npy")
        M = np.load(os.path.join(dr, "M.npy")).astype(np.float32)
        Clnc = np.load(os.path.join(dr, "lnc_ortho.npy")).astype(np.float32)
        Cdis = np.load(os.path.join(dr, "disease_emb.npy")).astype(np.float32)
        n_l, n_d = M.shape
        lf, df = folds(n_l, 5, seed=SEED), folds(n_d, 5, seed=SEED)

        row = {}
        for cname, scn, qaxis in SCN:
            vals = []
            for fi in range(5):
                tr_l, tr_d, ev_l, ev_d = scenario_indices(scn, lf[fi], df[fi], n_l, n_d)
                S = vghb_build(dev).fit(M, Clnc, Cdis, tr_l, tr_d).predict()
                r = eval_block(S, M, ev_l, ev_d, qaxis, seed=SEED)
                vals.append(float(r["AUPR_1to1"]))
            row[cname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "folds": vals}
            print("  %-6s %-8s %-7s %.4f±%.4f" % (tag, v, cname, row[cname]["mean"],
                                                  row[cname]["std"]), flush=True)
        out["variants"][v] = row
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)
    print("VGHB-%s DONE -> %s" % (tag, path), flush=True)


if __name__ == "__main__":
    main()
