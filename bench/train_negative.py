"""Training-side negative schemes for CDELDA on l2d5 (Supplementary Table S7).

Distinct from bench/true_negative.py, which varies the *evaluation* negatives and
only re-scores fixed predictions. Here the model is refitted under each scheme,
because unobserved pairs are unlabeled rather than confirmed negatives and how a
trainer treats them is a modelling decision in its own right.

The schemes are already knobs on the model (PEFT_NEG_MODE / PEFT_NEG_RATIO /
PU_PI); this driver just sweeps them under the frozen harness and writes the
result so the table has an archived source.

Run:  python -m bench.train_negative
Out:  results_negatives/train_negative_schemes_l2d5.json
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DR = f"{_DATA}/data_rd_l2d5"
OUT = os.path.join(_PAPER, "results_negatives")
SEED = 2026

os.environ["CCDIFF_DIS_DOIDS"] = "%s/disease_doids.txt" % DR
os.environ["CCDIFF_DIS_TEXTS"] = "%s/disease_texts.json" % DR
os.environ["CCDIFF_LNC_EXPR"] = "%s/lnc_expr.npy" % DR
os.environ["CCDIFF_DIS_SEMSIM"] = "%s/disease_semsim.npy" % DR
os.environ.setdefault("PEFT_EPOCHS", "100")
os.environ.setdefault("PEFT_R", "8")
os.environ.setdefault("TT_M", "128")
os.environ.setdefault("TT_WD", "1e-3")
os.environ.setdefault("TT_DROPOUT", "0.2")
os.environ.setdefault("TT_EPOCHS", "500")

from twoside_common import folds, scenario_indices, eval_block   # noqa: E402

# label -> (PEFT_NEG_MODE, PEFT_NEG_RATIO, PU_PI)
ARMS = [
    ("weighted-all (default)", "weighted", None, None),
    ("uniform 1:1", "uniform", 1, None),
    ("uniform 1:5", "uniform", 5, None),
    ("uniform 1:10", "uniform", 10, None),
    ("degree-matched 1:5", "degree", 5, None),
    ("hard-negative 1:5", "hard", 5, None),
    ("nnPU (pi = 0.02)", "pu", None, 0.02),
    ("nnPU (pi = 0.05)", "pu", None, 0.05),
    ("nnPU (pi = 0.10)", "pu", None, 0.10),
    ("nnPU (pi = 0.20)", "pu", None, 0.20),
]
# (column label, scenario, ranking axis) -- the axes the runner uses, so the
# numbers are directly comparable with Table 2.
SCN = [("C-dis", "disease", "disease"),
       ("C-lnc", "lncRNA", "lncRNA"),
       ("C-both", "both", "disease")]


def main():
    from ccdiff_models import get_device
    from bench.hero_peft import PEFTDiseaseHero

    dev = get_device()
    M = np.load("%s/M.npy" % DR).astype(np.float32)
    Clnc = np.load("%s/lnc_ortho.npy" % DR).astype(np.float32)
    Cdis = np.load("%s/disease_emb.npy" % DR).astype(np.float32)
    n_l, n_d = M.shape
    lf, df = folds(n_l, 5, seed=SEED), folds(n_d, 5, seed=SEED)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "train_negative_schemes_l2d5.json")
    out = {"variant": "l2d5", "seed": SEED, "metric": "AUPR_1to1",
           "model": "CDELDA (LoRA-disease), 100 epochs", "arms": {}}
    if os.path.exists(path):                       # resume, arm by arm
        out = json.load(open(path))

    for label, mode, ratio, pi in ARMS:
        if label in out["arms"]:
            print("[skip] %s" % label, flush=True)
            continue
        os.environ["PEFT_NEG_MODE"] = mode
        if ratio is not None:
            os.environ["PEFT_NEG_RATIO"] = str(ratio)
        if pi is not None:
            os.environ["PU_PI"] = str(pi)
        row = {}
        for cname, scn, qaxis in SCN:
            vals = []
            for fi in range(5):
                tr_l, tr_d, ev_l, ev_d = scenario_indices(scn, lf[fi], df[fi], n_l, n_d)
                S = PEFTDiseaseHero(device=dev).fit(M, Clnc, Cdis, tr_l, tr_d).predict()
                r = eval_block(S, M, ev_l, ev_d, qaxis, seed=SEED)
                vals.append(float(r["AUPR_1to1"]))
            row[cname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "folds": vals}
            print("  %-24s %-7s %.4f±%.4f" % (label, cname, row[cname]["mean"],
                                              row[cname]["std"]), flush=True)
        out["arms"][label] = row
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)                      # atomic, so a kill loses at most one arm
    print("TRAIN-NEG DONE -> %s" % path, flush=True)


if __name__ == "__main__":
    main()
