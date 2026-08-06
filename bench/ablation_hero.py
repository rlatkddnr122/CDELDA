"""Main-model ablation on l2d5, warm + cold (C2/C3/C4).

The main model is the hero = TwoTower + LoRA-disease (S-BioBERT). Two ablations:
  - axis:     content on lncRNA / disease / both / neither (free-emb where off)
  - modality: lncRNA content channel = ortho / kmer / rnafm / struct / expr
Reuses the frozen harness (warm_one / cold_one). AUPR(1:1), seed 2026.
Writes results_ablation_hero/{axis,modality}.json.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DR = f"{_DATA}/data_rd_l2d5"
os.environ["CCDIFF_DIS_DOIDS"] = f"{DR}/disease_doids.txt"
os.environ["CCDIFF_DIS_TEXTS"] = f"{DR}/disease_texts.json"
os.environ.setdefault("PEFT_EPOCHS", "100"); os.environ.setdefault("PEFT_R", "8")
os.environ.setdefault("TT_M", "128")
SEED = 2026; K = 5
OUT = f"{_REPO}/results_ablation_hero"
os.makedirs(OUT, exist_ok=True)

from twoside_common import folds
from bench.runner import warm_one, cold_one
from bench.hero_peft import PEFTDiseaseHero

M = np.load(f"{DR}/M.npy").astype(np.float32)
Cdis = np.load(f"{DR}/disease_emb.npy").astype(np.float32)
n_l, n_d = M.shape
POS = np.argwhere(M > 0)


def run_config(name, ctor, Clnc, tag):
    warm_folds = np.array_split(np.random.default_rng(SEED).permutation(len(POS)), K)
    lf = folds(n_l, K, seed=SEED); df = folds(n_d, K, seed=SEED)
    w = warm_one(ctor, M, Clnc, Cdis, warm_folds, POS, SEED)
    c = cold_one(ctor, M, Clnc, Cdis, lf, df, SEED)
    def met(x):
        auc = x.get("AUC_1to1") or x.get("AUROC_1to1")   # cold: AUC_1to1, warm: AUROC_1to1
        return {"AUPR": round(x["AUPR_1to1"]["mean"], 4), "AUC": round(auc["mean"], 4)}
    row = {"warm": met(w), "C2": met(c["disease"]), "C3": met(c["lncRNA"]), "C4": met(c["both"])}
    print(f"[{tag}] {name:16} "
          f"warm {row['warm']['AUPR']:.3f}/{row['warm']['AUC']:.3f}  "
          f"C2 {row['C2']['AUPR']:.3f}/{row['C2']['AUC']:.3f}  "
          f"C3 {row['C3']['AUPR']:.3f}/{row['C3']['AUC']:.3f}  "
          f"C4 {row['C4']['AUPR']:.3f}/{row['C4']['AUC']:.3f}  (AUPR/AUC)", flush=True)
    return row


def main():
    # --- axis ablation (hero with content toggles) ---
    Clnc = np.load(f"{DR}/lnc_ortho.npy").astype(np.float32)
    axis = {}
    for name, cl, cd in [("both", True, True), ("lncRNA-only", True, False),
                         ("disease-only", False, True), ("neither", False, False)]:
        ctor = (lambda cl=cl, cd=cd: PEFTDiseaseHero(content_l=cl, content_d=cd))
        axis[name] = run_config(name, ctor, Clnc, "AXIS")
        json.dump(axis, open(f"{OUT}/axis.json", "w"), indent=2)
    print("%%% AXIS DONE %%%", flush=True)

    # --- modality ablation (hero, lncRNA channel swapped) ---
    mod = {}
    for mname in ["ortho", "kmer", "rnafm", "struct", "expr"]:
        Cl = np.load(f"{DR}/lnc_{mname}.npy").astype(np.float32)
        ctor = (lambda: PEFTDiseaseHero(content_l=True, content_d=True))
        mod[mname] = run_config(mname, ctor, Cl, "MODALITY")
        json.dump(mod, open(f"{OUT}/modality.json", "w"), indent=2)
    print("%%% MODALITY DONE %%%", flush=True)
    print("%%% ABLATION HERO DONE %%%", flush=True)


if __name__ == "__main__":
    main()
