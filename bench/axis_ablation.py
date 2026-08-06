"""E5 axis ablation on l2d5 cold: which AXIS carries cold generalization?
content on lncRNA-only / disease-only / both (vs neither = free-emb baseline).
Reports C2/C3/C4 AUPR(1:1), 5-fold mean, seed=2026.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sklearn.metrics import average_precision_score
from twoside_common import folds, scenario_indices
from twoside_models import TwoTowerContent

_VAR = os.environ.get("CCDIFF_AXIS_VARIANT", "l2d5")
DR = f"{_DATA}/data_rd_{_VAR}"
SEED = 2026
os.environ.setdefault("TT_M", "128"); os.environ.setdefault("TT_WD", "1e-3")
os.environ.setdefault("TT_DROPOUT", "0.2"); os.environ.setdefault("TT_EPOCHS", "500")
M = np.load(f"{DR}/M.npy").astype(np.float32)
Cl = np.load(f"{DR}/lnc_ortho.npy").astype(np.float32)
Cd = np.load(f"{DR}/disease_emb.npy").astype(np.float32)
n_l, n_d = M.shape
SCN = [("disease", "disease", "C2"), ("lncRNA", "lncRNA", "C3"), ("both", "disease", "C4")]


def aupr(S, ev_l, ev_d):
    sub = M[np.ix_(ev_l, ev_d)].ravel().astype(int); sc = S[np.ix_(ev_l, ev_d)].ravel()
    pos = np.where(sub == 1)[0]; neg = np.where(sub == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    neg = np.random.default_rng(SEED).choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    return average_precision_score(sub[idx], sc[idx])


def run(cl_flag, cd_flag, label):
    lf = folds(n_l, 5, seed=SEED); df = folds(n_d, 5, seed=SEED)
    res = {}
    for scn, qax, tag in SCN:
        a = []
        for fi in range(5):
            tr_l, tr_d, ev_l, ev_d = scenario_indices(scn, lf[fi], df[fi], n_l, n_d)
            S = TwoTowerContent(content_l=cl_flag, content_d=cd_flag, device="cuda").fit(
                M, Cl, Cd, tr_l, tr_d).predict()
            a.append(aupr(S, ev_l, ev_d))
        res[tag] = np.nanmean(a)
    print(f"  {label:24} C2 {res['C2']:.3f}  C3 {res['C3']:.3f}  C4 {res['C4']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    print(f"E5 axis ablation — {_VAR} cold (content on which axis carries cold generalization):", flush=True)
    run(True, True, "both (lnc+dis)")
    run(True, False, "lncRNA-only")
    run(False, True, "disease-only")
    run(False, False, "neither (free-emb)")
    print("AXIS DONE", flush=True)
