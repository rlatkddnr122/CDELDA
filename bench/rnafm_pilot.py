"""Single both-cold fold pilot on l2d5: RNA-FM-LoRA lncRNA hero vs frozen dot hero.
Same fold-0 split (runner convention). Reports C4 AUPR(1:1) for both + timing.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, time, importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sklearn.metrics import average_precision_score
from twoside_common import folds, scenario_indices
from twoside_models import TwoTowerContent

DR = f"{_DATA}/data_rd_l2d5"
os.environ["CCDIFF_LNC_NAMES"] = f"{DR}/lnc_names.txt"
os.environ["CCDIFF_LNC_SEQ"] = f"{DR}/lncrna_seq.json"
os.environ.setdefault("RNAFM_MAXSTEPS", "300")
os.environ.setdefault("RNAFM_MAXWIN", "6")
SEED = 2026

M = np.load(f"{DR}/M.npy").astype(np.float32)
Cl = np.load(f"{DR}/lnc_ortho.npy").astype(np.float32)
Cd = np.load(f"{DR}/disease_emb.npy").astype(np.float32)
n_l, n_d = M.shape
lf = folds(n_l, 5, seed=SEED); df = folds(n_d, 5, seed=SEED)
tr_l, tr_d, ev_l, ev_d = scenario_indices("both", lf[0], df[0], n_l, n_d)
print(f"l2d5 both-cold fold0: train {len(tr_l)}x{len(tr_d)}  eval {len(ev_l)}x{len(ev_d)}", flush=True)


def aupr(S):
    sub = M[np.ix_(ev_l, ev_d)].ravel().astype(int); sc = S[np.ix_(ev_l, ev_d)].ravel()
    pos = np.where(sub == 1)[0]; neg = np.where(sub == 0)[0]
    neg = np.random.default_rng(SEED).choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    return average_precision_score(sub[idx], sc[idx])


# frozen dot hero
t = time.time()
S_frozen = TwoTowerContent(content_l=True, content_d=True, device="cuda").fit(M, Cl, Cd, tr_l, tr_d).predict()
print(f"[frozen dot]  C4 fold0 AUPR = {aupr(S_frozen):.4f}   ({time.time()-t:.0f}s)", flush=True)

# RNA-FM LoRA hero
spec = importlib.util.spec_from_file_location("rna", os.path.join(_HERE, "hero_peft_rna.py"))
rna = importlib.util.module_from_spec(spec); spec.loader.exec_module(rna)
t = time.time()
mdl = rna.build("cuda").fit(M, Cl, Cd, tr_l, tr_d)
tfit = time.time() - t
t = time.time()
S_rna = np.asarray(mdl.predict())
print(f"[RNA-FM LoRA] C4 fold0 AUPR = {aupr(S_rna):.4f}   (fit {tfit:.0f}s, predict {time.time()-t:.0f}s, "
      f"steps={os.environ['RNAFM_MAXSTEPS']}, win={os.environ['RNAFM_MAXWIN']})", flush=True)
print("PILOT DONE", flush=True)
