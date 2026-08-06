"""Phase 3 (low-cost arms) — true-negative / open-world robustness on l2d5 both-cold (C4).

For key models, fit on the C4 folds, get the score matrix S, and re-score the same
predictions under multiple negative schemes to show the both-cold collapse (content-blind
-> 0.5) and the hero margin are NOT artifacts of the uniform-1:1 negative choice:
  - uniform 1:1 AUPR        (the primary scheme)
  - degree-matched 1:1 AUPR (E-TN1: negatives matched to positive popularity -> removes
                             popularity inflation; Popularity should drop most)
  - native-prevalence AUPR  (E-TN6: all negatives, operational)
  - precision@20            (E-TN6: operational top-k)
Seed=2026, l2d5.
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
from sklearn.metrics import average_precision_score
from twoside_common import folds, scenario_indices

DR = f"{_DATA}/data_rd_l2d5"
os.environ["CCDIFF_DIS_DOIDS"] = f"{DR}/disease_doids.txt"
os.environ["CCDIFF_DIS_TEXTS"] = f"{DR}/disease_texts.json"
os.environ.setdefault("PEFT_EPOCHS", "100"); os.environ.setdefault("PEFT_R", "8")
os.environ.setdefault("TT_M", "128"); os.environ.setdefault("TT_WD", "1e-3")
os.environ.setdefault("TT_DROPOUT", "0.2"); os.environ.setdefault("TT_EPOCHS", "500")
os.environ["CCDIFF_LNC_EXPR"] = f"{DR}/lnc_expr.npy"; os.environ["CCDIFF_DIS_SEMSIM"] = f"{DR}/disease_semsim.npy"
SEED = 2026
OUT = os.path.join(_PAPER, "results_negatives")

M = np.load(f"{DR}/M.npy").astype(np.float32)
Clnc = np.load(f"{DR}/lnc_ortho.npy").astype(np.float32)
Cdis = np.load(f"{DR}/disease_emb.npy").astype(np.float32)
n_l, n_d = M.shape
lnc_deg = M.sum(1); dis_deg = M.sum(0)             # full-matrix popularity


def uniform_aupr(sub, sc, rng):
    pos = np.where(sub == 1)[0]; neg = np.where(sub == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    return average_precision_score(sub[idx], sc[idx])


def degree_matched_aupr(sub, sc, pop, rng):
    """1:1 but each negative drawn to match the popularity-bin of a positive."""
    pos = np.where(sub == 1)[0]; neg = np.where(sub == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # bin cells by popularity deciles
    edges = np.quantile(pop, np.linspace(0, 1, 11))
    binid = np.clip(np.digitize(pop, edges[1:-1]), 0, 9)
    chosen = []
    for b in binid[pos]:
        cand = neg[binid[neg] == b]
        if len(cand) == 0:
            cand = neg
        chosen.append(rng.choice(cand))
    idx = np.concatenate([pos, np.array(chosen)])
    return average_precision_score(sub[idx], sc[idx])


def native_aupr(sub, sc):
    if sub.sum() == 0 or sub.sum() == len(sub):
        return np.nan
    return average_precision_score(sub, sc)


def precision_at_k(sub, sc, k=20):
    if len(sc) < k:
        k = len(sc)
    top = np.argsort(-sc)[:k]
    return sub[top].mean()


def evaluate(S, ev_l, ev_d, rng):
    sub = M[np.ix_(ev_l, ev_d)].ravel().astype(int)
    sc = np.asarray(S)[np.ix_(ev_l, ev_d)].ravel()
    pop = (lnc_deg[ev_l][:, None] * dis_deg[ev_d][None, :]).ravel()
    return (uniform_aupr(sub, sc, rng), degree_matched_aupr(sub, sc, pop, rng),
            native_aupr(sub, sc), precision_at_k(sub, sc, 20))


def make_models(dev):
    from bench.interface import Popularity2, TwoTowerContent
    from bench.hero_peft import PEFTDiseaseHero
    from bench.baselines.vgaelda import build as vgae_build
    from bench.baselines.katzlda_content import build as katzc_build
    return [
        ("HERO LoRA-disease", lambda: PEFTDiseaseHero(device=dev)),
        ("TwoTower dot", lambda: TwoTowerContent(content_l=True, content_d=True, device=dev)),
        ("KATZLDA+content", lambda: katzc_build(dev)),
        ("VGAELDA (blind)", lambda: vgae_build(dev)),
        ("Popularity2", lambda: Popularity2()),
    ]


def main():
    from ccdiff_models import get_device
    dev = get_device()
    lf = folds(n_l, 5, seed=SEED); df = folds(n_d, 5, seed=SEED)
    print("Phase 3 — true-negative robustness, l2d5 both-cold (C4), seed=2026")
    print(f"{'model':20}{'uniform':>10}{'degree-m':>10}{'native':>9}{'P@20':>8}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "eval_negative_schemes_l2d5.json")
    out = {"variant": "l2d5", "protocol": "both", "seed": SEED,
           "schemes": ["uniform 1:1", "degree-matched 1:1", "native prevalence", "precision@20"],
           "models": {}}
    if os.path.exists(path):                       # resume, model by model
        out = json.load(open(path))
    for name, ctor in make_models(dev):
        if name in out["models"]:
            print("[skip] %s" % name, flush=True)
            continue
        rows = {"u": [], "d": [], "n": [], "p": []}
        for fi in range(5):
            tr_l, tr_d, ev_l, ev_d = scenario_indices("both", lf[fi], df[fi], n_l, n_d)
            S = ctor().fit(M, Clnc, Cdis, tr_l, tr_d).predict()
            rng = np.random.default_rng(SEED + fi)
            u, d, nat, p = evaluate(S, ev_l, ev_d, rng)
            rows["u"].append(u); rows["d"].append(d); rows["n"].append(nat); rows["p"].append(p)
        out["models"][name] = {
            k2: {"mean": float(np.nanmean(rows[k1])), "std": float(np.nanstd(rows[k1], ddof=1)),
                 "folds": [float(x) for x in rows[k1]]}
            for k1, k2 in (("u", "uniform"), ("d", "degree_matched"),
                           ("n", "native"), ("p", "precision@20"))}
        print(f"{name:20}{np.nanmean(rows['u']):>10.3f}{np.nanmean(rows['d']):>10.3f}"
              f"{np.nanmean(rows['n']):>9.3f}{np.nanmean(rows['p']):>8.3f}", flush=True)
        tmp = path + ".tmp"                        # atomic, so a kill costs one model
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)
    print("TN DONE -> %s" % path, flush=True)


if __name__ == "__main__":
    main()
