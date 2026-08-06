"""Unified, checkpointed benchmark runner.

Usage:
    CCDIFF_SEED=2026 TT_M=128 \\
      python -m bench.runner --dataset canon --protocol warm
    python -m bench.runner --dataset ld --protocol cold --only "TwoTower (content)"

  --dataset  canon | ld | rd     (-> data / data_ld / data_rd)
  --protocol warm  | cold

Model registry = simple references (Random, kNN-content, MF free-emb, Popularity2,
TwoTower content) + every auto-discovered baseline in bench/baselines/*.py.

Checkpointing: per (dataset, protocol, MODEL) results are written incrementally to
results/bench_<dataset>_<protocol>.json. A model already present in that file is
SKIPPED, so an interrupted run resumes exactly where it stopped. A model that
raises is logged and skipped (not written), so it is retried on the next run.

Best hyper-parameters are passed via env (TT_M=128, TT_WD=1e-3, TT_DROPOUT=0.2,
TT_LR=1e-3, TT_EPOCHS=500 -- the pre-registered hero HP); runner sets these as
DEFAULTS only (setdefault), so an explicit env override always wins.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os
import sys
import json
import glob
import time
import argparse
import importlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../twotower/bench
_PAPER = os.path.dirname(_HERE)                             # .../twotower
# Shared datasets live in the sibling ccdiff project (large, gitignored, regenerated there).
# Override with CCDIFF_DATA_ROOT if the data ever moves.
_DATA_ROOT = os.environ.get("CCDIFF_DATA_ROOT",
                            _DATA)
_SNAP = os.path.join(_PAPER, "snapshot_src")
for _p in (_PAPER, _SNAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench.interface import (RandomScorer, KNNContent, Popularity2,          # noqa: E402
                             TwoTowerContent, SEED)
from bench.dattn import DualAttnHero                                          # noqa: E402
from bench.hero_contrastive import ContrastiveHero                           # noqa: E402
from bench.hero_ncf import NCFHero                                           # noqa: E402
from bench.hero_bilinear import BilinearHero                                 # noqa: E402
from bench.hero_peft import PEFTDiseaseHero                                  # noqa: E402
from bench.hero_peft_methods import PEFTMethodHero                          # noqa: E402
from bench.metrics import warm_eval, cold_eval                               # noqa: E402
from twoside_common import folds, scenario_indices                          # noqa: E402
from ccdiff_models import get_device                                        # noqa: E402

DATA_DIRS = {"canon": "data", "ld": "data_ld", "rd": "data_rd"}
RESULTS = os.environ.get("CCDIFF_RESULTS") or os.path.join(_PAPER, "results")
K = 5
COLD_SCN = [("disease", "disease"), ("lncRNA", "lncRNA"), ("both", "disease")]
WARM_METRICS = ["AUPR_1to1", "AUROC_1to1", "AUPR_all", "Recall@20"]
COLD_METRICS = ["AUPR_1to1", "AUC_1to1", "AUPR_all", "Recall@20", "personalization"]


# ---------------------------------------------------------------------------
def _apply_hp_defaults():
    os.environ.setdefault("TT_M", "128")
    os.environ.setdefault("TT_WD", "1e-3")
    os.environ.setdefault("TT_DROPOUT", "0.2")
    os.environ.setdefault("TT_LR", "1e-3")
    os.environ.setdefault("TT_EPOCHS", "500")


def load_dataset(dataset,
                 lnc_file=None, dis_file=None):
    lnc_file = lnc_file or os.environ.get("CCDIFF_LNC_FILE", "lnc_ortho.npy")
    dis_file = dis_file or os.environ.get("CCDIFF_DIS_FILE", "disease_emb.npy")
    d = os.path.join(_DATA_ROOT, DATA_DIRS[dataset])
    M = np.load(os.path.join(d, "M.npy")).astype(np.float32)
    Clnc = np.load(os.path.join(d, lnc_file)).astype(np.float32)
    Cdis = np.load(os.path.join(d, dis_file)).astype(np.float32)
    assert Clnc.shape[0] == M.shape[0] and Cdis.shape[0] == M.shape[1], \
        f"shape mismatch M{M.shape} Clnc{Clnc.shape} Cdis{Cdis.shape}"
    return M, Clnc, Cdis


def discover_baselines():
    """Return [(NAME, module), ...] for every bench/baselines/<name>.py (name not '_'-prefixed)."""
    out = []
    for f in sorted(glob.glob(os.path.join(_HERE, "baselines", "*.py"))):
        name = os.path.basename(f)[:-3]
        if name.startswith("_"):
            continue
        mod = importlib.import_module(f"bench.baselines.{name}")
        if hasattr(mod, "NAME") and hasattr(mod, "build"):
            out.append((mod.NAME, mod))
        else:
            print(f"[warn] baseline {name}.py missing NAME/build -> ignored", flush=True)
    return out


def build_registry(device):
    """references (7-month roster) + auto-discovered baselines. Each entry: (name, ctor)."""
    refs = [
        ("Random", lambda: RandomScorer()),
        ("kNN-content", lambda: KNNContent()),
        ("MF (free-emb)", lambda: TwoTowerContent(content_l=False, content_d=False, device=device)),
        ("Popularity2", lambda: Popularity2()),
        ("TwoTower (content)", lambda: TwoTowerContent(content_l=True, content_d=True, device=device)),
        ("TwoTower-DualAttn (content)", lambda: DualAttnHero(h=128, heads=4, device=device)),
        ("TwoTower-Contrastive (content)", lambda: ContrastiveHero(device=device)),
        ("TwoTower-NCF (content)", lambda: NCFHero(device=device)),
        ("TwoTower-Bilinear (content)", lambda: BilinearHero(device=device)),
        ("TwoTower-PEFT-disease (LoRA S-BioBERT)", lambda: PEFTDiseaseHero(device=device)),
        ("TwoTower-PEFT-method (disease)", lambda: PEFTMethodHero(device=device)),
    ]
    base = [(nm, (lambda m=mod: m.build(device))) for nm, mod in discover_baselines()]
    return refs + base


def _check_finite(S, shape, tag):
    S = np.asarray(S)
    assert S.shape == tuple(shape), f"{tag}: predict shape {S.shape} != {tuple(shape)}"
    assert np.isfinite(S).all(), f"{tag}: predict returned non-finite (NaN/Inf) scores"


def _summ(vals):
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
            "folds": [float(x) for x in vals]}


# ---------------------------------------------------------------------------
def warm_one(ctor, M, Clnc, Cdis, warm_folds, pos, seed):
    n_l, n_d = M.shape
    all_l, all_d = np.arange(n_l), np.arange(n_d)
    acc = {m: [] for m in WARM_METRICS}
    for fold in warm_folds:
        test_pos = pos[fold]
        M_train = M.copy()
        M_train[test_pos[:, 0], test_pos[:, 1]] = 0.0        # hide held-out positives (nodes stay warm)
        mdl = ctor().fit(M_train, Clnc, Cdis, all_l, all_d)  # ALL nodes seen; masked matrix = sub-block
        S = mdl.predict()
        _check_finite(S, (n_l, n_d), "warm.predict")
        r = warm_eval(S, M, test_pos, seed=seed)             # evaluate vs FULL true M
        for m in WARM_METRICS:
            acc[m].append(r[m])
    return {m: _summ(acc[m]) for m in WARM_METRICS}


def cold_one(ctor, M, Clnc, Cdis, lfolds, dfolds, seed):
    n_l, n_d = M.shape
    out = {}
    for scn, qaxis in COLD_SCN:
        acc = {m: [] for m in COLD_METRICS}
        for fi in range(len(lfolds)):
            tr_l, tr_d, ev_l, ev_d = scenario_indices(scn, lfolds[fi], dfolds[fi],
                                                      n_lnc=n_l, n_dis=n_d)
            mdl = ctor().fit(M, Clnc, Cdis, tr_l, tr_d)
            S = mdl.predict()
            _check_finite(S, (n_l, n_d), f"cold-{scn}.predict")
            r = cold_eval(S, M, ev_l, ev_d, qaxis, seed=seed)
            for m in COLD_METRICS:
                acc[m].append(r[m])
        out[scn] = {m: _summ(acc[m]) for m in COLD_METRICS}
    return out


# ---------------------------------------------------------------------------
def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_protocol(dataset, protocol, only=None, data=None, results_dir=None,
                 seed=None, device=None, verbose=True):
    """Run one (dataset, protocol), checkpointing per model. Returns the results dict.

    data        : optional (M, Clnc, Cdis) override (used by the smoke test).
    results_dir : optional output dir override (smoke uses a scratch dir).
    """
    _apply_hp_defaults()
    seed = int(seed if seed is not None else os.environ.get("CCDIFF_SEED", SEED))
    device = device or get_device()
    results_dir = results_dir or RESULTS
    M, Clnc, Cdis = data if data is not None else load_dataset(dataset)
    n_l, n_d = M.shape

    registry = build_registry(device)
    if only:
        want = set(only)
        registry = [(nm, c) for nm, c in registry if nm in want]

    path = os.path.join(results_dir, f"bench_{dataset}_{protocol}.json")
    res = _load_json(path) or {}
    res.setdefault("dataset", dataset)
    res.setdefault("protocol", protocol)
    res.setdefault("seed", seed)
    res.setdefault("device", str(device))
    res.setdefault("M_shape", [int(n_l), int(n_d)])
    res.setdefault("n_pos", int((M > 0).sum()))
    res.setdefault("models", {})

    # shared, fully deterministic folds (identical for every model)
    if protocol == "warm":
        pos = np.argwhere(M > 0)
        warm_folds = np.array_split(np.random.default_rng(seed).permutation(len(pos)), K)
    else:
        lfolds = folds(n_l, K, seed=seed)
        dfolds = folds(n_d, K, seed=seed)

    if verbose:
        print(f"[runner] dataset={dataset} protocol={protocol} device={device} seed={seed} "
              f"M=({n_l},{n_d}) n_pos={int((M>0).sum())} models={len(registry)}", flush=True)

    for name, ctor in registry:
        if name in res["models"]:
            if verbose:
                print(f"  - skip (checkpoint) {name}", flush=True)
            continue
        t0 = time.time()
        try:
            if protocol == "warm":
                one = warm_one(ctor, M, Clnc, Cdis, warm_folds, pos, seed)
            else:
                one = cold_one(ctor, M, Clnc, Cdis, lfolds, dfolds, seed)
        except Exception as e:                     # noqa: BLE001  (resumable: skip, don't write)
            print(f"  ! ERROR {name}: {type(e).__name__}: {e}", flush=True)
            continue
        res["models"][name] = one
        _save_json(path, res)                      # incremental checkpoint
        if verbose:
            print(f"  + {name}  ({time.time()-t0:.1f}s) -> {os.path.basename(path)}", flush=True)
    if verbose and results_dir == RESULTS:
        _print_table(res, protocol)
    return res


def _print_table(res, protocol):
    models = res.get("models", {})
    if not models:
        return
    print(f"\n=== bench {res.get('dataset')} / {protocol} (seed={res.get('seed')}) ===")
    if protocol == "warm":
        print(f"  {'model':<22}" + "".join(f"{m:>12}" for m in WARM_METRICS))
        for nm, r in models.items():
            print(f"  {nm:<22}" + "".join(f"{r[m]['mean']:>12.4f}" for m in WARM_METRICS))
    else:
        for scn, _ in COLD_SCN:
            print(f"  -- {scn}-cold --")
            print(f"  {'model':<22}" + "".join(f"{m:>14}" for m in COLD_METRICS))
            for nm, r in models.items():
                s = r[scn]
                print(f"  {nm:<22}" + "".join(f"{s[m]['mean']:>14.4f}" for m in COLD_METRICS))


def main():
    ap = argparse.ArgumentParser(description="Paper benchmark runner (checkpointed).")
    ap.add_argument("--dataset", required=True, choices=list(DATA_DIRS))
    ap.add_argument("--protocol", required=True, choices=["warm", "cold"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these model names (exact match)")
    a = ap.parse_args()
    run_protocol(a.dataset, a.protocol, only=a.only)


if __name__ == "__main__":
    main()
