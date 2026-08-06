"""Run the benchmark (warm + cold) over the RNADisease k-core threshold sweep.

Loads each variant's M / lnc_ortho (Clnc) / disease_emb (Cdis) from ccdiff/data_rd_l{L}d{D}
and calls run_protocol with data= and results_dir= overrides, so results land in
results_sweep/bench_rd{L}{D}_{protocol}.json (checkpointed per model, fully resumable).

Usage:
  python -m bench.sweep_run                         # all variants, both protocols, all models
  python -m bench.sweep_run --variants l3d20 l3d10  # subset of variants
  python -m bench.sweep_run --only "LDAformer ..."  # subset of models (exact name match)
  python -m bench.sweep_run --protocol cold
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench.runner import run_protocol, build_registry      # noqa: E402
from ccdiff_models import get_device                       # noqa: E402

_DATA_ROOT = os.environ.get("CCDIFF_DATA_ROOT", _DATA)
RESULTS_SWEEP = os.environ.get("CCDIFF_SWEEP_RESULTS", os.path.join(_PAPER, "results_sweep"))
# ascending by size so results flow fast and large/risky variants come last
VARIANTS = ["l3d20", "l3d10", "l3d5", "l2d20", "l2d10", "l2d5"]


def load_variant(v):
    d = os.path.join(_DATA_ROOT, f"data_rd_{v}")
    lnc_file = os.environ.get("CCDIFF_LNC_FILE", "lnc_ortho.npy")   # for content-modality ablation
    dis_file = os.environ.get("CCDIFF_DIS_FILE", "disease_emb.npy")
    M = np.load(os.path.join(d, "M.npy")).astype(np.float32)
    Clnc = np.load(os.path.join(d, lnc_file)).astype(np.float32)
    Cdis = np.load(os.path.join(d, dis_file)).astype(np.float32)
    assert Clnc.shape[0] == M.shape[0] and Cdis.shape[0] == M.shape[1]
    return M, Clnc, Cdis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--protocol", nargs="*", default=["warm", "cold"], choices=["warm", "cold"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=None,
                    help="substrings; any model whose name contains one is skipped")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("CCDIFF_SEED", "2026")))
    a = ap.parse_args()
    dev = get_device()
    os.makedirs(RESULTS_SWEEP, exist_ok=True)
    only = a.only
    if a.skip:
        names = [nm for nm, _ in build_registry(dev)]
        only = [nm for nm in names if not any(s.lower() in nm.lower() for s in a.skip)]
        print(f"[sweep] --skip active; running {len(only)} models (skipping "
              f"{[nm for nm in names if nm not in only]})", flush=True)
    for v in a.variants:
        M, Clnc, Cdis = load_variant(v)
        # per-variant LITERAL content-similarity paths for content-equipped baselines
        # (harmless for other models, which never read these env vars)
        os.environ["CCDIFF_LNC_EXPR"] = os.path.join(_DATA_ROOT, f"data_rd_{v}", "lnc_expr.npy")
        os.environ["CCDIFF_DIS_SEMSIM"] = os.path.join(_DATA_ROOT, f"data_rd_{v}", "disease_semsim.npy")
        # raw disease text/doids for the PEFT LoRA encoder hero (harmless for other models)
        os.environ["CCDIFF_DIS_DOIDS"] = os.path.join(_DATA_ROOT, f"data_rd_{v}", "disease_doids.txt")
        os.environ["CCDIFF_DIS_TEXTS"] = os.path.join(_DATA_ROOT, f"data_rd_{v}", "disease_texts.json")
        for proto in a.protocol:
            t0 = time.time()
            print(f"\n########## variant={v} protocol={proto} M={M.shape} "
                  f"n_pos={int((M>0).sum())} dev={dev} seed={a.seed} ##########", flush=True)
            run_protocol(f"rd{v}", proto, only=only, data=(M, Clnc, Cdis),
                         results_dir=RESULTS_SWEEP, seed=a.seed, device=dev, verbose=True)
            print(f"########## {v}/{proto} done in {time.time()-t0:.1f}s ##########", flush=True)


if __name__ == "__main__":
    main()
