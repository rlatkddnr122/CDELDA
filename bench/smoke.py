"""Smoke test -- synthetic tiny data, CPU-forced, NO real full-data training.

Verifies, without ever touching real data or crashing the host:
  1. every reference model's fit()/predict() returns a finite (n_l, n_d) float array;
  2. SUB-BLOCK INVARIANCE: scrambling every entry OUTSIDE M[np.ix_(train_lnc, train_dis)]
     leaves predict() unchanged -> the model used ONLY the train sub-block (no leakage);
  3. the runner's warm and cold paths run end-to-end and emit metric dicts;
  4. checkpoint resume: a second run_protocol() call skips already-done models;
  5. the auto-discovery convention (NAME/build) works and the _template is excluded.

Run:  CUDA_VISIBLE_DEVICES= python /home/.../paper/bench/smoke.py
"""
import os
import sys
import warnings

# ---- force CPU + tiny/fast BEFORE importing torch-backed modules -----------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CCDIFF_SEED", "2026")
os.environ["TT_EPOCHS"] = "5"           # keep neural refs fast on the tiny toy data
os.environ["TT_M"] = "16"
os.environ["BENCH_KNN_K"] = "5"
warnings.filterwarnings("ignore")       # silence all-NaN nanmean on sparse toy folds

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench.runner import run_protocol, build_registry, discover_baselines   # noqa: E402
from bench.baselines import _template                                       # noqa: E402

DEV = "cpu"
N_L, N_D = 30, 20
D_L, D_D = 702, 768


def synth(seed=2026):
    rng = np.random.default_rng(seed)
    M = (rng.random((N_L, N_D)) < 0.25).astype(np.float32)     # ~25% dense {0,1}
    M[0, 0] = 1.0                                              # guarantee >=1 positive
    Clnc = rng.standard_normal((N_L, D_L)).astype(np.float32)
    Cdis = rng.standard_normal((N_D, D_D)).astype(np.float32)
    return M, Clnc, Cdis


def check(cond, msg):
    print(("  [ok]  " if cond else "  [FAIL] ") + msg, flush=True)
    if not cond:
        raise AssertionError(msg)


def test_reference_shapes(M, Clnc, Cdis):
    print("\n[1] reference fit/predict -> finite (n_l, n_d)")
    all_l, all_d = np.arange(N_L), np.arange(N_D)
    for name, ctor in build_registry(DEV):
        S = ctor().fit(M, Clnc, Cdis, all_l, all_d).predict()
        S = np.asarray(S)
        check(S.shape == (N_L, N_D) and np.isfinite(S).all() and S.dtype == np.float32,
              f"{name}: shape={S.shape} dtype={S.dtype} finite={np.isfinite(S).all()}")


def test_subblock_invariance(M, Clnc, Cdis):
    print("\n[2] sub-block invariance (cold split): off-block scramble must not change predict()")
    train_lnc = np.arange(0, 24)          # rows 24..29 are held out (cold)
    train_dis = np.arange(0, 16)          # cols 16..19 are held out (cold)
    # boolean mask of everything OUTSIDE the train sub-block
    off = np.ones((N_L, N_D), bool)
    off[np.ix_(train_lnc, train_dis)] = False
    rng = np.random.default_rng(7)
    M2 = M.copy()
    M2[off] = (rng.random(off.sum()) < 0.5).astype(np.float32)   # scramble held-out region only
    for name, ctor in build_registry(DEV):
        S1 = np.asarray(ctor().fit(M, Clnc, Cdis, train_lnc, train_dis).predict())
        S2 = np.asarray(ctor().fit(M2, Clnc, Cdis, train_lnc, train_dis).predict())
        same = np.allclose(S1, S2, atol=1e-4, rtol=1e-3)
        check(same, f"{name}: predict invariant to off-sub-block content (max|d|={np.max(np.abs(S1-S2)):.2e})")


def test_template_contract():
    print("\n[3] baseline auto-discovery convention (NAME/build) + _template excluded")
    disc = discover_baselines()
    check(all(not n.startswith("TEMPLATE") for n, _ in disc),
          f"_template not auto-discovered (discovered={[n for n,_ in disc]})")
    M, Clnc, Cdis = synth()
    mdl = _template.build(DEV)
    check(hasattr(_template, "NAME") and callable(_template.build),
          "_template exposes NAME:str and build(device)")
    S = np.asarray(mdl.fit(M, Clnc, Cdis, np.arange(24), np.arange(16)).predict())
    check(S.shape == (N_L, N_D) and np.isfinite(S).all(),
          f"_template model obeys contract: shape={S.shape} finite={np.isfinite(S).all()}")


def test_runner_paths(M, Clnc, Cdis):
    print("\n[4] runner warm + cold end-to-end (scratch results dir)")
    scratch = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bench_smoke_results")
    os.makedirs(scratch, exist_ok=True)
    for f in os.listdir(scratch):
        os.remove(os.path.join(scratch, f))

    res_w = run_protocol("canon", "warm", data=(M, Clnc, Cdis),
                         results_dir=scratch, device=DEV, seed=2026, verbose=True)
    ref = res_w["models"]["TwoTower (content)"]
    check(set(ref) >= {"AUPR_1to1", "AUROC_1to1", "AUPR_all", "Recall@20"},
          f"warm metric keys present: {sorted(ref)}")
    check(all(np.isfinite(ref[m]["mean"]) for m in ["AUPR_1to1", "AUROC_1to1", "AUPR_all"]),
          "warm TwoTower means finite")
    print("     warm TwoTower(content):",
          {m: round(ref[m]["mean"], 4) for m in ["AUPR_1to1", "AUROC_1to1", "AUPR_all", "Recall@20"]})

    res_c = run_protocol("canon", "cold", data=(M, Clnc, Cdis),
                         results_dir=scratch, device=DEV, seed=2026, verbose=True)
    cc = res_c["models"]["TwoTower (content)"]
    check(set(cc) == {"disease", "lncRNA", "both"}, f"cold scenarios present: {sorted(cc)}")
    check(set(cc["both"]) >= {"AUPR_1to1", "AUC_1to1", "AUPR_all", "Recall@20", "personalization"},
          f"cold metric keys present: {sorted(cc['both'])}")
    print("     cold TwoTower(content) both-cold:",
          {m: round(cc["both"][m]["mean"], 4) for m in ["AUPR_1to1", "AUC_1to1", "Recall@20"]})

    # [5] checkpoint resume: re-run must skip every model (no recompute)
    print("\n[5] checkpoint resume (second call skips finished models)")
    before = dict(res_w["models"])
    res_w2 = run_protocol("canon", "warm", data=(M, Clnc, Cdis),
                          results_dir=scratch, device=DEV, seed=2026, verbose=False)
    check(res_w2["models"].keys() == before.keys() and len(res_w2["models"]) == len(before),
          f"resume kept exactly {len(before)} checkpointed models")


def main():
    print("SMOKE: synthetic tiny data, CPU-forced, no real full-data training.")
    M, Clnc, Cdis = synth()
    print(f"synthetic M{M.shape} n_pos={int(M.sum())} Clnc{Clnc.shape} Cdis{Cdis.shape}")
    test_reference_shapes(M, Clnc, Cdis)
    test_subblock_invariance(M, Clnc, Cdis)
    test_template_contract()
    test_runner_paths(M, Clnc, Cdis)
    print("\nSMOKE PASS ✓  all reference models + runner warm/cold paths green.")


if __name__ == "__main__":
    main()
