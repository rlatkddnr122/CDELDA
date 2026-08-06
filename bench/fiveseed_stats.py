"""Aggregate the 5-seed l2d5 cold results: per-model mean±std of C2/C3/C4,
and the paired significance of PEFT-disease vs frozen dot hero at C4/C2
(paired t-test + bootstrap 95% CI of the mean difference).
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, json
import numpy as np

_RES = f"{_REPO}/results_5seed"
SEEDS = [2026, 1, 2, 3, 4]
V = "l2d5"
MODELS = {
    "TwoTower dot (frozen)": "TwoTower (content)",
    "TwoTower dual-attn": "TwoTower-DualAttn (content)",
    "TwoTower PEFT-disease (LoRA)": "TwoTower-PEFT-disease (LoRA S-BioBERT)",
    "KATZLDA+content": "KATZLDA-content (semsim+expr)",
    "kNN-content": "kNN-content",
    "VGAELDA (content-blind)": "VGAELDA (co-trained VGAE + label-prop resolvent, content-blind)",
}
SCN = {"C2": "disease", "C3": "lncRNA", "C4": "both"}


def load(seed, key, scn):
    f = os.path.join(_RES, f"seed{seed}", f"bench_rd{V}_cold.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))["models"].get(key)
    return m[scn]["AUPR_1to1"]["mean"] if m else None


def series(key, scn):
    return np.array([load(s, key, scn) for s in SEEDS], float)


def paired_t(diffs):
    d = diffs[~np.isnan(diffs)]
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return d.mean(), float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_rel(d + 0, np.zeros(n))  # one-sample t on diffs vs 0
    return d.mean(), t, p


def boot_ci(diffs, B=10000, seed=0):
    d = diffs[~np.isnan(diffs)]
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    print(f"5-seed {SEEDS} on {V} cold — mean±std AUPR(1:1)\n")
    print(f"{'model':30}{'C2':>16}{'C3':>16}{'C4':>16}")
    data = {}
    for lbl, key in MODELS.items():
        row = ""
        for sc in ("C2", "C3", "C4"):
            x = series(key, SCN[sc]); data[(lbl, sc)] = x
            xx = x[~np.isnan(x)]
            row += f"{xx.mean():.3f}±{xx.std(ddof=1):.3f}".rjust(16) if len(xx) > 1 else "  n/a".rjust(16)
        print(f"{lbl:30}{row}")

    print("\n=== PEFT-disease(LoRA) vs frozen dot — paired significance ===")
    fr = "TwoTower dot (frozen)"; pf = "TwoTower PEFT-disease (LoRA)"
    for sc in ("C2", "C4"):
        d = data[(pf, sc)] - data[(fr, sc)]
        mean, t, p = paired_t(d)
        lo, hi = boot_ci(d)
        persd = "  ".join(f"{x:+.3f}" for x in d)
        print(f"  {sc}: Δ(PEFT−frozen) per-seed = [{persd}]")
        print(f"      mean Δ = {mean:+.4f}  paired-t p = {p:.4f}  bootstrap95% CI = [{lo:+.4f}, {hi:+.4f}]")

    print("\n=== headline: TwoTower dot C4 & margin vs best content-baseline ===")
    dotc4 = data[("TwoTower dot (frozen)", "C4")]
    katz = data[("KATZLDA+content", "C4")]
    dd = dotc4 - katz
    print(f"  dot C4 = {dotc4[~np.isnan(dotc4)].mean():.3f}±{dotc4[~np.isnan(dotc4)].std(ddof=1):.3f}")
    lo, hi = boot_ci(dd); _, t, p = paired_t(dd)
    print(f"  dot − KATZ+content = {np.nanmean(dd):+.4f}  paired-t p = {p:.4f}  CI [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
