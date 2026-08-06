"""Table 3 — 5-seed cold-start CI across the single-source sweep (l2d5, l3d5, l2d10).

For each variant: per-model mean±std of C2/C3/C4 AUPR(1:1), plus the two headline
paired contrasts (PEFT-disease vs frozen dot; frozen dot vs best content baseline)
with paired t-test + bootstrap 95% CI. Confirms the hero effect spans the sweep
rather than living on a single dataset. CPU-only; reads results_5seed/seed*/ JSON.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, json
import numpy as np

_RES = f"{_REPO}/results_5seed"
SEEDS = [2026, 1, 2, 3, 4]
VARIANTS = ["l2d5", "l3d5", "l2d10"]
MODELS = {
    "TwoTower dot (frozen)": "TwoTower (content)",
    "TwoTower dual-attn": "TwoTower-DualAttn (content)",
    "TwoTower PEFT-disease (LoRA)": "TwoTower-PEFT-disease (LoRA S-BioBERT)",
    "KATZLDA+content": "KATZLDA-content (semsim+expr)",
    "kNN-content": "kNN-content",
    "VGAELDA (content-blind)": "VGAELDA (co-trained VGAE + label-prop resolvent, content-blind)",
}
SCN = {"C2": "disease", "C3": "lncRNA", "C4": "both"}


def load(variant, seed, key, scn):
    f = os.path.join(_RES, f"seed{seed}", f"bench_rd{variant}_cold.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))["models"].get(key)
    return m[scn]["AUPR_1to1"]["mean"] if m else None


def series(variant, key, scn):
    return np.array([load(variant, s, key, scn) for s in SEEDS], float)


def paired_t(diffs):
    d = diffs[~np.isnan(diffs)]
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return d.mean(), float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_rel(d + 0, np.zeros(n))
    return d.mean(), t, p


def boot_ci(diffs, B=10000, seed=0):
    d = diffs[~np.isnan(diffs)]
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cohens_d(diffs):
    d = diffs[~np.isnan(diffs)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return d.mean() / d.std(ddof=1)


def run_variant(variant):
    print(f"\n{'='*70}\n### {variant} — 5-seed {SEEDS} cold, mean±std AUPR(1:1)\n{'='*70}")
    print(f"{'model':30}{'C2':>16}{'C3':>16}{'C4':>16}")
    data = {}
    n_missing = 0
    for lbl, key in MODELS.items():
        row = ""
        for sc in ("C2", "C3", "C4"):
            x = series(variant, key, SCN[sc]); data[(lbl, sc)] = x
            xx = x[~np.isnan(x)]
            if len(xx) < len(SEEDS):
                n_missing += (len(SEEDS) - len(xx))
            row += f"{xx.mean():.3f}±{xx.std(ddof=1):.3f}".rjust(16) if len(xx) > 1 else "  n/a".rjust(16)
        print(f"{lbl:30}{row}")
    if n_missing:
        print(f"  [!] {n_missing} seed-cells missing for {variant} (run incomplete)")

    print(f"\n-- {variant}: PEFT-disease(LoRA) vs frozen dot --")
    fr, pf = "TwoTower dot (frozen)", "TwoTower PEFT-disease (LoRA)"
    for sc in ("C2", "C4"):
        d = data[(pf, sc)] - data[(fr, sc)]
        mean, t, p = paired_t(d); lo, hi = boot_ci(d); dz = cohens_d(d)
        persd = "  ".join(f"{x:+.3f}" for x in d)
        print(f"  {sc}: Δ per-seed [{persd}]")
        print(f"      mean Δ = {mean:+.4f}  paired-t p = {p:.4f}  boot95% CI [{lo:+.4f},{hi:+.4f}]  d = {dz:+.2f}")

    print(f"\n-- {variant}: frozen dot vs KATZLDA+content (headline margin) --")
    dd = data[("TwoTower dot (frozen)", "C4")] - data[("KATZLDA+content", "C4")]
    mean, t, p = paired_t(dd); lo, hi = boot_ci(dd); dz = cohens_d(dd)
    print(f"  C4: mean Δ = {mean:+.4f}  paired-t p = {p:.4f}  boot95% CI [{lo:+.4f},{hi:+.4f}]  d = {dz:+.2f}")
    return data


def main():
    for v in VARIANTS:
        run_variant(v)
    print(f"\n{'='*70}\nFIVESEED_ALL DONE\n{'='*70}")


if __name__ == "__main__":
    main()
