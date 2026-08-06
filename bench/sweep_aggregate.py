"""Merge the LDAformer-only sweep into the main sweep results, then build a
threshold-robustness summary across all 6 RNADisease k-core variants.

Outputs (to manuscript/tables/):
  sweep_warm.md   - warm AUPR_1to1 per model x variant
  sweep_cold.md   - cold C2/C3/C4 AUPR_1to1 per model x variant
  sweep_robustness.md - headline: TwoTower vs best content-blind baseline at C4,
                        and warm competitiveness, across the threshold grid.
Also prints a compact console summary.
"""
import os, sys, json, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
MAIN = os.path.join(_PAPER, "results_sweep")
LDAF = os.path.join(_PAPER, "results_sweep_ldaf")
TABLES = os.path.join(_PAPER, "manuscript", "tables")
VARIANTS = ["l2d5", "l2d10", "l2d20", "l3d5", "l3d10", "l3d20"]
LDAF_KEY = "LDAformer (node-token multi-hop Transformer, GIP-only)"
HERO = "TwoTower (content)"


def _load(d, v, proto):
    p = os.path.join(d, f"bench_rd{v}_{proto}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def merge_ldaformer():
    n = 0
    for v in VARIANTS:
        for proto in ("warm", "cold"):
            main = _load(MAIN, v, proto)
            ldaf = _load(LDAF, v, proto)
            if main is None or ldaf is None:
                continue
            if LDAF_KEY in ldaf.get("models", {}) and LDAF_KEY not in main["models"]:
                main["models"][LDAF_KEY] = ldaf["models"][LDAF_KEY]
                tmp = os.path.join(MAIN, f"bench_rd{v}_{proto}.json") + ".tmp"
                json.dump(main, open(tmp, "w"), indent=2)
                os.replace(tmp, os.path.join(MAIN, f"bench_rd{v}_{proto}.json"))
                n += 1
    print(f"[merge] injected LDAformer into {n} result files")


def short(nm):
    return nm.split(" (")[0]


def build_tables():
    os.makedirs(TABLES, exist_ok=True)
    # collect
    warm, cold = {}, {}
    for v in VARIANTS:
        w = _load(MAIN, v, "warm"); c = _load(MAIN, v, "cold")
        if w:
            for nm, r in w["models"].items():
                warm.setdefault(nm, {})[v] = r["AUPR_1to1"]["mean"]
        if c:
            for nm, r in c["models"].items():
                cold.setdefault(nm, {})[v] = {s: r[s]["AUPR_1to1"]["mean"] for s in ("disease", "lncRNA", "both")}

    models = list(warm.keys())
    hdr = "| model | " + " | ".join(VARIANTS) + " |"
    sep = "|" + "---|" * (len(VARIANTS) + 1)

    # warm table
    lines = ["# Sweep: warm AUPR (1:1), seed=2026", "", hdr, sep]
    for nm in models:
        row = [f"{warm[nm].get(v, float('nan')):.3f}" for v in VARIANTS]
        lines.append(f"| {short(nm)} | " + " | ".join(row) + " |")
    open(os.path.join(TABLES, "sweep_warm.md"), "w").write("\n".join(lines) + "\n")

    # cold table (C4 both)
    lines = ["# Sweep: cold both-cold (C4) AUPR (1:1), seed=2026", "", hdr, sep]
    for nm in models:
        row = [f"{cold.get(nm, {}).get(v, {}).get('both', float('nan')):.3f}" for v in VARIANTS]
        lines.append(f"| {short(nm)} | " + " | ".join(row) + " |")
    lines += ["", "## C2 disease-cold", "", hdr, sep]
    for nm in models:
        row = [f"{cold.get(nm, {}).get(v, {}).get('disease', float('nan')):.3f}" for v in VARIANTS]
        lines.append(f"| {short(nm)} | " + " | ".join(row) + " |")
    lines += ["", "## C3 lncRNA-cold", "", hdr, sep]
    for nm in models:
        row = [f"{cold.get(nm, {}).get(v, {}).get('lncRNA', float('nan')):.3f}" for v in VARIANTS]
        lines.append(f"| {short(nm)} | " + " | ".join(row) + " |")
    open(os.path.join(TABLES, "sweep_cold.md"), "w").write("\n".join(lines) + "\n")

    # robustness headline: TwoTower C4 vs best BASELINE C4 (exclude Random/MF/kNN refs? keep all non-hero)
    refs_floor = {"Random", "MF (free-emb)", "Popularity2"}
    lines = ["# Threshold-robustness headline (seed=2026)", "",
             "TwoTower both-cold (C4) vs the best content-blind baseline, across the k-core grid.",
             "If the margin stays positive across all 6 thresholds, the C4 conclusion does not",
             "depend on the specific (lnc, dis) cut.", "",
             "| variant | shape | TwoTower C4 | best baseline C4 (name) | margin | TwoTower warm | best warm (name) |",
             "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        c = _load(MAIN, v, "cold"); w = _load(MAIN, v, "warm")
        if not c or not w:
            continue
        shape = "x".join(map(str, c["M_shape"]))
        tt_c4 = c["models"].get(HERO, {}).get("both", {}).get("AUPR_1to1", {}).get("mean", float("nan"))
        base_c4 = {nm: r["both"]["AUPR_1to1"]["mean"] for nm, r in c["models"].items()
                   if nm != HERO and short(nm) not in refs_floor}
        bnm = max(base_c4, key=base_c4.get) if base_c4 else "-"
        bval = base_c4.get(bnm, float("nan"))
        tt_w = w["models"].get(HERO, {}).get("AUPR_1to1", {}).get("mean", float("nan"))
        base_w = {nm: r["AUPR_1to1"]["mean"] for nm, r in w["models"].items() if nm != HERO}
        wnm = max(base_w, key=base_w.get) if base_w else "-"
        wval = base_w.get(wnm, float("nan"))
        lines.append(f"| {v} | {shape} | {tt_c4:.3f} | {bval:.3f} ({short(bnm)}) | "
                     f"{tt_c4-bval:+.3f} | {tt_w:.3f} | {wval:.3f} ({short(wnm)}) |")
    open(os.path.join(TABLES, "sweep_robustness.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[tables] wrote sweep_warm.md, sweep_cold.md, sweep_robustness.md -> {TABLES}")


if __name__ == "__main__":
    merge_ldaformer()
    build_tables()
