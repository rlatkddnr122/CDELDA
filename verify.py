"""Check the published tables against the frozen result JSON.

Every number in Table 2, Table 3, and the 16-variant grid (Table S3 / Figure 3) of the article is asserted against results*/ here, so
the claim that the released artifact reproduces the article can be checked in one
command, without a GPU and without regenerating any dataset:

    python verify.py

One subtlety this script pins down. A few models were re-run into a dedicated
directory after first appearing in results_sweep/, and the two copies do not agree --
VGAELDA-contentfull differs in all sixteen variants. The article uses the dedicated
directory, so PREFERENCE below encodes that, and reading results_sweep/ first would
silently give a different "best content-aware" value on three variants.
"""
import glob
import json
import os
import sys

# a model present in more than one directory is taken from the first match here
PREFERENCE = ["results_sweep_content", "results_sweep_contentfull",
              "results_sweep_peft", "results_sweep_vghb", "results_sweep"]

MAIN = "TwoTower-PEFT-disease (LoRA S-BioBERT)"          # CDELDA (LoRA-disease)
DOT = "TwoTower (content)"                               # CDELDA (frozen)
VGHB = "LDA-VGHB (SVD+VGAE+SnapBoost)"
# the reproduced families, each in the content-aware form the article gives it
FAMILY = ("DSCMF-content", "IPCARF-content", "KATZLDA-content",
          "SIMCLDA-content", "VGAELDA-content")

# Table 2 -- headline variant l2d5, AUPR(1:1) mean +- fold-std
TABLE2 = [
    ("CDELDA (LoRA-disease)", MAIN, (0.922, 0.001), (0.755, 0.058), (0.911, 0.003), (0.717, 0.056)),
    ("CDELDA (frozen)", DOT, (0.923, 0.001), (0.743, 0.054), (0.911, 0.002), (0.704, 0.058)),
    ("DSCMF", "DSCMF-contentfull (cold-equipped)", (0.807, 0.003), (0.749, 0.047), (0.813, 0.005), (0.629, 0.036)),
    ("KATZLDA", "KATZLDA-content (semsim+expr)", (0.901, 0.002), (0.765, 0.023), (0.821, 0.003), (0.621, 0.055)),
    ("SIMCLDA", "SIMCLDA-content (semsim+expr)", (0.741, 0.002), (0.628, 0.018), (0.802, 0.003), (0.601, 0.037)),
    ("VGAELDA", "VGAELDA-contentfull (cold-equipped)", (0.893, 0.004), (0.601, 0.044), (0.792, 0.030), (0.588, 0.036)),
    ("IPCARF", "IPCARF-content (semsim+expr)", (0.889, 0.004), (0.724, 0.054), (0.803, 0.014), (0.555, 0.018)),
    ("LDA-VGHB", VGHB, (0.876, 0.003), (0.712, 0.069), (0.393, 0.025), (0.527, 0.035)),
    ("LDAformer", "LDAformer (node-token multi-hop Transformer, GIP-only)",
     (0.907, 0.001), (0.500, 0.000), (0.500, 0.000), (0.500, 0.000)),
    ("kNN-content", "kNN-content", (0.768, 0.002), (0.688, 0.040), (0.757, 0.009), (0.633, 0.028)),
    ("Popularity", "Popularity2", (0.888, 0.003), (0.500, 0.000), (0.500, 0.000), (0.500, 0.000)),
    ("Random", "Random", (0.499, 0.004), (0.502, 0.004), (0.502, 0.003), (0.504, 0.010)),
]

# Table S3 / Figure 3 -- both-cold across the 16-variant grid:
# CDELDA, LDA-VGHB, best content-aware baseline, margin
GRID = [
    ("l2d2", 0.701, 0.514, 0.673, 0.028), ("l2d3", 0.679, 0.546, 0.661, 0.019),
    ("l2d4", 0.654, 0.560, 0.637, 0.018), ("l2d5", 0.717, 0.527, 0.629, 0.087),
    ("l3d2", 0.690, 0.525, 0.669, 0.021), ("l3d3", 0.662, 0.521, 0.651, 0.011),
    ("l3d4", 0.649, 0.538, 0.633, 0.015), ("l4d2", 0.651, 0.538, 0.637, 0.014),
    ("l3d5", 0.661, 0.498, 0.612, 0.049), ("l4d3", 0.655, 0.535, 0.630, 0.024),
    ("l5d2", 0.655, 0.548, 0.643, 0.012), ("l4d4", 0.607, 0.523, 0.622, -0.015),
    ("l5d3", 0.663, 0.534, 0.631, 0.032), ("l4d5", 0.628, 0.502, 0.630, -0.002),
    ("l5d4", 0.645, 0.593, 0.629, 0.017), ("l5d5", 0.626, 0.522, 0.610, 0.016),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def read(variant, protocol):
    """{model: {split: (mean, std)}} for one variant, honouring PREFERENCE."""
    splits = ["disease", "lncRNA", "both"] if protocol == "cold" else ["warm"]
    seen = {}
    for path in glob.glob(os.path.join(HERE, "results*", "bench_rd%s_%s.json" % (variant, protocol))):
        src = os.path.basename(os.path.dirname(path))
        if src not in PREFERENCE:
            continue
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for name, entry in doc.get("models", {}).items():
            if not isinstance(entry, dict):
                continue
            for split in splits:
                node = entry.get(split, entry if split == "warm" else None)
                if not isinstance(node, dict):
                    continue
                cell = node.get("AUPR_1to1")
                if isinstance(cell, dict):
                    seen.setdefault((name, split), {})[src] = (cell["mean"], cell["std"])
    out = {}
    for (name, split), by_src in seen.items():
        for pref in PREFERENCE:
            if pref in by_src:
                out.setdefault(name, {})[split] = by_src[pref]
                break
    return out


def main():
    fails = []

    cold, warm = read("l2d5", "cold"), read("l2d5", "warm")
    print("Table 2 -- main benchmark on l2d5, AUPR(1:1) mean +- fold-std")
    for label, key, *cells in TABLE2:
        got = []
        for split, (want_m, want_s) in zip(["warm", "disease", "lncRNA", "both"], cells):
            src = warm if split == "warm" else cold
            pair = src.get(key, {}).get(split)
            ok = pair is not None and round(pair[0], 3) == want_m and round(pair[1], 3) == want_s
            got.append("%.3f+-%.3f" % (round(pair[0], 3), round(pair[1], 3)) if pair else "missing")
            if not ok:
                fails.append("Table 2 / %s / %s" % (label, split))
        print("  %-24s %s" % (label, "  ".join(got)))

    print("\nTable S3 / Figure 3 -- both-cold across the 16-variant grid")
    for variant, want_main, want_vghb, want_eq, want_margin in GRID:
        by_model = read(variant, "cold")
        main = by_model.get(MAIN, {}).get("both")
        vghb = by_model.get(VGHB, {}).get("both")
        family = [v["both"][0] for n, v in by_model.items()
                  if n.startswith(FAMILY) and "both" in v]
        best = max(family) if family else None
        margin = main[0] - best if (main and best is not None) else None
        for name, value, want in (("CDELDA", main and main[0], want_main),
                                  ("LDA-VGHB", vghb and vghb[0], want_vghb),
                                  ("best content-aware", best, want_eq),
                                  ("margin", margin, want_margin)):
            if value is None or round(value, 3) != want:
                fails.append("Grid / %s / %s" % (variant, name))
        print("  %-6s %.3f  %.3f  %.3f  %+.3f" % (variant, main[0], vghb[0], best, margin))

    print("\nTable 3 -- LDA-VGHB feature-computation leakage contrast on l2d5")
    LEAK = {"original": {"C-dis": 0.887, "C-lnc": 0.973, "C-both": 0.884},
            "safe": {"C-dis": 0.715, "C-lnc": 0.401, "C-both": 0.503}}
    leak_n = 0
    for tag, want in LEAK.items():
        path = os.path.join(HERE, "results_vghb_leakage_l2d5", "vghb_%s.json" % tag)
        try:
            doc = json.load(open(path, encoding="utf-8"))
            var = doc["variants"]["l2d5"]
        except (OSError, ValueError, KeyError):
            fails.append("Table 3 / %s / missing" % tag)
            continue
        row = []
        for scn, want_m in want.items():
            got = round(var[scn]["mean"], 3)
            row.append("%.3f" % got)
            leak_n += 1
            if got != want_m:
                fails.append("Table 3 / %s / %s" % (tag, scn))
        print("  %-9s %s" % (tag, "  ".join(row)))

    total = len(TABLE2) * 4 + len(GRID) * 4 + leak_n
    print("\n%d of %d published values reproduced from results*/" % (total - len(fails), total))
    for f in fails:
        print("  MISMATCH  " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
