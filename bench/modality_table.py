"""Table 4a — lncRNA content-modality ablation (which modality carries cold signal).
Reads results_ablation{,_l2d10}/lnc_<mod>/bench_rd<V>_cold.json, TwoTower (content),
prints C2/C3/C4 AUPR(1:1) per modality per dataset. seed=2026 (single-seed ablation).
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, json
import numpy as np

CASES = [
    ("l2d5", f"{_REPO}/results_ablation"),
    ("l2d10", f"{_REPO}/results_ablation_l2d10"),
]
MODS = ["ortho", "kmer", "rnafm", "struct", "expr"]
KEY = "TwoTower (content)"
SCN = {"C2": "disease", "C3": "lncRNA", "C4": "both"}


def val(root, mod, variant, scn):
    f = os.path.join(root, f"lnc_{mod}", f"bench_rd{variant}_cold.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))["models"].get(KEY)
    return m[scn]["AUPR_1to1"]["mean"] if m else None


def main():
    for variant, root in CASES:
        print(f"\n### {variant} — lncRNA modality ablation (TwoTower content), cold seed=2026")
        print(f"{'modality':10}{'C2':>9}{'C3':>9}{'C4':>9}")
        for mod in MODS:
            r = [val(root, mod, variant, SCN[s]) for s in ("C2", "C3", "C4")]
            cells = "".join((f"{x:.3f}".rjust(9) if x is not None else "  n/a".rjust(9)) for x in r)
            print(f"{mod:10}{cells}")
    print("\nMODALITY_TABLE DONE")


if __name__ == "__main__":
    main()
