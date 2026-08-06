"""Turn results_case_study/case_cdis_l2d5.json into the manuscript table and the
supplementary top-20 listings.

Main table  : five cancers x four methods, hits@20 / recall@50 with the chance
              expectation and an exact hypergeometric p-value beside them.
Supplement  : per-cancer top-20 lncRNA symbols with held-out status, for the
              main model only (the other methods' listings stay in the JSON).
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
RES = os.path.join(_PAPER, "results_case_study", "case_cdis_l2d5.json")

DISPLAY = {
    "TwoTower-PEFT-disease (LoRA S-BioBERT)": "Dual-encoder main (LoRA-disease)",
    "TwoTower (content)": "Dual-encoder (dot)",
    "KATZLDA-content (semsim+expr)": "KATZLDA + content",
    "KATZLDA (Chen 2015; closed-form Katz inverse, in-fold beta, GIP-only)":
        "KATZLDA (content-blind)",
    "Popularity2": "Popularity",
}
CANCERS = [("DOID:684", "Hepatocellular carcinoma"), ("DOID:10534", "Gastric cancer"),
           ("DOID:1612", "Breast cancer"), ("DOID:9256", "Colorectal cancer"),
           ("DOID:3908", "Non-small cell lung cancer")]


def fmt_p(p):
    if p is None:
        return "-"
    if p < 1e-4:
        return f"{p:.0e}".replace("e-0", "e-")
    return f"{p:.3f}"


def main():
    d = json.load(open(RES))
    models = d["models"]
    print(f"# C-dis case study, variant {d['variant']}, seed {d['seed']}, "
          f"{d['M_shape'][0]} lncRNA candidates per query")
    print(f"# leakage check (lncRNA symbols inside the target disease texts): "
          f"{d['text_leak_check'] or 'none'}\n")

    hdr = f"| cancer | curated lncRNAs | method | hits@20 | expected | p | recall@50 |"
    print(hdr)
    print("|---|---|---|---|---|---|---|")
    for do, name in CANCERS:
        first = True
        for key, disp in DISPLAY.items():
            r = models.get(key, {}).get(do)
            if r is None:
                continue
            if r.get("degenerate_ranking"):
                cells = "n/a (no ranking) | - | - | -"
                print(f"| {name if first else ''} | {r['n_pos'] if first else ''} | "
                      f"{disp} | {cells} |".replace(" |  |", " | |"))
            else:
                print(f"| {name if first else ''} | {r['n_pos'] if first else ''} | {disp} | "
                      f"{r['hits@20']} | {r['expected_hits@20']:.1f} | "
                      f"{fmt_p(r['hypergeom_p@20'])} | {r['recall@50']:.3f} |")
            first = False
        print("|  |  |  |  |  |  |  |")

    main_key = "TwoTower-PEFT-disease (LoRA S-BioBERT)"
    print("\n\n# Supplementary: main-model top-20 per cancer "
          "(* = held-out curated association recovered)\n")
    for do, name in CANCERS:
        r = models[main_key][do]
        print(f"## {name} ({do}) - {r['n_pos']} curated lncRNAs held out, "
              f"{r['hits@20']} recovered in top-20")
        row = []
        for e in r["top50"][:20]:
            row.append(f"{e['rank']}. {e['lncRNA']}{'*' if e['held_out_positive'] else ''}")
        print("   " + "; ".join(row) + "\n")

    # candidates for external validation: top-20 entries that are NOT held-out positives
    novel = {}
    for do, name in CANCERS:
        novel[name] = [e["lncRNA"] for e in models[main_key][do]["top50"][:20]
                       if not e["held_out_positive"]]
    out = os.path.join(_PAPER, "results_case_study", "novel_candidates.json")
    with open(out, "w") as f:
        json.dump(novel, f, indent=1)
    print(f"# wrote external-validation worklist -> {out}")
    for k, v in novel.items():
        print(f"#   {k}: {len(v)} candidates to check")


if __name__ == "__main__":
    main()
