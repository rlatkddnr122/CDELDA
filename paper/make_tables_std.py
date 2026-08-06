import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
#!/usr/bin/env python
"""Generate headline tables WITH fold-level std (mean±std over the 5 node-hold-out folds,
seed 2026). Warm uses the AUROC/AUPR std from the warm edge-CV. Numbers from JSON only.
Emits manuscript/tables/std_table{1,1b,4,6}.md.
"""
import json, os
TT = _REPO
OUT = os.path.join(TT, "manuscript", "tables"); os.makedirs(OUT, exist_ok=True)
VARIANTS = ["l2d5", "l2d10", "l3d5", "l2d20", "l3d10", "l3d20"]
DENS = {"l2d5": "1.62", "l2d10": "2.37", "l3d5": "3.24", "l2d20": "3.58", "l3d10": "4.87", "l3d20": "7.23"}


def load(sub, v, proto):
    f = os.path.join(TT, sub, f"bench_rd{v}_{proto}.json")
    return json.load(open(f))["models"] if os.path.exists(f) else {}


def find(m, n):
    return next((k for k in m if n in k), None)


def cell(sub, v, needle, scn, metric="AUPR_1to1"):
    proto = "warm" if scn == "warm" else "cold"
    m = load(sub, v, proto); k = find(m, needle)
    if not k:
        return None
    node = m[k] if scn == "warm" else m[k].get(scn)
    if not node:
        return None
    a = node.get(metric) or node.get({"AUC_1to1": "AUROC_1to1", "AUROC_1to1": "AUC_1to1"}.get(metric, ""))
    return (a["mean"], a["std"]) if a else None


def fmt(t, bold=False):
    if t is None:
        return "—"
    s = f"{t[0]:.3f}±{t[1]:.3f}"
    return f"**{s}**" if bold else s


# rows: (label, sub, needle)  — content per Table 1 footnotes
R = [("Dual-encoder main (LoRA-disease)", "results_sweep_peft", "PEFT-disease"),
     ("Dual-encoder (dot)", "results_sweep", "Dual-encoder (content)"),
     ("DSCMF", "results_sweep_contentfull", "DSCMF"),
     ("KATZLDA", "results_sweep_content", "KATZLDA"),
     ("SIMCLDA", "results_sweep_content", "SIMCLDA"),
     ("VGAELDA", "results_sweep_contentfull", "VGAELDA"),
     ("IPCARF", "results_sweep_content", "IPCARF"),
     ("LDAformer", "results_sweep_ldaf", "LDAformer"),
     ("kNN-content", "results_sweep", "kNN-content"),
     ("Popularity", "results_sweep", "Popularity"),
     ("Random", "results_sweep", "Random")]


def table1(metric, fname, title):
    L = [title, "",
         "| Method | warm | C2 (dis-cold) | C3 (lnc-cold) | C4 (both-cold) |",
         "|---|---|---|---|---|"]
    c4s = [cell(sub, "l2d5", nd, "both", metric) for _, sub, nd in R]
    best = max(x[0] for x in c4s if x)
    for lbl, sub, nd in R:
        c4 = cell(sub, "l2d5", nd, "both", metric)
        L.append(f"| {lbl} | {fmt(cell(sub,'l2d5',nd,'warm',metric))} | "
                 f"{fmt(cell(sub,'l2d5',nd,'disease',metric))} | {fmt(cell(sub,'l2d5',nd,'lncRNA',metric))} | "
                 f"{fmt(c4, c4 and c4[0]==best)} |")
    open(os.path.join(OUT, fname), "w").write("\n".join(L) + "\n")


def table4():
    heads = [("inner product (dot)", "results_sweep", "Dual-encoder (content)"),
             ("neural matching (NCF)", "results_ablation_hero", "NCF"),
             ("bilinear", "results_ablation_hero", "Bilinear"),
             ("cross-attention (dual-attn)", "results_ablation_hero", "DualAttn"),
             ("contrastive (InfoNCE)", "results_ablation_hero", "Contrastive")]
    vals = [(lbl, cell(sub, "l2d5", nd, "both")) for lbl, sub, nd in heads]
    best = max(x[0] for _, x in vals if x)
    L = ["**Table 4.** Interaction-head ablation at both-cold (l2d5, C4, mean±fold-std).", "",
         "| interaction head | C4 |", "|---|---|"]
    for lbl, v in vals:
        L.append(f"| {lbl} | {fmt(v, v and v[0]==best)} |")
    open(os.path.join(OUT, "std_table4.md"), "w").write("\n".join(L) + "\n")


def table6():
    L = ["**Table 6.** Both-cold (C4) across the six-point k-core sweep (seed 2026, mean±fold-std), "
         "ordered by graph density.", "",
         "| variant | density % | main (LoRA) | Dual-encoder dot | best content-equipped | content-blind |",
         "|---|---|---|---|---|---|"]
    for v in sorted(VARIANTS, key=lambda x: float(DENS[x])):
        hero = cell("results_sweep_peft", v, "PEFT-disease", "both")
        dot = cell("results_sweep", v, "Dual-encoder (content)", "both")
        eq = []
        for n in ["KATZLDA", "SIMCLDA", "IPCARF", "DSCMF", "VGAELDA"]:
            for src in ("results_sweep_contentfull", "results_sweep_content"):
                c = cell(src, v, n, "both")
                if c:
                    eq.append(c); break
        best_eq = max(eq, key=lambda x: x[0]) if eq else None
        blind = cell("results_sweep", v, "KATZLDA (Chen 2015", "both")
        L.append(f"| {v} | {DENS[v]} | {fmt(hero)} | {fmt(dot)} | {fmt(best_eq)} | {fmt(blind)} |")
    open(os.path.join(OUT, "std_table6.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    table1("AUPR_1to1", "std_table1.md",
           "**Table 1.** Faithful (as-published) reproduction on l2d5 (seed 2026), AUPR(1:1) as "
           "mean±fold-std over the 5 node-hold-out folds. **Bold** = best in column.")
    table1("AUROC_1to1", "std_table1b.md",
           "**Table 1b (AUROC companion).** Same reproductions/protocol as Table 1 (l2d5), AUROC(1:1) mean±fold-std.")
    table4(); table6()
    print("wrote std_table1/1b/4/6.md")
