#!/usr/bin/env python
"""Auto-generate the manuscript main tables + headline figures from the 5-seed
benchmark JSONs (results/ = seed 2026 headline; results/seed{1..4} = robustness).

Single source of truth: numbers come ONLY from the result JSONs (no hand-typed
values). Run:  python manuscript/make_tables_figures.py   (from the paper dir).
Outputs: manuscript/tables/*.md, manuscript/figures/*.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
SEEDS = {2026: "results", 1: "results/seed1", 2: "results/seed2", 3: "results/seed3", 4: "results/seed4"}
DS = ["canon", "ld", "rd"]
DSLABEL = {"canon": "canonical\n240x412", "ld": "LncRNADisease\n817x161", "rd": "RNADisease-core\n5035x164"}
TABLES = os.path.join(HERE, "tables"); FIGS = os.path.join(HERE, "figures")
os.makedirs(TABLES, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

def _load(d, proto, seed):
    return json.load(open(os.path.join(PAPER, SEEDS[seed], f"bench_{d}_{proto}.json")))["models"]
def _sc(v):
    return v.get("mean", v.get("value")) if isinstance(v, dict) else v
def warm(d, m, s, k="AUPR_1to1"): return _sc(_load(d, "warm", s)[m].get(k))
def cold(d, m, s, scn, k="AUPR_1to1"): return _sc(_load(d, "cold", s)[m][scn].get(k))
ORDER = list(_load("canon", "warm", 2026).keys())
def short(n): return n.split(" (")[0].strip()
def mstd(getter):
    v = np.array([getter(s) for s in SEEDS], float)
    return v.mean(), v.std(ddof=1)

def _pick(pfx):
    for m in ORDER:
        if m.startswith(pfx):
            return m
    return pfx
# Classify by prefix so the scripts survive baseline NAME changes.
CONTENT = [m for m in ORDER if m.startswith("Dual-encoder") or m.startswith("kNN")]
REFS = [m for m in ORDER if m.startswith("Popularity") or m.startswith("MF ") or m.startswith("Random")]
BLIND = [m for m in ORDER if m not in CONTENT and m not in REFS]   # the 6 reproduced baselines

# ---------------- TABLES (markdown, mean +/- std over 5 seeds) ----------------
def cell(mn, sd): return f"{mn:.3f}±{sd:.3f}"

def table_warm():
    L = ["# Table 1 — Warm-start (transductive 5-fold), 5-seed mean±std\n",
         "| Model | " + " | ".join(DS) + " | AUROC (canon/ld/rd) |", "|---|" + "---|" * (len(DS) + 1)]
    for m in ORDER:
        au = [cell(*mstd(lambda s, d=d, m=m: warm(d, m, s))) for d in DS]
        ro = [f"{mstd(lambda s, d=d, m=m: warm(d, m, s, 'AUROC_1to1'))[0]:.3f}" for d in DS]
        L.append(f"| {short(m)} | " + " | ".join(au) + " | " + "/".join(ro) + " |")
    return "\n".join(L) + "\n\n_Primary metric AUPR (1:1 balanced). AUROC shown for reference._\n"

def table_cold():
    L = ["# Table 2 — Cold-start (inductive node hold-out), AUPR 1:1, 5-seed mean±std\n",
         "| Model | C2 disease-cold (canon/ld/rd) | C3 lncRNA-cold | C4 both-cold |", "|---|---|---|---|"]
    for m in ORDER:
        def col(scn, m=m): return "/".join(cell(*mstd(lambda s, d=d, m=m, scn=scn: cold(d, m, s, scn))) for d in DS)
        L.append(f"| {short(m)} | {col('disease')} | {col('lncRNA')} | {col('both')} |")
    return "\n".join(L) + "\n\n_C4 both-cold is the strict headline. Content-blind methods collapse to 0.500 (chance)._\n"

def table_gap():
    L = ["# Table 3 — Warm→Cold gap (5-seed mean AUPR)\n",
         "| Model | dataset | warm | C4 both-cold | Δ (cold−warm) |", "|---|---|---|---|---|"]
    for m in [_pick("Dual-encoder"), _pick("kNN"), _pick("Popularity2"), _pick("VGAELDA")]:
        for d in DS:
            w = mstd(lambda s, d=d, m=m: warm(d, m, s))[0]
            c = mstd(lambda s, d=d, m=m: cold(d, m, s, "both"))[0]
            L.append(f"| {short(m)} | {d} | {w:.3f} | {c:.3f} | {c-w:+.3f} |")
    return "\n".join(L) + "\n"

def table_headline():
    L = ["# Table 0 — seed=2026 HEADLINE (single representative seed)\n",
         "| Model | warm AUPR (canon/ld/rd) | C4 both-cold AUPR |", "|---|---|---|"]
    for m in ORDER:
        w = "/".join(f"{warm(d, m, 2026):.3f}" for d in DS)
        c = "/".join(f"{cold(d, m, 2026, 'both'):.3f}" for d in DS)
        L.append(f"| {short(m)} | {w} | {c} |")
    return "\n".join(L) + "\n\n_Headline = seed=2026 single seed (prereg); Tables 1-2 give 5-seed robustness mean±std._\n"


for fn, name in [(table_headline, "table0_headline.md"), (table_warm, "table1_warm.md"), (table_cold, "table2_cold.md"), (table_gap, "table3_gap.md")]:
    open(os.path.join(TABLES, name), "w").write(fn())
print("tables ->", TABLES)

# ---------------- FIGURE 1 — warm vs both-cold, the headline ------------------
sel = [_pick("Dual-encoder"), _pick("kNN"), _pick("VGAELDA"), _pick("DSCMF"), _pick("Popularity2"), _pick("MF ")]
colors = {_pick("Dual-encoder"): "#c0392b", _pick("kNN"): "#e67e22"}
def barcolor(m): return colors.get(m, "#7f8c8d")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(len(DS)); w = 0.13
for ax, proto, title in [(axes[0], "warm", "WARM (5-fold transductive)"),
                         (axes[1], "cold", "BOTH-COLD C4 (strict node hold-out)")]:
    for i, m in enumerate(sel):
        get = (lambda d, s, m=m: warm(d, m, s)) if proto == "warm" else (lambda d, s, m=m: cold(d, m, s, "both"))
        mn = np.array([np.mean([get(d, s) for s in SEEDS]) for d in DS])
        sd = np.array([np.std([get(d, s) for s in SEEDS], ddof=1) for d in DS])
        ax.bar(x + (i - len(sel)/2) * w + w/2, mn, w, yerr=sd, capsize=2,
               color=barcolor(m), edgecolor="black", linewidth=0.4,
               label=(short(m) if ax is axes[0] else None))
    ax.axhline(0.5, ls="--", lw=1, color="black", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([DSLABEL[d] for d in DS], fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylim(0.45, 1.0)
axes[0].set_ylabel("AUPR (1:1)")
axes[0].legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9)
axes[1].text(0.02, 0.52, "chance floor 0.5", fontsize=7, color="black", alpha=0.7)
fig.suptitle("Content is the lever of generalization: warm-competitive, cold-exclusive", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig1_warm_vs_cold.{ext}"), dpi=160, bbox_inches="tight")
print("fig1 ->", os.path.join(FIGS, "fig1_warm_vs_cold.png"))

# ---------------- FIGURE 2 — C4 collapse wall (all models, per dataset) -------
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
models_all = CONTENT + BLIND + REFS
for ax, d in zip(axes2, DS):
    mn = [np.mean([cold(d, m, s, "both") for s in SEEDS]) for m in models_all]
    cols = ["#c0392b" if m in CONTENT else ("#2980b9" if m in BLIND else "#95a5a6") for m in models_all]
    ax.barh(range(len(models_all)), mn, color=cols, edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, ls="--", lw=1, color="black", alpha=0.6)
    ax.set_yticks(range(len(models_all))); ax.set_yticklabels([short(m) for m in models_all], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0.45, 0.78); ax.set_title(d, fontsize=10)
    ax.set_xlabel("C4 both-cold AUPR")
fig2.suptitle("Strict both-cold: only content-using methods (red) clear the 0.5 floor; "
              "content-blind SOTA (blue) collapse", fontsize=10)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig2.savefig(os.path.join(FIGS, f"fig2_c4_collapse.{ext}"), dpi=160, bbox_inches="tight")
print("fig2 ->", os.path.join(FIGS, "fig2_c4_collapse.png"))
print("DONE")
