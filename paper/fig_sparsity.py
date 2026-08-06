import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
#!/usr/bin/env python
"""Figure 3: both-cold (C4) across the 16-variant lncRNA x disease k-core grid (seed 2026).
Left panel: absolute both-cold AUPR of the main model, the strongest content-equipped classical
baseline, and the content-blind floor (0.500), ordered by graph density. Right panel: the margin
(main - best content-equipped), largest on the sparsest variant l2d5 and hovering near zero
elsewhere, occasionally dipping below. Outputs manuscript/figures/fig2_sparsity.{png,pdf}.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TT = _REPO
FIGS = os.path.join(TT, "manuscript", "figures")
os.makedirs(FIGS, exist_ok=True)
INK = "#22303a"

# --- Self-contained plot data (manuscript Table 2; seed 2026, both-cold AUPR 1:1) ---
# Committed here so the figure regenerates WITHOUT the gitignored results_sweep_* dirs.
# If you re-run the experiments, refresh these values from Table 2 / figures/figure_data.json.
# columns: variant, density %, main (dual-encoder + LoRA), best content-equipped classical
DATA = [
    ("l2d2", 1.03, 0.701, 0.673), ("l2d3", 1.22, 0.679, 0.661),
    ("l2d4", 1.41, 0.654, 0.637), ("l2d5", 1.62, 0.717, 0.629),
    ("l3d2", 2.02, 0.690, 0.669), ("l3d3", 2.47, 0.662, 0.651),
    ("l3d4", 2.82, 0.649, 0.633), ("l4d2", 2.83, 0.651, 0.637),
    ("l3d5", 3.24, 0.661, 0.612), ("l4d3", 3.42, 0.655, 0.630),
    ("l5d2", 3.57, 0.655, 0.643), ("l4d4", 3.95, 0.607, 0.622),
    ("l5d3", 4.41, 0.663, 0.631), ("l4d5", 4.54, 0.628, 0.630),
    ("l5d4", 5.07, 0.645, 0.629), ("l5d5", 5.83, 0.626, 0.610),
]
DATA.sort(key=lambda r: r[1])
vs = [r[0] for r in DATA]; xd = [r[1] for r in DATA]
main = [r[2] for r in DATA]; beq = [r[3] for r in DATA]
margin = [round(m - b, 3) for m, b in zip(main, beq)]

# Drawn at the size it is printed at: IEEE Access sets a page-wide figure at
# 7.16 in, and a figure authored wider than that has all of its text scaled down
# by the same factor when it is placed. At the previous 12.4 in the 6.6 pt tick
# labels landed under 4 pt on the page. Type 42 embeds the fonts as TrueType;
# matplotlib's default Type 3 is not accepted.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
FIGW = 7.16          # IEEE Access page width, inches
fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIGW, 3.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---- Left: absolute AUPR ----
axL.axhline(0.5, color="#9aa2a9", lw=0.9, ls=":", zorder=1)
axL.text(xd[-1], 0.504, "content-blind floor (all methods)", va="bottom", ha="right",
         fontsize=5.8, color="#7b838a")
axL.plot(xd, beq, "-", color="#c1922f", lw=1.2, marker="o", ms=3.2,
         label="best content-equipped classical", zorder=3)
axL.plot(xd, main, "-", color="#22615e", lw=1.8, marker="o", ms=4.6,
         label="main model (dual-encoder + LoRA)", zorder=5)
# highlight the headline variant l2d5 (per-point variant labels removed to avoid overlap where
# densities cluster; panel B lists all 16 variants with their densities)
i5 = vs.index("l2d5")
axL.scatter([xd[i5]], [main[i5]], s=70, facecolors="none", edgecolors="#b5623c", lw=1.3, zorder=6)
axL.annotate("l2d5", (xd[i5], main[i5]), xytext=(xd[i5] + 0.12, main[i5] + 0.010),
             fontsize=6.5, color="#b5623c", weight="bold")
axL.set_xlabel("retained association-graph density  (%)", fontsize=7.5)
axL.set_ylabel("both-cold AUPR (C-both), seed 2026", fontsize=7.5)
axL.set_title("A  Absolute both-cold performance", fontsize=8.2, weight="bold", color=INK, loc="left")
axL.set_ylim(0.47, 0.75)
axL.spines[["top", "right"]].set_visible(False)
axL.spines[["left", "bottom"]].set_color("#c7ccd1")
axL.tick_params(colors="#55606a", labelsize=6.5, length=2.2, width=0.6)
axL.grid(axis="y", color="#eceef0", lw=0.6, zorder=0)
axL.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.6)

# ---- Right: margin ----
axR.axhline(0.0, color="#9aa2a9", lw=0.9, ls="-", zorder=1)
cols = ["#b5623c" if v == "l2d5" else "#5f7d8c" for v in vs]
axR.bar(range(len(vs)), margin, color=cols, width=0.72, zorder=3)
axR.set_xticks(range(len(vs)))
# variant name only: two-line labels collide at this width, and the density is
# already the x-axis of panel A and a column of Table 3
axR.set_xticklabels(vs, fontsize=6, rotation=90)
axR.annotate("l2d5  +0.087", (i5, margin[i5]), xytext=(i5 + 0.3, margin[i5] + 0.006),
             fontsize=6.5, color="#b5623c", weight="bold")
axR.set_ylabel("margin: main − best content-equipped", fontsize=7.5)
axR.set_title("B  Advantage peaks on the sparsest data", fontsize=8.2, weight="bold",
              color=INK, loc="left")
axR.set_ylim(-0.03, 0.10)
axR.spines[["top", "right"]].set_visible(False)
axR.spines[["left", "bottom"]].set_color("#c7ccd1")
axR.tick_params(colors="#55606a", labelsize=6.5, length=2.2, width=0.6)
axR.grid(axis="y", color="#eceef0", lw=0.6, zorder=0)

# Two lines on purpose: as one line this runs wider than the 7.16 in page, and
# bbox_inches="tight" would then save a canvas wider than the figure itself.
fig.suptitle("Both-cold across the 16-variant k-core grid: content clears the floor\n"
             "everywhere, and the learned edge peaks where the graph is sparsest",
             fontsize=8.2, weight="bold", color=INK, y=0.998, va="top", linespacing=1.25)
# rect reserves the strip the two-line title sits in. Saving without
# bbox_inches="tight" is what keeps the file exactly 7.16 in wide: "tight"
# re-measures every artist and can hand back a canvas wider than the page.
fig.tight_layout(rect=[0, 0, 1, 0.945])
# 600 dpi is IEEE's line-art threshold and comfortably above the 300 dpi it
# asks of colour figures, so one raster file satisfies either classification.
fig.savefig(os.path.join(FIGS, "fig2_sparsity.png"), dpi=600)
fig.savefig(os.path.join(FIGS, "fig2_sparsity.pdf"))
fig.savefig(os.path.join(FIGS, "fig2_sparsity.tif"), dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
print(f"wrote fig2_sparsity.{{png,pdf,tif}} with {len(DATA)} variants")
