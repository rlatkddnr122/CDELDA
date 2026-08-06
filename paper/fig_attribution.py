import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
#!/usr/bin/env python
"""Figure 2 — attribution study on the MAIN model (dual-encoder + LoRA-disease), l2d5.
Four ablation panels, all at both-cold (C4) with warm shown for contrast where available:
  A content-axis   : both / lncRNA-only / disease-only / neither(free-emb)  -> content is the lever
  B architecture   : dot / dual-attn / NCF / bilinear / contrastive          -> head does not matter
  C lncRNA modality: ortho(composite) / k-mer / RNA-FM / structure / expr    -> signal is distributed
  D encoder adapt. : frozen / LoRA / IA3 / VeRA / prompt                      -> LoRA is the sweet spot
Reads the ablation JSONs; run after ablation_hero.py completes.
Outputs manuscript/figures/fig3_attribution.{png,pdf}.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TT = _REPO
FIGS = os.path.join(TT, "manuscript", "figures")
os.makedirs(FIGS, exist_ok=True)
TEAL, TEAL_D, CLAY, FLOORC, GRID = "#2f7e7a", "#22615e", "#b5623c", "#a8524a", "#c7ccd1"
INK, MUT = "#22303a", "#66716c"


# --- Self-contained plot data (main model = dual-encoder + LoRA-disease, l2d5, both-cold AUPR 1:1,
#     seed 2026). Committed here so the figure regenerates WITHOUT the gitignored ablation-results
#     dirs. Values: panel A = Table 3; B = Supplementary Table S2; C/D = the group's ablation runs. ---
A = [("both", 0.717), ("lncRNA\nonly", 0.489),
     ("disease\nonly", 0.505), ("neither\n(free-emb)", 0.507)]
B = [("dot", 0.704), ("dual-attn", 0.683), ("NCF", 0.694),
     ("bilinear", 0.689), ("contrastive", 0.677)]
C = [("ortho\n(composite)", 0.717), ("k-mer", 0.681),
     ("RNA-FM", 0.688), ("structure", 0.679), ("expr", 0.690)]
D = [("frozen\n(dot)", 0.704), ("LoRA", 0.717), ("IA3", 0.693),
     ("VeRA", 0.687), ("prompt", 0.676)]

PANELS = [("A", "content axis (ablate content)", A, TEAL_D, "content is the lever"),
          ("B", "interaction head (ablate architecture)", B, "#5f7d8c", "head does not matter"),
          ("C", "lncRNA modality (ablate channel)", C, TEAL, "signal is distributed"),
          ("D", "encoder adaptation (ablate LoRA)", D, CLAY, "LoRA is the sweet spot")]

# Drawn at the width it is printed at. IEEE Access sets a page-wide figure at
# 7.16 in; authored wider, every label is scaled down by the same factor when
# placed. Type 42 embeds the fonts as TrueType, which IEEE requires -- the
# matplotlib default, Type 3, is not accepted.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
FIGW = 7.16          # IEEE Access page width, inches
fig, axes = plt.subplots(2, 2, figsize=(FIGW, 4.9))
fig.suptitle("Attribution study on CDELDA (LoRA-disease), both-cold AUPR at l2d5",
             fontsize=8.6, weight="bold", color=INK, y=0.985)

for ax, (tag, title, data, col, take) in zip(axes.ravel(), PANELS):
    labels = [d[0] for d in data]
    vals = [d[1] if d[1] is not None else 0 for d in data]
    miss = [d[1] is None for d in data]
    xs = range(len(labels))
    bars = ax.bar(xs, vals, color=col, width=0.66, zorder=3, edgecolor="white", lw=0.5)
    # emphasise the best/reference bar per panel
    for i, b in enumerate(bars):
        if miss[i]:
            b.set_color(GRID); b.set_hatch("//")
    ax.axhline(0.5, color=FLOORC, lw=0.9, ls=":", zorder=2)
    ax.text(0.012, 0.5, "floor 0.500", transform=ax.get_yaxis_transform(), color=FLOORC,
            fontsize=5.8, va="bottom", ha="left")
    for i, v in enumerate(vals):
        if not miss[i]:
            ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.2, color=INK, weight="bold")
        else:
            ax.text(i, 0.52, "pending", ha="center", fontsize=6, color=MUT, rotation=90)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylim(0.47, 0.77)
    ax.set_title(f"{tag}. {title}", fontsize=7.4, weight="bold", color=INK, loc="left", pad=4)
    ax.text(0.99, 0.94, take, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.2, style="italic", color=MUT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=6, length=2.2, width=0.6)
    ax.set_ylabel("both-cold (C-both) AUPR", fontsize=6.4, color=MUT)

fig.tight_layout(rect=[0, 0, 1, 0.955])
# saving without bbox_inches="tight" keeps the file exactly 7.16 in wide
fig.savefig(os.path.join(FIGS, "fig3_attribution.png"), dpi=600)
fig.savefig(os.path.join(FIGS, "fig3_attribution.pdf"))
fig.savefig(os.path.join(FIGS, "fig3_attribution.tif"), dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
print("wrote fig3_attribution.{png,pdf,tif}")
