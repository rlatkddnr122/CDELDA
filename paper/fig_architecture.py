#!/usr/bin/env python
"""Figure: content-based dual-encoder architecture (single-source paper).
Publication schematic — node-intrinsic content -> shared latent -> bilinear score.
Outputs manuscript/figures/fig_architecture.{png,pdf}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGS, exist_ok=True)

# palette (chosen, slightly warm neutrals + two towers in muted teal / clay)
INK = "#22303a"
LNC = "#2f7e7a"      # teal  — lncRNA tower
LNC_L = "#dcecea"
DIS = "#b5623c"      # clay  — disease tower
DIS_L = "#f2e2d8"
SHARE = "#3d4a55"
LATENT_L = "#e7 e9ec".replace(" ", "")
GRID = "#c7ccd1"

# Drawn at the width it is printed at: IEEE Access sets a page-wide figure at
# 7.16 in, and a schematic authored wider has all of its type scaled down by the
# same factor when placed. Type 42 embeds the fonts as TrueType, which IEEE
# requires -- matplotlib's default Type 3 is not accepted.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.linewidth": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FIGW = 7.16          # IEEE Access page width, inches
fig, ax = plt.subplots(figsize=(FIGW, 3.05))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis("off")


def box(x, y, w, h, fc, ec, text, fs=8.2, tc=INK, weight="normal", rad=1.6, lw=1.1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    # linespacing 1.15 rather than matplotlib's 1.2, and the heights below are
    # set from the text: three lines at 7.2 pt occupy about 0.36 in, so a box
    # much deeper than that is padding rather than structure
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15,
            fontsize=fs, color=tc, weight=weight, zorder=3, wrap=True)


def arrow(x1, y1, x2, y2, color=INK, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=9, color=color, lw=lw, zorder=1,
                 shrinkA=1, shrinkB=1))


# ---- title ----
ax.text(50, 93.5, "CDELDA: a content-based dual-encoder with LoRA-disease", ha="center", va="center",
        fontsize=11.5, weight="bold", color=INK)

# Box widths come from the text they hold. The widest line in a tower is
# "RNA-FM sequence emb. \u2295" at 1.32 in and "S-BioBERT encoder + LoRA" at
# 1.26; a 23-unit box is 1.72 in, so each side carries about 0.2 in of padding
# rather than the 0.46 it had when every box was 30 units wide.
TW = 23                      # tower width
ax.set_xlim(2, 98); ax.set_ylim(34.5, 97)

# ================= lncRNA tower (left) =================
lx = 7
lc = lx + TW / 2
ax.text(lc, 85.5, "lncRNA encoder", ha="center", fontsize=9.3, weight="bold", color=LNC)
box(lx, 72.5, TW, 8.5, LNC_L, LNC,
    "RNA-FM sequence emb.  \u2295\n ViennaRNA structure  \u2295\n GTEx expression", fs=7.2)
arrow(lc, 72.5, lc, 67)
# the dimension belongs to the vector travelling down the arrow, so it is
# annotated there rather than hung off the side of the box
ax.text(lc + 1.6, 69.75, "702-d", ha="left", va="center", fontsize=6.2, color="#6a7178")
box(lx, 60.5, TW, 6.5, "#ffffff", LNC, "content MLP\n(dropout)", fs=7.5)
arrow(lc, 60.5, lc, 55)
ax.text(lc + 1.6, 57.75, "MLP\n702\u2192128", ha="left", va="center", fontsize=6.0,
        color="#3f6f6c", linespacing=1.15)
box(lc - 6, 50, 12, 5, LNC, LNC, "e  \u2208  \u211d\u00b9\u00b2\u2078", fs=8.6, tc="white", weight="bold")

# ================= disease tower (right) =================
dx = 68
dc = dx + TW / 2
ax.text(dc, 85.5, "disease encoder", ha="center", fontsize=9.3, weight="bold", color=DIS)
box(dx, 72.5, TW, 8.5, DIS_L, DIS,
    "Disease Ontology\ndefinition text\n(raw input)", fs=7.3)
arrow(dc, 72.5, dc, 68)
box(dx, 60, TW, 8, DIS_L, DIS,
    "S-BioBERT encoder + LoRA\n(r=8, \u03b1=16) \u00b7 mean-pool\nbase frozen \u00b7 LoRA-only", fs=6.5, lw=1.4)
arrow(dc, 60, dc, 55)
ax.text(dc - 1.6, 57.5, "768-d\nLinear 768\u2192128", ha="right", va="center", fontsize=6.0,
        color="#8f4a2c", linespacing=1.15)
box(dc - 6, 50, 12, 5, DIS, DIS, "q  \u2208  \u211d\u00b9\u00b2\u2078", fs=8.6, tc="white", weight="bold")

# ================= shared latent + score =================
arrow(lc, 50, 44, 43.5, color=SHARE)
arrow(dc, 50, 56, 43.5, color=SHARE)
box(39, 36.5, 22, 7, "#eef1f3", SHARE, "inner-product score\n\u0177 = \u03c3( \u27e8 e , q \u27e9 )",
    fs=8.6, weight="bold")

fig.savefig(os.path.join(FIGS, "fig_architecture.png"), dpi=600)
fig.savefig(os.path.join(FIGS, "fig_architecture.pdf"))
fig.savefig(os.path.join(FIGS, "fig_architecture.tif"), dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
print("wrote fig_architecture.{png,pdf,tif}")
