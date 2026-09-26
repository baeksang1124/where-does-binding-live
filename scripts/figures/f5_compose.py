"""Compose the Fig. 5 qualitative grid from tiles rendered by f5_candidates.py (no GPU).

Rows (F5_ROWS, default "sd15,sd3"; add "pixart" for PixArt-Sigma, tiles from f5_pixart_tiles.py): SD1.5 (UNet),\nPixArt-Sigma (DiT, isolated cross-attention), SD3.5-medium (MM-DiT, joint attention).
Columns: clean | top-1 image-facing head K/V swap | text stream, blocks 9-15 (other blocks pinned clean; SD3.5 only)
| swapped prompt (target). Every tile is a held-out pair rendered at the measurement protocol; the per-pair scores it
shows are the stored ones (re-graded in results/figs/f5_candidates/candidates.json).

Env: F5_SD15_ID, F5_SD3_ID (pair ids), F5_OUT (pdf path; a .png is written next to it). Tiles are looked up in
results/figs/f5_candidates/ and results/figs/f5_candidates_sd15extra/.
Run: PYTHONPATH=. $PY scripts/figures/f5_compose.py
"""
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
import matplotlib.pyplot as plt
from PIL import Image

DIRS = ["results/figs/f5_candidates", "results/figs/f5_candidates_sd15extra"]
SD15 = int(os.environ.get("F5_SD15_ID", 16))    # base-set pair, tiles from f5_sd15_tiles.py
SD3 = int(os.environ.get("F5_SD3_ID", 5029))
OUT = os.environ.get("F5_OUT", "results/figs/F5_grids.pdf")
FS = float(os.environ.get("F5_FS", 8.5))   # label size; keep FS x (include width / figure width) >= 6 pt
COLS = ["clean", "top-1\nhead swap", "text stream\nblocks 9–15", "swapped\nprompt"]


def tile(pid, model, arm):
    for d in DIRS:
        p = f"{d}/{pid}_{model}_{arm}.png"
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"{pid}_{model}_{arm}.png")


PIX = int(os.environ.get("F5_PIX_ID", SD3))
ALL = {"sd15": ("SD1.5 (UNet)", SD15, "sd15", ["clean", "top1", None, "target"],
                ["", "flips", "n/a\n(isolated\ncross-attention)", ""]),
       "pixart": ("PixArt-Σ (DiT)", PIX, "pixart", ["clean", "top1", None, "target"],
                  ["", "no flip", "n/a\n(isolated\ncross-attention)", ""]),
       "sd3": ("SD3.5 (MM-DiT)", SD3, "sd3", ["clean", "top1", "win915", "target"],
               ["", "no flip", "flips", ""])}
rows = [ALL[k] for k in os.environ.get("F5_ROWS", "sd15,pixart,sd3").split(",")]

fig, axes = plt.subplots(len(rows), 4, figsize=(4.9, 0.25 + 1.25 * len(rows)))
for r, (name, pid, model, arms, notes) in enumerate(rows):
    for c, arm in enumerate(arms):
        ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if arm is None:
            ax.imshow(Image.new("RGB", (512, 512), (241, 240, 236)))   # same-size blank keeps the grid aligned
            ax.text(0.5, 0.5, notes[c], ha="center", va="center", fontsize=FS, color="#52514e", transform=ax.transAxes)
        else:
            ax.imshow(tile(pid, model, arm))
            if notes[c]:
                ax.text(0.03, 0.04, notes[c], ha="left", va="bottom", fontsize=FS, color="white", weight="bold",
                        transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.2", fc="#0b0b0b", ec="none", alpha=0.75))
        if r == 0:
            ax.set_title(COLS[c], fontsize=FS, pad=3)
    axes[r][0].set_ylabel(name, fontsize=FS + 0.5)
fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.35)
fig.savefig(OUT, dpi=300, bbox_inches="tight"); fig.savefig(OUT[:-4] + ".png", dpi=200, bbox_inches="tight")
print("F5 composed ->", OUT, "| SD1.5 pair", SD15, "| SD3.5 pair", SD3)
