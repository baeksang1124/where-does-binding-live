"""Re-wrap the F5 qualitative grid into a VECTOR-container PDF (figures in vector format). The 6 panels are ACTUAL diffusion-generated samples (photographs) -> the
pixels are inherently raster and cannot be vectorized; we embed them at full resolution via
imshow and render the panel labels as crisp VECTOR text, saving a PDF. No GPU: reuses the
existing composite results/figs/F5_grids.png (crops out the burned-in label bars)."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3 (Springer PDF checks)
import matplotlib.pyplot as plt
from PIL import Image

W = H = 512; GAP = 10; LAB = 26
import os
comp = Image.open(os.environ.get("F5_IN", "results/figs/F5_grids.png"))
xs = [0, W + GAP, 2 * (W + GAP)]           # 0, 522, 1044
ys = [0, H + GAP]                          # row PixArt, row SD3.5
# PixArt row = pair 27 (top-1 swap scores 1.0 on the strict two-object metric; both objects flip)
labels = [["PixArt-Σ: clean", "+top-1 head (flips)", "full swap (ref)"],
          ["SD3.5: clean", "+top-1 head (no flip)", "+stream, all blocks (flips)"]]
row_tag = ["isolated CA (DiT)", "joint attn (MM-DiT)"]

# drawn near its printed size (0.84 textwidth ~ 4 in) so the lettering stays >= 6 pt (template rule)
fig, axes = plt.subplots(2, 3, figsize=(5.4, 3.84))
for r in range(2):
    for c in range(3):
        cell = comp.crop((xs[c], ys[r] + LAB, xs[c] + W, ys[r] + H))   # drop burned label bar
        ax = axes[r][c]
        ax.imshow(cell); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(labels[r][c], fontsize=8.5, pad=3)
    axes[r][0].set_ylabel(row_tag[r], fontsize=8.5)
fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
fig.savefig(os.environ.get("F5_PDF", "results/figs/F5_grids.pdf"), dpi=300, bbox_inches="tight")
print("F5 re-wrapped -> results/figs/F5_grids.pdf (vector labels + embedded raster samples)")
