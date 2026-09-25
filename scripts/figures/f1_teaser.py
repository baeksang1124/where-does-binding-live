"""F1 teaser: the binding-circuit MIGRATION cartoon across three architectures (CPU only).
UNet (SD1.5) head -> DiT (PixArt) head -> MM-DiT (SD3.5) text-stream/block-9, with the
single-head recovery number under each. Schematic, matplotlib."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("results/figs", exist_ok=True)
C = {"img": "#c6dbef", "txt": "#fdd0a2", "head": "#d73027", "stream": "#762a83"}

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
titles = ["tested SD1.5  (UNet)", "tested PixArt-$\\Sigma$  (DiT, isolated CA)",
          "tested SD3.5  (MM-DiT, JOINT attn)"]
subs = ["sparse single\nimage-facing HEAD handle", "not localized under\nour head-level probe",
        "text-stream handle\n(block ~9-11)"]
top1 = ["held-out head: 0.34", "held-out head: 0.10", "held-out head: 0.03 | stream 0.98"]
locus = ["head", "nohead", "stream"]

for ax, ttl, sub, t1, loc in zip(axes, titles, subs, top1, locus):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(ttl, fontsize=10, weight="bold")
    # image tokens block
    ax.add_patch(FancyBboxPatch((1, 5.5), 3.2, 2.2, boxstyle="round,pad=0.1",
                 fc=C["img"], ec="k", lw=1)); ax.text(2.6, 6.6, "image\ntokens", ha="center", fontsize=9)
    # text tokens block
    ax.add_patch(FancyBboxPatch((5.8, 5.5), 3.2, 2.2, boxstyle="round,pad=0.1",
                 fc=C["txt"], ec="k", lw=1)); ax.text(7.4, 6.6, "text\ntokens", ha="center", fontsize=9)
    if loc == "head":
        # cross-attention arrow with a highlighted HEAD
        ax.add_patch(FancyArrowPatch((5.7, 6.6), (4.3, 6.6), arrowstyle="-|>", mutation_scale=16,
                     color=C["head"], lw=2.5))
        ax.text(5.0, 7.4, "binding\nhead", ha="center", fontsize=8, color=C["head"], weight="bold")
    elif loc == "nohead":
        # cross-attention present but NO single binding head (greyed, crossed out)
        ax.add_patch(FancyArrowPatch((5.7, 6.6), (4.3, 6.6), arrowstyle="-|>", mutation_scale=13,
                     color="gray", lw=1.4, linestyle=(0, (3, 3))))
        ax.text(5.0, 7.4, "no single\nbinding head", ha="center", fontsize=7.5, color="gray")
        ax.plot([4.5, 5.5], [7.0, 6.2], color=C["head"], lw=2.2)
        ax.plot([4.5, 5.5], [6.2, 7.0], color=C["head"], lw=2.2)
    else:
        # joint attention: binding inside the evolving text stream
        ax.add_patch(FancyArrowPatch((5.7, 6.0), (4.3, 6.0), arrowstyle="-|>", mutation_scale=12,
                     color="gray", lw=1.2, linestyle=(0, (4, 3))))
        ax.add_patch(FancyBboxPatch((5.8, 3.0), 3.2, 1.6, boxstyle="round,pad=0.1",
                     fc=C["stream"], ec="k", lw=1, alpha=0.85))
        ax.text(7.4, 3.8, "text stream\n(self-attn+MLP)\nblock ~9-11", ha="center", fontsize=8,
                color="white", weight="bold")
        ax.add_patch(FancyArrowPatch((7.4, 5.4), (7.4, 4.7), arrowstyle="-|>", mutation_scale=14,
                     color=C["stream"], lw=2.2))
    ax.text(5.0, 2.0, sub, ha="center", fontsize=8.5)
    ax.text(5.0, 0.7, t1, ha="center", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="none"))

# migration arrows between panels
fig.text(0.345, 0.5, "$\\Rightarrow$", fontsize=22, ha="center", color="gray")
fig.text(0.665, 0.5, "$\\Rightarrow$", fontsize=22, ha="center", color=C["stream"])
fig.suptitle("The single-head binding handle appears in the UNet and does not transfer to "
             "diffusion transformers; in MM-DiT binding is depth-localized in the text stream",
             fontsize=10.5, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig("results/figs/F1_teaser.pdf", bbox_inches="tight")
print("F1 teaser saved -> results/figs/F1_teaser.pdf (vector)")
