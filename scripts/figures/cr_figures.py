"""Paper figures F1-F4. Run from repo root:
    PYTHONPATH=. python scripts/figures/cr_figures.py
Outputs -> results/figs/{F1_teaser,F2_localization_curves,F3_granularity,F4_commitment}.{pdf,png}
Data: F1 is a schematic whose numbers are typed into fig1(); F2-F4 are read from the result JSONs.
Style: one fixed categorical palette (validated, CVD-safe); 95% bootstrap CI bands/bars in F3 and
F4 (F2 has no CIs); titles say what was measured (claims live in captions), direct labels + legend.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "results/figs"; os.makedirs(OUT, exist_ok=True)
# fixed categorical slots (dataviz reference palette, validated light-mode adjacent pairs)
C = {"joint": "#2a78d6", "textonly": "#eb6834", "third": "#1baf7a",
     "sd15": "#2a78d6", "pixart": "#eb6834", "sd35": "#1baf7a",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#9a9891", "grid": "#e6e5e1"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6, "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.labelcolor": C["ink"], "text.color": C["ink"], "grid.color": C["grid"], "grid.linewidth": 0.5,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
TEXTW = 4.8  # LNCS text width in inches


def load(p):
    return json.load(open(os.path.join("results", p)))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    print("saved", name)


def band(ax, x, ci, color, label):
    m = [c[0] for c in ci]; lo = [c[1] for c in ci]; hi = [c[2] for c in ci]
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, "-o", color=color, lw=1.6, ms=4, mec="white", mew=0.8, label=label, zorder=3)
    return m


# ---------------- F3: granularity, not channel ----------------
def fig3():
    kv = load("kv_factorial.json"); we = load("window_extension_n75.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.45), gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) waterfall: image-facing K/V window -> + clean normalisation context -> + clean pinning -> text window
    A = kv["A_swtemb_nopin_ci"]; D = kv["D_cltemb_pin_ci"]
    temb = kv["paired"]["B_minus_A (temb | no pin)"]; pin = kv["paired"]["D_minus_B (pin | clean temb)"]
    xs = [0, 1, 2, 3]; labels = ["K/V\nwindow", "+clean\nnorm.", "+clean\npin.", "text\nwindow"]
    a.bar(0, A[0], color=C["joint"], width=0.6)
    a.bar(1, temb[0], bottom=A[0], color=C["muted"], width=0.6)
    a.bar(2, pin[0], bottom=A[0] + temb[0], color=C["third"], width=0.6)
    a.bar(3, D[0], color=C["joint"], width=0.6)
    a.errorbar([0, 3], [A[0], D[0]], yerr=[[A[0] - A[1], D[0] - D[1]], [A[2] - A[0], D[2] - D[0]]],
               fmt="none", ecolor=C["ink2"], elinewidth=0.8, capsize=2)
    a.errorbar([2], [A[0] + temb[0] + pin[0]], yerr=[[pin[0] - pin[1]], [pin[2] - pin[0]]], fmt="none",
               ecolor=C["ink2"], elinewidth=0.8, capsize=2)
    a.plot([0.3, 0.7], [A[0], A[0]], color=C["ink2"], lw=0.6, ls=":"); a.plot([1.3, 1.7], [A[0], A[0]], color=C["ink2"], lw=0.6, ls=":")
    a.plot([2.3, 2.7], [D[0], D[0]], color=C["ink2"], lw=0.6, ls=":")
    # value labels sit above the CI whisker so the whisker does not strike through them
    a.text(0, A[2] + 0.012, f"{A[0]:.2f}", ha="center", fontsize=7.5)
    a.text(1, A[0] - 0.035, "+0.00", ha="center", fontsize=7, color=C["ink2"])
    a.text(2, A[0] + temb[0] + pin[2] + 0.012, f"{pin[0]:+.2f}", ha="center", fontsize=7.5)
    a.text(3, D[2] + 0.012, f"{D[0]:.2f}", ha="center", fontsize=7.5)
    a.set_xticks(xs); a.set_xticklabels(labels, fontsize=6.5); a.set_ylim(0, 0.44)
    a.set_ylabel("binding-swap recovery"); a.set_title("(a) blocks 9–11, all heads", loc="left")

    # (b) read-window length: isolated window 9..e, joint-state vs text-only stream
    ends = we["ends"]; xs_b = list(range(len(ends)))
    mj = band(b, xs_b, [we["ci"][f"std_e{e}"] for e in ends], C["joint"], "joint-state stream")
    mt = band(b, xs_b, [we["ci"][f"hyb2_e{e}"] for e in ends], C["textonly"], "text-only stream\n(image pinned clean)")
    b.axhline(0.027, color=C["muted"], lw=0.8, ls="--"); b.text(xs_b[-1], 0.05, "top-ranked head 0.03", color=C["ink2"], fontsize=6.5, ha="right")
    b.set_xticks(xs_b); b.set_xticklabels([f"9–{e}" for e in ends]); b.set_xlim(-0.3, len(ends) - 0.7); b.set_ylim(0, 1.0)
    b.set_xlabel("isolated injection window (blocks)"); b.set_ylabel("")
    b.set_title("(b) window length", loc="left")
    b.legend(loc="upper left", handlelength=1.6)
    fig.tight_layout(w_pad=1.5); save(fig, "F3_granularity")


# ---------------- F4: commitment cliff ----------------
def fig4():
    p1 = load("hyb2_prefix.json"); p2 = load("hyb2_prefix_fine.json")
    assert p1["qualified"] == p2["qualified"]
    bs = sorted(p1["bs"] + p2["bs"])
    def ci(arm, b):
        src = p1 if b in p1["bs"] else p2
        return src["ci"][f"{arm}_b{b}"]
    fig, ax = plt.subplots(figsize=(TEXTW * 0.7, 2.3))
    ax.axvspan(9, 10, color=C["grid"], alpha=0.9, lw=0, zorder=0)
    mj = band(ax, bs, [ci("std", b) for b in bs], C["joint"], "joint-state stream")
    mt = band(ax, bs, [ci("hyb2", b) for b in bs], C["textonly"], "text-only stream")
    ax.text(10.4, 0.62, "cliff 9→10", fontsize=6.5, color=C["ink2"])
    ax.text(0, 1.035, "joint-state", color=C["joint"], fontsize=7)
    ax.text(0, 0.80, "text-only", color=C["textonly"], fontsize=7)
    ax.set_xticks([0, 3, 6, 9, 12, 15]); ax.set_xlim(-0.5, 16); ax.set_ylim(0, 1.1)
    ax.set_xlabel("inject swapped text stream from block b onward")
    ax.set_ylabel("binding-swap recovery")
    ax.set_title("prefix injection, block b onward", loc="left")
    ax.legend(loc="lower left", handlelength=1.6)
    fig.tight_layout(); save(fig, "F4_commitment")



# ---------------- F2: in-sample top-k head curves (ported from plot_figures.py, paper style) ----------------
def fig2():
    sd15 = load("sweep_results.json")["head_cum"]        # [[k,v]...], ranking set n=35
    px = load("pixart_single.json")["cum_mean"]          # {k:v}, curve n=11 (ranking n=8)
    sd35 = load("sd3_single.json")["cum_mean"]           # {k:v}, n=12
    kx15, ky15 = zip(*sd15)
    kpx = sorted(int(k) for k in px); vpx = [px[str(k)] for k in kpx]
    k35 = sorted(int(k) for k in sd35); v35 = [sd35[str(k)] for k in k35]
    # drawn at its printed width (0.72 textwidth) so all lettering is >= 7 pt
    fig, ax = plt.subplots(figsize=(0.72 * TEXTW, 2.35))
    ax.plot(kx15, ky15, "-o", color=C["sd15"], label="SD1.5 (UNet, iso)", lw=1.5, ms=3.2, mec="white", mew=0.6)
    ax.plot(kpx, vpx, "-s", color=C["pixart"], label="PixArt-Σ (DiT, iso)", lw=1.5, ms=3.2, mec="white", mew=0.6)
    ax.plot(k35, v35, "-^", color=C["sd35"], label="SD3.5 (MM-DiT, joint)", lw=1.5, ms=3.2, mec="white", mew=0.6)
    ax.axhline(0.70, ls=":", color=C["muted"], lw=0.8); ax.text(2 ** 7, 0.72, "70%", color=C["ink2"], fontsize=7, ha="right")
    ax.set_xscale("log", base=2)
    ks = [1, 2, 4, 8, 16, 32, 64, 128]; ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])  # no 2^k superscripts (< 6 pt)
    ax.minorticks_off()
    ax.set_xlabel("top-$k$ image-facing cross-attention heads swapped")
    ax.set_ylabel("recovery (protocol metric)")
    ax.set_ylim(-0.03, 1.05)
    # free region between the SD3.5 floor and the isolated-CA curves
    ax.legend(loc="lower center", bbox_to_anchor=(0.6, 0.09), handlelength=1.6)
    save(fig, "F2_localization_curves")


# ---------------- F1: teaser ----------------
def fig1():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.9))
    panels = [
        ("SD1.5 (UNet)", "single image-facing head", "held-out head 0.34", C["joint"], "head"),
        ("PixArt-Σ (DiT, isolated CA)", "no generalizing\nsingle-head handle", "held-out head 0.10", C["muted"], "none"),
        ("SD3.5 (MM-DiT, joint attn)", "text stream, block window", "head 0.03 · window 9–11 0.30\n9–15 0.62 · all blocks 0.98", C["textonly"], "window"),
    ]
    for ax, (title, sub, nums, col, kind) in zip(axes, panels):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
        ax.text(5, 9.8, title, ha="center", va="top", fontsize=7.2, weight="bold")
        for x, lab, fc in ((0.6, "image\ntokens", "#dbe7f7"), (5.9, "text\ntokens", "#fbe3d3")):
            ax.add_patch(FancyBboxPatch((x, 5.0), 3.5, 2.8, boxstyle="round,pad=0.1", fc=fc, ec=C["ink2"], lw=0.6))
            ax.text(x + 1.75, 6.4, lab, ha="center", va="center", fontsize=7)
        if kind == "head":
            ax.add_patch(FancyArrowPatch((5.9, 6.4), (4.2, 6.4), arrowstyle="-|>", mutation_scale=9, color=col, lw=1.8))
            ax.text(5.05, 8.05, "binding head", ha="center", va="bottom", fontsize=7, color=col, weight="bold")
        elif kind == "none":
            ax.add_patch(FancyArrowPatch((5.9, 6.4), (4.2, 6.4), arrowstyle="-|>", mutation_scale=9, color=col, lw=1.2, ls="--"))
            ax.plot([4.7, 5.4], [5.9, 6.9], color="#e34948", lw=1.4); ax.plot([4.7, 5.4], [6.9, 5.9], color="#e34948", lw=1.4)
        else:
            ax.add_patch(FancyArrowPatch((5.9, 6.4), (4.2, 6.4), arrowstyle="-|>", mutation_scale=9, color=C["muted"], lw=1.0, ls="--"))
            ax.add_patch(FancyBboxPatch((4.4, 2.9), 5.5, 1.5, boxstyle="round,pad=0.1", fc=col, ec="none"))
            ax.text(7.15, 3.65, "read window\nblocks ~9–15", ha="center", va="center", fontsize=6.9, color="white", weight="bold")
            ax.add_patch(FancyArrowPatch((7.65, 4.5), (7.65, 4.95), arrowstyle="-|>", mutation_scale=7, color=col, lw=1.2))
        # min 6.9 pt so the lettering stays >= 6 pt at the 0.9\linewidth include (LNCS template minimum)
        ax.text(5, 2.75, sub, ha="center", va="top", fontsize=6.9, color=C["ink"], linespacing=1.1)
        ax.text(5, 1.35, nums, ha="center", va="top", fontsize=6.9, color=C["ink2"], style="italic", linespacing=1.1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    save(fig, "F1_teaser")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
