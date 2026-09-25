"""Generate the quantitative paper figures from the result JSONs.
F2: 3-model single-object localization curves (top-k heads).
F3: SD3.5 channel contrast (image-K/V vs text-stream) + isolated single-block depth curve.
F4: PREFIX depth-commitment curves (color + material).
Outputs -> results/figs/*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figs", exist_ok=True)
C = {"sd15": "#2166ac", "pixart": "#1a9850", "sd35": "#d73027", "stream": "#762a83",
     "material": "#e08214"}


def load(p):
    return json.load(open(p))


# ---------- F2: 3-model single-object localization curves ----------
def fig2():
    sd15 = load("results/sweep_results.json")["head_cum"]        # [[k,v]...]
    px = load("results/pixart_single.json")["cum_mean"]          # {k:v}
    sd35 = load("results/sd3_single.json")["cum_mean"]
    kx15, ky15 = zip(*sd15)
    kpx = sorted(int(k) for k in px); vpx = [px[str(k)] for k in kpx]
    k35 = sorted(int(k) for k in sd35); v35 = [sd35[str(k)] for k in k35]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(kx15, ky15, "-o", color=C["sd15"], label="SD1.5 (UNet, isolated CA)", lw=2, ms=4)
    ax.plot(kpx, vpx, "-s", color=C["pixart"], label="PixArt-Σ (DiT, isolated CA)", lw=2, ms=4)
    ax.plot(k35, v35, "-^", color=C["sd35"], label="SD3.5 (MM-DiT, JOINT attn)", lw=2, ms=4)
    ax.axhline(0.70, ls=":", color="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("top-k image-facing cross-attention heads swapped")
    ax.set_ylabel("binding-swap recovery (single-object)")
    ax.set_title("A few heads appear to carry binding in-sample —\nonly the UNet's survives held-out", fontsize=11)
    ax.set_ylim(-0.03, 1.08)
    # curves start low-left (<=0.33 at k=1) -> upper-left corner is empty; legend fits there
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig("results/figs/F2_localization_curves.pdf")
    print("F2 saved (vector pdf)")


# ---------- F3: channel contrast + isolated depth curve ----------
def fig3():
    ci = load("results/sd3_stream_ci.json")                    # n=30 per-pair (P1a)
    coarse = load("results/sd3_coarse.json")
    kv_ceiling = max(coarse["block_mean_flip"].values())        # image-facing K/V per-block max
    stream_ref = float(np.mean(ci["allmatch"]))                # all-block text-stream inject, n=30
    iso_pp = ci["iso_per_pair"]                                 # {block:[per-pair]}
    blocks = sorted(int(b) for b in iso_pp)
    iv = [float(np.mean(iso_pp[str(b)])) for b in blocks]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    # (a) channel bar
    a1.bar(["image-facing\nK/V (per-head)", "text stream\n(all-block)"],
           [kv_ceiling, stream_ref], color=[C["sd35"], C["stream"]])
    a1.set_ylabel("binding-swap recovery"); a1.set_ylim(0, 1.08)
    a1.set_title("SD3.5: binding not localizable\nin the image channel", fontsize=11)
    for i, v in enumerate([kv_ceiling, stream_ref]):
        a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
    # (b) isolated single-block depth. Bars peak center (block 9); right side (blocks>15) empty.
    a2.bar(blocks, iv, color=C["stream"])
    a2.axhline(kv_ceiling, ls=":", color=C["sd35"], lw=1.2, label=f"image-K/V ceiling {kv_ceiling:.2f}")
    a2.set_xlabel("transformer block (isolated single-block text-stream inject)")
    a2.set_ylabel("binding-swap recovery")
    a2.set_title("Text-stream binding is depth-localized (block ~9-11)", fontsize=11)
    a2.set_ylim(0, max(iv) * 1.25)
    a2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig("results/figs/F3_channel_depth.pdf")
    print("F3 saved (vector pdf)")


# ---------- F4: PREFIX depth-commitment (color + material) ----------
def fig4():
    col_pp = load("results/sd3_stream_ci.json")["prefix_per_pair"]   # n=30 (P1a)
    mat = load("results/sd3_stream_material.json")["prefix_mean"]
    kc = sorted(int(b) for b in col_pp); vc = [float(np.mean(col_pp[str(b)])) for b in kc]
    km = sorted(int(b) for b in mat); vm = [mat[str(b)] for b in km]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(kc, vc, "-o", color=C["sd35"], label="color", lw=2, ms=5)
    ax.plot(km, vm, "-s", color=C["material"], label="material (n=4, preliminary)", lw=2, ms=5)
    ax.axvspan(9, 15, color="gray", alpha=0.12)
    ax.text(13.5, 0.72, "commit\nwindow", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("inject swapped text stream from block b onward")
    ax.set_ylabel("binding-swap recovery")
    ax.set_title("Binding commits by block ~15\n(later text edits do nothing)", fontsize=11)
    ax.set_ylim(-0.03, 1.12)
    # curves are high-left, drop to floor for x>15 -> center-right is empty
    ax.legend(fontsize=9, loc="center right", framealpha=0.9)
    fig.tight_layout(); fig.savefig("results/figs/F4_depth_commitment.pdf")
    print("F4 saved (vector pdf)")


if __name__ == "__main__":
    fig2(); fig3(); fig4()
    print("all figures -> results/figs/")
