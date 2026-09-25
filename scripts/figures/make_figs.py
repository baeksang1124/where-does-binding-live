"""Plots + example image grids from sweep_results.json. Run AFTER run_sweep.py."""
import json, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from binding_patch import load_pipe, install, encode, generate, Controller

R = json.load(open("results/sweep_results.json"))
DEVICE = "cuda"; STEPS = 30


def plots():
    layers = R["layers"]; lr = R["layer_rec"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (a) layer recovery
    vals = [lr[l] for l in layers]
    ax[0,0].bar(range(len(layers)), vals, color="steelblue")
    ax[0,0].set_title("Single-layer recovery (all 8 heads of one CA layer)")
    ax[0,0].set_xlabel("CA layer index (down->mid->up)"); ax[0,0].set_ylabel("mean recovery")
    ax[0,0].axhline(0, color="k", lw=0.5)

    # (b) layer cumulative
    lc = R["layer_cum"]
    ax[0,1].plot(range(1, len(lc)+1), lc, "o-", color="darkorange")
    ax[0,1].axhline(0.8, ls="--", color="gray"); ax[0,1].axhline(1.0, ls=":", color="green")
    ax[0,1].set_title("Layer cumulative recovery (top-k layers together)")
    ax[0,1].set_xlabel("# layers patched (ranked)"); ax[0,1].set_ylabel("recovery")

    # (c) sorted single-head recovery (heavy-tail check)
    hr = sorted(R["head_rec"].values(), reverse=True)
    ax[1,0].bar(range(len(hr)), hr, color="indianred")
    ax[1,0].set_title(f"Single-head recovery, sorted ({R['total_heads']} heads)")
    ax[1,0].set_xlabel("head rank"); ax[1,0].set_ylabel("recovery")
    ax[1,0].axhline(0.05, ls="--", color="gray", label="0.05 threshold"); ax[1,0].legend()

    # (d) head cumulative vs fraction of heads
    hc = R["head_cum"]; total = R["total_heads"]
    ks = [k for k, _ in hc]; ms = [m for _, m in hc]
    ax[1,1].plot([100*k/total for k in ks], ms, "o-", color="purple")
    ax[1,1].axhline(0.8, ls="--", color="gray"); ax[1,1].axhline(1.0, ls=":", color="green")
    ax[1,1].set_title("Head cumulative recovery (top-k heads together)")
    ax[1,1].set_xlabel("% of cross-attn heads patched"); ax[1,1].set_ylabel("recovery")

    plt.tight_layout(); plt.savefig("results/localization_curves.png", dpi=110)
    print("saved results/localization_curves.png")


def grids():
    pairs = json.load(open("results/qualified.json"))
    top_cells = R["ranked_cells"][:3]
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    ex = pairs[:5]
    rows = []
    for cellstr in top_cells:
        ln, hs = cellstr.split("|h"); cell = (ln, int(hs))
        rows.append((cellstr, cell))
    ncol = 2 + len(rows)
    fig, axes = plt.subplots(len(ex), ncol, figsize=(3*ncol, 3*len(ex)))
    for i, p in enumerate(ex):
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1,4,64,64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        axes[i,0].imshow(ic); axes[i,0].set_ylabel(f"p{p['id']} {p['grade_obj']}\n{p['clean_color']}->{p['swap_color']}", fontsize=8)
        axes[i,0].set_title("clean" if i==0 else "", fontsize=9)
        axes[i,1].imshow(ia); axes[i,1].set_title("swap-ALL" if i==0 else "", fontsize=9)
        for j, (cs, cell) in enumerate(rows):
            ip = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=[cell], steps=STEPS)
            axes[i,2+j].imshow(ip)
            if i==0: axes[i,2+j].set_title(cs.split(".attentions")[0].replace("_blocks","")+f".h{cell[1]}", fontsize=7)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig("results/top_head_grids.png", dpi=100)
    print("saved results/top_head_grids.png")


if __name__ == "__main__":
    plots()
    grids()
