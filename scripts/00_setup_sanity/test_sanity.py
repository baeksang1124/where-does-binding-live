"""Correctness brackets for the patcher. Must pass before trusting any sweep number."""
import torch, numpy as np
from binding_patch import load_pipe, install, encode, generate, Controller

DEVICE = "cuda"
STEPS = 30


def img_arr(im):
    return np.asarray(im).astype(np.int16)


def main():
    pipe = load_pipe(device=DEVICE)
    ctrl = Controller()
    layers = install(pipe, ctrl)
    print(f"{len(layers)} cross-attn layers; heads per layer:")
    for ln in layers:
        print("  ", ln, ctrl.layer_heads[ln])
    total_heads = sum(ctrl.layer_heads.values())
    print("TOTAL CROSS-ATTN HEADS:", total_heads)

    clean = "a red cube and a blue sphere"
    swapped = "a blue cube and a red sphere"
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    latents = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)

    swap_embeds = encode(pipe, swapped, DEVICE)

    # vanilla references (processor present but disabled paths)
    img_clean = generate(pipe, ctrl, clean, swap_embeds, latents.clone(), cells=None, steps=STEPS)
    img_swapall = generate(pipe, ctrl, clean, swap_embeds, latents.clone(), swap_all=True, steps=STEPS)

    # independent vanilla swapped-prompt generation (no swap, just run swapped prompt)
    ctrl.reset()
    img_vanilla_swapped = pipe(swapped, num_inference_steps=STEPS, guidance_scale=7.5,
                               latents=latents.clone()).images[0]

    d_swapall = np.abs(img_arr(img_swapall) - img_arr(img_vanilla_swapped)).max()
    print(f"\n[BRACKET] swap-ALL vs vanilla-swapped  max|pixel diff| = {d_swapall} (expect ~0)")

    # single-head isolation: patch one head, compare to clean -- some change, but bounded
    one_cell = [(layers[6], 0)]
    img_one = generate(pipe, ctrl, clean, swap_embeds, latents.clone(), cells=one_cell, steps=STEPS)
    d_one = np.abs(img_arr(img_one) - img_arr(img_clean)).max()
    print(f"[ISOLATION] 1 head ({layers[6]} h0) vs clean  max|pixel diff| = {d_one} (expect >0, small)")

    # two different single heads differ from each other
    img_one_b = generate(pipe, ctrl, clean, swap_embeds, latents.clone(), cells=[(layers[6], 1)], steps=STEPS)
    d_heads = np.abs(img_arr(img_one) - img_arr(img_one_b)).max()
    print(f"[ISOLATION] head0 vs head1 same layer  max|pixel diff| = {d_heads} (expect >0)")

    img_clean.save("results/sanity_clean.png")
    img_swapall.save("results/sanity_swapall.png")
    img_vanilla_swapped.save("results/sanity_vanilla_swapped.png")
    img_one.save("results/sanity_onehead.png")

    ok = d_swapall <= 2 and d_one > 0 and d_heads > 0
    print("\nSANITY", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
