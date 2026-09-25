"""PixArt-Sigma cross-attn patcher sanity brackets (isolated cross-attn, fixed T5 embed).

  A = vanilla clean
  B = generate(clean, no cells)        -> MUST be bit-identical to A
  S = vanilla swapped                  -> reference swapped image
  C = generate(clean, swap_all K&V)    -> should reproduce S
  D = generate(clean, single head)     -> differs from A, far less than C

Reports pixel L1 + per-gen time. Saves a contact sheet.
"""
import torch, time, os, numpy as np
from PIL import Image
from binding_patch_pixart import load_pipe, install, Controller, encode, generate

DEVICE = "cuda"; SEED = 0; STEPS = 20; GUID = 4.5; H = W = 512
CLEAN = "a red cube next to a blue sphere"
SWAP = "a blue cube next to a red sphere"


def l1(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


@torch.no_grad()
def vanilla(pipe, prompt, latents):
    g = None
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, latents=latents).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); names = install(pipe, ctrl)
    print(f"installed {len(names)} cross-attn layers, {sum(ctrl.layer_heads.values())} heads")

    # latent channels from transformer config
    lc = pipe.transformer.config.in_channels
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    lat = torch.randn((1, lc, H // 8, W // 8), generator=g, device=DEVICE, dtype=torch.float16)

    swap_emb, swap_mask = encode(pipe, SWAP, DEVICE)
    print(f"swap_emb shape {tuple(swap_emb.shape)}")

    tA = time.time(); A = vanilla(pipe, CLEAN, lat.clone()); dtA = time.time() - tA
    print(f"[A] vanilla clean: {dtA:.1f}s")
    S = vanilla(pipe, SWAP, lat.clone())
    print(f"[S] vanilla swapped done")

    tB = time.time(); B = generate(pipe, ctrl, CLEAN, swap_emb, lat.clone(), cells=None,
                                   steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
    dtB = time.time() - tB; dAB = l1(A, B)
    print(f"[B] zero-swap: {dtB:.1f}s ; L1(A,B)={dAB:.4f}  (expect ~0)")

    C = generate(pipe, ctrl, CLEAN, swap_emb, lat.clone(), swap_all=True, swap_k=True, swap_v=True,
                 steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
    dAC, dSC = l1(A, C), l1(S, C)
    print(f"[C] swap_all K&V: L1(A,C)={dAC:.3f}  L1(S,C)={dSC:.3f}  (expect C far from A, near S)")

    cell = (sorted(ctrl.layer_heads)[14], 0)  # mid-depth block, head 0
    D = generate(pipe, ctrl, CLEAN, swap_emb, lat.clone(), cells=[cell], swap_k=True, swap_v=True,
                 steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
    dAD = l1(A, D)
    print(f"[D] single head {cell[0]} h{cell[1]}: L1(A,D)={dAD:.3f}  (expect 0<<{dAC:.1f})")

    sheet = Image.new("RGB", (W * 5 + 40, H), "white")
    for i, im in enumerate([A, S, B, C, D]):
        sheet.paste(im, (i * (W + 10), 0))
    sheet.save("results/sanity_pixart.png")
    print("saved results/sanity_pixart.png (A_clean|S_swap|B_zero|C_all|D_1head)")

    ok_B = dAB < 1.0
    ok_C = dAC > 5.0 and dSC < dAC
    ok_D = 0.3 < dAD < dAC
    print(f"\nVERDICT: zero_identical={ok_B}  all_flips={ok_C}  head_isolates={ok_D}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
