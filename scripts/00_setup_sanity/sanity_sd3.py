"""SD3.5 joint-attn patcher sanity brackets.

  A = vanilla clean (controller off)
  B = generate(clean, no cells)        -> MUST be bit-identical to A
  C = generate(clean, swap_all K&V)    -> should reproduce the SWAPPED image (approx)
  S = capture_swapped (the real swapped-prompt image; reference)
  D = generate(clean, single head)     -> differs from A but far less than C

Reports pixel L1 distances + per-gen wall time. Saves a contact sheet.
"""
import torch, time, numpy as np
from PIL import Image
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate

DEVICE = "cuda"
SEED = 0
STEPS = 28
H = W = 512
CLEAN = "a photo of a red cube next to a blue sphere"
SWAP = "a photo of a blue cube next to a red sphere"


def l1(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


@torch.no_grad()
def vanilla(pipe, prompt):
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=7.0,
                height=H, width=W, generator=g).images[0]


def main():
    t0 = time.time()
    pipe = load_pipe(device=DEVICE)
    ctrl = Controller()
    blocks = install(pipe, ctrl)
    print(f"installed on {len(blocks)} blocks, heads/block={ctrl.layer_heads[0]} "
          f"=> {len(blocks)*ctrl.layer_heads[0]} total heads")

    tA = time.time(); A = vanilla(pipe, CLEAN); dtA = time.time()-tA
    print(f"[A] vanilla clean: {dtA:.1f}s")

    tS = time.time(); S = capture_swapped(pipe, ctrl, SWAP, SEED, steps=STEPS,
                                          height=H, width=W, device=DEVICE); dtS = time.time()-tS
    nsteps_cached = len(ctrl.swap_ctx[0])
    print(f"[S] capture swapped: {dtS:.1f}s ; cached {nsteps_cached} steps x {len(blocks)} blocks ; "
          f"ctx shape {tuple(ctrl.swap_ctx[0][0].shape)}")

    tB = time.time(); B = generate(pipe, ctrl, CLEAN, SEED, cells=None,
                                   steps=STEPS, height=H, width=W, device=DEVICE); dtB = time.time()-tB
    dAB = l1(A, B)
    print(f"[B] zero-swap: {dtB:.1f}s ; L1(A,B)={dAB:.4f}  (expect ~0 bit-identical)")

    tC = time.time(); C = generate(pipe, ctrl, CLEAN, SEED, swap_all=True, swap_k=True, swap_v=True,
                                   steps=STEPS, height=H, width=W, device=DEVICE); dtC = time.time()-tC
    dAC, dSC = l1(A, C), l1(S, C)
    print(f"[C] swap_all K&V: {dtC:.1f}s ; L1(A,C)={dAC:.3f}  L1(S,C)={dSC:.3f}  "
          f"(expect C far from clean A, close to swapped S)")

    # single head: pick block 12, head 0 as an arbitrary probe
    cell = (12, 0)
    tD = time.time(); D = generate(pipe, ctrl, CLEAN, SEED, cells=[cell], swap_k=True, swap_v=True,
                                   steps=STEPS, height=H, width=W, device=DEVICE); dtD = time.time()-tD
    dAD = l1(A, D)
    print(f"[D] single head {cell}: {dtD:.1f}s ; L1(A,D)={dAD:.3f}  "
          f"(expect 0 < L1(A,D) << L1(A,C)={dAC:.3f})")

    # contact sheet
    sheet = Image.new("RGB", (W*5 + 40, H), "white")
    for i, im in enumerate([A, S, B, C, D]):
        sheet.paste(im, (i*(W+10), 0))
    sheet.save("results/sanity_sd3.png")
    print("\nsaved results/sanity_sd3.png  (A_clean | S_swapped | B_zero | C_all | D_1head)")

    # verdict
    ok_B = dAB < 1.0
    ok_C = dAC > 5.0 and dSC < dAC
    ok_D = 0.5 < dAD < dAC
    print(f"\nVERDICT: zero-swap_identical={ok_B}  all-swap_flips={ok_C}  single-head_isolates={ok_D}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
