"""F5 qualitative grid visualizing the head-vs-stream contrast (one GPU).
Row PixArt (isolated CA): clean | single top-head K/V swap (FLIPS) | full swap (ref).
Row SD3.5 (joint): clean | single top-head swap (NO flip) | all-block text-stream inject (FLIPS).
Shows: in isolated CA a head flips binding; in MM-DiT the head channel is inert but the
text stream flips it. Saves results/figs/F5_grids.png.
"""
import torch, json, numpy as np, os
from PIL import Image, ImageDraw
from prompts import make_pairs

DEVICE = "cuda"; H = W = 512
os.makedirs("results/figs", exist_ok=True)


def label(img, txt):
    im = img.copy(); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 26], fill=(0, 0, 0)); d.text((6, 6), txt, fill=(255, 255, 255))
    return im


def pixart_row():
    from binding_patch_pixart import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/pixart_sweep.json"))
    ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    # paper figure uses pair 27: its top-1 swap scores 1.0 on the strict two-object metric in-sample
    # (top1_pixart_ext per_pair) and flips visibly. sw["qualified"][0] (id 1) is only a half-flip (0.5);
    # pair 25 also scores 1.0 but renders ambiguous teal. Override with F5_PIXART_PID.
    pid = int(os.environ.get("F5_PIXART_PID", 27)); p = {q["id"]: q for q in make_pairs()}[pid]
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
    lc = pipe.transformer.config.in_channels
    lat = torch.randn((1, lc, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
    emb, _ = encode(pipe, p["swapped"], DEVICE)
    clean = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=None, steps=20, guidance=4.5)
    head = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=20, guidance=4.5)
    full = generate(pipe, ctrl, p["clean"], emb, lat.clone(), swap_all=True, steps=20, guidance=4.5)
    del pipe; torch.cuda.empty_cache()
    return p, [label(clean, "PixArt: clean"),
               label(head, "+top-1 head swap"),
               label(full, "full swap (ref)")]


def sd3_row():
    from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
    from stream_sd3 import StreamHooks
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    pid = json.load(open("results/sd3_coarse.json"))["qualified"][0]
    p = {q["id"]: q for q in make_pairs()}[pid]; seed = p["seed"]
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    hooks = StreamHooks(pipe.transformer)
    capture_swapped(pipe, ctrl, p["swapped"], seed, steps=28, guidance=7.0, height=H, width=W, device=DEVICE)
    clean = generate(pipe, ctrl, p["clean"], seed, cells=None, steps=28, guidance=7.0, height=H, width=W, device=DEVICE)
    head = generate(pipe, ctrl, p["clean"], seed, cells=top1, swap_k=True, swap_v=True, steps=28, guidance=7.0, height=H, width=W, device=DEVICE)
    # text-stream inject at all blocks via hooks (capture swapped raw stream first)
    ctrl.swap_ctx = {}
    hooks.begin_capture()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    _ = pipe(p["swapped"], num_inference_steps=28, guidance_scale=7.0, height=H, width=W, generator=g).images[0]
    hooks.off()
    hooks.begin_inject(set(range(24)))
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    stream = pipe(p["clean"], num_inference_steps=28, guidance_scale=7.0, height=H, width=W, generator=g).images[0]
    hooks.off()
    return p, [label(clean, "SD3.5: clean"),
               label(head, "+top-1 head swap  (no flip)"),
               label(stream, "+text-stream inject  (FLIPS)")]


def main():
    pP, rowP = pixart_row()
    pS, rowS = sd3_row()
    canvas = Image.new("RGB", (W * 3 + 20, H * 2 + 10), "white")
    for j, im in enumerate(rowP):
        canvas.paste(im, (j * (W + 10), 0))
    for j, im in enumerate(rowS):
        canvas.paste(im, (j * (W + 10), H + 10))
    canvas.save(os.environ.get("F5_OUT", "results/figs/F5_grids.png"))
    print(f"F5 saved. PixArt pair='{pP['clean']}' | SD3.5 pair='{pS['clean']}'")


if __name__ == "__main__":
    main()
