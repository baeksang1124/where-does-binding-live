"""Pilot of the SD3.5 text-stream depth experiments on a NON-COLOR attribute (material or
size): isolated single-block and prefix text-stream injection, plus a per-block image-facing
K/V sweep for comparison. The shipped material run (results/sd3_stream_material.json) is small
(n=4) and its stored verdict is weak/different from color, so it makes no claim of generality;
the paper's material result is the widened set in material_ext.py.

Reuses StreamHooks from stream_sd3.py. Attribute chosen via ATTR env (material|size).
One GPU.
"""
import torch, json, time, os, numpy as np
from prompts_attr import make_pairs, ATTRS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
ATTR = os.environ.get("ATTR", "material")
SUBSET = int(os.environ.get("ATTR_NPAIRS", "0"))
LOG = open(f"results/sd3_stream_{ATTR}.log", "a")
CFG = ATTRS[ATTR]


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower()
    for sub, base in CFG["canon"].items():
        if sub in ans:
            return base
    return "other"


def q(obj):
    return CFG["question"].format(obj=obj)


def flip2(qg, img, p):
    a1 = qg._ask(img, q(p["o1"])); a2 = qg._ask(img, q(p["o2"]))
    k1, k2 = canon(a1), canon(a2); f = 0.0
    if k1 == p["a2"] and k1 != p["a1"]:
        f += 0.5
    if k2 == p["a1"] and k2 != p["a2"]:
        f += 0.5
    return f, (a1, a2)


def is_clean(p, a1, a2):
    return canon(a1) == p["a1"] and canon(a2) == p["a2"]


def is_swap(p, a1, a2):
    return canon(a1) == p["a2"] and canon(a2) == p["a1"]


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    pairs = make_pairs(ATTR)
    if SUBSET:
        pairs = pairs[:SUBSET]
    log(f"\n===== SD3.5 STREAM generalization: attr={ATTR}, {len(pairs)} candidate pairs =====")

    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); kv_layers = install(pipe, ctrl)   # processor for K/V coarse
    hooks = StreamHooks(pipe.transformer)                  # block-level stream hooks
    qg = QwenGrader(device=DEVICE)

    PREFIX_B = [0, 3, 6, 9, 12, 15, 18, 21]
    iso_flip = {b: [] for b in range(NBLOCKS)}
    prefix_flip = {b: [] for b in PREFIX_B}
    kv_block_flip = {b: [] for b in range(NBLOCKS)}
    allstream, qualified = [], []

    for p in pairs:
        seed = p["seed"]
        # qualify: clean binds + swap binds (use processor path with swap_all for swap image)
        hooks.off()
        A = gen(pipe, p["clean"], seed)
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)  # also fills ctrl cache for K/V
        S = gen(pipe, p["swapped"], seed)
        _, (ca1, ca2) = flip2(qg, A, p); _, (sa1, sa2) = flip2(qg, S, p)
        ok = is_clean(p, ca1, ca2) and is_swap(p, sa1, sa2)
        log(f"pair{p['id']:2d} clean=({ca1},{ca2}) swap=({sa1},{sa2}) -> {'QUALIFY' if ok else 'skip'} | {p['clean']}")
        if not ok:
            hooks.cache = {}; hooks.clean = {}; ctrl.swap_ctx = {}
            continue
        qualified.append(p["id"])

        # --- STREAM hooks: capture swapped + clean raw block-input streams ---
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        # all-block stream inject (channel reference)
        hooks.begin_inject(set(range(NBLOCKS))); f, _ = flip2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        allstream.append(f)
        # ISOLATED single-block (propagation removed)
        for b in range(NBLOCKS):
            hooks.begin_inject({b}, isolate=True); ff, _ = flip2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            iso_flip[b].append(ff)
        # PREFIX depth
        for b in PREFIX_B:
            hooks.begin_inject(set(range(b, NBLOCKS))); ff, _ = flip2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            prefix_flip[b].append(ff)
        hooks.cache = {}; hooks.clean = {}

        # --- image-facing K/V coarse (low-channel baseline), per block all heads ---
        # reuse ctrl.swap_ctx captured above (same seed)
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        for b in range(NBLOCKS):
            cells = [(b, h) for h in range(ctrl.layer_heads[b])]
            img = generate(pipe, ctrl, p["clean"], seed, cells=cells, swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            ff, _ = flip2(qg, img, p); kv_block_flip[b].append(ff)
        ctrl.swap_ctx = {}
        im = {b: float(np.mean(iso_flip[b])) for b in range(NBLOCKS)}
        top = sorted(im, key=im.get, reverse=True)[:4]
        log(f"  allstream={allstream[-1]:.2f} | top ISO blocks: "
            + ", ".join(f"b{b}={im[b]:.2f}" for b in top))

    iso_mean = {b: float(np.mean(iso_flip[b])) if iso_flip[b] else 0.0 for b in range(NBLOCKS)}
    prefix_mean = {b: float(np.mean(prefix_flip[b])) if prefix_flip[b] else 0.0 for b in PREFIX_B}
    kv_mean = {b: float(np.mean(kv_block_flip[b])) if kv_block_flip[b] else 0.0 for b in range(NBLOCKS)}
    ranked = sorted(range(NBLOCKS), key=lambda b: iso_mean[b], reverse=True)
    log(f"\n===== {ATTR}: qualified {len(qualified)}/{len(pairs)} =====")
    log(f"CHANNEL: all-block STREAM inject={np.mean(allstream) if allstream else 0:.3f}  vs  "
        f"image-K/V per-block max={max(kv_mean.values()):.3f}")
    log(f"DEPTH (ISOLATED single-block, top 6):")
    for b in ranked[:6]:
        log(f"  block {b:2d}: iso={iso_mean[b]:.3f}  (image-K/V block={kv_mean[b]:.3f})")
    log(f"PREFIX depth (swapped text from block b onward):")
    for b in PREFIX_B:
        log(f"  from block {b:2d}: {prefix_mean[b]:.3f}")

    best_iso_b = ranked[0]
    reproduces = (np.mean(allstream) >= 0.5 if allstream else False) and iso_mean[best_iso_b] >= 0.2
    verdict = (f"GENERALIZES: {ATTR} binding is also text-stream (all-block {np.mean(allstream):.2f} >> "
               f"K/V {max(kv_mean.values()):.2f}), depth-concentrated at block {best_iso_b} "
               f"(iso {iso_mean[best_iso_b]:.2f}) -> migration+depth NOT color-specific"
               if reproduces else
               f"WEAK/DIFFERENT for {ATTR}: all-block {np.mean(allstream) if allstream else 0:.2f}, "
               f"best iso block {best_iso_b}={iso_mean[best_iso_b]:.2f} -- inspect")
    log(f"\n  VERDICT: {verdict}")

    json.dump(dict(attr=ATTR, qualified=qualified,
                   allstream=float(np.mean(allstream)) if allstream else 0.0,
                   iso_mean=iso_mean, prefix_mean=prefix_mean, kv_block_mean=kv_mean,
                   ranked_blocks=ranked, verdict=verdict),
              open(f"results/sd3_stream_{ATTR}.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_stream_{ATTR}.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
