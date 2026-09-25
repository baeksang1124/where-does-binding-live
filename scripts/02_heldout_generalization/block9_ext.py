"""Statistical separation of the SD3.5 depth result (block-9 text-stream vs the
image-facing whole-block ceiling), on HELD-OUT pairs with a PAIRED design.

On the SD3.5 ext qualified pairs (n~75, disjoint from the n=30 in-sample depth curve), per pair:
  - iso text-stream inject block 9        (single block, clean pinned elsewhere)
  - iso text-stream inject block 11
  - iso text-stream inject blocks {9,10,11} (contiguous window -- how much does the region recover?)
  - image-facing whole-block-9 K/V swap   (the per-block image ceiling, SAME pair -> paired diff)
Reports ext-only + combined-with-n=30 CIs, and the PAIRED per-pair difference
(text-block9 - image-block9) bootstrap CI, which compares the two intervention sites at
matched (single-block) granularity. Also: this is a HELD-OUT test of the depth locus itself
(block 9 was identified on the in-sample pairs). Env knob: B9_NMAX (max pairs, default 75).
Outputs: results/block9_ext.json, results/block9_ext.log. One GPU.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
NMAX = int(os.environ.get("B9_NMAX", "75"))
LOG = open("results/block9_ext.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip2(qg, img, p):
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    qids = json.load(open("results/top1_sd3_ext.json"))["qualified"][:NMAX]   # ext held-out qualified
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== block-9 HELD-OUT expansion: {len(qids)} ext pairs =====")

    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); install(pipe, ctrl)          # image-facing K/V machinery
    hooks = StreamHooks(pipe.transformer)             # text-stream machinery
    qg = QwenGrader(device=DEVICE)
    WB9 = [(9, h) for h in range(24)]                 # whole image-facing block 9 (the ceiling block)

    iso9, iso11, win911, img_b9 = [], [], [], []
    for pid in qids:
        p = P[pid]; seed = p["seed"]
        # image-facing whole-block-9 K/V swap (ctrl path)
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        img = generate(pipe, ctrl, p["clean"], seed, cells=WB9, swap_k=True, swap_v=True,
                       steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        img_b9.append(flip2(qg, img, p)); ctrl.swap_ctx = {}
        # text-stream captures (hooks path)
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        swap_cache = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        hooks.cache = swap_cache
        for blocks, acc in [({9}, iso9), ({11}, iso11), ({9, 10, 11}, win911)]:
            hooks.begin_inject(set(blocks), isolate=True)
            acc.append(flip2(qg, gen(pipe, p["clean"], seed), p)); hooks.off()
        hooks.cache = {}; hooks.clean = {}
        log(f"pair{pid} img_b9={img_b9[-1]:.2f} iso9={iso9[-1]:.2f} iso11={iso11[-1]:.2f} win911={win911[-1]:.2f}"
            + (f"  [n={len(iso9)} run-means img{np.mean(img_b9):.3f} t9 {np.mean(iso9):.3f} w911 {np.mean(win911):.3f}]"
               if len(iso9) % 10 == 0 else ""))

    rng = np.random.default_rng(1234)

    def ci(x):
        x = np.array(x, float); bt = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return (float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))

    def paired_diff(a, b):
        d = np.array(a, float) - np.array(b, float)
        bt = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        return (float(d.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)),
                float((bt > 0).mean()))

    old9 = json.load(open("results/sd3_stream_ci.json"))["iso_per_pair"]["9"]
    comb9 = old9 + iso9
    log(f"\niso block9  EXT n={len(iso9)}: {ci(iso9)}")
    log(f"iso block9  COMBINED n={len(comb9)}: {ci(comb9)}   (in-sample n=30 was mean 0.200)")
    log(f"iso block11 EXT n={len(iso11)}: {ci(iso11)}")
    log(f"window 9-11 EXT n={len(win911)}: {ci(win911)}")
    log(f"image whole-block-9 SAME PAIRS n={len(img_b9)}: {ci(img_b9)}")
    d, lo, hi, pgt = paired_diff(iso9, img_b9)
    log(f"PAIRED text-b9 - image-b9: {d:+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={pgt:.3f}")
    d2, lo2, hi2, pgt2 = paired_diff(win911, img_b9)
    log(f"PAIRED window911 - image-b9: {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}] P(>0)={pgt2:.3f}")
    json.dump(dict(qualified=qids, iso9=iso9, iso11=iso11, win911=win911, img_b9=img_b9,
                   iso9_ext_ci=ci(iso9), iso9_combined_ci=ci(comb9), iso11_ext_ci=ci(iso11),
                   win911_ci=ci(win911), img_b9_ci=ci(img_b9),
                   paired_t9_minus_img=[d, lo, hi, pgt], paired_w911_minus_img=[d2, lo2, hi2, pgt2]),
              open("results/block9_ext.json", "w"), indent=2)


if __name__ == "__main__":
    main()
