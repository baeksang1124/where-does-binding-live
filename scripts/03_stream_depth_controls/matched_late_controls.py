"""Two controls in one run: a granularity-matched image-facing window and late-block placebo potency.

A) MATCHED 3-block image-facing control: swap the ENTIRE image-facing K/V of blocks 9-11
   (3 blocks x 24 heads) on the SAME 75 held-out pairs as block9_ext.py -> paired
   comparison text-window-9-11 vs image-window-9-11 at EQUAL block granularity.
   (Removes the granularity mismatch of comparing a 3-block text window with 1 image block.)

B) LATE-BLOCK PLACEBO POTENCY: inject the third-color placebo stream from block b onward
   (b in {9,15,18}) on the 30 coarse pairs; grade whether the placebo colors INSTALL
   (potency vs c3/c4) and whether the c1/c2 binding flips (expect 0). If late text is
   still read (potency at b>=15 high) while matched-swap flip is 0.00 there, the
   commitment is binding-specific; if late potency ~0, late text is no longer read and
   "commitment" is better described as readout closure. Either outcome is informative.
   One GPU. Writes results/matched_late.{json,log}.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
LOG = open("results/matched_late.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade2(qg, img, p):
    return canon(qg._ask_color(img, p["o1"])), canon(qg._ask_color(img, p["o2"]))


def flip_vs(k1, k2, c1, c2):
    f = 0.0
    if k1 == c2 and k1 != c1:
        f += 0.5
    if k2 == c1 and k2 != c2:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); install(pipe, ctrl)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    # ---------- A) matched image-facing 3-block window (same pairs/order as block9_ext) ----------
    qids = json.load(open("results/block9_ext.json"))["qualified"]
    Pe = {p["id"]: p for p in make_ext_pairs()}
    WB911 = [(b, h) for b in (9, 10, 11) for h in range(24)]
    img_b911 = []
    log(f"\n===== A) image-facing 3-block window (blocks 9-11), {len(qids)} pairs =====")
    for pid in qids:
        p = Pe[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        img = generate(pipe, ctrl, p["clean"], seed, cells=WB911, swap_k=True, swap_v=True,
                       steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        k1, k2 = grade2(qg, img, p); img_b911.append(flip_vs(k1, k2, p["c1"], p["c2"])); ctrl.swap_ctx = {}
        if len(img_b911) % 15 == 0:
            log(f"  n={len(img_b911)} running mean={np.mean(img_b911):.3f}")

    b9 = json.load(open("results/block9_ext.json"))
    win911_text = b9["win911"]                       # same pairs, same order
    rng = np.random.default_rng(1234)

    def ci(x):
        x = np.array(x, float); bt = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return (float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))

    def paired(a, b):
        d = np.array(a, float) - np.array(b, float)
        bt = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        return (float(d.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)),
                float((bt > 0).mean()))

    log(f"image 3-block window 9-11: {ci(img_b911)}")
    log(f"text  3-block window 9-11: {ci(win911_text)}  (from block9_ext)")
    d = paired(win911_text, img_b911)
    log(f"PAIRED text-window - image-window (matched 3v3): {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] mass>0={d[3]:.3f}")

    # ---------- B) late-block placebo potency ----------
    cq = json.load(open("results/sd3_coarse.json"))["qualified"][:30]
    Po = {p["id"]: p for p in make_pairs()}
    BS = [9, 15, 18]
    pot = {b: [] for b in BS}; pflip = {b: [] for b in BS}
    log(f"\n===== B) late-block placebo potency, {len(cq)} pairs, prefix b in {BS} =====")
    for pid in cq:
        p = Po[pid]; seed = p["seed"]
        others = [c for c in COLORS if c not in (p["c1"], p["c2"])]
        c3, c4 = (others + COLORS)[:2]
        pl_prompt = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"
        hooks.begin_capture(); _ = gen(pipe, pl_prompt, seed); hooks.off()
        for b in BS:
            hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
            k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            pot[b].append((int(k1 == c3) + int(k2 == c4)) / 2.0)
            pflip[b].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        hooks.cache = {}; hooks.clean = {}
    for b in BS:
        log(f"  prefix b={b:2d}: placebo POTENCY {ci(pot[b])}  cross-flip {np.mean(pflip[b]):.3f}")

    json.dump(dict(img_b911=img_b911, img_b911_ci=ci(img_b911), text_win911_ci=ci(win911_text),
                   paired_text_minus_img_window=list(d),
                   late_potency={str(b): pot[b] for b in BS},
                   late_potency_ci={str(b): ci(pot[b]) for b in BS},
                   late_crossflip={str(b): float(np.mean(pflip[b])) for b in BS}),
              open("results/matched_late.json", "w"), indent=2)
    log("-> results/matched_late.json")


if __name__ == "__main__":
    main()
