"""Window-level placebo and multi-seed checks for the SD3.5 text-stream window {9,10,11}.

A) WINDOW-LEVEL PLACEBO: the binding-specificity placebo is otherwise measured at
   all-block granularity; the block 9-11 window (held-out flip 0.30, block9_ext.py) gets
   its own control here. For 30 coarse pairs: inject the third-color placebo stream at the
   ISOLATED window {9,10,11} (clean pinned elsewhere) -> cross-flip vs c1/c2 (expect ~0)
   and placebo potency vs c3/c4 (how much the window alone can install).

B) 3-SEED for the window + prefix horizon: extends seed robustness from the all-block flip
   to the window and prefix flips. For 15 base pairs x 3 fresh seed offsets: re-qualify,
   then measure isolated window {9,10,11} flip and prefix flips from b=9 and b=15.
Env knobs: WP_NA (pairs for A, default 30), WP_NB (pairs for B, default 15).
Outputs: results/window_placebo_seeds.json, results/window_placebo_seeds.log.
One GPU.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
WIN = {9, 10, 11}
SEED_OFFSETS = [10000, 20000, 30000]
NP_A = int(os.environ.get("WP_NA", "30")); NP_B = int(os.environ.get("WP_NB", "15"))
LOG = open("results/window_placebo_seeds.log", "a")


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


def ci(x, rng):
    x = np.array(x, float); bt = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))


def main():
    os.makedirs("results", exist_ok=True)
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)
    rng = np.random.default_rng(1234)
    P = {p["id"]: p for p in make_pairs()}
    cq = json.load(open("results/sd3_coarse.json"))["qualified"]

    # ---------- A) window placebo ----------
    log(f"\n===== A) WINDOW {sorted(WIN)} placebo: {NP_A} pairs =====")
    wflip, wpot = [], []
    for pid in cq[:NP_A]:
        p = P[pid]; seed = p["seed"]
        others = [c for c in COLORS if c not in (p["c1"], p["c2"])]
        c3, c4 = (others + COLORS)[:2]
        pl = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"
        hooks.begin_capture(); _ = gen(pipe, pl, seed); hooks.off()
        pl_cache = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        hooks.cache = pl_cache
        hooks.begin_inject(set(WIN), isolate=True)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        wflip.append(flip_vs(k1, k2, p["c1"], p["c2"]))
        wpot.append((int(k1 == c3) + int(k2 == c4)) / 2.0)
        hooks.cache = {}; hooks.clean = {}
    log(f"window placebo cross-flip: {ci(wflip, rng)}  nonzero pairs {sum(1 for x in wflip if x>0)}/{len(wflip)}")
    log(f"window placebo potency  : {ci(wpot, rng)}")

    # ---------- B) 3-seed window + prefix ----------
    log(f"\n===== B) 3-seed window+prefix: {NP_B} pairs x {SEED_OFFSETS} =====")
    res = {o: dict(win=[], pre9=[], pre15=[]) for o in SEED_OFFSETS}
    for off in SEED_OFFSETS:
        for pid in cq[:NP_B]:
            p = P[pid]; seed = p["seed"] + off
            ck = grade2(qg, gen(pipe, p["clean"], seed), p)
            if not (ck[0] == p["c1"] and ck[1] == p["c2"]):
                continue
            sk = grade2(qg, gen(pipe, p["swapped"], seed), p)
            if not (sk[0] == p["c2"] and sk[1] == p["c1"]):
                continue
            hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
            sw = {b: list(v) for b, v in hooks.cache.items()}
            hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
            hooks.cache = sw
            hooks.begin_inject(set(WIN), isolate=True)
            k = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            res[off]["win"].append(flip_vs(*k, p["c1"], p["c2"]))
            for b, key in [(9, "pre9"), (15, "pre15")]:
                hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
                k = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
                res[off][key].append(flip_vs(*k, p["c1"], p["c2"]))
            hooks.cache = {}; hooks.clean = {}
        n = len(res[off]["win"])
        log(f"  seed+{off}: n={n} win={np.mean(res[off]['win']) if n else 0:.3f} "
            f"pre9={np.mean(res[off]['pre9']) if n else 0:.3f} pre15={np.mean(res[off]['pre15']) if n else 0:.3f}")

    means = {k: [float(np.mean(res[o][k])) for o in SEED_OFFSETS if res[o][k]] for k in ["win", "pre9", "pre15"]}
    for k in means:
        log(f"{k}: per-seed means {['%.3f'%x for x in means[k]]}  mean-of-means {np.mean(means[k]):.3f} SD {np.std(means[k]):.3f}")
    json.dump(dict(window_placebo_flip=wflip, window_placebo_potency=wpot,
                   wflip_ci=ci(wflip, rng), wpot_ci=ci(wpot, rng),
                   seeds={str(o): res[o] for o in SEED_OFFSETS},
                   per_seed_means=means),
              open("results/window_placebo_seeds.json", "w"), indent=2)
    log("-> results/window_placebo_seeds.json")


if __name__ == "__main__":
    main()
