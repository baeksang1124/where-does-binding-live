"""Pilot: does the text-stream locus hold for a NON-COLOUR attribute at usable sample size?

The earlier material probe (stream_attr_sd3.py) qualified only n=4, because prompts_attr.py
has just 12 candidate pairs. Here the candidate pool is widened to the 8-object set used for
the colour held-out split; the script measures the qualification funnel and -- on whatever
qualifies -- runs the same block 9-11 isolated-window injection plus an all-block reference. Purely a pilot to size the effect.

Distinct id/seed space (7000+) so it cannot collide with colour pairs.
One GPU.
"""
import torch, json, os, time, gc
from itertools import permutations
import numpy as np
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from prompts_attr import ATTRS
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
WIN = {9, 10, 11}
OBJECTS = ["cube", "sphere", "car", "cup", "book", "ball", "bottle", "box"]
CFGM = ATTRS["material"]
VALS, CANON, QUESTION = CFGM["values"], CFGM["canon"], CFGM["question"]
NCAND = int(os.environ.get("MAT_N", "48"))
os.makedirs("results", exist_ok=True)
LOG = open("results/material_ext.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def make_material_pairs(limit=0):
    """Same construction as the colour held-out set: distinct object pairs x rotating
    distinct material values, de-duplicated by clean text."""
    pairs, i, seen = [], 0, set()
    for k, (o1, o2) in enumerate(permutations(OBJECTS, 2)):
        a1 = VALS[k % len(VALS)]
        a2 = VALS[(k + 1) % len(VALS)]
        clean = f"a {a1} {o1} next to a {a2} {o2}"
        if clean in seen:
            continue
        seen.add(clean)
        pairs.append(dict(id=7000 + i, seed=7000 + i, clean=clean,
                          swapped=f"a {a2} {o1} next to a {a1} {o2}",
                          o1=o1, o2=o2, a1=a1, a2=a2))
        i += 1
        if limit and i >= limit:
            break
    return pairs


def canon_mat(ans):
    ans = (ans or "").lower()
    hits = {base for k, base in CANON.items() if k in ans}
    return hits.pop() if len(hits) == 1 else "other"


def grade2(qg, img, p):
    return (canon_mat(qg._ask(img, QUESTION.format(obj=p["o1"]))),
            canon_mat(qg._ask(img, QUESTION.format(obj=p["o2"]))))


def flip_vs(k1, k2, a1, a2):
    f = 0.0
    if k1 == a2 and k1 != a1:
        f += 0.5
    if k2 == a1 and k2 != a2:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    t0 = time.time()
    cands = make_material_pairs(NCAND)
    log(f"\n===== MATERIAL widened pilot: {len(cands)} candidates (was 12 -> n=4 qualified) =====")
    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    # ---- stage 1: qualification funnel (two-object gate, same as SD3.5 colour) ----
    qualified, funnel = [], dict(clean_ok=0, swap_ok=0, both=0)
    for n, p in enumerate(cands):
        k1c, k2c = grade2(qg, gen(pipe, p["clean"], p["seed"]), p)
        k1s, k2s = grade2(qg, gen(pipe, p["swapped"], p["seed"]), p)
        c_ok = (k1c == p["a1"] and k2c == p["a2"])
        s_ok = (k1s == p["a2"] and k2s == p["a1"])
        funnel["clean_ok"] += c_ok; funnel["swap_ok"] += s_ok
        if c_ok and s_ok:
            funnel["both"] += 1; qualified.append(p)
        if (n + 1) % 12 == 0:
            log(f"  screened {n+1}/{len(cands)}: qualified {len(qualified)}")
    log(f"FUNNEL: clean-binds {funnel['clean_ok']}/{len(cands)}, "
        f"swap-binds {funnel['swap_ok']}/{len(cands)}, BOTH (qualified) "
        f"{funnel['both']}/{len(cands)} = {funnel['both']/len(cands):.0%}")

    # ---- stage 2: window + all-block injection on qualified pairs ----
    win, allb = [], []
    for n, p in enumerate(qualified):
        seed = p["seed"]
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        hooks.begin_inject(set(WIN), isolate=True)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        win.append(flip_vs(k1, k2, p["a1"], p["a2"]))
        hooks.begin_inject(set(range(NBLOCKS)), isolate=False)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        allb.append(flip_vs(k1, k2, p["a1"], p["a2"]))
        hooks.cache = {}; hooks.clean = {}; gc.collect()
        log(f"  id{p['id']} [{n+1}/{len(qualified)}] win={win[-1]:.2f} all={allb[-1]:.2f} "
            f"| running win={np.mean(win):.3f} all={np.mean(allb):.3f}")

    rng = np.random.default_rng(20260828)
    out = dict(n_candidates=len(cands), funnel=funnel, qualified=[p["id"] for p in qualified],
               n_qualified=len(qualified), win911=win, allblock=allb,
               win911_ci=ci(win, rng), allblock_ci=ci(allb, rng))
    log(f"\nMATERIAL n={len(qualified)}  window 9-11 {out['win911_ci']}  all-block {out['allblock_ci']}")
    log("   (colour reference: window 0.30 [0.22,0.38], all-block 0.98)")
    json.dump(out, open("results/material_ext.json", "w"), indent=2)
    log(f"-> results/material_ext.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
