"""Does the LATE text->image read window close ENTIRELY, or only for attribute binding?

The prefix-injection sweep shows that swapped (and placebo) text injected from block ~15
onward no longer changes binding. That establishes a horizon after which binding cannot be
REWRITTEN by this intervention -- not that the image stops reading the text stream at all.
Late injections could fail because the image trajectory is hard to overwrite, while late
text->image influence for OTHER information remains.

Direct test: run the identical prefix-injection machinery with three source streams and
measure how much the IMAGE changes (pixel L1, no grader needed):
  swap    : the attribute-swapped prompt          (binding edit -- known to die after ~15)
  placebo : a third-colour prompt, same objects   (binding-neutral colour edit)
  alien   : different colours AND different objects (large semantic edit)

Read:
  alien L1 stays large at b=15,18 while swap flip = 0  -> the read window is still OPEN; what
      closes is binding REWRITEABILITY, not text->image reading.
  alien L1 also collapses -> the read window itself closes.

Also grades the swap arm (flip) so "large L1, zero flip" can be shown on the SAME images, and
probes whether the alien objects actually appear.
One GPU.
"""
import torch, json, os, time, re, gc
import numpy as np
from prompts_ext import make_ext_pairs, OBJECTS
from prompts import COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
BLOCKS = [9, 12, 15, 18]
N = int(os.environ.get("LRW_N", "20"))
os.makedirs("results", exist_ok=True)
LOG = open("results/late_readwindow.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip_vs(qg, img, p):
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


def l1(a, b):
    return float(np.abs(np.asarray(a, np.float32) / 255.0 - np.asarray(b, np.float32) / 255.0).mean())


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def arms_for(p):
    """swap / placebo / alien source prompts, all on the same two-object template."""
    oc = [c for c in COLORS if c not in (p["c1"], p["c2"])]
    oo = [o for o in OBJECTS if o not in (p["o1"], p["o2"])]
    c3, c4 = (oc + COLORS)[:2]
    o3, o4 = oo[:2]
    return {
        "swap":    dict(prompt=p["swapped"]),
        "placebo": dict(prompt=f"a {c3} {p['o1']} next to a {c4} {p['o2']}"),
        "alien":   dict(prompt=f"a {c3} {o3} next to a {c4} {o4}", o3=o3, o4=o4),
    }


def ci(x, rng):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    t0 = time.time()
    b9 = json.load(open("results/block9_ext.json"))
    qids = b9["qualified"][:N]
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== LATE READ-WINDOW direct test: {len(qids)} pairs, prefix blocks {BLOCKS} =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    ARMS = ["swap", "placebo", "alien"]
    L1 = {a: {b: [] for b in BLOCKS} for a in ARMS}
    FLIP = {b: [] for b in BLOCKS}          # swap arm only
    ALIEN_HIT = {b: [] for b in BLOCKS}     # alien arm: do the alien objects appear?
    ceiling = []                            # L1(full alien image, clean image) = upper reference

    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        A = arms_for(p)
        img_clean = gen(pipe, p["clean"], seed)
        ceiling.append(l1(gen(pipe, A["alien"]["prompt"], seed), img_clean))

        for a in ARMS:
            hooks.begin_capture(); _ = gen(pipe, A[a]["prompt"], seed); hooks.off()
            for b in BLOCKS:
                hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
                img = gen(pipe, p["clean"], seed)
                hooks.off()
                L1[a][b].append(l1(img, img_clean))
                if a == "swap":
                    FLIP[b].append(flip_vs(qg, img, p))
                elif a == "alien":
                    ans = (qg._ask(img, f"Is there a {A['alien']['o3']} or a "
                                        f"{A['alien']['o4']} in this image? Answer yes or no.") or "").lower()
                    ALIEN_HIT[b].append(1.0 if "yes" in ans else 0.0)
            hooks.cache = {}
            gc.collect()

        log(f"pair{pid} [{n+1}/{len(qids)}] ceiling={ceiling[-1]:.4f} | " + " ".join(
            f"b{b}: sw={L1['swap'][b][-1]:.4f}/f{FLIP[b][-1]:.1f} pl={L1['placebo'][b][-1]:.4f} "
            f"al={L1['alien'][b][-1]:.4f}/h{ALIEN_HIT[b][-1]:.0f}" for b in BLOCKS))

    rng = np.random.default_rng(20260828)
    out = dict(qualified=qids, blocks=BLOCKS, n=len(qids),
               ceiling_l1=ci(ceiling, rng), ceiling_raw=ceiling,
               l1={a: {str(b): L1[a][b] for b in BLOCKS} for a in ARMS},
               l1_ci={a: {str(b): ci(L1[a][b], rng) for b in BLOCKS} for a in ARMS},
               swap_flip={str(b): FLIP[b] for b in BLOCKS},
               swap_flip_ci={str(b): ci(FLIP[b], rng) for b in BLOCKS},
               alien_hit={str(b): ALIEN_HIT[b] for b in BLOCKS},
               alien_hit_ci={str(b): ci(ALIEN_HIT[b], rng) for b in BLOCKS})
    # paired alien-vs-swap L1 at each block: is the channel still carrying non-binding info?
    for b in BLOCKS:
        d = np.array(L1["alien"][b], float) - np.array(L1["swap"][b], float)
        bd = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        out.setdefault("paired_alien_minus_swap_l1", {})[str(b)] = [
            float(d.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)),
            float((bd > 0).mean())]

    log(f"\nfull-alien-image L1 ceiling: {out['ceiling_l1']}")
    for b in BLOCKS:
        log(f"prefix b={b:2d} | swap L1 {out['l1_ci']['swap'][str(b)]} flip {out['swap_flip_ci'][str(b)]}")
        log(f"           | plac L1 {out['l1_ci']['placebo'][str(b)]}")
        log(f"           | ALIEN L1 {out['l1_ci']['alien'][str(b)]} objhit {out['alien_hit_ci'][str(b)]}"
            f"  paired alien-swap {out['paired_alien_minus_swap_l1'][str(b)]}")
    json.dump(out, open("results/late_readwindow.json", "w"), indent=2)
    log(f"-> results/late_readwindow.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
