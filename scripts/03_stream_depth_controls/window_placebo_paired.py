"""Paired swap-vs-placebo at the block 9-11 text window, on the SAME 75 held-out pairs used
by block9_ext.py (which already stored the matched-swap window flip per pair). Here we
measure the third-color PLACEBO window flip on those identical pairs/seeds, so we can report
the swap-specific EXCESS = swap - placebo as a PAIRED bootstrap CI.
One GPU.
"""
import torch, json, os, numpy as np, re
from prompts_ext import make_ext_pairs
from prompts import COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
WIN = {9, 10, 11}
LOG = open("results/window_placebo_paired.log", "a")


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
    b9 = json.load(open("results/block9_ext.json"))
    qids = b9["qualified"]; swap_win = b9["win911"]          # same order
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== window {sorted(WIN)} PLACEBO on same {len(qids)} held-out pairs =====")
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    plac_win, plac_pot = [], []
    for pid in qids:
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
        plac_win.append(flip_vs(k1, k2, p["c1"], p["c2"]))
        plac_pot.append((int(k1 == c3) + int(k2 == c4)) / 2.0)
        hooks.cache = {}; hooks.clean = {}
        if len(plac_win) % 15 == 0:
            log(f"  n={len(plac_win)} placebo-win running {np.mean(plac_win):.3f}")

    rng = np.random.default_rng(1234)
    sw = np.array(swap_win, float); pl = np.array(plac_win, float)

    def ci(x):
        b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return (float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))

    d = sw - pl
    bd = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
    log(f"\nwindow SWAP    n={len(sw)}: {ci(sw)}")
    log(f"window PLACEBO n={len(pl)}: {ci(pl)}   (potency {ci(np.array(plac_pot,float))})")
    log(f"PAIRED swap-specific EXCESS = swap - placebo: {d.mean():+.3f} "
        f"[{np.percentile(bd,2.5):+.3f},{np.percentile(bd,97.5):+.3f}] mass>0={float((bd>0).mean()):.3f}")
    json.dump(dict(qualified=qids, swap_win=swap_win, placebo_win=plac_win, placebo_potency=plac_pot,
                   swap_ci=ci(sw), placebo_ci=ci(pl),
                   paired_excess=[float(d.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)),
                                  float((bd > 0).mean())]),
              open("results/window_placebo_paired.json", "w"), indent=2)
    log("-> results/window_placebo_paired.json")


if __name__ == "__main__":
    main()
