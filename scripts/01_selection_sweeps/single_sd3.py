"""SD3.5 single-object-instance flip re-grade, for apples-to-apples with SD1.5/PixArt.

Reuses the all-576 head ranking in results/sd3_headsweep.json. Per qualified pair:
capture swapped stream, then cumulative top-k head swap (K&V), graded with the per-OBJECT
flip metric (fraction of the 2 objects reading their swapped color). Gives SD3.5's top-1
and top-k single-object curve to place beside PixArt (0.27) and SD1.5 (0.33).
One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0
KS = [1, 2, 4, 8, 16, 32, 64, 128]
LOG = open("results/sd3_single.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def obj_flips(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    return int(k1 == p["c2"] and k1 != p["c1"]), int(k2 == p["c1"] and k2 != p["c2"])


def parse(s):
    b, h = s.split("|h"); return (int(b), int(h))


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    hs = json.load(open("results/sd3_headsweep.json"))
    ranked = [parse(s) for s in hs["ranked"]]
    coarse = json.load(open("results/sd3_coarse.json"))
    sub = int(os.environ.get("SD3_SUBSET", "12"))
    qids = coarse["qualified"][:sub]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== SD3.5 SINGLE-OBJECT re-grade: {len(qids)} pairs, top1={ranked[0]} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    cum = {k: [] for k in KS}
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        for k in KS:
            img = generate(pipe, ctrl, p["clean"], seed, cells=ranked[:k], swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            f1, f2 = obj_flips(qg, img, p)
            cum[k] += [f1, f2]
        ctrl.swap_ctx = {}
        log(f"pair{pid:2d}: " + ", ".join(f"k{k}={np.mean(cum[k][-2:]):.1f}" for k in KS))

    cum_mean = {k: float(np.mean(cum[k])) for k in KS}
    log(f"\n--- SD3.5 mean single-object-instance flip vs top-k HEADS ---")
    for k in KS:
        log(f"  top-{k:3d}: {cum_mean[k]:.3f} ({cum_mean[k]*100:.0f}%)")
    json.dump(dict(metric="single_object_instance_flip", qualified=qids, ks=KS, cum_mean=cum_mean),
              open("results/sd3_single.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_single.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
