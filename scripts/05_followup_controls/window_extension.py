"""How far must the isolated injection window extend for binding to flip?

On the first 40 held-out pairs the text-only (hybrid_v2) stream flips 0.09 in the isolated
9-11 window but 0.675 when injected from block 9 onward (hyb2_prefix); the joint-state (std)
stream gives 0.30 vs 0.90. This script sweeps the isolated window 9..e for e in {11,13,15,17,23}
with both streams, paired on the same pairs.
Brackets: e=11 reproduces hybrid_v2 (std_win / hyb2_win) per pair (40/40 each). e=23 (isolated,
blocks 0-8 pinned to the clean cache) is NOT a per-pair reproduction of hyb2_prefix b=9:
results/window_extension.json records 33/40 (std) and 27/40 (hyb2) agreement, because pinning
blocks 0-8 to the clean cache is not a no-op once the injection has changed the latent at later
steps.
Env: WE_N (pairs, default 40), WE_OFFSET (start index into the block9_ext pair list, default 0),
WE_SUFFIX (output suffix). One GPU. Writes results/window_extension{WE_SUFFIX}.json (overwritten on
re-run) and appends to the matching .log.
"""
import torch, json, os, time, gc
import numpy as np
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader
from hybrid_v2 import ImagePins, gen, grade2, flip_vs

DEVICE = "cuda"; START = 9
ENDS = [11, 13, 15, 17, 23]
N = int(os.environ.get("WE_N", "40")); OFF = int(os.environ.get("WE_OFFSET", "0"))
SUF = os.environ.get("WE_SUFFIX", "")
LOG = open(f"results/window_extension{SUF}.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + \
           ([float((b > 0).mean())] if paired else [])


def main():
    t0 = time.time()
    v1 = json.load(open("results/hybrid_stream.json")); v2 = json.load(open("results/hybrid_v2.json"))
    pf = json.load(open("results/hyb2_prefix.json"))
    b9 = json.load(open("results/block9_ext.json"))
    qids = b9["qualified"][OFF:OFF + N]          # same order as block9_ext (v1/v2 are its first 40)
    ref = {"std_e11": dict(zip(b9["qualified"], b9["win911"])), "hyb2_e11": dict(zip(v2["qualified"], v2["hyb2_win"])),
           "std_e23": dict(zip(pf["qualified"], pf["std_b9"])), "hyb2_e23": dict(zip(pf["qualified"], pf["hyb2_b9"]))}
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== ISOLATED window {START}..e, e in {ENDS}: joint-state (std) vs text-only (hyb2), {len(qids)} pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer); pins = ImagePins(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    res = {f"{arm}_e{e}": [] for arm in ("std", "hyb2") for e in ENDS}
    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        lat = []
        hooks.begin_capture_clean(); pins.begin_capture(); _ = gen(pipe, p["clean"], seed, collect=lat); pins.off(); hooks.off()
        clean_stream = {b: list(v) for b, v in hooks.clean.items()}
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        std = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture(); pins.begin_inject(); _ = gen(pipe, p["swapped"], seed, latents_pin=lat); pins.off(); hooks.off()
        hyb2 = {b: list(v) for b, v in hooks.cache.items()}

        hooks.clean = clean_stream
        for arm, src in (("std", std), ("hyb2", hyb2)):
            hooks.cache = src
            for e in ENDS:
                hooks.begin_inject(set(range(START, e + 1)), isolate=True)
                k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
                res[f"{arm}_e{e}"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        hooks.cache = {}; hooks.clean = {}; pins.store = {}
        del std, hyb2, clean_stream, lat; gc.collect()
        br = " ".join(f"{k}:{res[k][-1] == ref[k][pid]}" for k in ref if pid in ref[k])
        log(f"pair{pid} [{n+1}/{len(qids)}] std " + " ".join(f"e{e}={res[f'std_e{e}'][-1]:.1f}" for e in ENDS)
            + " | hyb2 " + " ".join(f"e{e}={res[f'hyb2_e{e}'][-1]:.1f}" for e in ENDS)
            + f" | brackets {br} | running hyb2 " + " ".join(f"e{e}={np.mean(res[f'hyb2_e{e}']):.2f}" for e in ENDS))

    rng = np.random.default_rng(20260919)
    out = dict(qualified=qids, n=len(qids), start=START, ends=ENDS, **res)
    out["brackets"] = {k: [int(sum(a == ref[k][i] for i, a in zip(qids, res[k]) if i in ref[k])), int(sum(i in ref[k] for i in qids))] for k in ref}
    out["ci"] = {k: ci(v, rng) for k, v in res.items()}
    out["paired_hyb2_minus_std"] = {f"e{e}": ci(np.array(res[f"hyb2_e{e}"]) - np.array(res[f"std_e{e}"]), rng, True) for e in ENDS}
    out["paired_vs_e11"] = {f"{arm}_e{e}": ci(np.array(res[f"{arm}_e{e}"]) - np.array(res[f"{arm}_e11"]), rng, True)
                            for arm in ("std", "hyb2") for e in ENDS[1:]}
    log(f"\nbrackets (per-pair reproduction): {out['brackets']}")
    log("window 9..e :   std (joint-state)        hyb2 (text-only)        paired hyb2-std")
    for e in ENDS:
        s, h, d = out["ci"][f"std_e{e}"], out["ci"][f"hyb2_e{e}"], out["paired_hyb2_minus_std"][f"e{e}"]
        log(f"  e={e:2d} : {s[0]:.3f} [{s[1]:.2f},{s[2]:.2f}]   {h[0]:.3f} [{h[1]:.2f},{h[2]:.2f}]   {d[0]:+.3f} [{d[1]:+.2f},{d[2]:+.2f}] P(>0)={d[3]:.2f}")
    for k, v in out["paired_vs_e11"].items():
        log(f"  {k} - e11: {v[0]:+.3f} [{v[1]:+.2f},{v[2]:+.2f}] P(>0)={v[3]:.2f}")
    json.dump(out, open(f"results/window_extension{SUF}.json", "w"), indent=2)
    log(f"-> results/window_extension{SUF}.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
