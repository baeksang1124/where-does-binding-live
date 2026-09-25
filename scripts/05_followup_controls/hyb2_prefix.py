"""Prefix-injection sweep following hybrid_v2: WHERE is text-only content read?

In hybrid_v2 the block 9-11 window flips 0.09 when the injected text stream has only ever
attended to clean image content (vs 0.30 for the standard joint-state stream), while the
same text-only stream injected at ALL blocks still flips 0.875. This script sweeps PREFIX
injection (blocks b..23) of the v2 text-only stream and, paired on the same pairs, of the
standard (joint-state) stream, b in {0,3,6,9,12,15}. b=0 reproduces hyb2_all / std_all.
Same 40 held-out pairs as hybrid_stream / hybrid_v2. One GPU.
Writes results/hyb2_prefix.{json,log}; never touches existing results.
"""
import torch, json, os, time, gc
import numpy as np
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader
from hybrid_v2 import ImagePins, gen, grade2, flip_vs, stream_maxdiff, NBLOCKS

DEVICE = "cuda"
BS = [int(x) for x in os.environ.get("HP_BS", "0,3,6,9,12,15").split(",")]
N = int(os.environ.get("HP_N", "40")); SUF = os.environ.get("HP_SUFFIX", "")
LOG = open(f"results/hyb2_prefix{SUF}.log", "a")


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
    qids = v2["qualified"][:N]
    v1_std_all = dict(zip(v1["qualified"], v1["std_all"])); v2_all = dict(zip(v2["qualified"], v2["hyb2_all"]))
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== PREFIX sweep b in {BS}: text-only (v2) vs joint-state (std) stream, {len(qids)} pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer); pins = ImagePins(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    res = {f"{arm}_b{b}": [] for arm in ("std", "hyb2") for b in BS}
    pins_applied, b9diff = [], []
    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        lat = []
        hooks.begin_capture_clean(); pins.begin_capture(); _ = gen(pipe, p["clean"], seed, collect=lat); pins.off(); hooks.off()
        clean_stream = {b: list(v) for b, v in hooks.clean.items()}
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        std = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture(); pins.begin_inject(); _ = gen(pipe, p["swapped"], seed, latents_pin=lat); pins.off(); hooks.off()
        hyb2 = {b: list(v) for b, v in hooks.cache.items()}
        pins_applied.append(pins.applied); b9diff.append(stream_maxdiff(hyb2, std, [9]))

        hooks.clean = clean_stream
        for arm, src in (("std", std), ("hyb2", hyb2)):
            hooks.cache = src
            for b in BS:
                hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
                k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
                res[f"{arm}_b{b}"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        hooks.cache = {}; hooks.clean = {}; pins.store = {}
        del std, hyb2, clean_stream, lat; gc.collect()
        log(f"pair{pid} [{n+1}/{len(qids)}] std " + " ".join(f"b{b}={res[f'std_b{b}'][-1]:.1f}" for b in BS)
            + " | hyb2 " + " ".join(f"b{b}={res[f'hyb2_b{b}'][-1]:.1f}" for b in BS)
            + (f" | b0 repro std {res['std_b0'][-1] == v1_std_all[pid]} hyb2 {res['hyb2_b0'][-1] == v2_all[pid]}" if 0 in BS else "")
            + " | running hyb2 " + " ".join(f"b{b}={np.mean(res[f'hyb2_b{b}']):.2f}" for b in BS))

    rng = np.random.default_rng(20260919)
    out = dict(qualified=qids, n=len(qids), bs=BS, **res, pins_applied=sorted(set(pins_applied)),
               b9_hyb2_vs_std_min=float(min(b9diff)),
               std_b0_reproduces_v1=[int(sum(a == v1_std_all[i] for i, a in zip(qids, res["std_b0"]))), len(qids)] if 0 in BS else None,
               hyb2_b0_reproduces_v2=[int(sum(a == v2_all[i] for i, a in zip(qids, res["hyb2_b0"]))), len(qids)] if 0 in BS else None)
    out["ci"] = {k: ci(v, rng) for k, v in res.items()}
    out["paired_hyb2_minus_std"] = {f"b{b}": ci(np.array(res[f"hyb2_b{b}"]) - np.array(res[f"std_b{b}"]), rng, True) for b in BS}
    log(f"\npins/capture {out['pins_applied']} (expect {NBLOCKS*28}); std b0 reproduces v1 all {out['std_b0_reproduces_v1']}; "
        f"hyb2 b0 reproduces v2 all {out['hyb2_b0_reproduces_v2']}")
    log("prefix from b :   std (joint-state)        hyb2 (text-only)        paired hyb2-std")
    for b in BS:
        s, h, d = out["ci"][f"std_b{b}"], out["ci"][f"hyb2_b{b}"], out["paired_hyb2_minus_std"][f"b{b}"]
        log(f"  b={b:2d} : {s[0]:.3f} [{s[1]:.2f},{s[2]:.2f}]   {h[0]:.3f} [{h[1]:.2f},{h[2]:.2f}]   {d[0]:+.3f} [{d[1]:+.2f},{d[2]:+.2f}] P(>0)={d[3]:.2f}")
    json.dump(out, open(f"results/hyb2_prefix{SUF}.json", "w"), indent=2)
    log(f"-> results/hyb2_prefix{SUF}.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
