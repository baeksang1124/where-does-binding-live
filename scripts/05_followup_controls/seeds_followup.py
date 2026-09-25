"""Seed robustness of the window headline numbers, with per-seed pair IDs preserved.
Same protocol as window_placebo_seeds.py: for each of the 40 held-out
pairs x 3 fresh seed offsets {10000,20000,30000}, re-qualify (clean binds AND full swap binds
under the two-object gate), then measure isolated windows 9-11 and 9-15 with the joint-state
(std) and text-only (hyb2) streams. Base-seed references: std e11 .300 / hyb2 e11 .0875 /
std e15 .625 / hyb2 e15 .5625 (results/window_extension.json).
One GPU. Writes results/seeds_followup.json (overwritten on re-run) and appends to results/seeds_followup.log.
"""
import torch, json, os, time, gc
import numpy as np
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader
from hybrid_v2 import ImagePins, gen, grade2, flip_vs

DEVICE = "cuda"
SEED_OFFSETS = [10000, 20000, 30000]
WINDOWS = {"e11": set(range(9, 12)), "e15": set(range(9, 16))}
N = int(os.environ.get("SF_N", "40"))
LOG = open("results/seeds_followup.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    t0 = time.time()
    qids = json.load(open("results/hybrid_v2.json"))["qualified"][:N]
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== SEED robustness: {len(qids)} pairs x offsets {SEED_OFFSETS}, windows {sorted(WINDOWS)} =====")
    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer); pins = ImagePins(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    res = {off: {"qualified": [], "failed": [], **{f"{a}_{w}": [] for a in ("std", "hyb2") for w in WINDOWS}} for off in SEED_OFFSETS}
    for off in SEED_OFFSETS:
        R = res[off]
        for n, pid in enumerate(qids):
            p = P[pid]; seed = p["seed"] + off
            ck = grade2(qg, gen(pipe, p["clean"], seed), p)
            sk = grade2(qg, gen(pipe, p["swapped"], seed), p)
            if not (ck == (p["c1"], p["c2"]) and sk == (p["c2"], p["c1"])):
                R["failed"].append(dict(id=pid, seed=seed, clean=list(ck), swap=list(sk))); continue
            R["qualified"].append(pid)
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
                for w, blocks in WINDOWS.items():
                    hooks.begin_inject(set(blocks), isolate=True)
                    k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
                    R[f"{arm}_{w}"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
            hooks.cache = {}; hooks.clean = {}; pins.store = {}
            del std, hyb2, clean_stream, lat; gc.collect()
            if len(R["qualified"]) % 5 == 0:
                log(f"  seed+{off}: screened {n+1}/{len(qids)}, qualified {len(R['qualified'])} | running "
                    + " ".join(f"{k}={np.mean(R[k]):.3f}" for k in R if k.startswith(("std", "hyb2"))))
        log(f"seed+{off}: qualified {len(R['qualified'])}/{len(qids)} | " + " ".join(f"{k}={np.mean(R[k]) if R[k] else float('nan'):.3f}" for k in R if k.startswith(("std", "hyb2"))))

    rng = np.random.default_rng(20260919)
    keys = [f"{a}_{w}" for a in ("std", "hyb2") for w in WINDOWS]
    base = json.load(open("results/window_extension.json"))
    base_means = {k: float(np.mean(base[k])) for k in keys}
    out = dict(qualified_ids=qids, seed_offsets=SEED_OFFSETS, base_means=base_means,
               seeds={str(o): res[o] for o in SEED_OFFSETS},
               per_seed_ci={str(o): {k: ci(res[o][k], rng) for k in keys} for o in SEED_OFFSETS})
    out["per_seed_means"] = {k: [float(np.mean(res[o][k])) for o in SEED_OFFSETS] for k in keys}
    out["mean_of_means_sampleSD"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in out["per_seed_means"].items()}
    out["pooled_ci"] = {k: ci(sum((res[o][k] for o in SEED_OFFSETS), []), rng) for k in keys}
    log(f"\nqualified per seed: {[len(res[o]['qualified']) for o in SEED_OFFSETS]} of {len(qids)}")
    log("metric     base(seed0)   per-seed means            mean±sampleSD     pooled CI")
    for k in keys:
        m = out["per_seed_means"][k]; mm = out["mean_of_means_sampleSD"][k]; pc = out["pooled_ci"][k]
        log(f"  {k:9s}  {base_means[k]:.3f}        {['%.3f' % x for x in m]}   {mm[0]:.3f}±{mm[1]:.3f}     {pc[0]:.3f} [{pc[1]:.2f},{pc[2]:.2f}]")
    json.dump(out, open("results/seeds_followup.json", "w"), indent=2)
    log(f"-> results/seeds_followup.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
