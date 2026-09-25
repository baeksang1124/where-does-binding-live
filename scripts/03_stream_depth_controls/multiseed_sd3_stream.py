"""Multi-seed reproduction of the central SD3.5 result: the all-block text-stream flip
(~0.95), checking that it is not a single-seed artifact. For each of K fresh generation
seeds (disjoint from the base p['seed']), we RE-QUALIFY each pair at that seed (clean
renders correct + swapped renders the swap) and then measure the matched all-block
text-stream injection flip on the newly-qualified set. Reports per-seed mean and the
pooled cross-seed mean+SD. MS_NPAIRS env sets the number of base pairs. One GPU.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
NPAIRS = int(os.environ.get("MS_NPAIRS", "15"))
SEED_OFFSETS = [10000, 20000, 30000]      # three fresh seeds per pair, disjoint from base 1000+i
LOG = open("results/multiseed_sd3_stream.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade2(qg, img, p):
    return canon(qg._ask_color(img, p["o1"])), canon(qg._ask_color(img, p["o2"]))


def flip2(qg, img, p):
    k1, k2 = grade2(qg, img, p); f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    coarse = json.load(open("results/sd3_coarse.json"))
    qids = coarse["qualified"][:NPAIRS]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== SD3.5 MULTI-SEED all-block text-stream flip: {len(qids)} base pairs x {len(SEED_OFFSETS)} seeds =====")

    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    per_seed = {}      # offset -> list of flips over pairs that requalified at that seed
    per_seed_nqual = {}
    for off in SEED_OFFSETS:
        flips = []
        for pid in qids:
            p = pairs[pid]; seed = p["seed"] + off
            # re-qualify at this fresh seed
            ck1, ck2 = grade2(qg, gen(pipe, p["clean"], seed), p)
            if not (ck1 == p["c1"] and ck2 == p["c2"]):
                continue
            sk1, sk2 = grade2(qg, gen(pipe, p["swapped"], seed), p)
            if not (sk1 == p["c2"] and sk2 == p["c1"]):
                continue
            # capture this pair's swapped stream at this seed, inject all-block on clean
            hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
            swap_cache = {b: list(v) for b, v in hooks.cache.items()}
            hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
            hooks.cache = swap_cache
            hooks.begin_inject(set(range(NBLOCKS)))
            f = flip2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            hooks.cache = {}; hooks.clean = {}
            flips.append(f)
        m = float(np.mean(flips)) if flips else 0.0
        per_seed[off] = flips; per_seed_nqual[off] = len(flips)
        log(f"  seed+{off}: n_qual={len(flips)}  all-block flip mean={m:.3f}")

    means = [np.mean(per_seed[o]) for o in SEED_OFFSETS if per_seed[o]]
    pooled = [f for o in SEED_OFFSETS for f in per_seed[o]]
    log(f"\nper-seed means = {[f'{x:.3f}' for x in means]}")
    log(f"cross-seed mean of means = {np.mean(means):.3f}  SD = {np.std(means):.3f}")
    log(f"pooled n={len(pooled)}  mean={np.mean(pooled):.3f}")
    json.dump(dict(seed_offsets=SEED_OFFSETS, base_pairs=qids,
                   per_seed_flips={str(o): per_seed[o] for o in SEED_OFFSETS},
                   per_seed_nqual=per_seed_nqual,
                   per_seed_mean={str(o): float(np.mean(per_seed[o])) if per_seed[o] else None for o in SEED_OFFSETS},
                   mean_of_means=float(np.mean(means)), sd_of_means=float(np.std(means)),
                   pooled_n=len(pooled), pooled_mean=float(np.mean(pooled))),
              open("results/multiseed_sd3_stream.json", "w"), indent=2)


if __name__ == "__main__":
    main()
