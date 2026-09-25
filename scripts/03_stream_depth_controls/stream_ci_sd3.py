"""SD3.5 stream experiment with PER-PAIR storage (for bootstrap CIs) + a PLACEBO control.

Adds uncertainty quantification to the block-depth curves (per-pair flips for bootstrap CIs)
and a placebo control: is the matched all-block flip binding-specific, or does injecting *any*
text stream disrupt binding?

Per qualified pair (reuse coarse-qualified set):
  - capture swapped + clean raw block-input streams
  - ISOLATED single-block injection over all 24 blocks (clean pinned elsewhere) -> per-pair iso[b]
  - PREFIX injection from block b onward -> per-pair prefix[b]
  - PLACEBO: inject at ALL blocks the stream of a third-color prompt (same objects/template,
    colors c3/c4 disjoint from c1/c2; "mismatch" in variable/log names) -> should NOT flip
    THIS pair's binding; contrasts with the matched all-block flip.
Stores full per-pair arrays so CIs are computed offline (bootstrap_ci.py).
The paper's in-sample depth curves use SD3_SUBSET=30; the default (10) reproduces only the pilot
and overwrites results/sd3_stream_ci.json. One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "10"))
LOG = open("results/sd3_stream_ci.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip2(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2); f = 0.0
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
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    coarse = json.load(open("results/sd3_coarse.json"))
    qids = coarse["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    PREFIX_B = [0, 3, 6, 9, 12, 15, 18, 21]
    log(f"\n===== SD3.5 CI+placebo: {len(qids)} pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    iso_pp = {b: [] for b in range(NBLOCKS)}        # per-pair isolated single-block
    prefix_pp = {b: [] for b in PREFIX_B}
    allmatch, placebo = [], []

    for i, pid in enumerate(qids):
        p = pairs[pid]; seed = p["seed"]
        # PLACEBO prompt: SAME objects, SAME template, but a disjoint THIRD color pair
        # (not the c1<->c2 swap). Injecting it is a non-trivial text edit that is NOT the
        # binding swap; the two objects stay gradeable. Expect flip ~0 on this pair's metric.
        others = [c for c in COLORS if c not in (p["c1"], p["c2"])]
        c3, c4 = (others + COLORS)[:2]
        placebo_prompt = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        swap_cache = {b: list(v) for b, v in hooks.cache.items()}      # THIS pair's swapped stream
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        # capture the PLACEBO (third-color) stream, same seed
        hooks.begin_capture(); _ = gen(pipe, placebo_prompt, seed); hooks.off()
        mismatch_cache = {b: list(v) for b, v in hooks.cache.items()}
        hooks.cache = swap_cache                                        # restore matched swapped

        # matched all-block (reference)
        hooks.begin_inject(set(range(NBLOCKS))); allmatch.append(flip2(qg, gen(pipe, p["clean"], seed), p)); hooks.off()
        # placebo: inject the third-color stream at all blocks
        hooks.cache = mismatch_cache
        hooks.begin_inject(set(range(NBLOCKS))); placebo.append(flip2(qg, gen(pipe, p["clean"], seed), p)); hooks.off()
        hooks.cache = swap_cache
        # isolated single-block
        for b in range(NBLOCKS):
            hooks.begin_inject({b}, isolate=True); iso_pp[b].append(flip2(qg, gen(pipe, p["clean"], seed), p)); hooks.off()
        # prefix
        for b in PREFIX_B:
            hooks.begin_inject(set(range(b, NBLOCKS))); prefix_pp[b].append(flip2(qg, gen(pipe, p["clean"], seed), p)); hooks.off()
        hooks.cache = {}; hooks.clean = {}
        log(f"pair{pid:2d} allmatch={allmatch[-1]:.2f} placebo={placebo[-1]:.2f} | "
            f"iso b9={np.mean(iso_pp[9]):.2f} b11={np.mean(iso_pp[11]):.2f}")

    out = dict(qualified=qids,
               allmatch=allmatch, placebo=placebo,
               iso_per_pair={b: iso_pp[b] for b in range(NBLOCKS)},
               prefix_per_pair={b: prefix_pp[b] for b in PREFIX_B})
    json.dump(out, open("results/sd3_stream_ci.json", "w"), indent=2)
    log(f"\nall-block MATCHED flip mean={np.mean(allmatch):.3f} | PLACEBO (mismatched) mean={np.mean(placebo):.3f}")
    log(f"iso block9 mean={np.mean(iso_pp[9]):.3f} block11={np.mean(iso_pp[11]):.3f}")
    log(f"results -> results/sd3_stream_ci.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
