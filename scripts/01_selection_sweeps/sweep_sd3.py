"""SD3.5 coarse localization sweep (block level) + pair qualification, in one pass.

Per pair:
  capture swapped (S, fills cache) ; vanilla clean (A).
  QUALIFY: A binds clean (o1=c1,o2=c2 distinct) AND S binds swapped (o1=c2,o2=c1 distinct).
  If qualified, using the SAME cache:
    b1 = swap_all K&V (full-effect reference, expect proper~1)
    for each block 0..23: swap ALL 24 heads in that block (K&V) -> two-object grade.
  Clear cache.

Metric per image (two-object joint grade, Qwen-VL):
  flip = fraction of the two objects now reading their SWAPPED color (0/.5/1)  [smooth, for ranking]
  proper = 1 iff o1 reads c2 AND o2 reads c1 (genuine distinct rebind)         [strict]

Output: results/sd3_coarse.json (qualified pairs, per-block mean flip/proper, ranked blocks).
One GPU (select it with CUDA_VISIBLE_DEVICES).
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"
STEPS = 28
H = W = 512
GUID = 7.0
NBLOCKS = 24
LOG = open("results/sd3_coarse.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower()
    hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade(qg, img, p):
    """Return (flip_frac, proper, a1, a2). clean: o1=c1,o2=c2 ; swap target: o1=c2,o2=c1."""
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    sw1, sw2 = p["c2"], p["c1"]
    flip = 0.0
    if k1 == sw1 and k1 != p["c1"]:
        flip += 0.5
    if k2 == sw2 and k2 != p["c2"]:
        flip += 0.5
    proper = int(k1 == sw1 and k2 == sw2)
    return flip, proper, a1, a2


def is_clean_bind(p, a1, a2):
    return canon(a1) == p["c1"] and canon(a2) == p["c2"]


def is_swap_bind(p, a1, a2):
    return canon(a1) == p["c2"] and canon(a2) == p["c1"]


def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    pairs = make_pairs()
    nlim = int(os.environ.get("SD3_NPAIRS", "0"))
    if nlim:
        pairs = pairs[:nlim]
    log(f"\n===== SD3.5 coarse sweep: {len(pairs)} candidate pairs, {STEPS} steps, {H}px =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    qualified = []
    block_flip = {b: [] for b in range(NBLOCKS)}    # per qualified pair flip score
    block_proper = {b: [] for b in range(NBLOCKS)}
    full_flip, full_proper = [], []                 # swap_all reference
    per_pair = []

    for p in pairs:
        seed = p["seed"]
        S = capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                            height=H, width=W, device=DEVICE)
        A = generate(pipe, ctrl, p["clean"], seed, cells=None, steps=STEPS, guidance=GUID,
                     height=H, width=W, device=DEVICE)
        _, _, sa1, sa2 = grade(qg, S, p)
        _, _, ca1, ca2 = grade(qg, A, p)
        ok = is_clean_bind(p, ca1, ca2) and is_swap_bind(p, sa1, sa2)
        log(f"pair{p['id']:2d} clean=({ca1},{ca2}) swap=({sa1},{sa2}) -> "
            f"{'QUALIFY' if ok else 'skip'} | {p['clean']}")
        if not ok:
            ctrl.swap_ctx = {}      # free cache
            continue
        qualified.append(p["id"])

        bf = generate(pipe, ctrl, p["clean"], seed, swap_all=True, swap_k=True, swap_v=True,
                      steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        ff, fp, _, _ = grade(qg, bf, p)
        full_flip.append(ff); full_proper.append(fp)

        rec = {"id": p["id"], "full_flip": ff, "full_proper": fp, "blocks": {}}
        for b in range(NBLOCKS):
            cells = [(b, h) for h in range(ctrl.layer_heads[b])]
            img = generate(pipe, ctrl, p["clean"], seed, cells=cells, swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            f, pr, _, _ = grade(qg, img, p)
            block_flip[b].append(f); block_proper[b].append(pr)
            rec["blocks"][b] = {"flip": f, "proper": pr}
        per_pair.append(rec)
        # incremental top-5 by flip so far
        means = {b: float(np.mean(block_flip[b])) for b in range(NBLOCKS)}
        top = sorted(means, key=means.get, reverse=True)[:5]
        log(f"  full_flip={ff:.2f} | running top-5 blocks by flip: "
            + ", ".join(f"b{b}={means[b]:.2f}" for b in top))
        ctrl.swap_ctx = {}

    nq = len(qualified)
    log(f"\n===== qualified {nq}/{len(pairs)} ; swap_all ref flip={np.mean(full_flip):.3f} "
        f"proper={np.mean(full_proper):.3f} =====")
    block_mean_flip = {b: float(np.mean(block_flip[b])) if block_flip[b] else 0.0 for b in range(NBLOCKS)}
    block_mean_proper = {b: float(np.mean(block_proper[b])) if block_proper[b] else 0.0 for b in range(NBLOCKS)}
    ranked = sorted(range(NBLOCKS), key=lambda b: block_mean_flip[b], reverse=True)
    log("\n--- blocks ranked by mean flip (recovery of swapped binding) ---")
    for b in ranked:
        log(f"  block {b:2d}: flip={block_mean_flip[b]:.3f}  proper={block_mean_proper[b]:.3f}")

    out = dict(n_qualified=nq, qualified=qualified, steps=STEPS, px=H,
               swap_all_flip=float(np.mean(full_flip)), swap_all_proper=float(np.mean(full_proper)),
               block_mean_flip=block_mean_flip, block_mean_proper=block_mean_proper,
               ranked_blocks=ranked, per_pair=per_pair)
    json.dump(out, open("results/sd3_coarse.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_coarse.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
