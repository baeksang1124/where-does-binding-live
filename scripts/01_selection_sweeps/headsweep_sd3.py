"""SD3.5 PER-HEAD localization (the SD1.5-style decisive test).

Block-level greedy was ambiguous because a whole-block swap can hide head-level
sparsity (within-block +/- head cancellation). This sweeps INDIVIDUAL heads.

Candidate heads = all 24 heads of the greedy-implicated blocks {9,5,11,2,4,13}
(the blocks that actually raised flip in greedy; block 0/1 were tie-break noise).

Pass 1 (rank): per pair, capture swapped (cache); for each candidate (block,head)
  swap that single head K&V on the clean run -> two-object flip. Mean over pairs
  -> per-head mean single-flip. Rank.
Pass 2 (cumulative): re-capture per pair; swap the global top-k heads together for
  k in KS -> cumulative flip curve. LOCALIZED iff small k recovers >=70% of swap_all.

Env: SD3_SUBSET (pairs, default 12), SD3_BLOCKS (comma list, default 9,5,11,2,4,13), SD3_KS (cumulative
k list), SD3_PIDS (explicit pair ids; overrides SD3_SUBSET). The shipped 576-head results/sd3_headsweep.json used
SD3_BLOCKS=0,...,23 SD3_PIDS=0,2,3,8,10,11 SD3_KS=1,2,4,8,16,32,64,128 (see SCRIPTS.md). Every run overwrites
results/sd3_headsweep.json, whose ranking other scripts read. One GPU.
"""
import torch, json, time, os, numpy as np, re
from collections import defaultdict
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0
SUBSET = int(os.environ.get("SD3_SUBSET", "12"))
BLOCKS = [int(x) for x in os.environ.get("SD3_BLOCKS", "9,5,11,2,4,13").split(",")]
KS = [int(x) for x in os.environ.get("SD3_KS", "1,2,3,4,6,8,12,16,24,32").split(",")]
LOG = open("results/sd3_headsweep.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip_score(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2); f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    coarse = json.load(open("results/sd3_coarse.json"))
    pids_env = os.environ.get("SD3_PIDS", "")
    if pids_env:
        qids = [int(x) for x in pids_env.split(",")]
    else:
        qids = coarse["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    cells = [(b, h) for b in BLOCKS for h in range(24)]
    log(f"\n===== SD3.5 PER-HEAD sweep: {len(qids)} pairs, blocks={BLOCKS} ({len(cells)} heads) =====")
    log(f"  qualified subset: {qids}")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    # ---- PASS 1: per-head single-swap flip ----
    head_flips = defaultdict(list)   # (b,h) -> [flip per pair]
    refs = []
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        ref = flip_score(qg, generate(pipe, ctrl, p["clean"], seed, swap_all=True,
                         swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                         height=H, width=W, device=DEVICE), p)
        refs.append(ref)
        for (b, h) in cells:
            img = generate(pipe, ctrl, p["clean"], seed, cells=[(b, h)], swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            head_flips[(b, h)].append(flip_score(qg, img, p))
        ctrl.swap_ctx = {}
        means = {c: float(np.mean(head_flips[c])) for c in cells}
        top = sorted(means, key=means.get, reverse=True)[:6]
        log(f"pair{pid:2d} ref={ref:.2f} running top heads: "
            + ", ".join(f"b{b}h{h}={means[(b,h)]:.2f}" for (b, h) in top))

    head_mean = {c: float(np.mean(head_flips[c])) for c in cells}
    ranked = sorted(cells, key=lambda c: head_mean[c], reverse=True)
    ref_mean = float(np.mean(refs))
    log(f"\n--- per-head mean single-flip (top 20 of {len(cells)}) ; swap_all ref={ref_mean:.3f} ---")
    for c in ranked[:20]:
        log(f"  b{c[0]:2d} h{c[1]:2d}: {head_mean[c]:.3f}")
    nonzero = [c for c in cells if head_mean[c] > 0.05]
    log(f"  heads with mean single-flip > 0.05: {len(nonzero)} / {len(cells)}")

    # ---- PASS 2: cumulative top-k heads (global ranking) ----
    log(f"\n--- PASS 2: cumulative top-k heads ; KS={KS} ---")
    cum = {k: [] for k in KS}
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        for k in KS:
            topk = ranked[:k]
            img = generate(pipe, ctrl, p["clean"], seed, cells=topk, swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            cum[k].append(flip_score(qg, img, p))
        ctrl.swap_ctx = {}
        log(f"pair{pid:2d} cumulative: " + ", ".join(f"k{k}={cum[k][-1]:.2f}" for k in KS))

    cum_mean = {k: float(np.mean(cum[k])) for k in KS}
    log(f"\n--- mean cumulative flip vs top-k HEADS (ref={ref_mean:.3f}) ---")
    for k in KS:
        frac = cum_mean[k] / ref_mean if ref_mean > 0 else 0
        log(f"  top-{k:2d} heads: flip={cum_mean[k]:.3f} ({frac*100:.0f}% of swap_all)")
    k70 = next((k for k in KS if cum_mean[k] >= 0.70 * ref_mean), None)
    verdict = (f"HEAD-LOCALIZED (GO): top-{k70} heads reach >=70% of swap_all"
               if (k70 is not None and k70 <= 16)
               else f"NOT head-localized: needs >{KS[-1] if k70 is None else k70} heads for 70%")
    log(f"\n  VERDICT: {verdict}")

    json.dump(dict(blocks=BLOCKS, subset=qids, ref_mean=ref_mean,
                   head_mean={f"{b}|h{h}": head_mean[(b, h)] for (b, h) in cells},
                   ranked=[f"{b}|h{h}" for (b, h) in ranked], n_nonzero=len(nonzero),
                   ks=KS, cum_mean=cum_mean, verdict=verdict),
              open("results/sd3_headsweep.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_headsweep.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
