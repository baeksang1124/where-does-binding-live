"""SD3.5 DECISIVE localization test: block-level GREEDY cumulative recovery curve.

The coarse sweep swaps each block alone; if binding is super-additive across blocks
(single block ~0), that curve under-reads localization. This builds the cumulative
curve directly: greedily add the block that most increases the swapped-binding flip,
until we reach ~the swap_all reference. The #blocks needed to hit >=70% recovery is
the localization verdict.

  LOCALIZED (GO)   : few blocks (<=~4 of 24) recover >=70% -> sparse handle exists.
  DISTRIBUTED (NO) : needs many blocks (flat curve) -> no sparse block-level handle.

Per pair: capture swapped (cache) ; greedy over blocks (each step tries every
remaining block appended to the current set, all 24 heads, K&V) ; record order +
cumulative flip. Aggregate mean cumulative flip vs #blocks across the subset.

Env: SD3_SUBSET (n qualified pairs to use, default 12), SD3_MAXK (max blocks, default 6).
One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "12"))
MAXK = int(os.environ.get("SD3_MAXK", "6"))
LOG = open("results/sd3_greedy.log", "a")


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


def cells_for(blocks, ctrl):
    out = []
    for b in blocks:
        out += [(b, h) for h in range(ctrl.layer_heads[b])]
    return out


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    coarse = json.load(open("results/sd3_coarse.json"))
    qids = coarse["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== SD3.5 GREEDY cumulative localization: {len(qids)} pairs, MAXK={MAXK} =====")
    log(f"  qualified subset ids: {qids}")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    per_pair = []
    cum_curve = {k: [] for k in range(MAXK + 1)}   # k=0 -> clean baseline (0)
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        # swap_all reference for this pair
        ref = flip_score(qg, generate(pipe, ctrl, p["clean"], seed, swap_all=True,
                         swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                         height=H, width=W, device=DEVICE), p)
        chosen, curve = [], [0.0]
        cum_curve[0].append(0.0)
        cur = 0.0
        for k in range(1, MAXK + 1):
            best_b, best_f = None, -1.0
            for b in range(NBLOCKS):
                if b in chosen:
                    continue
                img = generate(pipe, ctrl, p["clean"], seed, cells=cells_for(chosen + [b], ctrl),
                               swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                               height=H, width=W, device=DEVICE)
                f = flip_score(qg, img, p)
                if f > best_f:
                    best_f, best_b = f, b
            chosen.append(best_b); cur = best_f; curve.append(cur)
            cum_curve[k].append(cur)
            if cur >= 0.99:
                # fill remaining k with saturated value for aggregation
                for kk in range(k + 1, MAXK + 1):
                    cum_curve[kk].append(cur)
                break
        per_pair.append({"id": pid, "ref_flip": ref, "order": chosen, "curve": curve})
        log(f"pair{pid:2d} ref={ref:.2f} greedy_order={chosen} curve={[round(x,2) for x in curve]}")

    log("\n--- mean cumulative flip vs #blocks ---")
    mean_curve = {k: float(np.mean(v)) if v else 0.0 for k, v in cum_curve.items()}
    ref_mean = float(np.mean([pp["ref_flip"] for pp in per_pair]))
    for k in range(MAXK + 1):
        frac = mean_curve[k] / ref_mean if ref_mean > 0 else 0.0
        log(f"  {k} blocks: flip={mean_curve[k]:.3f}  (={frac*100:.0f}% of swap_all {ref_mean:.3f})")
    # how many blocks to reach 70% of ref
    k70 = next((k for k in range(MAXK + 1) if mean_curve[k] >= 0.70 * ref_mean), None)
    verdict = (f"LOCALIZED (GO): {k70} blocks reach >=70% of swap_all" if (k70 is not None and k70 <= 4)
               else f"DISTRIBUTED-ish: needs >{MAXK if k70 is None else k70} blocks for 70% -> weak handle")
    log(f"\n  VERDICT: {verdict}")
    # most frequent first-choice blocks
    firsts = [pp["order"][0] for pp in per_pair if pp["order"]]
    from collections import Counter
    log(f"  greedy 1st-block frequency: {Counter(firsts).most_common()}")

    json.dump(dict(subset=qids, maxk=MAXK, ref_mean=ref_mean, mean_curve=mean_curve,
                   per_pair=per_pair, verdict=verdict,
                   first_block_freq=Counter(firsts).most_common()),
              open("results/sd3_greedy.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_greedy.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
