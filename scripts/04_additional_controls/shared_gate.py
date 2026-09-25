"""A genuinely SHARED-GATE common-prompt comparison.

The main protocol qualified SD1.5 with a single-object gate (o1 only) because SD1.5 rarely
binds two objects, while PixArt/SD3.5 used the two-object gate. The 3-way intersection of
qualified pairs therefore fixes the *population* but not the *gate*.

The two-object gate strictly implies the single-object one, so any pair passing it must
already be in SD1.5's 40 qualified held-out pairs. We therefore only re-adjudicate those 40:
regenerate the clean and full-swap images at the identical seed/latent and grade BOTH objects,
then apply the two-object gate used for PixArt/SD3.5. Head-swap per-pair outcomes are reused
unchanged from results/top1_sd15_ext.json (generation is deterministic).

Outputs the shared-gate qualification count, the shared-gate 3-way intersection, and the
paired SD1.5-SD3.5 comparison on it.
One GPU.
"""
import torch, json, os, re, time
import numpy as np
from prompts_ext import make_ext_pairs
from prompts import make_pairs, COLORS
from binding_patch import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 30
os.makedirs("results", exist_ok=True)
LOG = open("results/shared_gate.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = (ans or "").lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * (4 if paired else 3)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    out = [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out + [float((b > 0).mean())] if paired else out


def binom_upper(n, alpha=0.05):
    return 1.0 - (alpha / 2.0) ** (1.0 / n) if n else float("nan")


def main():
    t0 = time.time()
    s15 = json.load(open("results/top1_sd15_ext.json"))
    px = json.load(open("results/top1_pixart_ext.json"))
    s3 = json.load(open("results/top1_sd3_ext.json"))
    sd15_ho = {i: v for i, v in zip(s15["ids"], s15["per_pair"]) if i >= 5000}
    px_ho = {i: v for i, v in zip(px["qualified_ids"], px["per_pair"]) if i >= 5000}
    s3_ho = {i: v for i, v in zip(s3["qualified"], s3["ext_per"]) if i >= 5000}
    P = {p["id"]: p for p in make_ext_pairs() + make_pairs()}

    ids = sorted(sd15_ho)
    log(f"\n===== SHARED-GATE re-adjudication of SD1.5: {len(ids)} held-out pairs =====")
    log("  (two-object gate implies the single-object one, so this set is a superset of the answer)")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    passed, detail = [], {}
    for n, pid in enumerate(ids):
        p = P[pid]
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        k1c, k2c = canon(qg._ask_color(ic, p["o1"])), canon(qg._ask_color(ic, p["o2"]))
        k1s, k2s = canon(qg._ask_color(ia, p["o1"])), canon(qg._ask_color(ia, p["o2"]))
        one = (k1c == p["c1"]) and (k1s == p["c2"])          # original single-object SD1.5 gate
        two = one and (k2c == p["c2"]) and (k2s == p["c1"])  # gate used for PixArt/SD3.5
        detail[pid] = dict(clean=[k1c, k2c], full_swap=[k1s, k2s],
                           truth=[p["c1"], p["c2"]], one_obj=bool(one), two_obj=bool(two))
        if two:
            passed.append(pid)
        if (n + 1) % 10 == 0:
            log(f"  {n+1}/{len(ids)} re-adjudicated, shared-gate passes {len(passed)}")

    log(f"\nSD1.5 shared (two-object) gate: {len(passed)}/{len(ids)} of the previously qualified "
        f"held-out pairs -> {len(passed)}/84 of all held-out candidates "
        f"(submitted single-object gate was {len(ids)}/84)")

    rng = np.random.default_rng(20260830)
    out = dict(reexamined=ids, shared_gate_pass=passed, n_shared=len(passed), detail=detail)

    inter = sorted(set(passed) & set(px_ho) & set(s3_ho))
    out["shared_gate_intersection"] = inter
    log(f"SHARED-GATE 3-way intersection: n={len(inter)} (population-only intersection was 14)")
    if inter:
        for nm, D in (("SD1.5", sd15_ho), ("PixArt", px_ho), ("SD3.5", s3_ho)):
            v = [D[i] for i in inter]
            c = ci(v, rng); out[f"{nm}_shared"] = c
            extra = f"  [0/{len(v)}, exact-binom UB {binom_upper(len(v)):.2f}]" if c[0] == 0 else ""
            log(f"  {nm:7s} {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]{extra}")
        d = np.array([sd15_ho[i] for i in inter]) - np.array([s3_ho[i] for i in inter])
        out["paired_SD15_minus_SD35_shared"] = ci(d, rng, paired=True)
        log(f"  PAIRED SD1.5-SD3.5 {out['paired_SD15_minus_SD35_shared'][0]:+.3f} "
            f"[{out['paired_SD15_minus_SD35_shared'][1]:+.3f},"
            f"{out['paired_SD15_minus_SD35_shared'][2]:+.3f}] "
            f"mass>0={out['paired_SD15_minus_SD35_shared'][3]:.3f}")
        dp = np.array([sd15_ho[i] for i in inter]) - np.array([px_ho[i] for i in inter])
        out["paired_SD15_minus_PixArt_shared"] = ci(dp, rng, paired=True)
        log(f"  PAIRED SD1.5-PixArt {out['paired_SD15_minus_PixArt_shared'][0]:+.3f} "
            f"[{out['paired_SD15_minus_PixArt_shared'][1]:+.3f},"
            f"{out['paired_SD15_minus_PixArt_shared'][2]:+.3f}]")
    json.dump(out, open("results/shared_gate.json", "w"), indent=2)
    log(f"-> results/shared_gate.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
