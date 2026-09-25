"""SD3.5 A4 -- OV/QK (content vs routing) decomposition on the localized top heads.

Reads the 'ranked' head list in results/sd3_headsweep.json (no localization check is made;
the result JSON is not shipped). For top1 and top-K head sets, splits the FULL text-source swap
into its routing (K/PATTERN) and content (V/VALUE) parts:

  FULL     swap_k=T swap_v=T   reference (reproduces localization recovery)
  PATTERN  swap_k=T swap_v=F   swapped ROUTING (QK), clean content
  VALUE    swap_k=F swap_v=T   clean routing, swapped CONTENT (OV)
  BASELINE swap_k=F swap_v=F   no-op (~0)

Two-object metric (SD3.5 binds cleanly, unlike SD1.5): flip = frac of the two
objects reading their swapped color (0/.5/1) ; proper = both distinct rebound.
Bootstrap 95% CI on [PATTERN - VALUE] (difference of means over pairs).

Verdict: PATTERN~FULL & VALUE~base -> ROUTING(QK) ; VALUE~FULL & PATTERN~base ->
CONTENT(OV) ; both partial -> MIXED.
Env: SD3_SUBSET (pairs, default = all qualified), SD3_TOPK (default 16). One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0
TOPK = int(os.environ.get("SD3_TOPK", "16"))
NBOOT = 10000; BOOT_SEED = 1234
LOG = open("results/sd3_a4.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2); f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f, int(k1 == p["c2"] and k2 == p["c1"])


def parse(s):
    b, h = s.split("|h"); return (int(b), int(h))


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    hs = json.load(open("results/sd3_headsweep.json"))
    ranked = [parse(s) for s in hs["ranked"]]
    coarse = json.load(open("results/sd3_coarse.json"))
    pairs_all = {p["id"]: p for p in make_pairs()}
    sub = int(os.environ.get("SD3_SUBSET", "0"))
    qids = coarse["qualified"][:sub] if sub else coarse["qualified"]
    head_sets = {"top1": ranked[:1], f"top{TOPK}": ranked[:TOPK]}
    CONDS = {"FULL": (True, True), "PATTERN": (True, False),
             "VALUE": (False, True), "BASELINE": (False, False)}
    log(f"\n===== SD3.5 A4 OV/QK decomp: {len(qids)} pairs ; top1={ranked[0]} ; top{TOPK} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    # results[hs][cond] = list of flip per pair ; also proper
    res = {hn: {c: [] for c in CONDS} for hn in head_sets}
    proper = {hn: {c: [] for c in CONDS} for hn in head_sets}
    refs = []
    for pid in qids:
        p = pairs_all[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID,
                        height=H, width=W, device=DEVICE)
        rf, _ = grade(qg, generate(pipe, ctrl, p["clean"], seed, swap_all=True, swap_k=True,
                      swap_v=True, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE), p)
        refs.append(rf)
        for hn, cells in head_sets.items():
            for c, (sk, sv) in CONDS.items():
                f, pr = grade(qg, generate(pipe, ctrl, p["clean"], seed, cells=cells, swap_k=sk,
                              swap_v=sv, steps=STEPS, guidance=GUID, height=H, width=W,
                              device=DEVICE), p)
                res[hn][c].append(f); proper[hn][c].append(pr)
        ctrl.swap_ctx = {}
        log(f"pair{pid:2d} ref={rf:.2f} | "
            + " ".join(f"{hn}:{','.join(f'{c[0]}={res[hn][c][-1]:.1f}' for c in CONDS)}"
                       for hn in head_sets))

    ref_mean = float(np.mean(refs))
    rng = np.random.default_rng(BOOT_SEED)
    out = {"top1": ranked[0], "topk": TOPK, "n": len(qids), "ref_mean": ref_mean, "sets": {}}
    log(f"\n===== RESULTS (swap_all ref flip={ref_mean:.3f}) =====")
    for hn in head_sets:
        means = {c: float(np.mean(res[hn][c])) for c in CONDS}
        pmeans = {c: float(np.mean(proper[hn][c])) for c in CONDS}
        P = np.array(res[hn]["PATTERN"]); V = np.array(res[hn]["VALUE"])
        n = len(P); idx = rng.integers(0, n, size=(NBOOT, n))
        boots = P[idx].mean(1) - V[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5]); diff = float(P.mean() - V.mean())
        rng_ = max(means["FULL"] - means["BASELINE"], 1e-6)
        fp = (means["PATTERN"] - means["BASELINE"]) / rng_
        fv = (means["VALUE"] - means["BASELINE"]) / rng_
        v = ("ROUTING (QK)" if fp >= 0.6 and fv <= 0.4 else
             "CONTENT (OV)" if fv >= 0.6 and fp <= 0.4 else "MIXED")
        log(f"  [{hn}] FULL={means['FULL']:.3f} PATTERN={means['PATTERN']:.3f} "
            f"VALUE={means['VALUE']:.3f} BASE={means['BASELINE']:.3f} "
            f"proper(FULL)={pmeans['FULL']:.3f}")
        log(f"        PATTERN-VALUE={diff:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] "
            f"-> {v} (routing ~{fp*100:.0f}%, content ~{fv*100:.0f}%)")
        out["sets"][hn] = dict(means=means, proper=pmeans, pattern_minus_value=diff,
                               ci=[float(lo), float(hi)], frac_routing=fp, frac_content=fv,
                               verdict=v)
    json.dump(out, open("results/sd3_a4.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_a4.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
