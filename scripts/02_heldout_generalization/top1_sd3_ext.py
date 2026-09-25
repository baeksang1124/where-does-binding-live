"""Expand SD3.5 held-out top-1 n to tighten the 'no head handle' CI (the original n=12
measurement, results/top1_sd3.json, gives [0.00,0.25]). Qualify the extended prompt pairs on
SD3.5 and measure the top-1 head flip (per-object strict flip, mean over both objects);
combine with the original 12. Outputs: results/top1_sd3_ext.json, results/top1_sd3_ext.log.
One GPU.
"""
import torch, json, os, numpy as np, re
from prompts_ext import make_ext_pairs
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; GUID = 7.0; H = W = 512
LOG = open("results/top1_sd3_ext.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def grade2(qg, img, p):
    return canon(qg._ask_color(img, p["o1"])), canon(qg._ask_color(img, p["o2"]))


def main():
    os.makedirs("results", exist_ok=True)
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    ext = make_ext_pairs()
    # exclude the original coarse-qualified ids already measured (they are the in-sample-ish set);
    # ext ids are >=5000 so disjoint from base pairs anyway.
    log(f"\n===== SD3.5 top-1 n-expansion: {len(ext)} ext candidates, top1={hs['ranked'][0]} =====")
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)
    per, qids = [], []
    for p in ext:
        seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        A = generate(pipe, ctrl, p["clean"], seed, cells=None, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        S = generate(pipe, ctrl, p["clean"], seed, swap_all=True, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        ca1, ca2 = grade2(qg, A, p); sa1, sa2 = grade2(qg, S, p)
        ok = (ca1 == p["c1"] and ca2 == p["c2"] and sa1 == p["c2"] and sa2 == p["c1"])
        if not ok:
            ctrl.swap_ctx = {}; continue
        qids.append(p["id"])
        img = generate(pipe, ctrl, p["clean"], seed, cells=top1, swap_k=True, swap_v=True,
                       steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
        f1 = int(k1 == p["c2"] and k1 != p["c1"]); f2 = int(k2 == p["c1"] and k2 != p["c2"])
        per.append((f1 + f2) / 2.0); ctrl.swap_ctx = {}
        if len(per) % 5 == 0:
            log(f"  qualified {len(per)}  running mean={np.mean(per):.3f}")

    # combine with original held-out-style n=12
    orig = json.load(open("results/top1_sd3.json"))["per_pair"]
    comb = orig + per
    def ci(x):
        x = np.array(x, float); rng = np.random.default_rng(1234)
        bt = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
    m, lo, hi = ci(per); mc, loc, hic = ci(comb)
    log(f"\nEXT-only  n={len(per)} mean={m:.3f} CI[{lo:.3f},{hi:.3f}]")
    log(f"COMBINED  n={len(comb)} mean={mc:.3f} CI[{loc:.3f},{hic:.3f}]  (was n=12 [0.00,0.25])")
    json.dump(dict(top1=hs['ranked'][0], ext_n=len(per), ext_per=per, ext_ci=[lo, hi], ext_mean=m,
                   combined_n=len(comb), combined_mean=mc, combined_ci=[loc, hic], qualified=qids),
              open("results/top1_sd3_ext.json", "w"), indent=2)


if __name__ == "__main__":
    main()
