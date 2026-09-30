"""Per-pair distributions and complete-swap statistics for the window-control run (CPU only).

Reads the per-pair records ({OUT}.jsonl) and summary ({OUT}.json) written by window_controls.py, plus
kv_factorial.json (arm D), and writes {OUT}_stats.json: for every arm the 0/0.5/1 pair counts, the mean
with its CI, the complete-swap count and the CI of the complete-swap rate; paired differences against the
9-15 window on the mean and on complete swaps; and the pinned 9-11 window (D) and the pinned block 9 minus the pinned head on
complete swaps. Pair-level percentile bootstrap, 10^5 resamples, seed 20261002.
Run from the repo root:  PYTHONPATH=. $PY scripts/05_followup_controls/window_controls_stats.py
"""
import json
import os

import numpy as np

OUT = os.environ.get("WC_OUT", "results/window_controls")


def main():
    S = json.load(open(f"{OUT}.json"))
    recs = {}
    for line in open(f"{OUT}.jsonl"):
        r = json.loads(line); recs[r["id"]] = r
    q = S["qualified"]; R = [recs[i] for i in q]
    rng = np.random.default_rng(20261002)

    def ci(x, paired=False):
        x = np.asarray(x, float); b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + ([float((b > 0).mean())] if paired else [])

    def arm(x):
        x = np.asarray(x, float); c = (x == 1).astype(float)
        return dict(pairs_0_05_1=[int((x == 0).sum()), int((x == 0.5).sum()), int((x == 1).sum())],
                    mean_ci=ci(x), complete=int(c.sum()), complete_rate_ci=ci(c))

    col = lambda f: np.array([f(r) for r in R], float)  # noqa: E731
    out = dict(n=len(R), seed=20261002, resamples=100000, windows={}, windows_paired_vs_9_15={}, placebo={}, units={})
    b915 = col(lambda r: r["win"]["9-15"]["flip"])
    for k in R[0]["win"]:
        x = col(lambda r, k=k: r["win"][k]["flip"])
        out["windows"][k] = arm(x)
        if k != "9-15":
            out["windows_paired_vs_9_15"][k] = dict(mean=ci(x - b915, True),
                                                   complete=ci((x == 1).astype(float) - (b915 == 1), True))
    for k in ("9-15", "9-11"):
        x = col(lambda r, k=k: r["placebo"][k]["flip"]); p = col(lambda r, k=k: r["placebo"][k]["potency"])
        out["placebo"][k] = dict(crossflip=arm(x), potency_ci=ci(p), installed_fully=int((p == 1).sum()))
    for k in ("head_4h1_pinned", "block9_allheads_pinned", "textinject_block9"):
        out["units"][k] = arm(col(lambda r, k=k: r["unit"][k]["flip"]))
    kv = json.load(open("results/kv_factorial.json"))
    D = np.array([dict(zip(kv["qualified"], kv["D_cltemb_pin"]))[i] for i in q], float)
    hd = col(lambda r: r["unit"]["head_4h1_pinned"]["flip"])
    out["units"]["window911_pinned_D"] = arm(D)
    out["units"]["D_minus_head_complete"] = ci((D == 1).astype(float) - (hd == 1), True)
    bk = col(lambda r: r["unit"]["block9_allheads_pinned"]["flip"])
    out["units"]["block9_minus_head_complete"] = ci((bk == 1).astype(float) - (hd == 1), True)
    json.dump(out, open(f"{OUT}_stats.json", "w"), indent=1)
    for k, v in out["windows"].items():
        print(f"window {k:6s} {v['pairs_0_05_1']} mean {v['mean_ci'][0]:.2f} complete {v['complete']}/{len(R)} "
              f"[{v['complete_rate_ci'][1]:.2f},{v['complete_rate_ci'][2]:.2f}]")
    for k, v in out["placebo"].items():
        c = v["crossflip"]
        print(f"placebo {k} {c['pairs_0_05_1']} mean {c['mean_ci'][0]:.2f} complete {c['complete']} "
              f"[{c['complete_rate_ci'][1]:.2f},{c['complete_rate_ci'][2]:.2f}] installed fully {v['installed_fully']}")
    for k in ("head_4h1_pinned", "block9_allheads_pinned"):
        v = out["units"][k]; print(f"{k} {v['pairs_0_05_1']} mean {v['mean_ci'][0]:.2f} complete {v['complete']}")
    d = out["units"]["D_minus_head_complete"]; print(f"D - head (complete) {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]")
    d = out["units"]["block9_minus_head_complete"]; print(f"block 9 - head (complete) {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]")
    print("->", f"{OUT}_stats.json")


if __name__ == "__main__":
    main()
