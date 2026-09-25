"""Offline bootstrap 95% CIs from stored per-pair arrays (CPU only, no GPU).
Reads results/sd3_stream_ci.json (per-pair iso[block], prefix[block], allmatch, placebo).
Reports mean + 95% CI (percentile bootstrap over pairs) for the depth curve and the
matched-vs-placebo channel contrast, so the paper can print error bars.
"""
import json, numpy as np

NBOOT = 10000
rng = np.random.default_rng(1234)


def ci(arr):
    a = np.asarray(arr, float); n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    idx = rng.integers(0, n, size=(NBOOT, n))
    boots = a[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(a.mean()), float(lo), float(hi))


def main():
    d = json.load(open("results/sd3_stream_ci.json"))
    print(f"n pairs = {len(d['qualified'])}\n")

    m, lo, hi = ci(d["allmatch"])
    print(f"all-block MATCHED flip : {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    m, lo, hi = ci(d["placebo"])
    print(f"PLACEBO (3rd-color)    : {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    # difference matched - placebo
    A = np.array(d["allmatch"]); P = np.array(d["placebo"]); n = len(A)
    idx = rng.integers(0, n, size=(NBOOT, n))
    diff = A[idx].mean(1) - P[idx].mean(1)
    print(f"MATCHED - PLACEBO diff : {A.mean()-P.mean():.3f}  95% CI "
          f"[{np.percentile(diff,2.5):.3f}, {np.percentile(diff,97.5):.3f}]\n")

    print("ISOLATED single-block depth curve (block: mean [CI]):")
    iso = d["iso_per_pair"]
    order = sorted(iso.keys(), key=lambda b: -np.mean(iso[b]))
    for b in order[:8]:
        m, lo, hi = ci(iso[b])
        print(f"  block {int(b):2d}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\nblock9 vs block11 separation:")
    b9, b11 = np.array(iso["9"]), np.array(iso["11"])
    dd = b9[idx].mean(1) - b11[idx].mean(1)
    print(f"  block9 - block11 = {b9.mean()-b11.mean():+.3f}  95% CI "
          f"[{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}]  "
          f"({'separable' if np.percentile(dd,2.5)>0 else 'OVERLAPS 0 -> report as block ~9-11'})")

    print("\nPREFIX depth-commitment (from block b onward):")
    for b in sorted(d["prefix_per_pair"], key=lambda x: int(x)):
        m, lo, hi = ci(d["prefix_per_pair"][b])
        print(f"  from block {int(b):2d}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
