"""Construction of the selection and held-out prompt sets, and every held-out headline
recomputed without prompt-level overlap.

The held-out set (ids 5000+) was built combinatorially; 6 of its 84 prompt pairs are
TEXT-IDENTICAL (same ordered (clean, swapped) prompts) to pairs in the selection set (ids 0-35).
They carry different seeds (5000+i vs 1000+i), so the graded images are disjoint, but the
prompts overlap. This script recomputes EVERY held-out headline with those 6 ids removed,
using the stored per-pair arrays only (no GPU).

Also reports object-identity and (colour,object) overlap statistics between the two sets.
holdout_overlap13.py extends the overlap check to unordered {clean, swapped} pairs (13/84).
"""
import json
import numpy as np
from prompts import make_pairs
from prompts_ext import make_ext_pairs

RNG = np.random.default_rng(20260828)


def ci(x, paired=False):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    b = x[RNG.integers(0, len(x), (100000, len(x)))].mean(1)
    out = [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out + [float((b > 0).mean())] if paired else out


def binom_upper(n, alpha=0.05):
    """Clopper-Pearson two-sided upper bound for 0 successes out of n."""
    return 1.0 - (alpha / 2.0) ** (1.0 / n) if n else float("nan")


def report(tag, ids, vals, leak):
    ids = list(ids); vals = list(vals)
    keep = [(i, v) for i, v in zip(ids, vals) if i not in leak]
    drop = [(i, v) for i, v in zip(ids, vals) if i in leak]
    full = ci([v for _, v in zip(ids, vals)])
    pure = ci([v for _, v in keep])
    line = (f"{tag:28s} full n={len(ids):3d} {full[0]:.3f} [{full[1]:.3f},{full[2]:.3f}]   "
            f"|  leak-free n={len(keep):3d} {pure[0]:.3f} [{pure[1]:.3f},{pure[2]:.3f}]   "
            f"(dropped {len(drop)}: {[i for i, _ in drop]})")
    if pure[0] == 0.0 and len(keep):
        line += f"  [0/{len(keep)}, exact-binom UB {binom_upper(len(keep)):.3f}]"
    print(line)
    return dict(tag=tag, n_full=len(ids), full=full, n_pure=len(keep), pure=pure,
                dropped=[i for i, _ in drop])


def main():
    A, B = make_pairs(), make_ext_pairs()
    sel = {(p["clean"], p["swapped"]) for p in A}
    leak = {p["id"] for p in B if (p["clean"], p["swapped"]) in sel}

    oA = {o for p in A for o in (p["o1"], p["o2"])}
    oB = {o for p in B for o in (p["o1"], p["o2"])}
    cA = {(p[c], p[o]) for p in A for c, o in (("c1", "o1"), ("c2", "o2"))}
    cB = {(p[c], p[o]) for p in B for c, o in (("c1", "o1"), ("c2", "o2"))}

    print("=== SET CONSTRUCTION ===")
    print(f"selection set: n={len(A)} (ids 0-35, seed 1000+id) | held-out: n={len(B)} (ids 5000+, seed 5000+id)")
    print(f"object identities: selection {len(oA)} {sorted(oA)}")
    print(f"                   held-out  {len(oB)} (adds {sorted(oB - oA)})")
    print(f"(colour,object) combos: selection {len(cA)}, held-out {len(cB)}, "
          f"held-out-only {len(cB - cA)} -> combos DO overlap by design (same 4 colours)")
    print(f"TEXT-IDENTICAL pair leak: {len(leak)}/{len(B)} ids {sorted(leak)} "
          f"(different seeds, so images/gradings are disjoint)")
    print()

    out = {"leak_ids": sorted(leak), "rows": []}
    print("=== EVERY HELD-OUT HEADLINE, RECOMPUTED WITHOUT THE LEAKED PAIRS ===")

    # --- top-1 head, held-out, per model -------------------------------------------------
    s15 = json.load(open("results/top1_sd15_ext.json"))
    px = json.load(open("results/top1_pixart_ext.json"))
    s3 = json.load(open("results/top1_sd3_ext.json"))
    tri = {}
    for name, ids, vals in (("SD1.5 top-1 head", s15["ids"], s15["per_pair"]),
                            ("PixArt top-1 head", px["qualified_ids"], px["per_pair"]),
                            ("SD3.5 top-1 head", s3["qualified"], s3["ext_per"])):
        ho = [(i, v) for i, v in zip(ids, vals) if i >= 5000]
        out["rows"].append(report(name, [i for i, _ in ho], [v for _, v in ho], leak))
        tri[name.split()[0]] = dict(ho)

    # --- SD3.5 text-stream depth / window ------------------------------------------------
    b9 = json.load(open("results/block9_ext.json"))
    for key, lab in (("iso9", "SD3.5 iso block 9"), ("iso11", "SD3.5 iso block 11"),
                     ("win911", "SD3.5 window 9-11"), ("img_b9", "SD3.5 image-facing blk9")):
        out["rows"].append(report(lab, b9["qualified"], b9[key], leak))

    # --- paired window - image-facing, and swap - placebo ---------------------------------
    def paired(tag, ids, a, b):
        keep = [(x - y) for i, x, y in zip(ids, a, b) if i not in leak]
        full = [(x - y) for x, y in zip(a, b)]
        f, p = ci(full, paired=True), ci(keep, paired=True)
        print(f"{tag:28s} full n={len(full):3d} {f[0]:+.3f} [{f[1]:+.3f},{f[2]:+.3f}] m>0={f[3]:.3f} "
              f"|  leak-free n={len(keep):3d} {p[0]:+.3f} [{p[1]:+.3f},{p[2]:+.3f}] m>0={p[3]:.3f}")
        out["rows"].append(dict(tag=tag, full=f, pure=p, n_full=len(full), n_pure=len(keep)))

    paired("PAIRED window - imgfacing", b9["qualified"], b9["win911"], b9["img_b9"])
    wp = json.load(open("results/window_placebo_paired.json"))
    out["rows"].append(report("SD3.5 window placebo", wp["qualified"], wp["placebo_win"], leak))
    paired("PAIRED swap - placebo", wp["qualified"], wp["swap_win"], wp["placebo_win"])

    # --- 3-way common-set comparison (shared prompt population) ---------------------------
    print("\n=== 3-WAY COMMON QUALIFIED SET (shared prompt population) ===")
    common_full = sorted(set(tri["SD1.5"]) & set(tri["PixArt"]) & set(tri["SD3.5"]))
    for lab, ids in (("all", common_full), ("leak-free", [i for i in common_full if i not in leak])):
        row = {"scope": lab, "n": len(ids)}
        for m in ("SD1.5", "PixArt", "SD3.5"):
            v = [tri[m][i] for i in ids]
            row[m] = ci(v)
            if row[m][0] == 0.0:
                row[m].append(binom_upper(len(ids)))
        d = np.array([tri["SD1.5"][i] for i in ids]) - np.array([tri["SD3.5"][i] for i in ids])
        row["paired_SD15_minus_SD35"] = ci(d, paired=True)
        print(f"  [{lab:9s}] n={len(ids):2d} | " + " | ".join(
            f"{m} {row[m][0]:.3f} [{row[m][1]:.3f},{row[m][2]:.3f}]" for m in ("SD1.5", "PixArt", "SD3.5")))
        print(f"              paired SD1.5-SD3.5 {row['paired_SD15_minus_SD35'][0]:+.3f} "
              f"[{row['paired_SD15_minus_SD35'][1]:+.3f},{row['paired_SD15_minus_SD35'][2]:+.3f}] "
              f"mass>0={row['paired_SD15_minus_SD35'][3]:.3f}")
        out.setdefault("common_set", []).append(row)

    json.dump(out, open("results/holdout_purity.json", "w"), indent=2)
    print("\n-> results/holdout_purity.json")


if __name__ == "__main__":
    main()
