"""Held-out/selection overlap under UNORDERED prompt sets. holdout_purity.py counts a held-out
pair as overlapping only when its ORDERED (clean, swapped) text pair was used in selection (6/84).
Counting a pair as overlapping when the UNORDERED {clean, swapped} set was used in selection gives
13/84. Recompute every held-out headline without those 13, including the matched 3-block image
window (matched_late.json img_b911), which holdout_purity.py does not report, and the paired
contrasts of kv_factorial.py / hybrid_v2.py. CPU only.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_additional_controls"))  # holdout_purity.py lives in 04_additional_controls
from prompts import make_pairs
from prompts_ext import make_ext_pairs
from holdout_purity import ci, report, binom_upper

A, B = make_pairs(), make_ext_pairs()
sel_o = {(p["clean"], p["swapped"]) for p in A}
sel_u = {frozenset((p["clean"], p["swapped"])) for p in A}
leak_o = {p["id"] for p in B if (p["clean"], p["swapped"]) in sel_o}
leak = {p["id"] for p in B if frozenset((p["clean"], p["swapped"])) in sel_u}
print(f"ordered overlap {len(leak_o)}/{len(B)} {sorted(leak_o)}")
print(f"UNORDERED overlap {len(leak)}/{len(B)} {sorted(leak)}\n")
out = {"ordered_ids": sorted(leak_o), "unordered_ids": sorted(leak), "rows": []}

s15 = json.load(open("results/top1_sd15_ext.json")); px = json.load(open("results/top1_pixart_ext.json")); s3 = json.load(open("results/top1_sd3_ext.json"))
tri = {}
for name, ids, vals in (("SD1.5 top-1 head", s15["ids"], s15["per_pair"]), ("PixArt top-1 head", px["qualified_ids"], px["per_pair"]), ("SD3.5 top-1 head", s3["qualified"], s3["ext_per"])):
    ho = [(i, v) for i, v in zip(ids, vals) if i >= 5000]
    out["rows"].append(report(name, [i for i, _ in ho], [v for _, v in ho], leak)); tri[name.split()[0]] = dict(ho)
b9 = json.load(open("results/block9_ext.json")); ml = json.load(open("results/matched_late.json")); wp = json.load(open("results/window_placebo_paired.json"))
for key, lab in (("iso9", "SD3.5 iso block 9"), ("win911", "SD3.5 window 9-11"), ("img_b9", "SD3.5 image-facing blk9")):
    out["rows"].append(report(lab, b9["qualified"], b9[key], leak))
out["rows"].append(report("SD3.5 image window 9-11", b9["qualified"], ml["img_b911"], leak))
out["rows"].append(report("SD3.5 window placebo", wp["qualified"], wp["placebo_win"], leak))

def paired(tag, ids, a, b):
    keep = [(x - y) for i, x, y in zip(ids, a, b) if i not in leak]; full = [(x - y) for x, y in zip(a, b)]
    f, p = ci(full, paired=True), ci(keep, paired=True)
    print(f"{tag:28s} full n={len(full):3d} {f[0]:+.3f} [{f[1]:+.3f},{f[2]:+.3f}] m>0={f[3]:.3f} |  overlap-free n={len(keep):3d} {p[0]:+.3f} [{p[1]:+.3f},{p[2]:+.3f}] m>0={p[3]:.3f}")
    out["rows"].append(dict(tag=tag, full=f, pure=p, n_full=len(full), n_pure=len(keep)))
paired("PAIRED text win - image win", b9["qualified"], b9["win911"], ml["img_b911"])
paired("PAIRED swap - placebo", wp["qualified"], wp["swap_win"], wp["placebo_win"])
for f in ("kv_factorial", "hybrid_v2"):
    d = json.load(open(f"results/{f}.json"))
    if f == "kv_factorial":
        paired("PAIRED factorial D - A (pin)", d["qualified"], d["D_cltemb_pin"], d["A_swtemb_nopin"])
    else:
        v1 = json.load(open("results/hybrid_stream.json"))
        paired("PAIRED hyb2 - std window", d["qualified"], d["hyb2_win"], v1["std_win"][:d["n"]])
common = sorted(set(tri["SD1.5"]) & set(tri["PixArt"]) & set(tri["SD3.5"]))
for lab, ids in (("all", common), ("overlap-free", [i for i in common if i not in leak])):
    row = {"scope": lab, "n": len(ids)}
    for m in tri:
        row[m] = ci([tri[m][i] for i in ids])
    row["paired_SD15_minus_SD35"] = ci(np.array([tri["SD1.5"][i] for i in ids]) - np.array([tri["SD3.5"][i] for i in ids]), paired=True)
    print(f"common [{lab:12s}] n={len(ids):2d} | " + " | ".join(f"{m} {row[m][0]:.3f}" for m in tri) + f" | paired SD1.5-SD3.5 {row['paired_SD15_minus_SD35'][0]:+.3f} [{row['paired_SD15_minus_SD35'][1]:+.3f},{row['paired_SD15_minus_SD35'][2]:+.3f}]")
    out.setdefault("common_set", []).append(row)
json.dump(out, open("results/holdout_overlap13.json", "w"), indent=2)
print("\n-> results/holdout_overlap13.json")
