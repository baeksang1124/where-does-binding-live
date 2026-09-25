"""Paper numbers that no experiment script writes, recomputed from the shipped result JSONs only.

Every number below is derived on CPU from files already in results/ (no model, no GPU):
  1. Window-length sweep at n=75 (Sec. 4 "Recovery grows with the length of the isolated window", Fig. 3b,
     "Text-only content vs. joint state"): results/window_extension.json (40 pairs) + window_extension_p2.json
     (35 pairs) concatenated per pair, CIs recomputed with window_extension.py's ci() and rng seed 20260919 in
     the same call order. The merge goes into results/derived_stats.json and is compared to the shipped
     results/window_extension_n75.json (per-pair lists and all stored CIs); the shipped file is not rewritten.
  2. Table 1 PixArt top-1 split: top1_pixart_ext.json per_pair split by id>=5000 into held-out (n=26) and
     in-sample (n=9), mean + 1e5-resample percentile bootstrap, numpy default_rng(1234) per split; compared to
     the stored held_out / in_sample fields.
  3. Limitations, overlap sensitivity rows not in holdout_overlap13.json (overlap ids = its 'unordered_ids',
     re-derived here from the prompt sets as a check):
     (a) text-only minus joint-state 9-11 window (hyb2_e11 - std_e11, paired) from the n=75 merge, full vs
         overlap-excluded; (b) unified strict o1 for PixArt from dual_grade_full.json raw Qwen answers
         (canon() and exact binomial bootstrap CI imported from unified_strict_o1.py), full vs overlap-excluded.
  4. Material vs colour 9-11 window: material_ext.json win911 (n=19) minus block9_ext.json win911 (n=75),
     independent-means bootstrap, 1e5 resamples, seed 1234 (diff_ci() of unified_strict_o1.py).
  5. Seed robustness: sample SD (ddof=1) over the three per-seed means, window_placebo_seeds.json
     (in-sample window / prefix 9 / prefix 15) and seeds_followup.json (held-out, joint-state and text-only).
  6. SD3.5 unpinned whole-block key/value swap (supplementary (iii)): sd3_coarse.json per_pair, block-9 mean
     flip + percentile bootstrap (1e4 resamples as in the in-sample ranking scripts, seed 1234), both-objects
     flipped, max over the other 23 blocks, and the all-head bracket count (full_proper == 1).

Writes results/derived_stats.json (one entry per item plus the check table) and prints
"item | recomputed | paper value | match". A value matches when it rounds (half-up) to the paper's printed
value at the paper's precision; a Monte-Carlo bootstrap bound that misses by <= 0.01 is marked as resampling
noise. Exit status 1 if any row does not match.

CPU only. Run from the repo root:
    PYTHONPATH=. $PY scripts/05_followup_controls/derived_stats.py
"""
import json, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from prompts import make_pairs
from prompts_ext import make_ext_pairs
from unified_strict_o1 import canon, exact_boot_ci, diff_ci   # same directory; no side effects on import

OUT = "results/derived_stats.json"
ENDS = [11, 13, 15, 17, 23]
ARMS = [f"{arm}_e{e}" for arm in ("std", "hyb2") for e in ENDS]
ROWS = []


def load(name):
    return json.load(open(f"results/{name}.json"))


def ci(x, rng, paired=False):      # verbatim from window_extension.py
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + \
           ([float((b > 0).mean())] if paired else [])


def boot(x, seed=1234, nboot=100000):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (nboot, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def check(item, vals, paper, fmt, ci_idx=()):
    """vals: recomputed numbers; paper: the paper's printed strings (same order); fmt: display template.
    ci_idx: positions that are Monte-Carlo bootstrap bounds (<= 0.01 tolerated as resampling noise)."""
    status = "yes"
    for i, (v, p) in enumerate(zip(vals, paper)):
        if "." not in p:
            ok = int(v) == int(p)
        else:
            q = Decimal(1).scaleb(-len(p.split(".")[1]))
            ok = Decimal(repr(round(float(v), 9))).quantize(q, rounding=ROUND_HALF_UP) == Decimal(p)
        if not ok:
            if i in ci_idx and abs(float(v) - float(p)) <= 0.01 + 1e-9:
                status = "yes (CI bound within 0.01: resampling noise)"
            else:
                status = "NO"; break
    show = [str(int(v)) if "." not in p else (f"{v:+.3f}" if p[0] in "+-" else f"{v:.3f}") for v, p in zip(vals, paper)]
    ROWS.append(dict(item=item, recomputed=fmt.format(*show), paper=fmt.format(*paper), match=status))


def ident(item, ok, what):
    ROWS.append(dict(item=item, recomputed="identical" if ok else "DIFFERS", paper=what, match="yes" if ok else "NO"))


def item1():
    a, b, shipped = load("window_extension"), load("window_extension_p2"), load("window_extension_n75")
    res = {k: a[k] + b[k] for k in ARMS}
    rng = np.random.default_rng(20260919)       # same seed and call order as window_extension.py
    m = dict(qualified=a["qualified"] + b["qualified"], n=a["n"] + b["n"], ends=ENDS,
             source=["window_extension.json", "window_extension_p2.json"], **res)
    m["ci"] = {k: ci(v, rng) for k, v in res.items()}
    m["paired_hyb2_minus_std"] = {f"e{e}": ci(np.array(res[f"hyb2_e{e}"]) - np.array(res[f"std_e{e}"]), rng, True) for e in ENDS}
    m["paired_vs_e11"] = {f"{arm}_e{e}": ci(np.array(res[f"{arm}_e{e}"]) - np.array(res[f"{arm}_e11"]), rng, True)
                          for arm in ("std", "hyb2") for e in ENDS[1:]}
    eq = {"qualified": m["qualified"] == shipped["qualified"], "per_pair": all(res[k] == shipped[k] for k in ARMS),
          **{k: m[k] == shipped[k] for k in ("ci", "paired_hyb2_minus_std", "paired_vs_e11")}}
    m["equals_shipped_window_extension_n75"] = eq
    ident("1 merge vs shipped n75 (ids, 10 per-pair lists, 23 CIs)", all(eq.values()), "window_extension_n75.json")
    c, pd, pv = m["ci"], m["paired_hyb2_minus_std"], m["paired_vs_e11"]
    check("1 joint-state window 9-11, n=75", c["std_e11"], ["0.30", "0.23", "0.38"], "{} [{},{}]", (1, 2))
    check("1 joint-state 9-13 / 9-17 / 9-23", [c["std_e13"][0], c["std_e17"][0], c["std_e23"][0]], ["0.45", "0.63", "0.87"], "{} / {} / {}")
    check("1 joint-state window 9-15", c["std_e15"], ["0.62", "0.54", "0.70"], "{} [{},{}]", (1, 2))
    check("1 joint-state 9-15 minus 9-11 (paired)", pv["std_e15"][:3], ["+0.32", "+0.23", "+0.41"], "{} [{},{}]", (1, 2))
    check("1 joint-state 9-17 minus 9-15", [c["std_e17"][0] - c["std_e15"][0]], ["+0.01"], "{}")
    check("1 text-only window 9-11", c["hyb2_e11"], ["0.08", "0.03", "0.13"], "{} [{},{}]", (1, 2))
    check("1 text-only minus joint 9-11 (paired)", pd["e11"][:3], ["-0.22", "-0.31", "-0.14"], "{} [{},{}]", (1, 2))
    check("1 text-only window 9-15", [c["hyb2_e15"][0]], ["0.49"], "{}")
    check("1 text-only minus joint 9-15 (paired)", pd["e15"][:3], ["-0.13", "-0.23", "-0.02"], "{} [{},{}]", (1, 2))
    return m


def item2():
    px = load("top1_pixart_ext"); pp = list(zip(px["qualified_ids"], px["per_pair"]))
    out = {}
    for key, sel in (("held_out", lambda i: i >= 5000), ("in_sample", lambda i: i < 5000)):
        x = [v for i, v in pp if sel(i)]
        out[key] = boot(x); out[f"{key}_n"] = len(x)
    out["equals_stored_fields"] = all(out[k] == px[k] for k in ("held_out", "held_out_n", "in_sample", "in_sample_n"))
    ident("2 PixArt split vs stored held_out/in_sample", out["equals_stored_fields"], "top1_pixart_ext.json fields")
    check("2 PixArt top-1 held-out (id>=5000)", [out["held_out_n"]] + out["held_out"], ["26", "0.10", "0.02", "0.17"], "n={} {} [{},{}]", (2, 3))
    check("2 PixArt top-1 in-sample (id<5000)", [out["in_sample_n"], out["in_sample"][0]], ["9", "0.33"], "n={} {}")
    return out


def item3(m):
    A, B = make_pairs(), make_ext_pairs()
    sel_u = {frozenset((p["clean"], p["swapped"])) for p in A}
    leak = set(load("holdout_overlap13")["unordered_ids"])
    assert leak == {p["id"] for p in B if frozenset((p["clean"], p["swapped"])) in sel_u}, "overlap ids differ from prompts"
    ident("3 overlap ids = holdout_overlap13.json unordered_ids", True, "13 ids re-derived from prompts")
    # (a) text-only minus joint-state, isolated 9-11, from the n=75 merge
    ids = m["qualified"]; d = np.array(m["hyb2_e11"]) - np.array(m["std_e11"])
    keep = [x for i, x in zip(ids, d) if i not in leak]
    a = dict(full=m["paired_hyb2_minus_std"]["e11"], n_full=len(d),
             pure=ci(keep, np.random.default_rng(20260919), True), n_pure=len(keep),
             dropped=sorted(i for i in ids if i in leak))
    check("3a text-only - joint 9-11: full -> overlap-free", [a["full"][0], a["pure"][0], a["n_full"], a["n_pure"]],
          ["-0.22", "-0.21", "75", "63"], "{} -> {} (n {} -> {})")
    # (b) unified strict o1, PixArt, primary grader (Qwen)
    P = {p["id"]: p for p in A}; P.update({p["id"]: p for p in B})
    raw = load("dual_grade_full")["raw"]
    s = [(int(k.split("_")[1]), float(canon(raw[k]["qwen"]["a1"]) == P[int(k.split("_")[1])]["c2"]))
         for k in sorted(k for k in raw if k.startswith("pixart_"))]
    full, pure = [v for _, v in s], [v for i, v in s if i not in leak]
    b = dict(full=exact_boot_ci(full), n_full=len(full), pure=exact_boot_ci(pure), n_pure=len(pure),
             dropped=sorted(i for i, _ in s if i in leak))
    b["full_equals_unified_strict_o1_json"] = b["full"] == load("unified_strict_o1")["pixart"]["qwen_strict"]
    ident("3b full-set value vs unified_strict_o1.json", b["full_equals_unified_strict_o1_json"], "pixart qwen_strict")
    check("3b unified strict o1 PixArt: full -> overlap-free", [b["full"][0], b["pure"][0], b["n_full"], b["n_pure"]],
          ["0.12", "0.05", "26", "21"], "{} -> {} (n {} -> {})")
    return dict(overlap_ids=sorted(leak), textonly_minus_joint_e11=a, unified_strict_o1_pixart_qwen=b)


def item4():
    ma, co = load("material_ext")["win911"], load("block9_ext")["win911"]
    d = diff_ci(ma, co, np.random.default_rng(1234))    # 1e5 resamples, independent means
    out = dict(material_mean=float(np.mean(ma)), n_material=len(ma), colour_mean=float(np.mean(co)), n_colour=len(co),
               diff=d[:3], p_gt0=d[3], nboot=100000, seed=1234)
    check("4 material - colour window 9-11 (indep.)", d[:3], ["+0.02", "-0.16", "+0.21"], "{} [{},{}]", (1, 2))
    return out


def item5():
    out = {}
    for name, keys, paper in (("window_placebo_seeds", ("win", "pre9", "pre15"), (("0.34", "0.08"), ("0.83", "0.14"), ("0.01", "0.02"))),
                              ("seeds_followup", ("std_e11", "std_e15", "hyb2_e11", "hyb2_e15"),
                               (("0.33", "0.04"), ("0.64", "0.02"), ("0.07", "0.03"), ("0.49", "0.01")))):
        d = load(name); r = {}
        for k, pv in zip(keys, paper):
            means = [float(np.mean(d["seeds"][s][k])) for s in sorted(d["seeds"], key=int)]
            r[k] = dict(per_seed_means=means, n_per_seed=[len(d["seeds"][s][k]) for s in sorted(d["seeds"], key=int)],
                        mean=float(np.mean(means)), sd_ddof1=float(np.std(means, ddof=1)))
            assert np.allclose(means, d["per_seed_means"][k]), f"{name} {k}: per-seed means differ from stored"
            check(f"5 {name} {k} (mean +- SD)", [r[k]["mean"], r[k]["sd_ddof1"]], list(pv), "{} +- {}")
        out[name] = r
    return out


def item6():
    c = load("sd3_coarse"); pp = c["per_pair"]
    flip = {b: [p["blocks"][str(b)]["flip"] for p in pp] for b in range(24)}
    prop = {b: [p["blocks"][str(b)]["proper"] for p in pp] for b in range(24)}
    assert all(abs(np.mean(flip[b]) - c["block_mean_flip"][str(b)]) < 1e-12 for b in range(24))
    others = {b: float(np.mean(flip[b])) for b in range(24) if b != 9}
    bmax = max(others, key=others.get)
    out = dict(n=len(pp), block9_flip=boot(flip[9], 1234, 10000), block9_both_flipped=float(np.mean(prop[9])),
               max_other_block=bmax, max_other_flip=others[bmax],
               allhead_bracket=[int(sum(p["full_proper"] == 1 for p in pp)), len(pp)])
    check("6 SD3.5 whole-block swap, block 9", out["block9_flip"], ["0.11", "0.04", "0.20"], "{} [{},{}]", (1, 2))
    check("6 block 9 both objects flipped", [out["block9_both_flipped"]], ["0.03"], "{}")
    ROWS.append(dict(item="6 every other block <= 0.03", recomputed=f"max {others[bmax]:.3f} (block {bmax})", paper="<= 0.03",
                     match="yes" if round(others[bmax], 2) <= 0.03 else "NO"))
    check("6 all-head bracket (full_proper==1)", out["allhead_bracket"], ["33", "35"], "{}/{}")
    return out


def main():
    m = item1()
    out = {"note": "paper numbers not written by any experiment script, recomputed from shipped results/*.json by "
                   "scripts/05_followup_controls/derived_stats.py (CPU); see its docstring for methods.",
           "1_window_extension_n75_merge": m, "2_pixart_top1_heldout_insample_split": item2(),
           "3_overlap_sensitivity_limitations": item3(m), "4_material_minus_colour_window911": item4(),
           "5_seed_robustness_sd_over_seed_means": item5(), "6_sd3_whole_block_unpinned_bound": item6()}
    out["checks"] = ROWS
    json.dump(out, open(OUT, "w"), indent=1)
    w = [max(len(r[k]) for r in ROWS + [dict(item="item", recomputed="recomputed", paper="paper value", match="match")]) for k in ("item", "recomputed", "paper")]
    print(f"{'item':{w[0]}} | {'recomputed':{w[1]}} | {'paper value':{w[2]}} | match (within rounding)")
    print("-" * (sum(w) + 34))
    for r in ROWS:
        print(f"{r['item']:{w[0]}} | {r['recomputed']:{w[1]}} | {r['paper']:{w[2]}} | {r['match']}")
    bad = [r["item"] for r in ROWS if r["match"] == "NO"]
    print(f"\n{len(ROWS) - len(bad)}/{len(ROWS)} rows match" + (f"; MISMATCH: {bad}" if bad else "") + f"\n-> {OUT}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
