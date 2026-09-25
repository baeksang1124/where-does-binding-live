"""EXP A4 — OV-vs-QK decomposition of diffusion attribute-object binding.

The localization sweep (run_sweep.py) found that binding is carried by a few SD1.5 cross-attn
heads via a per-head text-source swap (recompute K AND V from the swapped prompt). This
experiment splits that swap into its PATTERN (routing / QK) and CONTENT (value / OV) parts
to ask which one actually carries the binding.

Conditions (per selected head):
  FULL      K=swapped, V=swapped   -> reference; reproduces the localization recovery.
  PATTERN   K=swapped, V=clean     -> attends per swapped routing, delivers CLEAN content.
  VALUE     K=clean,   V=swapped   -> attends as clean, delivers SWAPPED content.
  BASELINE  K=clean,   V=clean     -> no-op; sanity must be ~0.

Verdict:
  ROUTING(QK):  PATTERN ~ FULL, VALUE ~ baseline.
  CONTENT(OV):  VALUE   ~ FULL, PATTERN ~ baseline.
  MIXED:        both partial.

Reuses load_pipe/install/encode/generate (now with independent swap_k/swap_v), the Qwen
grader, qualified.json, and the head ranking in sweep_results.json. Nothing is rebuilt.
"""
import torch, json, sys, time, numpy as np
from binding_patch import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"
STEPS = 30
N_BOOT = 10000
BOOT_SEED = 1234
LOG = open("results/a4_decomp.log", "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


def score(qg, img, p):
    s, _ = qg.object_swap_score(img, p["grade_obj"], p["clean_color"], p["swap_color"])
    return s


def parse_cell(s):
    ln, h = s.split("|h")
    return (ln, int(h))


def main():
    t0 = time.time()
    pairs = json.load(open("results/qualified.json"))
    sweep = json.load(open("results/sweep_results.json"))
    ranked = [parse_cell(c) for c in sweep["ranked_cells"]]
    top1 = [ranked[0]]
    top16 = ranked[:16]
    head_sets = {"top1": top1, "top16": top16}
    log(f"\n===== A4 OV-vs-QK decomp: {len(pairs)} pairs, {STEPS} steps =====")
    log(f"  top1 head : {top1[0][0]} h{top1[0][1]}")
    log(f"  top16 set : {[f'{l.split(chr(46))[0]}..h{h}' for l,h in top16]}")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    # ---- per-pair baselines + cache (latents, swap_embeds) : identical recipe to sweep ----
    ctx = []
    for p in pairs:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        b0, b1 = score(qg, ic, p), score(qg, ia, p)
        ctx.append(dict(p=p, lat=lat, se=se, b0=b0, b1=b1))
        log(f"  baseline pair{p['id']:2d} b0={b0:.2f} b1={b1:.2f} | {p['clean']}")
    valid = [c for c in ctx if (c["b1"] - c["b0"]) >= 0.5]
    log(f"  valid (b1-b0>=0.5): {len(valid)}/{len(ctx)}")

    def recov(c, s):
        return (s - c["b0"]) / (c["b1"] - c["b0"])

    # conditions: (swap_k, swap_v)
    CONDS = {"FULL": (True, True), "PATTERN": (True, False),
             "VALUE": (False, True), "BASELINE": (False, False)}

    # ---- independence assertion: snapshot post-intervention K/V on the top-1 layer ----
    log("\n--- SANITY: K-source / V-source are independently controlled ---")
    cl, ch = top1[0]
    c0 = valid[0]
    snaps = {}
    for cond, (sk, sv) in CONDS.items():
        _ = generate(pipe, ctrl, c0["p"]["clean"], c0["se"], c0["lat"].clone(),
                     cells=top1, swap_k=sk, swap_v=sv, capture_layer=cl, steps=STEPS)
        cap = ctrl._captured
        snaps[cond] = (cap["key"][ch], cap["value"][ch])  # [seq, dim] for the patched head
    def eq(a, b): return bool(torch.equal(a, b))
    kB, vB = snaps["BASELINE"]; kF, vF = snaps["FULL"]
    kP, vP = snaps["PATTERN"];  kV, vV = snaps["VALUE"]
    checks = {
        "PATTERN swaps K (key != baseline)":      not eq(kP, kB),
        "PATTERN keeps V (value == baseline)":         eq(vP, vB),
        "VALUE keeps K (key == baseline)":             eq(kV, kB),
        "VALUE swaps V (value != baseline)":      not eq(vV, vB),
        "FULL K == PATTERN K (same swapped K)":        eq(kF, kP),
        "FULL V == VALUE V (same swapped V)":          eq(vF, vV),
    }
    for k, v in checks.items():
        log(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        log("  !! independence assertion FAILED — K/V source split is wired wrong. ABORT.")
        sys.exit(1)
    log("  independence assertion: ALL PASS (K-source and V-source are independent).")

    # ---- main: recovery per (head_set x condition) ----
    results = {}
    for hs_name, cells in head_sets.items():
        log(f"\n--- HEAD SET: {hs_name} ({len(cells)} head(s)) ---")
        results[hs_name] = {}
        for cond, (sk, sv) in CONDS.items():
            rs = []
            for c in valid:
                img = generate(pipe, ctrl, c["p"]["clean"], c["se"], c["lat"].clone(),
                               cells=cells, swap_k=sk, swap_v=sv, steps=STEPS)
                rs.append(recov(c, score(qg, img, c["p"])))
            rs = np.array(rs)
            results[hs_name][cond] = rs.tolist()
            log(f"  {cond:9s} recovery mean={rs.mean():+.3f}  (n={len(rs)})")

    # ---- paired bootstrap CI on [PATTERN - VALUE] over the valid pairs ----
    log("\n--- BOOTSTRAP 95% CI on [PATTERN - VALUE] (paired, resampling pairs) ---")
    rng = np.random.default_rng(BOOT_SEED)
    boot = {}
    nval = len(valid)
    for hs_name in head_sets:
        P = np.array(results[hs_name]["PATTERN"])
        V = np.array(results[hs_name]["VALUE"])
        diff = float(P.mean() - V.mean())
        idx = rng.integers(0, nval, size=(N_BOOT, nval))
        boots = P[idx].mean(1) - V[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        boot[hs_name] = dict(diff=diff, lo=float(lo), hi=float(hi))
        log(f"  {hs_name:6s}: PATTERN-VALUE = {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ---- verdict per head set ----
    def verdict(hs):
        r = {k: float(np.mean(results[hs][k])) for k in CONDS}
        full = r["FULL"]; base = r["BASELINE"]
        if full - base < 0.10:
            return "INCONCLUSIVE (FULL~BASELINE; no effect to decompose)", r
        # fraction of FULL effect carried by each route (clamped above baseline)
        rng_ = max(full - base, 1e-6)
        fp = (r["PATTERN"] - base) / rng_
        fv = (r["VALUE"] - base) / rng_
        if fp >= 0.6 and fv <= 0.4:
            v = "ROUTING (QK)"
        elif fv >= 0.6 and fp <= 0.4:
            v = "CONTENT (OV)"
        else:
            v = "MIXED"
        return f"{v}  (routing carries ~{fp*100:.0f}% of FULL, content ~{fv*100:.0f}%)", r

    log("\n===== VERDICT =====")
    verds = {}
    for hs in head_sets:
        vtext, r = verdict(hs)
        verds[hs] = vtext
        log(f"  [{hs}] FULL={r['FULL']:+.3f} PATTERN={r['PATTERN']:+.3f} "
            f"VALUE={r['VALUE']:+.3f} BASELINE={r['BASELINE']:+.3f}")
        log(f"        -> {vtext}")

    out = dict(head_sets={k: [f"{l}|h{h}" for l, h in v] for k, v in head_sets.items()},
               n_valid=nval, conditions=list(CONDS),
               recovery_means={hs: {c: float(np.mean(results[hs][c])) for c in CONDS}
                               for hs in head_sets},
               recovery_per_pair=results, bootstrap=boot, verdict=verds,
               pair_ids=[c["p"]["id"] for c in valid])
    json.dump(out, open("results/a4_decomp.json", "w"), indent=2)
    log(f"\n  results -> results/a4_decomp.json ; elapsed {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
