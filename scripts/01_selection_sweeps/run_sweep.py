"""SD1.5 localization sweep.

Coarse-to-fine cross-attention patching to decide whether attribute-object binding is
LOCALIZED to a few heads or DISTRIBUTED across many.

Per qualified pair:
  b0 = binding_swap_score(clean baseline)   (~0)
  b1 = binding_swap_score(swap-all)         (~1)   [proven == swapped-prompt image]
  recovery(config) = (score(config) - b0) / (b1 - b0)

Phases:
  1. LAYER sweep  : swap all 8 heads of one CA layer at a time (16 layers).
  2. LAYER cumulative: patch top-k layers TOGETHER (ranked by mean recovery).
  3. HEAD sweep   : within candidate layers, swap one head at a time.
  4. HEAD cumulative: patch top-k heads TOGETHER.
Verdict from the cumulative curves.
"""
import torch, json, sys, time, numpy as np
from binding_patch import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 30
LOG = open("results/kt_localization.log", "a")

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


def score(qg, img, p):
    """Single-object binding score using the pair's recorded grade object."""
    s, _ = qg.object_swap_score(img, p["grade_obj"], p["clean_color"], p["swap_color"])
    return s


def main(qual_path="results/qualified.json"):
    t0 = time.time()
    pairs = json.load(open(qual_path))
    log(f"\n===== SWEEP start: {len(pairs)} qualified pairs, {STEPS} steps =====")
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); layers = install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)
    L = len(layers)

    # ---- per-pair baselines + cache (latents, swap_embeds) ----
    ctx = []
    for p in pairs:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1,4,64,64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        b0, b1 = score(qg, ic, p), score(qg, ia, p)
        ctx.append(dict(p=p, lat=lat, se=se, b0=b0, b1=b1))
        log(f"  baseline pair{p['id']:2d} b0={b0:.2f} b1={b1:.2f} | {p['clean']}")
    # keep only pairs with a real flip
    valid = [c for c in ctx if (c["b1"] - c["b0"]) >= 0.5]
    log(f"  valid (b1-b0>=0.5): {len(valid)}/{len(ctx)}")

    def recov(c, s):
        return (s - c["b0"]) / (c["b1"] - c["b0"])

    def eval_config(label, **kw):
        rs = []
        for c in valid:
            img = generate(pipe, ctrl, c["p"]["clean"], c["se"], c["lat"].clone(), steps=STEPS, **kw)
            rs.append(recov(c, score(qg, img, c["p"])))
        m = float(np.mean(rs))
        return m, rs

    # ---- Phase 1: layer sweep ----
    log("\n--- PHASE 1: layer sweep ---")
    layer_rec = {}
    for ln in layers:
        m, _ = eval_config(ln, swap_layer=ln)
        layer_rec[ln] = m
        log(f"  LAYER {ln:55s} recovery={m:+.3f}")
    ranked_layers = sorted(layers, key=lambda l: layer_rec[l], reverse=True)
    log("\n  layers ranked by recovery:")
    for ln in ranked_layers:
        log(f"    {layer_rec[ln]:+.3f}  {ln}")

    # ---- Phase 2: layer cumulative ----
    log("\n--- PHASE 2: layer cumulative (top-k layers together) ---")
    layer_cum = []
    for k in range(1, L + 1):
        cells = [(ln, h) for ln in ranked_layers[:k] for h in range(ctrl.layer_heads[ln])]
        m, _ = eval_config(f"top{k}L", cells=cells)
        layer_cum.append(m)
        log(f"  top-{k:2d} layers ({k*8:3d} heads) cumulative recovery={m:+.3f}")

    # ---- Phase 3: FULL per-head sweep over ALL 128 cells (no pre-selection) ----
    log("\n--- PHASE 3: FULL head sweep (all cells) ---")
    head_rec = {}
    for li, ln in enumerate(layers):
        for h in range(ctrl.layer_heads[ln]):
            m, rs = eval_config(f"{ln}.h{h}", cells=[(ln, h)])
            head_rec[(ln, h)] = m
            log(f"  HEAD L{li:2d}.h{h} rec={m:+.3f}  {ln}")
    ranked_cells = sorted(head_rec, key=lambda c: head_rec[c], reverse=True)
    log("\n  top-15 cells by single-head recovery:")
    for c in ranked_cells[:15]:
        log(f"    {head_rec[c]:+.3f}  {c[0]} h{c[1]}")

    # ---- Phase 4: head cumulative (greedy by single-head rank, patched together) ----
    log("\n--- PHASE 4: head cumulative (top-k cells together) ---")
    ncells = len(ranked_cells)
    head_cum = []
    ks = sorted(set([1,2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,ncells]))
    ks = [k for k in ks if k <= ncells]
    for k in ks:
        cells = ranked_cells[:k]
        m, _ = eval_config(f"top{k}H", cells=cells)
        head_cum.append((k, m))
        log(f"  top-{k:3d} cells ({100*k/ncells:4.0f}% of heads) cumulative recovery={m:+.3f}")

    # ---- save + verdict ----
    total = sum(ctrl.layer_heads.values())
    nonzero = sum(1 for v in head_rec.values() if v > 0.05)
    # smallest k reaching 70% / 80% of full-swap
    def k_for(thr):
        for k, m in head_cum:
            if m >= thr:
                return k
        return None
    k70, k80 = k_for(0.70), k_for(0.80)
    out = dict(layers=layers, layer_rec=layer_rec, ranked_layers=ranked_layers,
               layer_cum=layer_cum,
               head_rec={f"{c[0]}|h{c[1]}": v for c, v in head_rec.items()},
               ranked_cells=[f"{c[0]}|h{c[1]}" for c in ranked_cells],
               head_cum=head_cum, n_valid=len(valid), total_heads=total,
               nonzero_heads=nonzero, k70=k70, k80=k80)
    json.dump(out, open("results/sweep_results.json", "w"), indent=2)

    log("\n===== SUMMARY =====")
    log(f"  valid pairs={len(valid)}; total cross-attn heads={total}")
    log(f"  #single-heads with recovery>0.05: {nonzero}/{total}")
    log(f"  layer cumulative curve: {[round(x,2) for x in layer_cum]}")
    log(f"  head cumulative curve: {[(k,round(m,2)) for k,m in head_cum]}")
    log(f"  smallest k (heads patched together) reaching 70% recovery: {k70}")
    log(f"  smallest k reaching 80% recovery: {k80}")
    frac80 = (k80/total) if k80 else None
    log(f"  fraction of heads needed for 80%: {frac80}")
    if k80 is not None and k80 <= 10:
        verdict = "LOCALIZED -> GO"
    elif k80 is None or (frac80 is not None and frac80 > 0.5):
        verdict = "DISTRIBUTED -> NO-GO"
    else:
        verdict = "INTERMEDIATE -> see curve"
    log(f"  VERDICT(auto-heuristic): {verdict}")
    log(f"  results -> results/sweep_results.json ; elapsed {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qualified.json")
