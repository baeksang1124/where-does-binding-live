"""EXP A4b -- color-BLEED vs genuine two-object RE-BINDING control for the OV/QK decomp.

A4 (results/a4_decomp.json) graded ONLY one object (grade_obj = o1) with object_swap_score.
On the top-16 head set the VALUE-only route (K=clean, V=swapped) showed ~0.46 single-object
"recovery". Suspected confound: injecting swapped VALUEs across many heads can globally TINT
the image toward the swapped palette / collapse both objects to one color, so the single
object o1 reads the swap color WITHOUT a genuine, distinct two-object re-binding.

This experiment regrades with a TWO-OBJECT joint grader and separates:
  PROPER_REBIND -- o1 reads its swapped color (c2) AND o2 reads its swapped color (c1),
                   two DISTINCT correct re-assignments == genuine binding.
  BLEED/COLLAPSE -- o1 turned to the swap palette but o2 did NOT hold its complementary
                    distinct color (covers COLLAPSE: both objects the SAME color, and muddy
                    drift). Operationally a case the OLD single-object metric counted as a
                    flip that is NOT a clean distinct two-color swap.

Conditions (per head set top1 / top16): FULL(K,V) PATTERN(K) VALUE(V) BASELINE(none).
Reuses binding_patch.generate (independent swap_k/swap_v), QwenGrader, qualified.json, and
the EXACT top1/top16 head sets recorded in results/a4_decomp.json. Same seeds/latents as A4.

Verdict: VALUE high-BLEED / low-PROPER_REBIND while PATTERN high-PROPER_REBIND
  => the VALUE recovery was a bleed artifact => binding is ROUTING (clean headline).
  VALUE genuine PROPER_REBIND comparable to PATTERN => honest HYBRID.
"""
import torch, json, time, os, numpy as np, re
from PIL import Image
from binding_patch import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"
STEPS = 30
COLORS = ["red", "blue", "green", "yellow"]
GRID_PAIR_IDS = [0, 10, 14, 22]  # pairs the A4 top1 FULL flipped -> good visual bleed/rebind contrast
LOG = open("results/a4b_bleed.log", "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


def canon(ans):
    """Map a free-text color answer to one base color, or 'other'."""
    ans = ans.lower()
    hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"  # exactly one base color, else 'other'


@torch.no_grad()
def grade_two(qg, img, p):
    """Two-object joint grade. clean: o1=c1,o2=c2 ; swapped target: o1=c2,o2=c1.
    Returns dict with canon answers + the three metrics for this image."""
    a1 = qg._ask_color(img, p["o1"])
    a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    swap1, swap2 = p["c2"], p["c1"]          # swapped-prompt expected colors
    clean1, clean2 = p["c1"], p["c2"]

    proper = int(k1 == swap1 and k2 == swap2)                 # genuine distinct re-bind
    collapse = int(k1 == k2 and k1 in COLORS)                 # both objects same real color
    o1_flipped = int(k1 == swap1)                             # target object turned swap color
    # BLEED: target flipped to swap palette but partner did NOT cleanly hold its distinct
    # complementary color -> not a clean two-color swap (includes collapse & muddy drift).
    bleed = int(o1_flipped and k2 != swap2)

    # OLD single-object flip score (reproduces object_swap_score on grade_obj=o1)
    if swap1 in a1 and clean1 not in a1:
        old = 1.0
    elif clean1 in a1 and swap1 not in a1:
        old = 0.0
    else:
        old = 0.5
    return dict(a1=a1, a2=a2, k1=k1, k2=k2, proper=proper, collapse=collapse,
                bleed=bleed, old=old)


def parse_cell(s):
    ln, h = s.split("|h")
    return (ln, int(h))


def main():
    t0 = time.time()
    os.makedirs("results/a4b_grids", exist_ok=True)
    pairs = json.load(open("results/qualified.json"))
    a4 = json.load(open("results/a4_decomp.json"))
    head_sets = {k: [parse_cell(c) for c in v] for k, v in a4["head_sets"].items()}
    CONDS = {"FULL": (True, True), "PATTERN": (True, False),
             "VALUE": (False, True), "BASELINE": (False, False)}

    log(f"\n===== A4b color-bleed control: {len(pairs)} pairs, {STEPS} steps =====")
    log(f"  top1 : {a4['head_sets']['top1']}")
    log(f"  top16: {len(head_sets['top16'])} heads")
    log("  BLEED rule: o1 reads swap color c2 AND o2 != its swap color c1 "
        "(target flipped but no clean distinct partner; includes collapse).")
    log("  PROPER_REBIND rule: o1==c2 AND o2==c1 (two distinct correct re-assignments).")
    log("  grader Qs: 'What color is the {obj} in this image? Answer with a single color word.' x2")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    # cache latents + swapped embeds per pair (identical recipe to A4)
    ctx = []
    for p in pairs:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ctx.append(dict(p=p, lat=lat, se=se))

    # BASELINE is head-set-independent (no swap) -> compute once, reuse for both sets.
    log("\n--- BASELINE (no swap; clean prompt) graded once ---")
    base_g = []
    for c in ctx:
        img = generate(pipe, ctrl, c["p"]["clean"], c["se"], c["lat"].clone(),
                       cells=head_sets["top1"], swap_k=False, swap_v=False, steps=STEPS)
        base_g.append(grade_two(qg, img, c["p"]))

    metrics = ["proper", "bleed", "collapse", "old"]
    results = {}
    grid_imgs = {pid: {} for pid in GRID_PAIR_IDS}

    for hs_name, cells in head_sets.items():
        log(f"\n--- HEAD SET: {hs_name} ({len(cells)} head(s)) ---")
        results[hs_name] = {}
        for cond, (sk, sv) in CONDS.items():
            if cond == "BASELINE":
                grades = base_g
            else:
                grades = []
                for c in ctx:
                    img = generate(pipe, ctrl, c["p"]["clean"], c["se"], c["lat"].clone(),
                                   cells=cells, swap_k=sk, swap_v=sv, steps=STEPS)
                    grades.append(grade_two(qg, img, c["p"]))
                    if hs_name == "top16" and c["p"]["id"] in GRID_PAIR_IDS:
                        grid_imgs[c["p"]["id"]][cond] = img.copy()
            agg = {m: float(np.mean([g[m] for g in grades])) for m in metrics}
            results[hs_name][cond] = dict(agg=agg, per=[{k: g[k] for k in
                ("a1", "a2", "proper", "bleed", "old")} for g in grades])
            log(f"  {cond:9s} PROPER_REBIND={agg['proper']:.3f}  BLEED={agg['bleed']:.3f}  "
                f"COLLAPSE={agg['collapse']:.3f}  old_flip={agg['old']:.3f}")

    # ---- save example FULL/PATTERN/VALUE grids (top16) for visual bleed-vs-rebind check ----
    log("\n--- saving example grids (top16: FULL | PATTERN | VALUE) ---")
    for pid in GRID_PAIR_IDS:
        if set(["FULL", "PATTERN", "VALUE"]).issubset(grid_imgs[pid]):
            cols = [grid_imgs[pid][c] for c in ["FULL", "PATTERN", "VALUE"]]
            w, h = cols[0].size
            canvas = Image.new("RGB", (w * 3 + 20, h), "white")
            for i, im in enumerate(cols):
                canvas.paste(im, (i * (w + 10), 0))
            path = f"results/a4b_grids/grid_pair{pid}_FULL_PATTERN_VALUE.png"
            canvas.save(path)
            p = pairs[pid]
            log(f"    pair{pid}: clean='{p['clean']}' swap='{p['swapped']}' -> {path}")

    # ---- verdict ----
    log("\n===== VERDICT (top16 is where A4 saw the VALUE recovery) =====")
    def row(hs, cond):
        return results[hs][cond]["agg"]
    for hs in head_sets:
        F, P, V, B = (row(hs, c) for c in ("FULL", "PATTERN", "VALUE", "BASELINE"))
        log(f"  [{hs}] PROPER_REBIND  FULL={F['proper']:.3f} PATTERN={P['proper']:.3f} "
            f"VALUE={V['proper']:.3f} BASELINE={B['proper']:.3f}")
        log(f"  [{hs}] BLEED          FULL={F['bleed']:.3f} PATTERN={P['bleed']:.3f} "
            f"VALUE={V['bleed']:.3f} BASELINE={B['bleed']:.3f}")
        log(f"  [{hs}] old_flip       FULL={F['old']:.3f} PATTERN={P['old']:.3f} "
            f"VALUE={V['old']:.3f} BASELINE={B['old']:.3f}")

    Ph = row("top16", "PATTERN"); Vh = row("top16", "VALUE")
    if Vh["proper"] <= 0.5 * Ph["proper"] + 1e-9 and Vh["bleed"] >= Vh["proper"]:
        verdict = ("VALUE recovery is mostly COLOR-BLEED (low proper-rebind, bleed-dominated) "
                   "while PATTERN carries proper-rebind => BINDING IS ROUTING (clean headline).")
    elif Vh["proper"] >= 0.7 * Ph["proper"]:
        verdict = ("VALUE shows genuine two-object PROPER_REBIND comparable to PATTERN "
                   "=> binding is a HYBRID routing+content (honest, less clean headline).")
    else:
        verdict = ("MIXED: VALUE shows partial genuine rebind and partial bleed.")
    log("\n  VERDICT: " + verdict)

    out = dict(head_sets=a4["head_sets"], conds=list(CONDS), metrics=metrics,
               n_pairs=len(pairs),
               agg={hs: {c: results[hs][c]["agg"] for c in CONDS} for hs in head_sets},
               per_pair={hs: {c: results[hs][c]["per"] for c in CONDS} for hs in head_sets},
               grid_pair_ids=GRID_PAIR_IDS, verdict=verdict)
    json.dump(out, open("results/a4b_bleed.json", "w"), indent=2)
    log(f"\n  results -> results/a4b_bleed.json ; elapsed {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
