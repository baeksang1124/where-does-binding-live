"""Phase 0: qualify prompt pairs (single-object binding flip).

A (pair, seed) is GRADABLE if, for at least one object o, the BASELINE clean image renders
o in its clean color and the swap-all image (== swapped-prompt image, proven) renders o in
its swapped color -- i.e. a real per-object color flip exists to localize. Selection is on
baseline generation only, independent of any head => does not bias the verdict.
We record which object to grade and its clean/swap target colors.
"""
import torch, json, sys, time
from binding_patch import load_pipe, install, encode, generate, Controller
from prompts import make_pairs

DEVICE = "cuda"; STEPS = 30
SEEDS_PER_PAIR = 10
LOGF = open("results/qualify.log", "a")

def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOGF.write(m+"\n"); LOGF.flush()


def main(max_pairs=None, out="results/qualified.json", seeds_per=SEEDS_PER_PAIR):
    t0 = time.time()
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    from grader import QwenGrader
    qg = QwenGrader(device=DEVICE)
    log(f"\n===== QUALIFY start {time.strftime('%H:%M:%S')} =====")
    log(f"[mem] {torch.cuda.memory_allocated()/1e9:.1f}GB alloc")

    pairs = make_pairs()
    if max_pairs: pairs = pairs[:max_pairs]
    qualified = []
    for p in pairs:
        se = encode(pipe, p["swapped"], DEVICE)
        # objects with their (clean_color, swap_color)
        obj_specs = [(p["o1"], p["c1"], p["c2"]), (p["o2"], p["c2"], p["c1"])]
        rec = None
        for s in range(seeds_per):
            seed = p["seed"] + 100000 * s
            g = torch.Generator(device=DEVICE).manual_seed(seed)
            lat = torch.randn((1,4,64,64), generator=g, device=DEVICE, dtype=torch.float16)
            ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
            ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
            for (obj, cc, sc) in obj_specs:
                sclean, ac = qg.object_swap_score(ic, obj, cc, sc)  # want 0.0
                sswap, asw = qg.object_swap_score(ia, obj, cc, sc)  # want 1.0
                if sclean == 0.0 and sswap == 1.0:
                    rec = dict(p); rec.update(seed=seed, grade_obj=obj,
                                              clean_color=cc, swap_color=sc)
                    ic.save(f"results/qual_clean_{p['id']}.png")
                    ia.save(f"results/qual_swap_{p['id']}.png")
                    log(f"  pair{p['id']:2d} seed_off{s} QUAL obj={obj} {cc}->{sc} "
                        f"(clean='{ac}' swap='{asw}') | {p['clean']}")
                    break
            if rec: break
        if rec is None:
            log(f"  pair{p['id']:2d} no gradable flip in {seeds_per} seeds | {p['clean']}")
        else:
            qualified.append(rec)
    json.dump(qualified, open(out, "w"), indent=2)
    log(f"QUALIFIED {len(qualified)}/{len(pairs)} in {time.time()-t0:.0f}s -> {out}")
    return qualified


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_pairs=mp)
