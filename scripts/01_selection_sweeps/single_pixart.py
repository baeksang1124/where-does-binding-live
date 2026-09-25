"""PixArt localization re-analysis with the SINGLE-OBJECT flip metric (apples-to-apples
with SD1.5, which was declared LOCALIZED via single-object recovery top-1=0.33/top-8=0.70).

The two-object flip in sweep_pixart.py caps localization: a single head typically flips
ONE object's color; requiring BOTH understates the sparse handle. Here the metric is the
per-OBJECT-INSTANCE flip: over all (pair, object) instances, fraction now reading the
swapped color. Reuses the head ranking already in pixart_sweep.json (no re-ranking).

Passes:
  PERHEAD-single: top M ranked heads, single-head swap, per-object flip (refine the sparse set).
  CUMULATIVE-single: top-k heads (global ranking) -> per-object-instance flip curve.
One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_pixart import load_pipe, install, Controller, encode, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 20; GUID = 4.5; H = W = 512
KS = [1, 2, 4, 8, 16, 32, 64, 128]
TOPM = int(os.environ.get("PX_TOPM", "24"))
LOG = open("results/pixart_single.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def obj_flips(qg, img, p):
    """Return (o1_flipped, o2_flipped) as 0/1 ints: object reads its SWAPPED color."""
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    f1 = int(k1 == p["c2"] and k1 != p["c1"])
    f2 = int(k2 == p["c1"] and k2 != p["c2"])
    return f1, f2


def parse(s):
    ln, h = s.split("|h"); return (ln, int(h))


def latent_for(pipe, seed):
    lc = pipe.transformer.config.in_channels
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn((1, lc, H // 8, W // 8), generator=g, device=DEVICE, dtype=torch.float16)


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    sw = json.load(open("results/pixart_sweep.json"))
    ranked = [parse(s) for s in sw["ranked"]]
    qids = sw["qualified"]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== PixArt SINGLE-OBJECT re-analysis: {len(qids)} pairs, top1={ranked[0]} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    ctx = {}
    for pid in qids:
        p = pairs[pid]
        emb, _ = encode(pipe, p["swapped"], DEVICE)
        ctx[pid] = dict(p=p, lat=latent_for(pipe, p["seed"]), emb=emb)

    # ---- PER-HEAD single-object on top M ranked heads ----
    log(f"\n--- PER-HEAD single-object flip, top {TOPM} ranked heads ---")
    head_inst = {}   # cell -> list of per-instance flips (2 per pair)
    for cell in ranked[:TOPM]:
        inst = []
        for pid in qids:
            c = ctx[pid]; p = c["p"]
            img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=[cell],
                           swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                           height=H, width=W, device=DEVICE)
            f1, f2 = obj_flips(qg, img, p)
            inst += [f1, f2]
        head_inst[cell] = inst
        log(f"  {cell[0]} h{cell[1]}: inst-flip={np.mean(inst):.3f} "
            f"(sum {int(np.sum(inst))}/{len(inst)})")

    # ---- CUMULATIVE single-object with existing global ranking ----
    log(f"\n--- CUMULATIVE top-k heads, single-object-instance flip ; KS={KS} ---")
    cum = {k: [] for k in KS}
    for pid in qids:
        c = ctx[pid]; p = c["p"]
        for k in KS:
            img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=ranked[:k],
                           swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                           height=H, width=W, device=DEVICE)
            f1, f2 = obj_flips(qg, img, p)
            cum[k] += [f1, f2]
        log(f"pair{pid:2d}: " + ", ".join(f"k{k}={np.mean(cum[k][-2:]):.1f}" for k in KS))
    cum_mean = {k: float(np.mean(cum[k])) for k in KS}
    log(f"\n--- mean single-object-instance flip vs top-k HEADS ---")
    for k in KS:
        log(f"  top-{k:3d}: {cum_mean[k]:.3f} ({cum_mean[k]*100:.0f}%)")
    k70 = next((k for k in KS if cum_mean[k] >= 0.70), None)
    verdict = (f"LOCALIZED (single-obj): top-{k70} heads reach >=70%"
               if (k70 is not None and k70 <= 16)
               else f"partial: needs >{KS[-1] if k70 is None else k70} heads for 70% single-obj")
    log(f"\n  VERDICT: {verdict}")

    json.dump(dict(metric="single_object_instance_flip", qualified=qids, topm=TOPM,
                   head_inst_mean={f"{ln}|h{h}": float(np.mean(v)) for (ln, h), v in head_inst.items()},
                   ks=KS, cum_mean=cum_mean, verdict=verdict),
              open("results/pixart_single.json", "w"), indent=2)
    log(f"\nresults -> results/pixart_single.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
