"""PixArt-Sigma localization sweep -- the CONTROL arm of the 2x2 design (backbone x attention type).

Question: is attribute binding LOCALIZED to few heads in PixArt (isolated cross-attn DiT,
like SD1.5) or DISTRIBUTED (like SD3.5 MM-DiT joint attn)? This adjudicates whether the
SD1.5->SD3.5 dissolution is driven by ATTENTION COUPLING (joint vs isolated) rather than
UNet-vs-Transformer or scale.

Three passes (one script, incremental logging):
  QUALIFY  per pair: clean binds (o1=c1,o2=c2 distinct) AND swapped binds (o1=c2,o2=c1).
  COARSE   per qualified pair x 28 blocks: swap ALL 16 heads of one block -> block flip.
  PERHEAD  first N_PH qualified pairs x 448 heads: swap one head K&V -> per-head flip;
           rank heads; then cumulative top-k head curve. LOCALIZED iff small k >=70%.

Fixed-embedding regime (no activation cache): swap_embeds = caption_projection(swapped T5),
reused every block/step. generate() takes per-pair latents from the seed.
Metric: two-object flip (0/.5/1) + proper (both distinct rebound). One GPU.
"""
import torch, json, time, os, numpy as np, re
from collections import defaultdict
from prompts import make_pairs, COLORS
from binding_patch_pixart import load_pipe, install, Controller, encode, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 20; GUID = 4.5; H = W = 512
N_PH = int(os.environ.get("PX_NPH", "8"))          # pairs for per-head pass
NPAIRS = int(os.environ.get("PX_NPAIRS", "0"))     # limit qualify candidates (0=all)
KS = [1, 2, 4, 8, 16, 32, 64, 128]
LOG = open("results/pixart_sweep.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2); f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    proper = int(k1 == p["c2"] and k2 == p["c1"])
    return f, proper, (a1, a2)


def latent_for(pipe, seed):
    lc = pipe.transformer.config.in_channels
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn((1, lc, H // 8, W // 8), generator=g, device=DEVICE, dtype=torch.float16)


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    pairs = make_pairs()
    if NPAIRS:
        pairs = pairs[:NPAIRS]
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); names = install(pipe, ctrl)
    blocks = names                                   # 28 block layer-names
    all_cells = [(ln, h) for ln in blocks for h in range(ctrl.layer_heads[ln])]
    log(f"\n===== PixArt sweep: {len(pairs)} candidates, {len(blocks)} blocks, "
        f"{len(all_cells)} heads, {STEPS} steps =====")
    qg = QwenGrader(device=DEVICE)

    # ---- cache per-pair latent + swapped projected embed ----
    ctx = {}
    for p in pairs:
        emb, _ = encode(pipe, p["swapped"], DEVICE)
        ctx[p["id"]] = dict(p=p, lat=latent_for(pipe, p["seed"]), emb=emb)

    # ---- QUALIFY ----
    qualified = []
    for p in pairs:
        c = ctx[p["id"]]
        A = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=None,
                     steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        Simg = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), swap_all=True,
                        steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        _, _, (ca1, ca2) = grade(qg, A, p)
        _, _, (sa1, sa2) = grade(qg, Simg, p)
        ok = (canon(ca1) == p["c1"] and canon(ca2) == p["c2"]
              and canon(sa1) == p["c2"] and canon(sa2) == p["c1"])
        log(f"pair{p['id']:2d} clean=({ca1},{ca2}) swap=({sa1},{sa2}) -> {'QUALIFY' if ok else 'skip'}")
        if ok:
            qualified.append(p["id"])
    log(f"\nqualified {len(qualified)}/{len(pairs)}: {qualified}")

    # ---- COARSE block sweep ----
    log(f"\n--- COARSE: per-block (all 16 heads) flip, {len(qualified)} pairs ---")
    block_flip = defaultdict(list)
    for pid in qualified:
        c = ctx[pid]; p = c["p"]
        for ln in blocks:
            cells = [(ln, h) for h in range(ctrl.layer_heads[ln])]
            img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=cells,
                           swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                           height=H, width=W, device=DEVICE)
            f, _, _ = grade(qg, img, p)
            block_flip[ln].append(f)
    block_mean = {ln: float(np.mean(block_flip[ln])) for ln in blocks}
    ranked_blocks = sorted(blocks, key=lambda b: block_mean[b], reverse=True)
    log("  top blocks by flip:")
    for ln in ranked_blocks[:8]:
        log(f"    {ln}: {block_mean[ln]:.3f}")

    # ---- PER-HEAD sweep (first N_PH qualified pairs) ----
    ph_pairs = qualified[:N_PH]
    log(f"\n--- PER-HEAD: {len(all_cells)} heads x {len(ph_pairs)} pairs ---")
    head_flip = defaultdict(list)
    for pid in ph_pairs:
        c = ctx[pid]; p = c["p"]
        for cell in all_cells:
            img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=[cell],
                           swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                           height=H, width=W, device=DEVICE)
            f, _, _ = grade(qg, img, p)
            head_flip[cell].append(f)
        means = {cc: float(np.mean(head_flip[cc])) for cc in head_flip}
        top = sorted(means, key=means.get, reverse=True)[:5]
        log(f"pair{pid:2d} done; running top heads: "
            + ", ".join(f"{ln.split('.')[1]}h{h}={means[(ln,h)]:.2f}" for (ln, h) in top))

    head_mean = {cc: float(np.mean(head_flip[cc])) for cc in all_cells}
    ranked = sorted(all_cells, key=lambda cc: head_mean[cc], reverse=True)
    nonzero = [cc for cc in all_cells if head_mean[cc] > 0.05]
    log(f"\n--- per-head top 20 (of {len(all_cells)}) ; nonzero(>0.05): {len(nonzero)} ---")
    for cc in ranked[:20]:
        log(f"    {cc[0]} h{cc[1]}: {head_mean[cc]:.3f}")

    # ---- CUMULATIVE top-k head curve (on the per-head pairs) ----
    log(f"\n--- CUMULATIVE top-k heads ; KS={KS} ---")
    cum = {k: [] for k in KS}
    for pid in ph_pairs:
        c = ctx[pid]; p = c["p"]
        for k in KS:
            img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=ranked[:k],
                           swap_k=True, swap_v=True, steps=STEPS, guidance=GUID,
                           height=H, width=W, device=DEVICE)
            f, _, _ = grade(qg, img, p)
            cum[k].append(f)
        log(f"pair{pid:2d} cumulative: " + ", ".join(f"k{k}={cum[k][-1]:.2f}" for k in KS))
    cum_mean = {k: float(np.mean(cum[k])) for k in KS}
    log(f"\n--- mean cumulative flip vs top-k HEADS (ref=1.0 by qualify) ---")
    for k in KS:
        log(f"  top-{k:3d}: flip={cum_mean[k]:.3f} ({cum_mean[k]*100:.0f}%)")
    k70 = next((k for k in KS if cum_mean[k] >= 0.70), None)
    verdict = (f"LOCALIZED (GO): top-{k70} heads reach >=70%"
               if (k70 is not None and k70 <= 16)
               else f"NOT localized: needs >{KS[-1] if k70 is None else k70} heads for 70%")
    log(f"\n  VERDICT: {verdict}")

    json.dump(dict(n_qualified=len(qualified), qualified=qualified,
                   block_mean={ln: block_mean[ln] for ln in blocks},
                   ranked_blocks=ranked_blocks,
                   head_mean={f"{ln}|h{h}": head_mean[(ln, h)] for (ln, h) in all_cells},
                   ranked=[f"{ln}|h{h}" for (ln, h) in ranked], n_nonzero=len(nonzero),
                   ph_pairs=ph_pairs, ks=KS, cum_mean=cum_mean, verdict=verdict),
              open("results/pixart_sweep.json", "w"), indent=2)
    log(f"\nresults -> results/pixart_sweep.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
