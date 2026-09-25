"""PixArt A4: OV/QK (content vs routing) decomposition on the isolated-DiT binding core.

Parallels the SD1.5 A4 (a4_decomp.json: top head is ROUTING). Asks whether the PixArt core
heads (top-ranked, e.g. block13 h7 / block11 h1) carry binding via the attention PATTERN
(K / routing) or the VALUE (V / content). Wang et al. (2601.06338) trace QK and VO circuits;
DifFRACT uses transcoders (no OV/QK split). Reuses binding_patch_pixart (independent swap_k/swap_v).

Conditions per head-set: FULL(K,V) PATTERN(K) VALUE(V) BASELINE(none). Single-object flip on
the graded object; bootstrap 95% CI on [PATTERN - VALUE]. One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_pixart import load_pipe, install, Controller, encode, generate
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 20; GUID = 4.5; H = W = 512
NBOOT = 10000; rng = np.random.default_rng(1234)
LOG = open("results/pixart_a4.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def obj_flip(qg, img, p):
    """single-object flip on o1 (queried object): 1 if reads c2, 0 if c1, 0.5 ambiguous."""
    a = qg._ask_color(img, p["o1"]); k = canon(a)
    if k == p["c2"] and k != p["c1"]:
        return 1.0
    if k == p["c1"] and k != p["c2"]:
        return 0.0
    return 0.5


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
    head_sets = {"top1": ranked[:1], "top2": ranked[:2]}
    CONDS = {"FULL": (True, True), "PATTERN": (True, False),
             "VALUE": (False, True), "BASELINE": (False, False)}
    log(f"\n===== PixArt A4 OV/QK: {len(qids)} pairs; top1={ranked[0]} top2={ranked[1]} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)

    ctx = {}
    for pid in qids:
        p = pairs[pid]; emb, _ = encode(pipe, p["swapped"], DEVICE)
        ctx[pid] = dict(p=p, lat=latent_for(pipe, p["seed"]), emb=emb)

    res = {hn: {c: [] for c in CONDS} for hn in head_sets}
    for pid in qids:
        c = ctx[pid]; p = c["p"]
        for hn, cells in head_sets.items():
            for cond, (sk, sv) in CONDS.items():
                img = generate(pipe, ctrl, p["clean"], c["emb"], c["lat"].clone(), cells=cells,
                               swap_k=sk, swap_v=sv, steps=STEPS, guidance=GUID,
                               height=H, width=W, device=DEVICE)
                res[hn][cond].append(obj_flip(qg, img, p))
    log(f"graded {len(qids)} pairs x {len(head_sets)} sets x 4 conds")

    out = {"top1": sw["ranked"][0], "n": len(qids), "sets": {}}
    for hn in head_sets:
        means = {c: float(np.mean(res[hn][c])) for c in CONDS}
        P = np.array(res[hn]["PATTERN"]); V = np.array(res[hn]["VALUE"]); n = len(P)
        idx = rng.integers(0, n, size=(NBOOT, n)); boots = P[idx].mean(1) - V[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5]); diff = float(P.mean() - V.mean())
        rng_ = max(means["FULL"] - means["BASELINE"], 1e-6)
        fp = (means["PATTERN"] - means["BASELINE"]) / rng_
        fv = (means["VALUE"] - means["BASELINE"]) / rng_
        v = ("ROUTING (QK)" if fp >= 0.6 and fv <= 0.4 else
             "CONTENT (OV)" if fv >= 0.6 and fp <= 0.4 else "MIXED")
        log(f"  [{hn}] FULL={means['FULL']:.3f} PATTERN={means['PATTERN']:.3f} "
            f"VALUE={means['VALUE']:.3f} BASE={means['BASELINE']:.3f}")
        log(f"        PATTERN-VALUE={diff:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] -> {v} "
            f"(routing~{fp*100:.0f}%, content~{fv*100:.0f}%)")
        out["sets"][hn] = dict(means=means, pattern_minus_value=diff, ci=[float(lo), float(hi)],
                               frac_routing=fp, frac_content=fv, verdict=v)
    json.dump(out, open("results/pixart_a4.json", "w"), indent=2)
    log(f"results -> results/pixart_a4.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
