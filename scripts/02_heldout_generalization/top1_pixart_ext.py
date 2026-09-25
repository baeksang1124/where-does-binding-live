"""Power up the PixArt top-1 leg: qualify the extended candidate pairs on PixArt and
measure the top-1 head flip (per-object strict flip, mean over both objects), to tighten the
CI on the PixArt-SD3.5 gap (SD3.5 per-pair values from results/top1_sd3.json). The original
pairs (11 qualified in results/top1_pixart.json) are included as candidates and re-measured
under the same protocol; candidates are deduplicated by clean prompt.
Outputs: results/top1_pixart_ext.json, results/top1_pixart_ext.log. One GPU.
"""
import torch, json, os, numpy as np, re
from prompts_ext import make_ext_pairs
from prompts import make_pairs
from binding_patch_pixart import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 20; GUID = 4.5; H = W = 512
LOG = open("results/top1_pixart_ext.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(a):
    a = a.lower()
    from prompts import COLORS
    h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def grade2(qg, img, p):
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    return k1, k2


def latent(pipe, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    lc = pipe.transformer.config.in_channels
    return torch.randn((1, lc, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)


def main():
    os.makedirs("results", exist_ok=True)
    sw = json.load(open("results/pixart_sweep.json"))
    ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    # candidates = extended set + original pairs (to re-measure uniformly)
    ext = make_ext_pairs()
    orig = make_pairs()
    cand = ext + orig
    seen = set(); uniq = []
    for p in cand:
        if p["clean"] in seen:
            continue
        seen.add(p["clean"]); uniq.append(p)
    log(f"\n===== PixArt top-1 power-up: {len(uniq)} candidates, top1={sw['ranked'][0]} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)
    per = []; qualified_ids = []
    for p in uniq:
        emb, _ = encode(pipe, p["swapped"], DEVICE); lat = latent(pipe, p["seed"])
        A = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=None, steps=STEPS, guidance=GUID)
        S = generate(pipe, ctrl, p["clean"], emb, lat.clone(), swap_all=True, steps=STEPS, guidance=GUID)
        ca1, ca2 = grade2(qg, A, p); sa1, sa2 = grade2(qg, S, p)
        ok = (ca1 == p["c1"] and ca2 == p["c2"] and sa1 == p["c2"] and sa2 == p["c1"])
        if not ok:
            continue
        qualified_ids.append(p["id"])
        img = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True,
                       steps=STEPS, guidance=GUID)
        k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
        f1 = int(k1 == p["c2"] and k1 != p["c1"]); f2 = int(k2 == p["c1"] and k2 != p["c2"])
        per.append((f1 + f2) / 2.0)
        if len(per) % 5 == 0:
            log(f"  qualified so far: {len(per)}  running mean={np.mean(per):.3f}")

    per = [float(x) for x in per]; a = np.array(per); n = len(a)
    rng = np.random.default_rng(1234); idx = rng.integers(0, n, size=(100000, n))
    boots = a[idx].mean(1); lo, hi = np.percentile(boots, [2.5, 97.5])
    # difference vs SD3.5
    sd3 = np.array(json.load(open("results/top1_sd3.json"))["per_pair"])
    di = rng.integers(0, len(sd3), size=(100000, len(sd3)))
    diff = boots - sd3[di].mean(1)
    dlo, dhi = np.percentile(diff, [2.5, 97.5]); pg = (diff > 0).mean()
    out = dict(top1_cell=sw["ranked"][0], n=n, per_pair=per, qualified_ids=qualified_ids,
               mean=float(a.mean()), ci=[float(lo), float(hi)],
               diff_vs_sd3=float(a.mean() - sd3.mean()), diff_ci=[float(dlo), float(dhi)], p_gt0=float(pg))
    json.dump(out, open("results/top1_pixart_ext.json", "w"), indent=2)
    log(f"\nPixArt top-1: n={n} mean={a.mean():.3f} CI[{lo:.3f},{hi:.3f}]")
    log(f"PixArt-SD3.5 diff={a.mean()-sd3.mean():+.3f} CI[{dlo:+.3f},{dhi:+.3f}] P(>0)={pg:.3f}  "
        f"{'SEPARATED (excludes 0)' if dlo > 0 else 'still touches 0'}")


if __name__ == "__main__":
    main()
