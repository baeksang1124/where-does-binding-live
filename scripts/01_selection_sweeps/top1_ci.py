"""Per-pair top-1-head single-object flip for each model, so we can put a PAIR-CLUSTERED
bootstrap CI on the cross-model top-1-head contrast. MODEL env: sd15|pixart|sd3.
Stores per-pair flips -> results/top1_{model}.json. One GPU.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from grader import QwenGrader

DEVICE = "cuda"; MODEL = os.environ.get("MODEL", "sd15")


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def obj_flips(qg, img, p):
    """per-object flips (o1,o2): 1 if reads swapped color."""
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    return (int(k1 == p["c2"] and k1 != p["c1"]), int(k2 == p["c1"] and k2 != p["c2"]))


def run_sd15():
    from binding_patch import load_pipe, install, encode, generate, Controller
    sweep = json.load(open("results/sweep_results.json"))
    cell = sweep["ranked_cells"][0]; ln, h = cell.split("|h"); top1 = [(ln, int(h))]
    qp = json.load(open("results/qualified.json"))
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)
    per = []
    for p in qp:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        img = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=30)
        s, _ = qg.object_swap_score(img, p["grade_obj"], p["clean_color"], p["swap_color"])
        per.append(s)
    return cell, per


def run_pixart():
    from binding_patch_pixart import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/pixart_sweep.json"))
    ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    pairs = {p["id"]: p for p in make_pairs()}
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE); per = []
    for pid in sw["qualified"]:
        p = pairs[pid]
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lc = pipe.transformer.config.in_channels
        lat = torch.randn((1, lc, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        emb, _ = encode(pipe, p["swapped"], DEVICE)
        img = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True,
                       steps=20, guidance=4.5)
        f1, f2 = obj_flips(qg, img, p); per.append((f1 + f2) / 2.0)
    return sw["ranked"][0], per


def run_sd3():
    from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
    hs = json.load(open("results/sd3_headsweep.json"))
    b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    coarse = json.load(open("results/sd3_coarse.json"))
    qids = coarse["qualified"][:12]; pairs = {p["id"]: p for p in make_pairs()}
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE); per = []
    for pid in qids:
        p = pairs[pid]
        capture_swapped(pipe, ctrl, p["swapped"], p["seed"], steps=28, guidance=7.0, height=512, width=512, device=DEVICE)
        img = generate(pipe, ctrl, p["clean"], p["seed"], cells=top1, swap_k=True, swap_v=True,
                       steps=28, guidance=7.0, height=512, width=512, device=DEVICE)
        f1, f2 = obj_flips(qg, img, p); per.append((f1 + f2) / 2.0); ctrl.swap_ctx = {}
    return hs["ranked"][0], per


def main():
    os.makedirs("results", exist_ok=True)
    cell, per = {"sd15": run_sd15, "pixart": run_pixart, "sd3": run_sd3}[MODEL]()
    per = [float(x) for x in per]
    rng = np.random.default_rng(1234); a = np.array(per); n = len(a)
    idx = rng.integers(0, n, size=(10000, n)); boots = a[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = dict(model=MODEL, top1_cell=cell, n=n, per_pair=per,
               mean=float(a.mean()), ci=[float(lo), float(hi)])
    json.dump(out, open(f"results/top1_{MODEL}.json", "w"), indent=2)
    print(f"[{MODEL}] top1={cell} n={n} mean={a.mean():.3f} 95%CI[{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
