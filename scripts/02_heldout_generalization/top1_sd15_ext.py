"""SD1.5 held-out generalization test of the top-1 head (does 0.33 survive out-of-sample,
or was it selection bias like PixArt?). Single-object protocol on o1 (SD1.5 barely binds two
objects). Extended pairs (held-out) vs original. One GPU.
"""
import torch, json, os, numpy as np
from prompts_ext import make_ext_pairs
from prompts import make_pairs, COLORS
from binding_patch import load_pipe, install, encode, generate, Controller
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 30
LOG = open("results/top1_sd15_ext.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def main():
    os.makedirs("results", exist_ok=True)
    sweep = json.load(open("results/sweep_results.json"))
    c = sweep["ranked_cells"][0]; ln, h = c.split("|h"); top1 = [(ln, int(h))]
    ext = make_ext_pairs(); orig = make_pairs()
    for p in orig:  # give orig pairs the single-object fields
        p.setdefault("grade_obj", p["o1"])
    cand = []
    seen = set()
    for p in ext + orig:
        if p["clean"] in seen:
            continue
        seen.add(p["clean"])
        cand.append(dict(p, grade_obj=p["o1"], clean_color=p["c1"], swap_color=p["c2"]))
    log(f"\n===== SD1.5 held-out top-1: {len(cand)} candidates, top1={c} =====")

    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    qg = QwenGrader(device=DEVICE)
    per, ids = [], []
    for p in cand:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        b0, _ = qg.object_swap_score(ic, p["grade_obj"], p["clean_color"], p["swap_color"])
        b1, _ = qg.object_swap_score(ia, p["grade_obj"], p["clean_color"], p["swap_color"])
        if not (b0 == 0.0 and b1 == 1.0):
            continue
        img = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=STEPS)
        s, _ = qg.object_swap_score(img, p["grade_obj"], p["clean_color"], p["swap_color"])
        per.append(float(s)); ids.append(p["id"])
        if len(per) % 8 == 0:
            log(f"  qualified {len(per)}  running mean={np.mean(per):.3f}")

    a = np.array(per); n = len(a); rng = np.random.default_rng(1234)
    ext_v = [v for i, v in zip(ids, per) if i >= 5000]; ins_v = [v for i, v in zip(ids, per) if i < 5000]

    def ci(x):
        x = np.array(x, float)
        if len(x) == 0:
            return (0, 0, 0)
        b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return (float(x.mean()), *np.percentile(b, [2.5, 97.5]))
    log(f"COMBINED  n={n} mean={ci(per)[0]:.3f} CI[{ci(per)[1]:.3f},{ci(per)[2]:.3f}]")
    log(f"HELD-OUT  n={len(ext_v)} mean={ci(ext_v)[0]:.3f} CI[{ci(ext_v)[1]:.3f},{ci(ext_v)[2]:.3f}]")
    log(f"IN-SAMPLE n={len(ins_v)} mean={ci(ins_v)[0]:.3f} CI[{ci(ins_v)[1]:.3f},{ci(ins_v)[2]:.3f}]")
    json.dump(dict(top1_cell=c, n=n, per_pair=per, ids=ids,
                   combined=ci(per), held_out=ci(ext_v), in_sample=ci(ins_v)),
              open("results/top1_sd15_ext.json", "w"), indent=2)


if __name__ == "__main__":
    main()
