"""Small dual-grader check of the top-1 head contrast between SD1.5, PixArt and SD3.5.
Regenerates the top-1-head-swap image for SD1.5 / PixArt on held-out pairs (ids >= 5000) and for
SD3.5 on the first N_PAIRS base (in-sample) pairs of results/sd3_coarse.json, saves them, then
grades each with Qwen2.5-VL-7B AND LLaVA-1.5-7B to check the contrast is not specific to one
grader. The full held-out dual grading is dual_grade_full.py. Single-object flip on o1.
Memory-safe (models never co-resident). One GPU.
"""
import torch, json, os, gc, numpy as np, re
from PIL import Image
from prompts import make_pairs, COLORS
from prompts_ext import make_ext_pairs

DEVICE = "cuda"; IMGDIR = "results/validate_top1_imgs"; os.makedirs(IMGDIR, exist_ok=True)
N = int(os.environ.get("N_PAIRS", "8"))


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def held_out_ids(jf, key):
    ids = json.load(open(jf))[key]
    return [i for i in ids if i >= 5000][:N]


def pairs_by_id():
    d = {p["id"]: p for p in make_pairs()}
    d.update({p["id"]: p for p in make_ext_pairs()})
    return d


def gen_sd15(meta):
    from binding_patch import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/sweep_results.json")); ln, h = sw["ranked_cells"][0].split("|h"); top1 = [(ln, int(h))]
    ids = held_out_ids("results/top1_sd15_ext.json", "ids"); P = pairs_by_id()
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        p = P[pid]; g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        img = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=30)
        img.save(f"{IMGDIR}/sd15_{pid}.png"); meta.append(dict(model="sd15", pid=pid, o1=p["o1"], c1=p["c1"], c2=p["c2"]))
    del pipe; gc.collect(); torch.cuda.empty_cache()


def gen_pixart(meta):
    from binding_patch_pixart import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/pixart_sweep.json")); ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    ids = held_out_ids("results/top1_pixart_ext.json", "qualified_ids"); P = pairs_by_id()
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        p = P[pid]; g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lc = pipe.transformer.config.in_channels
        lat = torch.randn((1, lc, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        emb, _ = encode(pipe, p["swapped"], DEVICE)
        img = generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=20, guidance=4.5)
        img.save(f"{IMGDIR}/pixart_{pid}.png"); meta.append(dict(model="pixart", pid=pid, o1=p["o1"], c1=p["c1"], c2=p["c2"]))
    del pipe; gc.collect(); torch.cuda.empty_cache()


def gen_sd3(meta):
    from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    ids = json.load(open("results/sd3_coarse.json"))["qualified"][:N]; P = pairs_by_id()
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        p = P[pid]
        capture_swapped(pipe, ctrl, p["swapped"], p["seed"], steps=28, guidance=7.0, height=512, width=512, device=DEVICE)
        img = generate(pipe, ctrl, p["clean"], p["seed"], cells=top1, swap_k=True, swap_v=True, steps=28, guidance=7.0, height=512, width=512, device=DEVICE)
        img.save(f"{IMGDIR}/sd3_{pid}.png"); meta.append(dict(model="sd3", pid=pid, o1=p["o1"], c1=p["c1"], c2=p["c2"])); ctrl.swap_ctx = {}
    del pipe; gc.collect(); torch.cuda.empty_cache()


def flip1(grader, img, m):
    k = canon(grader._ask_color(img, m["o1"]))
    if k == m["c2"] and k != m["c1"]:
        return 1.0
    if k == m["c1"] and k != m["c2"]:
        return 0.0
    return 0.5


def grade(grader, meta):
    return {(m["model"], m["pid"]): flip1(grader, Image.open(f"{IMGDIR}/{m['model']}_{m['pid']}.png"), m) for m in meta}


def main():
    meta = []
    gen_sd15(meta); gen_pixart(meta); gen_sd3(meta)
    print(f"[gen] {len(meta)} images")
    from grader import QwenGrader
    qg = QwenGrader(device=DEVICE); q = grade(qg, meta); del qg; gc.collect(); torch.cuda.empty_cache()
    from grader import LlavaGrader
    lg = LlavaGrader(device=DEVICE); l = grade(lg, meta); del lg; gc.collect(); torch.cuda.empty_cache()

    out = {}
    print(f"\n=== top-1 held-out, dual grader ===")
    for mdl in ["sd15", "pixart", "sd3"]:
        keys = [(m["model"], m["pid"]) for m in meta if m["model"] == mdl]
        qv = [q[k] for k in keys]; lv = [l[k] for k in keys]
        agree = np.mean([int(q[k] == l[k]) for k in keys])
        out[mdl] = dict(qwen=float(np.mean(qv)), llava=float(np.mean(lv)), agree=float(agree), n=len(keys))
        print(f"  {mdl:7s} Qwen={np.mean(qv):.3f}  LLaVA={np.mean(lv):.3f}  agree={agree:.3f}  (n={len(keys)})")
    ok = out["sd15"]["llava"] - out["pixart"]["llava"] > 0.15 and out["sd15"]["qwen"] - out["pixart"]["qwen"] > 0.15
    print(f"  VERDICT: {'CORROBORATED (both graders: SD1.5 > PixArt~SD3.5)' if ok else 'inspect'}")
    json.dump(out, open("results/validate_top1_dual.json", "w"), indent=2)


if __name__ == "__main__":
    main()
