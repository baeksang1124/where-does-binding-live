"""FULL held-out dual-grader re-scoring of the top-1 head swap.

Regenerates the top-1-head-swap image for EVERY held-out qualified pair (SD1.5 n~40,
PixArt n~26, SD3.5 n~75; deterministic seeds), saves PNGs to results/dual_full_imgs/, then
grades ALL of them with Qwen2.5-VL-7B AND LLaVA-1.5-7B (models never co-resident).
Reports, per grader:
  - each model's PAPER metric (SD1.5 single-object o1; PixArt/SD3.5 two-object mean), CI
  - a UNIFIED o1-single-object metric for all three (same metric across models)
  - SD1.5-PixArt and SD1.5-SD3.5 difference-of-independent-means bootstrap CIs
  - per-image Qwen/LLaVA agreement; RAW grader answers are stored in the output JSON.
Outputs: results/dual_grade_full.json, results/dual_grade_full.log.
Resume-safe: skips PNGs already on disk. One GPU.
"""
import torch, json, os, gc, numpy as np, re
from PIL import Image
from prompts import make_pairs, COLORS
from prompts_ext import make_ext_pairs

DEVICE = "cuda"; IMGDIR = "results/dual_full_imgs"; os.makedirs(IMGDIR, exist_ok=True)
LOG = open("results/dual_grade_full.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def pairs_by_id():
    d = {p["id"]: p for p in make_pairs()}
    d.update({p["id"]: p for p in make_ext_pairs()})
    return d


def meta_for(mdl, pid, P):
    p = P[pid]
    return dict(model=mdl, pid=pid, o1=p["o1"], o2=p["o2"], c1=p["c1"], c2=p["c2"])


# ---------- pass A: regenerate held-out top-1-swap images ----------
def gen_sd15(meta, P):
    from binding_patch import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/sweep_results.json")); ln, h = sw["ranked_cells"][0].split("|h"); top1 = [(ln, int(h))]
    ids = [i for i in json.load(open("results/top1_sd15_ext.json"))["ids"] if i >= 5000]
    log(f"[gen sd15] {len(ids)} held-out pairs, top1={sw['ranked_cells'][0]}")
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        f = f"{IMGDIR}/sd15_{pid}.png"; meta.append(meta_for("sd15", pid, P))
        if os.path.exists(f):
            continue
        p = P[pid]; g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=top1, swap_k=True, swap_v=True, steps=30).save(f)
    del pipe; gc.collect(); torch.cuda.empty_cache()


def gen_pixart(meta, P):
    from binding_patch_pixart import load_pipe, install, encode, generate, Controller
    sw = json.load(open("results/pixart_sweep.json")); ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    ids = [i for i in json.load(open("results/top1_pixart_ext.json"))["qualified_ids"] if i >= 5000]
    log(f"[gen pixart] {len(ids)} held-out pairs, top1={sw['ranked'][0]}")
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        f = f"{IMGDIR}/pixart_{pid}.png"; meta.append(meta_for("pixart", pid, P))
        if os.path.exists(f):
            continue
        p = P[pid]; g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lc = pipe.transformer.config.in_channels
        lat = torch.randn((1, lc, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        emb, _ = encode(pipe, p["swapped"], DEVICE)
        generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True,
                 steps=20, guidance=4.5).save(f)
    del pipe; gc.collect(); torch.cuda.empty_cache()


def gen_sd3(meta, P):
    from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    ids = json.load(open("results/top1_sd3_ext.json"))["qualified"]      # ext held-out qualified (n~75)
    log(f"[gen sd3] {len(ids)} held-out pairs, top1={hs['ranked'][0]}")
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    for pid in ids:
        f = f"{IMGDIR}/sd3_{pid}.png"; meta.append(meta_for("sd3", pid, P))
        if os.path.exists(f):
            continue
        p = P[pid]
        capture_swapped(pipe, ctrl, p["swapped"], p["seed"], steps=28, guidance=7.0, height=512, width=512, device=DEVICE)
        generate(pipe, ctrl, p["clean"], p["seed"], cells=top1, swap_k=True, swap_v=True,
                 steps=28, guidance=7.0, height=512, width=512, device=DEVICE).save(f)
        ctrl.swap_ctx = {}
    del pipe; gc.collect(); torch.cuda.empty_cache()


# ---------- pass B/C: grade every image with one grader ----------
def grade_all(grader, meta):
    """Return {(model,pid): {'a1':raw,'a2':raw}} raw per-object color answers."""
    out = {}
    for m in meta:
        img = Image.open(f"{IMGDIR}/{m['model']}_{m['pid']}.png")
        out[(m["model"], m["pid"])] = dict(a1=grader._ask_color(img, m["o1"]),
                                           a2=grader._ask_color(img, m["o2"]))
    return out


# ---------- metrics (computed offline from raw answers) ----------
def m_o1(m, ans):           # unified single-object (paper SD1.5 metric): c2->1, c1->0, else 0.5
    k = canon(ans["a1"])
    return 1.0 if k == m["c2"] else (0.0 if k == m["c1"] else 0.5)


def m_two(m, ans):          # paper PixArt/SD3.5 metric: per-object strict flip, mean of 2
    k1, k2 = canon(ans["a1"]), canon(ans["a2"])
    f1 = 1.0 if (k1 == m["c2"] and k1 != m["c1"]) else 0.0
    f2 = 1.0 if (k2 == m["c1"] and k2 != m["c2"]) else 0.0
    return (f1 + f2) / 2.0


def ci(x, nboot=100000, seed=1234):
    x = np.array(x, float); rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (0.0, 0.0, 0.0)
    bt = x[rng.integers(0, len(x), (nboot, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))


def diff_ci(a, b, nboot=100000, seed=1234):   # independent-means bootstrap a-b
    a, b = np.array(a, float), np.array(b, float); rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (nboot, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (nboot, len(b)))].mean(1)
    d = da - db
    return (float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float((d > 0).mean()))


def main():
    P = pairs_by_id(); meta = []
    gen_sd15(meta, P); gen_pixart(meta, P); gen_sd3(meta, P)
    log(f"[gen] total {len(meta)} images")

    from grader import QwenGrader
    qg = QwenGrader(device=DEVICE); qraw = grade_all(qg, meta); del qg; gc.collect(); torch.cuda.empty_cache()
    log("[grade] Qwen done")
    from grader import LlavaGrader
    lg = LlavaGrader(device=DEVICE); lraw = grade_all(lg, meta); del lg; gc.collect(); torch.cuda.empty_cache()
    log("[grade] LLaVA done")

    res = dict(raw={f"{k[0]}_{k[1]}": dict(qwen=qraw[k], llava=lraw[k]) for k in qraw})
    scores = {}
    for grname, raw in [("qwen", qraw), ("llava", lraw)]:
        s = {}
        for mdl, paper_metric in [("sd15", m_o1), ("pixart", m_two), ("sd3", m_two)]:
            ms = [m for m in meta if m["model"] == mdl]
            s[mdl] = dict(paper=[paper_metric(m, raw[(mdl, m["pid"])]) for m in ms],
                          o1=[m_o1(m, raw[(mdl, m["pid"])]) for m in ms])
        scores[grname] = s

    log("\n===== FULL HELD-OUT DUAL-GRADER =====")
    summary = {}
    for grname in ["qwen", "llava"]:
        s = scores[grname]; row = {}
        for mdl in ["sd15", "pixart", "sd3"]:
            mp, lo, hi = ci(s[mdl]["paper"]); mo, olo, ohi = ci(s[mdl]["o1"])
            row[mdl] = dict(paper=[mp, lo, hi], o1=[mo, olo, ohi], n=len(s[mdl]["paper"]))
            log(f"  [{grname}] {mdl:7s} n={len(s[mdl]['paper']):3d} paper={mp:.3f} [{lo:.3f},{hi:.3f}]  o1={mo:.3f} [{olo:.3f},{ohi:.3f}]")
        # differences on the UNIFIED o1 metric (apples-to-apples)
        for other in ["pixart", "sd3"]:
            d, dlo, dhi, pgt = diff_ci(s["sd15"]["o1"], s[other]["o1"])
            row[f"sd15_minus_{other}_o1"] = [d, dlo, dhi, pgt]
            log(f"  [{grname}] sd15-{other} (o1 metric): {d:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P(>0)={pgt:.3f}")
        summary[grname] = row
    # agreement (on the unified o1 metric, exact match of per-image score)
    for mdl in ["sd15", "pixart", "sd3"]:
        ms = [m for m in meta if m["model"] == mdl]
        ag = np.mean([int(m_o1(m, qraw[(mdl, m['pid'])]) == m_o1(m, lraw[(mdl, m['pid'])])) for m in ms])
        summary.setdefault("agreement_o1", {})[mdl] = float(ag)
        log(f"  agreement[{mdl}] (o1 exact) = {ag:.3f}")

    res["summary"] = summary
    json.dump(res, open("results/dual_grade_full.json", "w"), indent=2)
    log("-> results/dual_grade_full.json")


if __name__ == "__main__":
    main()
