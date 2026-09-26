"""Candidate tiles for the qualitative Fig. 5 (one GPU).

  Row SD1.5  (UNet, isolated cross-attn): clean | top-1 head K/V swap | swapped prompt (target)
  Row SD3.5  (MM-DiT, joint attention)  : clean | top-1 head K/V swap | text stream, blocks 9-15
                                          (isolated, joint-state "std" arm e=15) | swapped prompt (target)

Every tile is produced by the SAME code path as the number it illustrates (functions/constants are
imported from the measurement scripts, nothing re-implemented):
  SD1.5 top-1  scripts/02_heldout_generalization/top1_sd15_ext.py  (DDIM, 30 steps, cfg 7.5, fp16 randn latent)
  SD3.5 top-1  scripts/02_heldout_generalization/top1_sd3_ext.py   (JointHeadSwapProcessor, 28 steps, cfg 7.0, 512^2)
  SD3.5 window scripts/05_followup_controls/window_extension.py, std arm e=15 (default processors +
               StreamHooks, isolate=True). The hyb2 capture / image pins of that script only feed the
               hyb2 arm, so they are not run here.
  targets      the same pipeline, swapped prompt, the pair's seed. The qualification brackets
               (clean / swap-all) are regenerated too, so each pair's qualification is re-checked.

Candidates (held-out ids >= 5000): SD1.5 top-1 flip == 1 (top1_sd15_ext.json) AND SD3.5 top-1 == 0
(top1_sd3_ext.json) AND SD3.5 window 9-15 std flip == 1 (window_extension_n75.json). If that
intersection has < 4 ids, each row is padded with pairs that satisfy only its own criteria.

Diffusion models are freed before grading; every tile is re-graded with QwenGrader using the grading rule
of its measurement (SD1.5: object_swap_score on o1; SD3.5: per-object colour question + whole-word canon,
flip_vs). Reads results/*.json only; writes only under results/figs/f5_candidates/.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/figures/f5_candidates.py      (F5C_PHASE=gen|grade|all, F5C_N)
"""
import os, sys, json, time, gc, shutil, tempfile, contextlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from prompts_ext import make_ext_pairs
from grader import QwenGrader
import binding_patch as BP15
import binding_patch_sd3 as BP3
from stream_sd3 import StreamHooks

OUT = os.environ.get("F5C_OUT", "results/figs/f5_candidates")
MAXN = int(os.environ.get("F5C_N", "8"))
PHASE = os.environ.get("F5C_PHASE", "all")
DEVICE = "cuda"
WIN_END = 15
sys.path[:0] = [os.path.abspath("scripts/02_heldout_generalization"), os.path.abspath("scripts/05_followup_controls")]


@contextlib.contextmanager
def _cwd(d):
    old = os.getcwd(); os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


# The measurement scripts open results/<name>.log in append mode at import time. Import them from a
# throw-away cwd so those handles land in a temp dir and no existing file under results/ is opened.
_SANDBOX = tempfile.mkdtemp(prefix="f5c_import_"); os.makedirs(f"{_SANDBOX}/results")
with _cwd(_SANDBOX):
    import top1_sd15_ext as T15
    import top1_sd3_ext as T3
    import hybrid_v2 as HV2
    import window_extension as WE
shutil.rmtree(_SANDBOX, ignore_errors=True)

ARMS = {"sd15": ["clean", "top1", "target"], "sd3": ["clean", "top1", "win915", "target"]}
EXTRA = {"sd15": ["swapall"], "sd3": ["swapall"]}       # qualification bracket tiles (not in the sheet)
TITLE = {"clean": "clean", "top1": "top-1 head K/V swap", "win915": "text stream, blocks 9-15",
         "target": "swapped prompt (target)", "swapall": "swap-all (qualification)"}


def tile_path(pid, model, arm):
    return f"{OUT}/{pid}_{model}_{arm}.png"


def candidates():
    a = json.load(open("results/top1_sd15_ext.json"))
    s15 = {i: v for i, v in zip(a["ids"], a["per_pair"]) if i >= 5000}
    b = json.load(open("results/top1_sd3_ext.json")); s3 = dict(zip(b["qualified"], b["ext_per"]))
    c = json.load(open("results/window_extension_n75.json")); we = dict(zip(c["qualified"], c["std_e15"]))
    ok15 = {i for i, v in s15.items() if v == 1.0}
    ok3 = {i for i, v in s3.items() if i >= 5000 and v == 0.0 and we.get(i) == 1.0}
    both = sorted(ok15 & ok3)[:MAXN]
    rows = {"sd15": list(both), "sd3": list(both)}
    if len(both) < 4:
        rows["sd3"] = (both + sorted(ok3 - set(both)))[:MAXN]
        rows["sd15"] = (both + sorted(ok15 - set(both)))[:MAXN]
    stored = {i: dict(sd15_top1=s15.get(i), sd3_top1=s3.get(i), sd3_win915_std=we.get(i),
                      sd3_win911_std=dict(zip(c["qualified"], c["std_e11"])).get(i))
              for i in sorted(set(rows["sd15"]) | set(rows["sd3"]))}
    return rows, both, stored, a["top1_cell"], b["top1"]


def maxdiff(x, y):
    return int(np.abs(np.asarray(x, np.int16) - np.asarray(y, np.int16)).max())


# ------------------------------------------------------------------ generation
def gen_sd15(ids, P, brackets):
    sweep = json.load(open("results/sweep_results.json"))
    c = sweep["ranked_cells"][0]; ln, h = c.split("|h"); top1 = [(ln, int(h))]
    pipe = BP15.load_pipe(device=DEVICE); ctrl = BP15.Controller(); BP15.install(pipe, ctrl)
    for pid in ids:
        p = P[pid]
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g, device=DEVICE, dtype=torch.float16)
        se = BP15.encode(pipe, p["swapped"], DEVICE)
        im = {}
        im["clean"] = BP15.generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=T15.STEPS)
        im["swapall"] = BP15.generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=T15.STEPS)
        im["top1"] = BP15.generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=top1, swap_k=True, swap_v=True,
                                   steps=T15.STEPS)
        im["target"] = BP15.generate(pipe, ctrl, p["swapped"], se, lat.clone(), cells=None, steps=T15.STEPS)
        for k, v in im.items():
            v.save(tile_path(pid, "sd15", k))
        brackets.setdefault(pid, {})["sd15_target_vs_swapall_maxabs"] = maxdiff(im["target"], im["swapall"])
        print(f"  sd15 {pid} done (target vs swap-all max|diff| {brackets[pid]['sd15_target_vs_swapall_maxabs']})", flush=True)
    del pipe, ctrl; gc.collect(); torch.cuda.empty_cache()
    return c


def gen_sd3(ids, P, brackets):
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    pipe = BP3.load_pipe(device=DEVICE)
    # (a) window_extension.py std arm: default joint-attn processors + StreamHooks, isolated window 9..15
    hooks = StreamHooks(pipe.transformer); win = set(range(WE.START, WIN_END + 1))
    wclean, wtarget = {}, {}
    for pid in ids:
        p = P[pid]; seed = p["seed"]; lat = []
        hooks.begin_capture_clean(); wclean[pid] = HV2.gen(pipe, p["clean"], seed, collect=lat); hooks.off()
        clean_stream = {k: list(v) for k, v in hooks.clean.items()}
        hooks.begin_capture(); wtarget[pid] = HV2.gen(pipe, p["swapped"], seed); hooks.off()
        std = {k: list(v) for k, v in hooks.cache.items()}
        hooks.clean = clean_stream; hooks.cache = std
        hooks.begin_inject(win, isolate=True); HV2.gen(pipe, p["clean"], seed).save(tile_path(pid, "sd3", "win915")); hooks.off()
        hooks.cache = {}; hooks.clean = {}
        del std, clean_stream, lat; gc.collect()
        print(f"  sd3 window {pid} done", flush=True)
    hooks.remove()
    # (b) top1_sd3_ext.py: JointHeadSwapProcessor on every block, capture_swapped then clean / swap-all / top-1
    ctrl = BP3.Controller(); BP3.install(pipe, ctrl)
    kw = dict(steps=T3.STEPS, guidance=T3.GUID, height=T3.H, width=T3.W, device=DEVICE)
    for pid in ids:
        p = P[pid]; seed = p["seed"]; im = {}
        im["target"] = BP3.capture_swapped(pipe, ctrl, p["swapped"], seed, **kw)
        im["clean"] = BP3.generate(pipe, ctrl, p["clean"], seed, cells=None, **kw)
        im["swapall"] = BP3.generate(pipe, ctrl, p["clean"], seed, swap_all=True, **kw)
        im["top1"] = BP3.generate(pipe, ctrl, p["clean"], seed, cells=top1, swap_k=True, swap_v=True, **kw)
        ctrl.swap_ctx = {}
        for k, v in im.items():
            v.save(tile_path(pid, "sd3", k))
        br = brackets.setdefault(pid, {})
        br["sd3_clean_top1path_vs_windowpath_maxabs"] = maxdiff(im["clean"], wclean[pid])
        br["sd3_target_top1path_vs_windowpath_maxabs"] = maxdiff(im["target"], wtarget[pid])
        print(f"  sd3 top-1 {pid} done (clean/target path max|diff| {br['sd3_clean_top1path_vs_windowpath_maxabs']}/"
              f"{br['sd3_target_top1path_vs_windowpath_maxabs']})", flush=True)
    del pipe, ctrl, hooks, wclean, wtarget; gc.collect(); torch.cuda.empty_cache()
    return hs["ranked"][0]


# ------------------------------------------------------------------ grading
def grade_all(rows, P):
    qg = QwenGrader(device=DEVICE); G = {}
    for pid in rows["sd15"]:
        p = P[pid]; d = G.setdefault(pid, {}).setdefault("sd15", {})
        for arm in ARMS["sd15"] + EXTRA["sd15"]:
            img = Image.open(tile_path(pid, "sd15", arm)).convert("RGB")
            s, a1 = qg.object_swap_score(img, p["o1"], p["c1"], p["c2"])       # top1_sd15_ext primary rule (o1)
            a2 = qg._ask_color(img, p["o2"])                                  # auxiliary: second object, canon
            d[arm] = dict(o1_swap_score=s, o1_answer=a1, o2_answer=a2, o2_canon=T3.canon(a2))
    for pid in rows["sd3"]:
        p = P[pid]; d = G.setdefault(pid, {}).setdefault("sd3", {})
        for arm in ARMS["sd3"] + EXTRA["sd3"]:
            img = Image.open(tile_path(pid, "sd3", arm)).convert("RGB")
            a1, a2 = qg._ask_color(img, p["o1"]), qg._ask_color(img, p["o2"])
            k1, k2 = T3.canon(a1), T3.canon(a2)                               # == top1_sd3_ext.grade2 / hybrid_v2.grade2
            d[arm] = dict(o1_answer=a1, o2_answer=a2, o1=k1, o2=k2, flip=HV2.flip_vs(k1, k2, p["c1"], p["c2"]))
        print(f"  graded {pid}", flush=True)
    del qg; gc.collect(); torch.cuda.empty_cache()
    return G


def reproduce(pid, p, g, st):
    r = {}
    if "sd15" in g:
        s = g["sd15"]
        r["sd15_qualification(clean=0,swapall=1)"] = s["clean"]["o1_swap_score"] == 0.0 and s["swapall"]["o1_swap_score"] == 1.0
        r["sd15_top1"] = dict(stored=st["sd15_top1"], regraded=s["top1"]["o1_swap_score"],
                              match=s["top1"]["o1_swap_score"] == st["sd15_top1"])
    if "sd3" in g:
        s = g["sd3"]
        r["sd3_qualification(clean=c1c2,swapall=c2c1)"] = (s["clean"]["o1"], s["clean"]["o2"]) == (p["c1"], p["c2"]) and \
                                                          (s["swapall"]["o1"], s["swapall"]["o2"]) == (p["c2"], p["c1"])
        r["sd3_top1"] = dict(stored=st["sd3_top1"], regraded=s["top1"]["flip"], match=s["top1"]["flip"] == st["sd3_top1"])
        r["sd3_win915_std"] = dict(stored=st["sd3_win915_std"], regraded=s["win915"]["flip"],
                                   match=s["win915"]["flip"] == st["sd3_win915_std"])
    return r


# ------------------------------------------------------------------ contact sheets
def _font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def sheet(pid, p, rows, grades=None, T=384):
    f, fs = _font(15), _font(13)
    lab_h, gap, head_h, left = 40, 6, 30, 70
    cols = 4
    have = [m for m in ("sd15", "sd3") if pid in rows[m]]
    W = left + cols * T + (cols - 1) * gap; H = head_h + len(have) * (T + lab_h + gap)
    cv = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(cv)
    d.text((6, 6), f"#{pid}  clean: \"{p['clean']}\"   swapped: \"{p['swapped']}\"   seed {p['seed']}", fill=(0, 0, 0), font=f)
    for r, m in enumerate(have):
        y = head_h + r * (T + lab_h + gap)
        d.text((6, y + lab_h + T // 2 - 8), "SD1.5" if m == "sd15" else "SD3.5", fill=(0, 0, 0), font=f)
        slots = ["clean", "top1", None, "target"] if m == "sd15" else ARMS["sd3"]
        for j, arm in enumerate(slots):
            if arm is None:
                continue
            x = left + j * (T + gap)
            cv.paste(Image.open(tile_path(pid, m, arm)).convert("RGB").resize((T, T), Image.LANCZOS), (x, y + lab_h))
            d.text((x + 2, y + 2), TITLE[arm], fill=(0, 0, 0), font=f)
            if grades and m in grades.get(pid, {}):
                gg = grades[pid][m][arm]
                if m == "sd15":
                    txt = f"{p['o1']}={gg['o1_answer']} (score {gg['o1_swap_score']:.1f})  {p['o2']}={gg['o2_canon']}"
                else:
                    txt = f"{p['o1']}={gg['o1']}  {p['o2']}={gg['o2']}  flip {gg['flip']:.1f}"
                d.text((x + 2, y + 21), txt, fill=(90, 90, 90), font=fs)
    cv.save(f"{OUT}/{pid}_sheet.png")


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    P = {p["id"]: p for p in make_ext_pairs()}
    rows, both, stored, c15, c3 = candidates()
    print(f"intersection {both}; rows {rows}", flush=True)
    jpath = f"{OUT}/candidates.json"
    meta = json.load(open(jpath)) if os.path.exists(jpath) else {}
    timing = meta.get("timing_s", {}); brackets = meta.get("brackets", {})
    brackets = {int(k): v for k, v in brackets.items()}
    if PHASE in ("gen", "all"):
        t = time.time(); gen_sd15(rows["sd15"], P, brackets); timing["gen_sd15"] = round(time.time() - t)
        t = time.time(); gen_sd3(rows["sd3"], P, brackets); timing["gen_sd3"] = round(time.time() - t)
        for pid in sorted(set(rows["sd15"]) | set(rows["sd3"])):
            sheet(pid, P[pid], rows)
    grades = {}
    if PHASE in ("grade", "all"):
        t = time.time(); grades = grade_all(rows, P); timing["grade"] = round(time.time() - t)
        for pid in sorted(set(rows["sd15"]) | set(rows["sd3"])):
            sheet(pid, P[pid], rows, grades)
    timing["total_this_run"] = round(time.time() - t0)
    cands = []
    for pid in sorted(set(rows["sd15"]) | set(rows["sd3"])):
        p = P[pid]
        e = dict(id=pid, seed=p["seed"], clean=p["clean"], swapped=p["swapped"], o1=p["o1"], c1=p["c1"], o2=p["o2"], c2=p["c2"],
                 rows=[m for m in ("sd15", "sd3") if pid in rows[m]], in_intersection=pid in both,
                 stored=stored[pid], brackets=brackets.get(pid, {}),
                 tiles={m: {a: tile_path(pid, m, a) for a in ARMS[m] + EXTRA[m]} for m in ("sd15", "sd3") if pid in rows[m]},
                 sheet=f"{OUT}/{pid}_sheet.png")
        if grades:
            e["regraded"] = grades[pid]; e["reproduction"] = reproduce(pid, p, grades[pid], stored[pid])
        cands.append(e)
    out = dict(created=time.strftime("%Y-%m-%d %H:%M"), script="scripts/figures/f5_candidates.py",
               sd15_top1_cell=c15, sd3_top1_cell=c3, window=f"{WE.START}-{WIN_END} isolated, std (joint-state) arm",
               settings=dict(sd15=dict(steps=T15.STEPS, guidance=7.5, scheduler="DDIM", latent="fp16 randn(seed) 1x4x64x64"),
                             sd3=dict(steps=T3.STEPS, guidance=T3.GUID, size=[T3.H, T3.W], scheduler="pipeline default (FlowMatch Euler)")),
               grading=dict(sd15="QwenGrader.object_swap_score on o1 (top1_sd15_ext rule); o2 asked for figure QA only",
                            sd3="QwenGrader._ask_color per object + whole-word canon + flip_vs (top1_sd3_ext / window_extension rule)"),
               intersection=both, rows=rows, timing_s=timing,
               brackets={str(k): v for k, v in brackets.items()}, candidates=cands)
    for k in ("ranking", "visual_notes", "layout_similarity", "summary"):
        if k in meta:
            out[k] = meta[k]
    json.dump(out, open(jpath, "w"), indent=2)
    print(f"-> {jpath}; elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
