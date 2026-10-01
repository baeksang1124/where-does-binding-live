"""Second-grader (LLaVA-1.5-7B) check of the SD3.5 window, placebo and cliff results.

Phase A (SD3.5 + Qwen): on the same 75 held-out pairs/seeds as block9_ext, regenerate and SAVE (PNG)
  clean, swapped (target reference), isolated joint-state windows 9-11 and 9-15, the third-color placebo in the
  isolated windows 9-15 and 9-11, and, on the first 40 pairs (the prefix-sweep set), the non-isolated injection from
  b=9 and b=10 onward. Every image is graded with Qwen exactly as before; brackets: 9-11 == block9_ext win911,
  9-15 == window_extension_n75 std_e15 (and image hash == window_controls.jsonl), placebos' image hashes ==
  window_controls.jsonl (9-11 placebo also == window_placebo_paired), b=9 == hyb2_prefix std_b9, b=10 == hyb2_prefix_fine std_b10.
Phase B (LLaVA): grade every saved image with LLaVA-1.5-7B (same questions as dual_grade_full.py), raw answers kept.
Phase C (CPU): per-arm LLaVA scores, LLaVA-Qwen agreement (exact per-image score, Cohen's kappa on {0,0.5,1}),
  LLaVA swap-specific excess over the placebos, LLaVA cliff drop b=9 -> b=10, the same on the pairs that pass the
  two-object gate under LLaVA, and stratified review sheets (PNG + CSV) for a human check.
Resumable: {OUT}_qwen.jsonl and {OUT}_llava.jsonl are appended per pair; images in {OUT}_imgs/.
Run from the repo root:  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/second_grader_windows.py
(SG_PHASE=a|b|c runs one phase; SG_N, SG_NPRE, SG_OUT)
"""
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

DEVICE = "cuda"; STEPS = 28; GUID = 7.0; H = W = 512; NBLOCKS = 24
N = int(os.environ.get("SG_N", "75")); NPRE = int(os.environ.get("SG_NPRE", "40"))
OUT = os.environ.get("SG_OUT", "results/second_grader_windows")
IMG = f"{OUT}_imgs"
ARMS = ["clean", "swapped", "win911", "win915", "plac915", "plac911", "pre9", "pre10"]
os.makedirs(IMG, exist_ok=True)
LOG = open(f"{OUT}.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def colors():
    from prompts import COLORS
    return COLORS


def canon(ans, C):
    ans = (ans or "").lower(); hit = [c for c in C if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip_vs(k1, k2, c1, c2):
    return 0.5 * (k1 == c2 and k1 != c1) + 0.5 * (k2 == c1 and k2 != c2)


def sha(img):
    return hashlib.sha256(np.asarray(img, np.uint8).tobytes()).hexdigest()[:16]


def load_jsonl(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); d[r["id"]] = r
    return d


def placebo_colors(p, C):
    others = [c for c in C if c not in (p["c1"], p["c2"])]
    return (others + C)[:2]


def phase_a(qids, P):
    from binding_patch_sd3 import load_pipe
    from grader import QwenGrader
    from stream_sd3 import StreamHooks
    C = colors()
    done = load_jsonl(f"{OUT}_qwen.jsonl")
    todo = [i for i in qids if i not in done]
    log(f"\n===== phase A: {len(todo)} of {len(qids)} pairs to generate =====")
    if not todo:
        return
    b9 = json.load(open("results/block9_ext.json")); we = json.load(open("results/window_extension_n75.json"))
    wp = json.load(open("results/window_placebo_paired.json"))
    hp = json.load(open("results/hyb2_prefix.json")); hf = json.load(open("results/hyb2_prefix_fine.json"))
    wc = load_jsonl("results/window_controls.jsonl")
    ref = dict(win911=dict(zip(b9["qualified"], b9["win911"])), win915=dict(zip(we["qualified"], we["std_e15"])),
               plac911=dict(zip(wp["qualified"], wp["placebo_win"])),
               pre9=dict(zip(hp["qualified"], hp["std_b9"])), pre10=dict(zip(hf["qualified"], hf["std_b10"])))
    pre_ids = set(qids[:NPRE]); assert pre_ids <= set(hp["qualified"]) and pre_ids <= set(hf["qualified"])
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    @torch.no_grad()
    def gen(prompt, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]

    t0 = time.time()
    for pid in todo:
        p = P[pid]; seed = p["seed"]; c3, c4 = placebo_colors(p, C)
        placebo_prompt = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"
        imgs = {}
        hooks.begin_capture_clean(); imgs["clean"] = gen(p["clean"], seed); hooks.off()
        clean = {b: list(v) for b, v in hooks.clean.items()}
        hooks.begin_capture(); imgs["swapped"] = gen(p["swapped"], seed); hooks.off()
        swap = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture(); _ = gen(placebo_prompt, seed); hooks.off()
        plac = {b: list(v) for b, v in hooks.cache.items()}
        hooks.clean = clean
        jobs = [("win911", swap, range(9, 12), True), ("win915", swap, range(9, 16), True),
                ("plac915", plac, range(9, 16), True), ("plac911", plac, range(9, 12), True)]
        if pid in pre_ids:
            jobs += [("pre9", swap, range(9, NBLOCKS), False), ("pre10", swap, range(10, NBLOCKS), False)]
        for arm, src, blocks, iso in jobs:
            hooks.cache = src; hooks.begin_inject(set(blocks), isolate=iso)
            imgs[arm] = gen(p["clean"], seed); hooks.off()
        rec = dict(id=pid, seed=seed, placebo_prompt=placebo_prompt, c3=c3, c4=c4, arms={})
        for arm, img in imgs.items():
            img.save(f"{IMG}/{pid}_{arm}.png")
            a1, a2 = qg._ask_color(img, p["o1"]), qg._ask_color(img, p["o2"])
            k1, k2 = canon(a1, C), canon(a2, C)
            rec["arms"][arm] = dict(a1=a1, a2=a2, k1=k1, k2=k2, flip=flip_vs(k1, k2, p["c1"], p["c2"]), sha=sha(img))
        w = wc[pid]
        rec["brackets"] = {k: rec["arms"][k]["flip"] == ref[k][pid] for k in ref if k in rec["arms"]}
        rec["brackets"].update(win915_sha=rec["arms"]["win915"]["sha"] == w["win"]["9-15"]["sha"],
                               plac915_sha=rec["arms"]["plac915"]["sha"] == w["placebo"]["9-15"]["sha"],
                               plac911_sha=rec["arms"]["plac911"]["sha"] == w["placebo"]["9-11"]["sha"])
        with open(f"{OUT}_qwen.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        hooks.cache = {}; hooks.clean = {}
        del clean, swap, plac, imgs; gc.collect(); torch.cuda.empty_cache()
        log(f"pair{pid} " + " ".join(f"{a}={rec['arms'][a]['flip']:.1f}" for a in ARMS if a in rec["arms"])
            + f" | brackets {sum(rec['brackets'].values())}/{len(rec['brackets'])}"
            + ("" if all(rec["brackets"].values()) else f" FAIL {[k for k, v in rec['brackets'].items() if not v]}")
            + f" | {time.time() - t0:.0f}s")
    del pipe, hooks, qg; gc.collect(); torch.cuda.empty_cache()


def phase_b(qids, P):
    from grader import LlavaGrader
    C = colors()
    done = load_jsonl(f"{OUT}_llava.jsonl"); qw = load_jsonl(f"{OUT}_qwen.jsonl")
    todo = [i for i in qids if i not in done]
    log(f"\n===== phase B (LLaVA): {len(todo)} of {len(qids)} pairs to grade =====")
    if not todo:
        return
    lg = LlavaGrader(device=DEVICE); t0 = time.time()
    for pid in todo:
        p = P[pid]; rec = dict(id=pid, arms={})
        for arm in qw[pid]["arms"]:
            img = Image.open(f"{IMG}/{pid}_{arm}.png").convert("RGB")
            assert sha(img) == qw[pid]["arms"][arm]["sha"], (pid, arm)
            a1, a2 = lg._ask_color(img, p["o1"]), lg._ask_color(img, p["o2"])
            k1, k2 = canon(a1, C), canon(a2, C)
            rec["arms"][arm] = dict(a1=a1, a2=a2, k1=k1, k2=k2, flip=flip_vs(k1, k2, p["c1"], p["c2"]))
        with open(f"{OUT}_llava.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    log(f"phase B done in {time.time() - t0:.0f}s")
    del lg; gc.collect(); torch.cuda.empty_cache()


def kappa(a, b, cats=(0.0, 0.5, 1.0)):
    a, b = np.asarray(a), np.asarray(b); po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def phase_c(qids, P):
    qw = load_jsonl(f"{OUT}_qwen.jsonl"); ll = load_jsonl(f"{OUT}_llava.jsonl")
    rng = np.random.default_rng(20261003)

    def ci(x, paired=False):
        x = np.asarray(x, float); b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + ([float((b > 0).mean())] if paired else [])

    def gate(ids, G):     # two-object gate: clean reads (c1, c2) and the swapped reference reads (c2, c1)
        ok = []
        for i in ids:
            p = P[i]; c, s = G[i]["arms"]["clean"], G[i]["arms"]["swapped"]
            ok.append(c["k1"] == p["c1"] and c["k2"] == p["c2"] and s["k1"] == p["c2"] and s["k2"] == p["c1"])
        return np.array(ok)

    out = dict(n=len(qids), n_prefix=NPRE, seed=20261003, resamples=100000)
    out["brackets"] = {k: [int(sum(qw[i]["brackets"][k] for i in qids if k in qw[i]["brackets"])),
                           int(sum(k in qw[i]["brackets"] for i in qids))] for k in qw[qids[0]]["brackets"]}
    out["gate"] = dict(qwen=int(gate(qids, qw).sum()), llava=int(gate(qids, ll).sum()))
    lg_ok = gate(qids, ll)

    def arm_ids(a):
        return [i for i in qids if a in qw[i]["arms"]]

    out["arms"] = {}
    for a in ARMS[2:]:
        ids = arm_ids(a)
        q = np.array([qw[i]["arms"][a]["flip"] for i in ids]); l = np.array([ll[i]["arms"][a]["flip"] for i in ids])
        sub = np.array([lg_ok[qids.index(i)] for i in ids])
        out["arms"][a] = dict(n=len(ids), qwen=ci(q), llava=ci(l), llava_complete=int((l == 1).sum()),
                              qwen_complete=int((q == 1).sum()),
                              llava_dist=[int((l == 0).sum()), int((l == 0.5).sum()), int((l == 1).sum())],
                              llava_minus_qwen=ci(l - q, True), exact_agreement=float((l == q).mean()),
                              kappa=kappa(q, l), complete_agreement=float(((l == 1) == (q == 1)).mean()),
                              llava_on_llava_gate=(ci(l[sub]) if sub.sum() else None), n_llava_gate=int(sub.sum()))

    def paired(a, b, G):
        ids = [i for i in arm_ids(a) if b in G[i]["arms"]]
        x = np.array([G[i]["arms"][a]["flip"] for i in ids]); y = np.array([G[i]["arms"][b]["flip"] for i in ids])
        return dict(mean=ci(x - y, True), complete=ci((x == 1).astype(float) - (y == 1), True), n=len(ids))

    out["contrasts"] = {g: {"win915-plac915": paired("win915", "plac915", G), "win911-plac911": paired("win911", "plac911", G),
                            "win915-win911": paired("win915", "win911", G), "pre9-pre10": paired("pre9", "pre10", G)}
                        for g, G in (("qwen", qw), ("llava", ll))}
    # per-object label agreement over all graded images (canonical colour words)
    pairs = [(qw[i]["arms"][a][k], ll[i]["arms"][a][k]) for i in qids for a in qw[i]["arms"] for k in ("k1", "k2")]
    out["object_label_agreement"] = float(np.mean([x == y for x, y in pairs]))
    out["llava_other_rate"] = float(np.mean([y == "other" for _, y in pairs]))
    json.dump(out, open(f"{OUT}.json", "w"), indent=1)
    log(f"\nbrackets {out['brackets']} | gate qwen {out['gate']['qwen']}/{len(qids)} llava {out['gate']['llava']}/{len(qids)}"
        f" | object-label agreement {out['object_label_agreement']:.3f}, LLaVA 'other' {out['llava_other_rate']:.3f}")
    for a, v in out["arms"].items():
        log(f"  {a:8s} n={v['n']} qwen {v['qwen'][0]:.2f} | llava {v['llava'][0]:.2f} [{v['llava'][1]:.2f},{v['llava'][2]:.2f}]"
            f" complete {v['llava_complete']} (qwen {v['qwen_complete']}) | agree {v['exact_agreement']:.2f} kappa {v['kappa']:.2f}"
            f" | llava-qwen {v['llava_minus_qwen'][0]:+.2f} [{v['llava_minus_qwen'][1]:+.2f},{v['llava_minus_qwen'][2]:+.2f}]")
    for g in ("qwen", "llava"):
        for k, v in out["contrasts"][g].items():
            log(f"  {g:5s} {k:15s} mean {v['mean'][0]:+.2f} [{v['mean'][1]:+.2f},{v['mean'][2]:+.2f}]"
                f" complete {v['complete'][0]:+.2f} [{v['complete'][1]:+.2f},{v['complete'][2]:+.2f}] n={v['n']}")
    review_sheets(qids, P, qw, ll)
    log(f"-> {OUT}.json")


def review_sheets(qids, P, qw, ll):
    """Stratified human-review material: per arm up to 3 pairs per Qwen score class, clean | target | arm image."""
    rng = np.random.default_rng(7); rows = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for a in ("win911", "win915", "plac915", "pre10"):
        picks = []
        for s in (0.0, 0.5, 1.0):
            ids = [i for i in qids if a in qw[i]["arms"] and qw[i]["arms"][a]["flip"] == s]
            picks += [(int(i), s) for i in rng.permutation(ids)[:3]]
        tile = 256; sheet = Image.new("RGB", (3 * tile + 520, max(1, len(picks)) * tile), "white"); d = ImageDraw.Draw(sheet)
        for r, (i, s) in enumerate(picks):
            for c, arm in enumerate(("clean", "swapped", a)):
                sheet.paste(Image.open(f"{IMG}/{i}_{arm}.png").resize((tile, tile)), (c * tile, r * tile))
            p = P[i]; q = qw[i]["arms"][a]; l = ll[i]["arms"][a]
            txt = (f"id {i}  [{a}]\nclean: {p['clean']}\nswapped: {p['swapped']}\n"
                   f"Qwen: {p['o1']}={q['a1']}, {p['o2']}={q['a2']} -> {q['flip']}\n"
                   f"LLaVA: {p['o1']}={l['a1']}, {p['o2']}={l['a2']} -> {l['flip']}")
            d.multiline_text((3 * tile + 10, r * tile + 10), txt, fill="black", font=font, spacing=6)
            rows.append(dict(id=i, arm=a, qwen_class=s, image=f"{IMG}/{i}_{a}.png", o1=p["o1"], o2=p["o2"],
                             c1=p["c1"], c2=p["c2"], qwen=q["flip"], llava=l["flip"], human_o1="", human_o2=""))
        sheet.save(f"{OUT}_review_{a}.png")
    with open(f"{OUT}_review.csv", "w") as f:
        keys = list(rows[0].keys()); f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    from prompts_ext import make_ext_pairs
    P = {p["id"]: p for p in make_ext_pairs()}
    qids = json.load(open("results/block9_ext.json"))["qualified"][:N]
    phase = os.environ.get("SG_PHASE", "all")
    if phase in ("all", "a"):
        phase_a(qids, P)
    if phase in ("all", "b"):
        phase_b(qids, P)
    if phase in ("all", "c"):
        phase_c(qids, P)


if __name__ == "__main__":
    main()
