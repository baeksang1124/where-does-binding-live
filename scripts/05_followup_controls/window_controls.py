"""Window and unit controls for SD3.5-medium on the same 75 held-out pairs/seeds as block9_ext, kv_factorial and
window_extension_n75, one pass per pair:

 (1) 9-15 placebo: inject a same-object third-color prompt's stream into the ISOLATED window 9-15 (other blocks
     pinned to the clean stream). Cross-flip = two-object flip toward the c1<->c2 swap; potency = its own colors
     installed. Bracket: the 9-11 placebo is re-run and must reproduce window_placebo_paired.json per pair.
 (2) Clean-pinned single units: the top-1 head (block 4, head 1) and all 24 heads of block 9, each swapped with the
     swapped run's text state re-normalised under the CLEAN run's temb, every block's text input pinned clean (the
     2x2's arm-D conditions, applied to one head / one block). Bracket: the block-9 arm must be pixel-identical to the
     isolated text-stream injection at block 9 (also generated here) and reproduce block9_ext iso9.
 (3) Shifted equal-length windows: isolated 7-block windows s..s+6 of the joint-state (swapped-run) stream for
     s in {0,3,6,8,9,10,12,14,17}. Bracket: s=9 (9-15) must reproduce window_extension_n75 std_e15 per pair.

Every graded image also stores Qwen's raw answers and the first 16 hex digits of the image's SHA-256. Per-pair
records are appended to results/window_controls.jsonl (resumable: finished pairs are skipped); the summary goes to
results/window_controls.json; window_controls_stats.py (CPU) adds per-pair distributions and complete-swap CIs.
Existing result files are only read.
Run from the repo root:  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/window_controls.py (WC_N=75)
"""
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import torch

from binding_patch_sd3 import Controller, generate, install, load_pipe
from grader import QwenGrader
from prompts import COLORS
from prompts_ext import make_ext_pairs
from stream_sd3 import StreamHooks

DEVICE = "cuda"; STEPS = 28; GUID = 7.0; H = W = 512
N = int(os.environ.get("WC_N", "75"))
WIN_STARTS = [0, 3, 6, 8, 9, 10, 12, 14, 17]; WLEN = 7
HEAD = (4, 1); BLOCK = 9
OUT = os.environ.get("WC_OUT", "results/window_controls")
LOG = open(f"{OUT}.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = (ans or "").lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip_vs(k1, k2, c1, c2):
    return 0.5 * (k1 == c2 and k1 != c1) + 0.5 * (k2 == c1 and k2 != c2)


def sha(img):
    return hashlib.sha256(np.asarray(img, np.uint8).tobytes()).hexdigest()[:16]


def pix(a, b):
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)); return float(d.max())


class TembTap:
    """temb (timestep + pooled-prompt projection) per denoising step, read at block 0."""
    def __init__(self, tf):
        self.on = False; self.tembs = []
        tf.transformer_blocks[0].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.on:
            self.tembs.append((kwargs["temb"] if "temb" in kwargs else args[2]).detach().to("cpu", copy=True))


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


@torch.no_grad()
def renorm(pipe, raw, tembs, blocks):
    """norm1_context(swapped raw block input, CLEAN temb): what the block's attention sees under text injection."""
    return {b: [pipe.transformer.transformer_blocks[b].norm1_context(r.to(DEVICE), emb=t.to(DEVICE))[0].to("cpu", copy=True)
                for r, t in zip(raw[b], tembs)] for b in blocks}


def main():
    t0 = time.time()
    b9 = json.load(open("results/block9_ext.json")); wp = json.load(open("results/window_placebo_paired.json"))
    we = json.load(open("results/window_extension_n75.json"))
    qids = b9["qualified"][:N]
    assert wp["qualified"] == b9["qualified"] == we["qualified"]
    ref_iso9 = dict(zip(b9["qualified"], b9["iso9"])); ref_pl911 = dict(zip(wp["qualified"], wp["placebo_win"]))
    ref_w915 = dict(zip(we["qualified"], we["std_e15"]))
    done = {}
    if os.path.exists(f"{OUT}.jsonl"):
        for line in open(f"{OUT}.jsonl"):
            r = json.loads(line); done[r["id"]] = r
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== window controls on {len(qids)} held-out pairs ({len(done)} already done) =====")

    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); install(pipe, ctrl)
    hooks = StreamHooks(pipe.transformer); tap = TembTap(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    def grade(img, p, c_a, c_b):
        a1, a2 = qg._ask_color(img, p["o1"]), qg._ask_color(img, p["o2"])
        k1, k2 = canon(a1), canon(a2)
        return dict(flip=flip_vs(k1, k2, p["c1"], p["c2"]), potency=(int(k1 == c_a) + int(k2 == c_b)) / 2.0,
                    a1=a1, a2=a2, sha=sha(img))

    for n, pid in enumerate(qids):
        if pid in done:
            continue
        p = P[pid]; seed = p["seed"]
        others = [c for c in COLORS if c not in (p["c1"], p["c2"])]
        c3, c4 = (others + COLORS)[:2]
        placebo_prompt = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"
        # captures: clean stream + clean temb; swapped (joint-state) stream; placebo stream
        hooks.begin_capture_clean(); tap.on = True; tap.tembs = []; _ = gen(pipe, p["clean"], seed); tap.on = False; hooks.off()
        clean = {b: list(v) for b, v in hooks.clean.items()}; tembs = list(tap.tembs)
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        swap = {b: list(v) for b, v in hooks.cache.items()}
        hooks.begin_capture(); _ = gen(pipe, placebo_prompt, seed); hooks.off()
        plac = {b: list(v) for b, v in hooks.cache.items()}
        matched = renorm(pipe, swap, tembs, [HEAD[0], BLOCK])
        hooks.clean = clean
        rec = dict(id=pid, seed=seed, placebo_prompt=placebo_prompt, win={}, placebo={}, unit={})

        hooks.cache = swap                                          # (3) shifted windows, joint-state stream
        for s in WIN_STARTS:
            hooks.begin_inject(set(range(s, s + WLEN)), isolate=True)
            rec["win"][f"{s}-{s + WLEN - 1}"] = grade(gen(pipe, p["clean"], seed), p, p["c2"], p["c1"]); hooks.off()
        hooks.cache = plac                                          # (1) placebo windows
        for name, blocks in (("9-15", range(9, 16)), ("9-11", range(9, 12))):
            hooks.begin_inject(set(blocks), isolate=True)
            rec["placebo"][name] = grade(gen(pipe, p["clean"], seed), p, c3, c4); hooks.off()
        hooks.cache = swap                                          # (2) text-stream injection at block 9 (bracket)
        hooks.begin_inject({BLOCK}, isolate=True); img_t9 = gen(pipe, p["clean"], seed); hooks.off()
        for name, cells, blk in (("head_4h1_pinned", [HEAD], HEAD[0]),
                                 ("block9_allheads_pinned", [(BLOCK, h) for h in range(24)], BLOCK)):
            ctrl.swap_ctx = {blk: matched[blk]}
            hooks.begin_inject(set(), isolate=True)                 # every block's text input pinned clean
            img = generate(pipe, ctrl, p["clean"], seed, cells=cells, swap_k=True, swap_v=True,
                           steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            hooks.off(); ctrl.swap_ctx = {}
            rec["unit"][name] = grade(img, p, p["c2"], p["c1"])
            if name.startswith("block9"):
                rec["unit"]["block9_vs_textinject_maxpix"] = pix(img, img_t9)
        rec["unit"]["textinject_block9"] = grade(img_t9, p, p["c2"], p["c1"])
        rec["brackets"] = dict(win915=rec["win"]["9-15"]["flip"] == ref_w915[pid],
                               placebo911=rec["placebo"]["9-11"]["flip"] == ref_pl911[pid],
                               iso9=rec["unit"]["textinject_block9"]["flip"] == ref_iso9[pid],
                               block9_pixel_identical=rec["unit"]["block9_vs_textinject_maxpix"] == 0.0)
        with open(f"{OUT}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        done[pid] = rec
        hooks.cache = {}; hooks.clean = {}
        del clean, swap, plac, matched, tembs; gc.collect(); torch.cuda.empty_cache()
        log(f"pair{pid} [{len(done)}/{len(qids)}] win " + " ".join(f"{k}={v['flip']:.1f}" for k, v in rec["win"].items())
            + f" | plac915 {rec['placebo']['9-15']['flip']:.1f}/pot{rec['placebo']['9-15']['potency']:.1f}"
            + f" | head {rec['unit']['head_4h1_pinned']['flip']:.1f} blk9 {rec['unit']['block9_allheads_pinned']['flip']:.1f}"
            + f" | brackets {rec['brackets']} | {time.time() - t0:.0f}s")

    # ---------------- summary ----------------
    R = [done[i] for i in qids]
    rng = np.random.default_rng(20261001)

    def ci(x, paired=False):
        x = np.asarray(x, float); b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + ([float((b > 0).mean())] if paired else [])

    col = lambda f: np.array([f(r) for r in R], float)  # noqa: E731
    out = dict(qualified=qids, n=len(R), window_len=WLEN, starts=WIN_STARTS, head=list(HEAD), block=BLOCK)
    out["windows"] = {k: dict(mean_ci=ci(col(lambda r, k=k: r["win"][k]["flip"])),
                              complete=int(sum(r["win"][k]["flip"] == 1.0 for r in R)))
                      for k in R[0]["win"]}
    sw915 = np.array([ref_w915[i] for i in qids], float); sw911 = np.array([dict(zip(b9["qualified"], b9["win911"]))[i] for i in qids], float)
    pl915 = col(lambda r: r["placebo"]["9-15"]["flip"]); pl911 = col(lambda r: r["placebo"]["9-11"]["flip"])
    out["placebo"] = {"9-15": dict(crossflip_ci=ci(pl915), potency_ci=ci(col(lambda r: r["placebo"]["9-15"]["potency"])),
                                   complete=int((pl915 == 1).sum()),
                                   excess_swap_minus_placebo=ci(sw915 - pl915, True),
                                   excess_complete=ci((sw915 == 1).astype(float) - (pl915 == 1).astype(float), True)),
                      "9-11": dict(crossflip_ci=ci(pl911), potency_ci=ci(col(lambda r: r["placebo"]["9-11"]["potency"])),
                                   excess_swap_minus_placebo=ci(sw911 - pl911, True))}
    hd = col(lambda r: r["unit"]["head_4h1_pinned"]["flip"]); bk = col(lambda r: r["unit"]["block9_allheads_pinned"]["flip"])
    kvD = np.array([dict(zip(json.load(open("results/kv_factorial.json"))["qualified"],
                             json.load(open("results/kv_factorial.json"))["D_cltemb_pin"]))[i] for i in qids], float)
    out["units"] = dict(head_4h1_pinned=dict(mean_ci=ci(hd), complete=int((hd == 1).sum())),
                        block9_allheads_pinned=dict(mean_ci=ci(bk), complete=int((bk == 1).sum())),
                        window911_pinned_D_minus_head=ci(kvD - hd, True),
                        window911_pinned_D_minus_block9=ci(kvD - bk, True),
                        block9_minus_head=ci(bk - hd, True))
    out["brackets"] = {k: [int(sum(r["brackets"][k] for r in R)), len(R)] for k in R[0]["brackets"]}
    w = out["windows"]; base = col(lambda r: r["win"]["9-15"]["flip"])
    out["windows_paired_vs_9_15"] = {k: ci(col(lambda r, k=k: r["win"][k]["flip"]) - base, True) for k in w if k != "9-15"}
    json.dump(out, open(f"{OUT}.json", "w"), indent=2)
    log(f"\nbrackets: {out['brackets']}")
    for k, v in w.items():
        log(f"  window {k:6s} {v['mean_ci'][0]:.3f} [{v['mean_ci'][1]:.2f},{v['mean_ci'][2]:.2f}] complete {v['complete']}/{len(R)}")
    log(f"  placebo 9-15: {out['placebo']['9-15']}")
    log(f"  placebo 9-11: {out['placebo']['9-11']}")
    log(f"  units: {out['units']}")
    log(f"-> {OUT}.json ; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
