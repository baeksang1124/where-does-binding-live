"""Attribute the gap between the text-stream window (0.30) and the image-facing K/V window (0.17).

In a JointTransformerBlock the IMAGE output reads the text tokens only through the text K/V,
so raw text-stream injection at block b (normalised with the CLEAN run's temb, other blocks
pinned clean) is mathematically the same intervention on the image branch as swapping all 24
heads' text K/V computed from that same state. The K/V-window arm of matched_late_controls.py
(img_b911) differs from that intervention in two ways:
  (temb)  its K/V were captured post-norm1_context in the SWAPPED run -> swapped pooled-CLIP temb
  (pin)   non-window blocks were NOT pinned to the clean text stream (text branch drifts)
This script runs the 2x2 factorial on the same 75 held-out pairs as block9_ext / matched_late:
  A  swapped temb, no pin   (= matched_late img_b911; bracket: reproduces matched_late.json)
  B  clean temb,   no pin
  C  swapped temb, clean pin
  D  clean temb,   clean pin (= text window by construction; bracket: pixel-identical to T)
  T  raw text-stream window 9-11 regenerated (bracket: reproduces block9_ext.json win911)
One GPU. Writes results/kv_factorial.json (overwritten on re-run) and appends to results/kv_factorial.log.
"""
import torch, json, os, time, re, gc
import numpy as np
from prompts_ext import make_ext_pairs
from prompts import COLORS
from binding_patch_sd3 import load_pipe, install, Controller, generate
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0
WIN = [9, 10, 11]
WB911 = [(b, h) for b in WIN for h in range(24)]
N = int(os.environ.get("KV_N", "75"))
os.makedirs("results", exist_ok=True)
LOG = open("results/kv_factorial.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = (ans or "").lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade2(qg, img, p):
    return canon(qg._ask_color(img, p["o1"])), canon(qg._ask_color(img, p["o2"]))


def flip_vs(k1, k2, c1, c2):
    f = 0.0
    if k1 == c2 and k1 != c1:
        f += 0.5
    if k2 == c1 and k2 != c2:
        f += 0.5
    return f


def pix(a, b):
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return float(d.max()), float(d.mean())


class TembTap:
    """Record temb (timestep + pooled-prompt projection) once per denoising step from block 0."""
    def __init__(self, transformer):
        self.on = False; self.tembs = []
        self.h = transformer.transformer_blocks[0].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.on:
            t = kwargs["temb"] if "temb" in kwargs else args[2]
            self.tembs.append(t.detach().to("cpu", copy=True))
        return None

    def start(self):
        self.on = True; self.tembs = []

    def stop(self):
        self.on = False


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


@torch.no_grad()
def renorm(pipe, raw_cache, tembs):
    """norm1_context(raw swapped block input, CLEAN temb) for the window blocks -> what the
    attention processor sees under raw text-stream injection."""
    out = {}
    for b in WIN:
        blk = pipe.transformer.transformer_blocks[b]
        out[b] = [blk.norm1_context(r.to(DEVICE), emb=t.to(DEVICE))[0].to("cpu", copy=True)
                  for r, t in zip(raw_cache[b], tembs)]
    return out


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + \
           ([float((b > 0).mean())] if paired else [])


def main():
    t0 = time.time()
    b9 = json.load(open("results/block9_ext.json")); ml = json.load(open("results/matched_late.json"))
    qids = b9["qualified"][:N]
    prev_win = dict(zip(b9["qualified"], b9["win911"]))
    prev_img = dict(zip(b9["qualified"], ml["img_b911"]))
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== K/V-window 2x2 factorial (temb x pin) on {len(qids)} held-out pairs =====")

    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); install(pipe, ctrl)
    hooks = StreamHooks(pipe.transformer)
    tap = TembTap(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    arms = ("A_swtemb_nopin", "B_cltemb_nopin", "C_swtemb_pin", "D_cltemb_pin", "T_text_window")
    res = {k: [] for k in arms}
    px_D_vs_T, px_A_vs_T = [], []
    repro_A, repro_T = [], []

    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        # clean run: clean text stream (for pinning) + clean temb per step
        hooks.begin_capture_clean(); tap.start(); _ = gen(pipe, p["clean"], seed); tap.stop(); hooks.off()
        clean_stream = {b: list(v) for b, v in hooks.clean.items()}
        tembs = list(tap.tembs)
        # swapped run: raw text stream (hooks) AND post-norm attention inputs (ctrl) in ONE pass
        hooks.begin_capture(); ctrl.begin_capture(); _ = gen(pipe, p["swapped"], seed); ctrl.off(); hooks.off()
        raw_swap = {b: list(v) for b, v in hooks.cache.items()}
        norm_swap = {b: list(v) for b, v in ctrl.swap_ctx.items()}
        matched = renorm(pipe, raw_swap, tembs)
        if n == 0:
            log(f"  shapes: raw {tuple(raw_swap[9][0].shape)} norm {tuple(norm_swap[9][0].shape)} "
                f"temb {tuple(tembs[0].shape)} steps {len(tembs)}")

        imgs = {}
        hooks.clean = clean_stream
        for tag, src, pin in (("A_swtemb_nopin", norm_swap, False), ("B_cltemb_nopin", matched, False),
                              ("C_swtemb_pin", norm_swap, True), ("D_cltemb_pin", matched, True)):
            ctrl.swap_ctx = src
            if pin:
                hooks.begin_inject(set(), isolate=True)      # every block's text input pinned clean
            imgs[tag] = generate(pipe, ctrl, p["clean"], seed, cells=WB911, swap_k=True, swap_v=True,
                                 steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
            hooks.off(); ctrl.swap_ctx = {}
        hooks.cache = raw_swap; hooks.begin_inject(set(WIN), isolate=True)
        imgs["T_text_window"] = gen(pipe, p["clean"], seed); hooks.off()

        px_D_vs_T.append(pix(imgs["D_cltemb_pin"], imgs["T_text_window"]))
        px_A_vs_T.append(pix(imgs["A_swtemb_nopin"], imgs["T_text_window"]))
        for tag in arms:
            if tag == "T_text_window" and px_D_vs_T[-1][0] == 0.0:
                res[tag].append(res["D_cltemb_pin"][-1])   # identical image -> identical grade
                continue
            k1, k2 = grade2(qg, imgs[tag], p); res[tag].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        repro_A.append(res["A_swtemb_nopin"][-1] == prev_img[pid])
        repro_T.append(res["T_text_window"][-1] == prev_win[pid])

        hooks.cache = {}; hooks.clean = {}; ctrl.swap_ctx = {}
        del raw_swap, norm_swap, matched, clean_stream, tembs, imgs; gc.collect()
        log(f"pair{pid} [{n+1}/{len(qids)}] A={res[arms[0]][-1]:.1f} B={res[arms[1]][-1]:.1f} "
            f"C={res[arms[2]][-1]:.1f} D={res[arms[3]][-1]:.1f} T={res[arms[4]][-1]:.1f} "
            f"| D==T px {px_D_vs_T[-1][0]:.0f} | reproA {repro_A[-1]} reproT {repro_T[-1]} "
            f"| running A={np.mean(res[arms[0]]):.3f} B={np.mean(res[arms[1]]):.3f} "
            f"C={np.mean(res[arms[2]]):.3f} D={np.mean(res[arms[3]]):.3f}")

    rng = np.random.default_rng(20260919)
    out = dict(qualified=qids, n=len(qids), **res,
               pixel_D_vs_T=px_D_vs_T, pixel_A_vs_T=px_A_vs_T,
               D_identical_to_T=[int(sum(d[0] == 0.0 for d in px_D_vs_T)), len(px_D_vs_T)],
               A_reproduces_matched_late=[int(sum(repro_A)), len(repro_A)],
               T_reproduces_block9_ext=[int(sum(repro_T)), len(repro_T)])
    for k in arms:
        out[f"{k}_ci"] = ci(res[k], rng)
    A, B, C, D = (np.array(res[k], float) for k in arms[:4])
    contrasts = {"D_minus_A (submitted gap)": D - A, "B_minus_A (temb | no pin)": B - A,
                 "C_minus_A (pin | swapped temb)": C - A, "D_minus_B (pin | clean temb)": D - B,
                 "D_minus_C (temb | pin)": D - C,
                 "main_temb ((B+D)-(A+C))/2": ((B + D) - (A + C)) / 2,
                 "main_pin ((C+D)-(A+B))/2": ((C + D) - (A + B)) / 2,
                 "interaction (D-C)-(B-A)": (D - C) - (B - A)}
    out["paired"] = {k: ci(v, rng, paired=True) for k, v in contrasts.items()}

    log(f"\nD pixel-identical to T: {out['D_identical_to_T']}   A reproduces matched_late: "
        f"{out['A_reproduces_matched_late']}   T reproduces block9_ext: {out['T_reproduces_block9_ext']}")
    for k in arms:
        log(f"  {k:16s} {out[f'{k}_ci']}")
    for k, v in out["paired"].items():
        log(f"  {k:32s} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] P(>0)={v[3]:.3f}")
    json.dump(out, open("results/kv_factorial.json", "w"), indent=2)
    log(f"-> results/kv_factorial.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
