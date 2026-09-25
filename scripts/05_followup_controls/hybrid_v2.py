"""Hybrid capture v2 -- pin the IMAGE hidden state entering EVERY block
to the clean run's value (per block, per step) while the text stream is recomputed under the
swapped prompt.

hybrid_stream.py (v1) pinned only the step-end latent, so within a step blocks 0..b-1 still
processed that latent with the swapped prompt: the image state entering block 9 equalled the
swapped run's, not the clean run's, and the final-image L1=0 bracket was tautological
(the last step's latent is overwritten). v2 pins hidden_states at every block input via a
forward pre-hook, so the captured text stream at block b has attended only to clean image K/V.
Brackets are at the TENSOR level, not the final image:
  B1  v2 capture under the CLEAN prompt reproduces the clean text stream bit-for-bit
  B2  the pin fired 24 blocks x 28 steps per capture
  B3  the v2 stream differs from both the std and the clean stream (it is a new object)
Then the v2 stream is injected at the 9-11 window (isolated) and at all blocks on the same 40
pairs as v1; std_* per-pair values are reused from results/hybrid_stream.json (determinism was
verified 40/40 there) and std_win is regenerated here as a determinism bracket.
One GPU. Writes results/hybrid_v2.json (overwritten on re-run) and appends to results/hybrid_v2.log.
"""
import torch, json, os, time, re, gc
import numpy as np
from prompts_ext import make_ext_pairs
from prompts import COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
WIN = {9, 10, 11}
N = int(os.environ.get("HYB2_N", "40"))
os.makedirs("results", exist_ok=True)
LOG = open("results/hybrid_v2.log", "a")


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


class ImagePins:
    """forward_pre_hooks on every transformer block for the IMAGE hidden_states (the text
    counterpart of StreamHooks). capture: store per (block, step). inject: overwrite the block's
    image input with the stored clean value."""
    def __init__(self, transformer):
        self.mode = "off"; self.store = {}; self.step = {}; self.applied = 0
        self.handles = [blk.register_forward_pre_hook(self._mk(i), with_kwargs=True)
                        for i, blk in enumerate(transformer.transformer_blocks)]

    def _mk(self, i):
        def hook(module, args, kwargs):
            if self.mode == "off":
                return None
            if "hidden_states" in kwargs:
                hs, where = kwargs["hidden_states"], "kw"
            elif len(args) >= 1:
                hs, where = args[0], "pos"
            else:
                return None
            if self.mode == "capture":
                self.store.setdefault(i, []).append(hs.detach().to("cpu", copy=True)); return None
            idx = self.step.get(i, 0); self.step[i] = idx + 1
            rep = self.store[i][idx].to(hs.device, hs.dtype); self.applied += 1
            if where == "kw":
                kwargs = dict(kwargs); kwargs["hidden_states"] = rep; return (args, kwargs)
            args = list(args); args[0] = rep; return (tuple(args), kwargs)
        return hook

    def begin_capture(self):
        self.mode = "capture"; self.store = {}; self.step = {}

    def begin_inject(self):
        self.mode = "inject"; self.step = {}; self.applied = 0

    def off(self):
        self.mode = "off"; self.step = {}


@torch.no_grad()
def gen(pipe, prompt, seed, latents_pin=None, collect=None):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    kw = dict(num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g)
    if latents_pin is not None or collect is not None:
        def cb(_pipe, i, _t, ckw):
            out = {}
            if collect is not None:
                collect.append(ckw["latents"].detach().clone())
            if latents_pin is not None and i < len(latents_pin):
                out["latents"] = latents_pin[i].to(ckw["latents"].device, ckw["latents"].dtype).clone()
            return out
        kw["callback_on_step_end"] = cb; kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(prompt, **kw).images[0]


def stream_maxdiff(a, b, blocks=None):
    blocks = blocks if blocks is not None else sorted(set(a) & set(b))
    return max(float((x.float() - y.float()).abs().max()) for bk in blocks for x, y in zip(a[bk], b[bk]))


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + \
           ([float((b > 0).mean())] if paired else [])


def main():
    t0 = time.time()
    v1 = json.load(open("results/hybrid_stream.json"))
    qids = v1["qualified"][:N]
    v1_std_win = dict(zip(v1["qualified"], v1["std_win"])); v1_std_all = dict(zip(v1["qualified"], v1["std_all"]))
    v1_hyb_win = dict(zip(v1["qualified"], v1["hyb_win"])); v1_hyb_all = dict(zip(v1["qualified"], v1["hyb_all"]))
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== HYBRID v2 (per-block clean IMAGE-state pinning) on {len(qids)} held-out pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)      # text stream capture / inject
    pins = ImagePins(pipe.transformer)         # image state capture / pin
    qg = QwenGrader(device=DEVICE)

    res = {"std_win_regen": [], "hyb2_win": [], "hyb2_all": []}
    brackets = {"B1_clean_noop_maxdiff": [], "B2_pins_applied": [], "B3_hyb2_vs_std_b9": [],
                "B3_hyb2_vs_clean_b9": [], "B3_hyb2_vs_std_b0": []}

    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        # (1) clean run: clean text stream + clean image state at every block input + latents
        lat = []
        hooks.begin_capture_clean(); pins.begin_capture()
        _ = gen(pipe, p["clean"], seed, collect=lat)
        pins.off(); hooks.off()
        clean_stream = {b: list(v) for b, v in hooks.clean.items()}
        # (2) std capture (standard joint-state protocol) -- text stream only, for B3 and the regen bracket
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        std = {b: list(v) for b, v in hooks.cache.items()}
        # (3) B1 on the first two pairs: v2 machinery under the CLEAN prompt must be a tensor no-op
        if n < 2:
            hooks.begin_capture(); pins.begin_inject()
            _ = gen(pipe, p["clean"], seed, latents_pin=lat)
            pins.off(); hooks.off()
            d = stream_maxdiff(hooks.cache, clean_stream)
            brackets["B1_clean_noop_maxdiff"].append(d)
            log(f"  B1 pair{pid}: v2-capture(clean prompt) vs clean text stream max|diff| = {d:.6f} "
                f"(pins applied {pins.applied})")
        # (4) HYBRID v2 capture: swapped prompt, image state pinned clean at every block + latent pin
        hooks.begin_capture(); pins.begin_inject()
        _ = gen(pipe, p["swapped"], seed, latents_pin=lat)
        pins.off(); hooks.off()
        hyb2 = {b: list(v) for b, v in hooks.cache.items()}
        brackets["B2_pins_applied"].append(pins.applied)
        brackets["B3_hyb2_vs_std_b9"].append(stream_maxdiff(hyb2, std, [9]))
        brackets["B3_hyb2_vs_clean_b9"].append(stream_maxdiff(hyb2, clean_stream, [9]))
        brackets["B3_hyb2_vs_std_b0"].append(stream_maxdiff(hyb2, std, [0]))
        # (5) inject: std window regen (determinism bracket), hyb2 window (isolated), hyb2 all-block
        hooks.clean = clean_stream
        hooks.cache = std; hooks.begin_inject(set(WIN), isolate=True)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        res["std_win_regen"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        hooks.cache = hyb2; hooks.begin_inject(set(WIN), isolate=True)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        res["hyb2_win"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
        hooks.begin_inject(set(range(NBLOCKS)), isolate=False)
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        res["hyb2_all"].append(flip_vs(k1, k2, p["c1"], p["c2"]))

        hooks.cache = {}; hooks.clean = {}; pins.store = {}
        del std, hyb2, clean_stream, lat; gc.collect()
        log(f"pair{pid} [{n+1}/{len(qids)}] win std={res['std_win_regen'][-1]:.2f}(v1 {v1_std_win[pid]:.2f}) "
            f"hyb2={res['hyb2_win'][-1]:.2f}(v1 hyb {v1_hyb_win[pid]:.2f}) | all hyb2={res['hyb2_all'][-1]:.2f}"
            f"(v1 hyb {v1_hyb_all[pid]:.2f}) | pins {pins.applied} b9 |hyb2-std| {brackets['B3_hyb2_vs_std_b9'][-1]:.2f} "
            f"|hyb2-clean| {brackets['B3_hyb2_vs_clean_b9'][-1]:.2f} "
            f"| running hyb2 win={np.mean(res['hyb2_win']):.3f} all={np.mean(res['hyb2_all']):.3f}")

    rng = np.random.default_rng(20260919)
    std_win = np.array([v1_std_win[i] for i in qids], float); std_all = np.array([v1_std_all[i] for i in qids], float)
    hyb1_win = np.array([v1_hyb_win[i] for i in qids], float); hyb1_all = np.array([v1_hyb_all[i] for i in qids], float)
    out = dict(qualified=qids, n=len(qids), **res, brackets=brackets,
               std_win_regen_matches_v1=[int(sum(a == b for a, b in zip(res["std_win_regen"], std_win))), len(qids)],
               std_win_ci=ci(std_win, rng), std_all_ci=ci(std_all, rng),
               hyb1_win_ci=ci(hyb1_win, rng), hyb1_all_ci=ci(hyb1_all, rng),
               hyb2_win_ci=ci(res["hyb2_win"], rng), hyb2_all_ci=ci(res["hyb2_all"], rng))
    h2w, h2a = np.array(res["hyb2_win"], float), np.array(res["hyb2_all"], float)
    out["paired"] = {"hyb2_minus_std_win": ci(h2w - std_win, rng, True), "hyb2_minus_std_all": ci(h2a - std_all, rng, True),
                     "hyb2_minus_hyb1_win": ci(h2w - hyb1_win, rng, True), "hyb2_minus_hyb1_all": ci(h2a - hyb1_all, rng, True)}
    log(f"\nB1 clean no-op max|diff|: {brackets['B1_clean_noop_maxdiff']}   B2 pins/capture: "
        f"{sorted(set(brackets['B2_pins_applied']))} (expect {NBLOCKS*STEPS})   std_win regen matches v1: {out['std_win_regen_matches_v1']}")
    log(f"B3 block9 |hyb2-std| min {min(brackets['B3_hyb2_vs_std_b9']):.3f}  |hyb2-clean| min {min(brackets['B3_hyb2_vs_clean_b9']):.3f}  "
        f"block0 |hyb2-std| max {max(brackets['B3_hyb2_vs_std_b0']):.3f} (expect 0: block-0 text input is the encoder output)")
    for k in ("std_win", "hyb1_win", "hyb2_win", "std_all", "hyb1_all", "hyb2_all"):
        log(f"  {k:9s} {out[f'{k}_ci']}")
    for k, v in out["paired"].items():
        log(f"  {k:20s} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] P(>0)={v[3]:.3f}")
    json.dump(out, open("results/hybrid_v2.json", "w"), indent=2)
    log(f"-> results/hybrid_v2.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
