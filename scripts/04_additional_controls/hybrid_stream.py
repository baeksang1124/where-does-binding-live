"""Is the MM-DiT text-stream effect confounded by the IMAGE trajectory?

The standard protocol captures the evolving text stream from a run driven by the SWAPPED
prompt -- so that stream has already attended to a swapped IMAGE trajectory. Injecting it
therefore shows that the text stream is a strong intervention point, but not by itself that
the text stream carries binding independently of the image stream.

Controlled analysis: a HYBRID capture in which the text stream is recomputed under the
swapped prompt while the image stream is PINNED to the CLEAN latent trajectory at every
denoising step (callback_on_step_end overwrite). The resulting text state has only ever seen
clean image content, so injecting it isolates the text-side contribution.

Both captures (STANDARD and HYBRID) are injected on the SAME pairs/seeds, at the block 9-11
window and at all blocks, giving a paired hybrid-vs-standard comparison.

Read:
  hyb ~= std  -> the text stream carries binding on its own; the image-trajectory conditioning
                 is not what drives the effect.
  hyb << std  -> the effect depends on the swapped image trajectory; what the text stream
                 exposes is a joint text/image state accessible via the text stream.

Bracket: pinning the clean trajectory while running the CLEAN prompt must be a no-op (L1 ~ 0).
One GPU.
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
N = int(os.environ.get("HYB_N", "40"))
os.makedirs("results", exist_ok=True)
LOG = open("results/hybrid_stream.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
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


def l1(a, b):
    return float(np.abs(np.asarray(a, np.float32) / 255.0 - np.asarray(b, np.float32) / 255.0).mean())


@torch.no_grad()
def gen(pipe, prompt, seed, latents_pin=None, collect=None):
    """Generate; optionally record the per-step latent trajectory (collect) or FORCE it
    onto a previously recorded one (latents_pin). Pinning happens at the end of every step,
    so the transformer input at every subsequent step is the pinned (clean) latent."""
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
        kw["callback_on_step_end"] = cb
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(prompt, **kw).images[0]


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] + \
           ([float((b > 0).mean())] if paired else [])


def main():
    t0 = time.time()
    b9 = json.load(open("results/block9_ext.json"))
    qids = b9["qualified"][:N]
    prev_win = dict(zip(b9["qualified"], b9["win911"]))
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== HYBRID (clean-latent-pinned) text-stream capture: {len(qids)} held-out pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    res = {k: [] for k in ("std_win", "hyb_win", "std_all", "hyb_all")}
    brackets = []

    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        # (1) clean run: capture the clean text stream AND the clean latent trajectory
        lat = []
        hooks.begin_capture_clean()
        img_clean = gen(pipe, p["clean"], seed, collect=lat)
        hooks.off()
        clean_stream = {b: list(v) for b, v in hooks.clean.items()}

        # (2) bracket on the first two pairs: pinning clean latents under the clean prompt = no-op
        if n < 2:
            brackets.append(l1(gen(pipe, p["clean"], seed, latents_pin=lat), img_clean))
            log(f"  bracket pair{pid}: pinned-clean vs clean L1={brackets[-1]:.5f}")

        # (3) STANDARD capture (standard protocol): swapped prompt, swapped image trajectory
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        std = {b: list(v) for b, v in hooks.cache.items()}

        # (4) HYBRID capture: swapped prompt, image trajectory pinned to the CLEAN one
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed, latents_pin=lat); hooks.off()
        hyb = {b: list(v) for b, v in hooks.cache.items()}

        # (5) inject both, at the window (isolated) and at all blocks
        for tag, src in (("std", std), ("hyb", hyb)):
            hooks.clean = clean_stream; hooks.cache = src
            hooks.begin_inject(set(WIN), isolate=True)
            k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            res[f"{tag}_win"].append(flip_vs(k1, k2, p["c1"], p["c2"]))
            hooks.begin_inject(set(range(NBLOCKS)), isolate=False)
            k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
            res[f"{tag}_all"].append(flip_vs(k1, k2, p["c1"], p["c2"]))

        hooks.cache = {}; hooks.clean = {}
        del std, hyb, clean_stream, lat
        gc.collect()
        log(f"pair{pid} [{n+1}/{len(qids)}] win std={res['std_win'][-1]:.2f} hyb={res['hyb_win'][-1]:.2f} "
            f"| all std={res['std_all'][-1]:.2f} hyb={res['hyb_all'][-1]:.2f} "
            f"| submitted win911={prev_win.get(pid):.2f} "
            f"| running win std={np.mean(res['std_win']):.3f} hyb={np.mean(res['hyb_win']):.3f}")

    rng = np.random.default_rng(20260828)
    out = dict(qualified=qids, n=len(qids), brackets=brackets, **res)
    for k in res:
        out[f"{k}_ci"] = ci(res[k], rng)
    for arm in ("win", "all"):
        d = np.array(res[f"hyb_{arm}"], float) - np.array(res[f"std_{arm}"], float)
        bd = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        out[f"paired_hyb_minus_std_{arm}"] = [float(d.mean()), float(np.percentile(bd, 2.5)),
                                              float(np.percentile(bd, 97.5)), float((bd > 0).mean())]
    # agreement with the stored per-pair window flips in results/block9_ext.json (determinism check)
    same = [prev_win[i] == v for i, v in zip(qids, res["std_win"]) if i in prev_win]
    out["std_reproduces_submitted"] = [int(sum(same)), len(same)]

    log(f"\nbracket (pinned-clean vs clean) L1: {brackets}")
    log(f"window  STANDARD {out['std_win_ci']}")
    log(f"window  HYBRID   {out['hyb_win_ci']}")
    log(f"window  PAIRED hyb-std {out['paired_hyb_minus_std_win']}")
    log(f"allblk  STANDARD {out['std_all_ci']}")
    log(f"allblk  HYBRID   {out['hyb_all_ci']}")
    log(f"allblk  PAIRED hyb-std {out['paired_hyb_minus_std_all']}")
    log(f"std reproduces submitted per-pair window: {out['std_reproduces_submitted']}")
    json.dump(out, open("results/hybrid_stream.json", "w"), indent=2)
    log(f"-> results/hybrid_stream.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
