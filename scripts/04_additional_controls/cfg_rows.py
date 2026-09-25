"""Pilot: conditional-only vs unconditional-only vs both-row CFG patching.

The default text-stream intervention replaces the stream on BOTH CFG rows. SD3's pipeline
batches prompt embeddings as [negative(uncond), positive(cond)], so row0=uncond, row1=cond.
This script repeats the block 9-11 isolated-window injection under three row policies on the
same held-out pairs and seeds, giving a paired per-row comparison.

Sanity bracket: rows=[0,1] must be bit-identical to the default rows=None path, and the
both-row arm must reproduce the stored per-pair window flips in results/block9_ext.json.
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
N = int(os.environ.get("CFG_N", "40"))
os.makedirs("results", exist_ok=True)
LOG = open("results/cfg_rows.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


class RowStreamHooks(StreamHooks):
    """StreamHooks + partial CFG-row injection. self.rows=None reproduces the default
    StreamHooks behaviour exactly; self.rows=[1] patches only the conditional row, [0] only uncond."""

    def __init__(self, transformer):
        self.rows = None
        super().__init__(transformer)

    def _mk(self, i):
        def hook(module, args, kwargs):
            ehs = None; where = None
            if "encoder_hidden_states" in kwargs:
                ehs = kwargs["encoder_hidden_states"]; where = "kw"
            elif len(args) >= 2:
                ehs = args[1]; where = "pos"
            if ehs is None:
                return None
            if self.mode == "capture":
                self.cache.setdefault(i, []).append(ehs.detach().to("cpu", copy=True)); return None
            if self.mode == "capture_clean":
                self.clean.setdefault(i, []).append(ehs.detach().to("cpu", copy=True)); return None
            if self.mode == "inject":
                idx = self.step.get(i, 0); self.step[i] = idx + 1
                src = None
                if i in self.active:
                    swapped = self.cache[i][idx]
                    if self.rows is None:
                        src = swapped
                    else:
                        base = self.clean[i][idx] if self.isolate else ehs.detach().to("cpu", copy=True)
                        src = base.clone()
                        for r in self.rows:
                            src[r] = swapped[r]
                elif self.isolate:
                    src = self.clean[i][idx]
                if src is None:
                    return None
                rep = src.to(ehs.device, ehs.dtype)
                if where == "kw":
                    kwargs = dict(kwargs); kwargs["encoder_hidden_states"] = rep
                    return (args, kwargs)
                args = list(args); args[1] = rep
                return (tuple(args), kwargs)
            return None
        return hook


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip_vs(qg, img, p):
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def ci(x, rng, paired=False):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
    out = [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out + [float((b > 0).mean())] if paired else out


def main():
    t0 = time.time()
    b9 = json.load(open("results/block9_ext.json"))
    qids = b9["qualified"][:N]
    prev = dict(zip(b9["qualified"], b9["win911"]))
    P = {p["id"]: p for p in make_ext_pairs()}
    log(f"\n===== CFG ROW ablation, window {sorted(WIN)}, {len(qids)} held-out pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = RowStreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    # arms: default path, explicit both-rows (must match), cond-only, uncond-only
    ARMS = [("both", None), ("both01", [0, 1]), ("cond", [1]), ("uncond", [0])]
    res = {a: [] for a, _ in ARMS}
    shape_checked = False

    for n, pid in enumerate(qids):
        p = P[pid]; seed = p["seed"]
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        if not shape_checked:
            s = hooks.cache[9][0].shape
            assert s[0] == 2, f"expected CFG batch 2, got {s}"
            log(f"  stream shape {tuple(s)}: row0=uncond, row1=cond (diffusers cat order)")
            shape_checked = True
        for tag, rows in ARMS:
            hooks.rows = rows
            hooks.begin_inject(set(WIN), isolate=True)
            res[tag].append(flip_vs(qg, gen(pipe, p["clean"], seed), p))
            hooks.off()
        hooks.rows = None
        hooks.cache = {}; hooks.clean = {}; gc.collect()
        log(f"pair{pid} [{n+1}/{len(qids)}] both={res['both'][-1]:.2f} both01={res['both01'][-1]:.2f} "
            f"cond={res['cond'][-1]:.2f} uncond={res['uncond'][-1]:.2f} | submitted={prev.get(pid):.2f} "
            f"| running both={np.mean(res['both']):.3f} cond={np.mean(res['cond']):.3f} "
            f"uncond={np.mean(res['uncond']):.3f}")

    rng = np.random.default_rng(20260828)
    out = dict(qualified=qids, n=len(qids), **{f"{a}": res[a] for a, _ in ARMS})
    for a, _ in ARMS:
        out[f"{a}_ci"] = ci(res[a], rng)
    out["bracket_both_equals_both01"] = res["both"] == res["both01"]
    out["both_reproduces_submitted"] = [int(sum(prev[i] == v for i, v in zip(qids, res["both"]))), len(qids)]
    for a in ("cond", "uncond"):
        d = np.array(res[a], float) - np.array(res["both"], float)
        bd = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        out[f"paired_{a}_minus_both"] = [float(d.mean()), float(np.percentile(bd, 2.5)),
                                         float(np.percentile(bd, 97.5)), float((bd > 0).mean())]
    log(f"\nbracket rows=None == rows=[0,1]: {out['bracket_both_equals_both01']}")
    log(f"both reproduces submitted: {out['both_reproduces_submitted']}")
    for a, _ in ARMS:
        log(f"  {a:7s} {out[f'{a}_ci']}")
    for a in ("cond", "uncond"):
        log(f"  PAIRED {a}-both {out[f'paired_{a}_minus_both']}")
    json.dump(out, open("results/cfg_rows.json", "w"), indent=2)
    log(f"-> results/cfg_rows.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
