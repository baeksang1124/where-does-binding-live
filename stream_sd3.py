"""SD3.5 text-stream injection engine (StreamHooks) and the 10-pair pilot that first used it.

StreamHooks registers a forward_pre_hook on every transformer block that captures or replaces the
raw text stream (encoder_hidden_states) ENTERING that block, so the block's context norm, joint
attention, residual and context MLP all see the injected text state. The image tokens read the
text only through the text key/value of joint attention, so at the injected blocks this is an
all-head text key/value swap; what separates it from the per-head / per-block key/value swaps of
binding_patch_sd3.py is granularity (all heads of a block or window) and, in isolate mode, the
clean-pinning of every other block -- not a different channel.

Probes run by main() (pilot, SD3_SUBSET pairs, default 10):
  ALL      inject the swapped stream at every block (reference).
  ISOLATED inject at block b only, every other block pinned to the clean stream (iso_mean).
  NAIVE    inject at block b only, no pinning: the swapped state propagates downstream (prop_mean;
           >= 0.95 for every block up to 9 on the pilot, i.e. propagation, not a per-block effect).
  PREFIX   inject at blocks b..23 -> depth curve.

Writes results/sd3_stream.json. The 'verdict' string stored there is a run-time heuristic from
exploration and is superseded by the paper's analysis. Hook-based capture/inject, independent of
the processor K/V machinery. One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "10"))
LOG = open("results/sd3_stream.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def flip2(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2); f = 0.0
    if k1 == p["c2"] and k1 != p["c1"]:
        f += 0.5
    if k2 == p["c1"] and k2 != p["c2"]:
        f += 0.5
    return f


class StreamHooks:
    """forward_pre_hooks on every transformer_block capturing/injecting the RAW text-stream
    input (encoder_hidden_states) at block granularity.

    Two caches: swapped-run stream (self.cache) and clean-run stream (self.clean). The
    ISOLATE mode pins EVERY block's input to a known stream (swapped for `active` blocks,
    clean for the rest) -> removes the downstream propagation confound so a single-block
    injection truly isolates that block's contribution."""
    def __init__(self, transformer):
        self.tf = transformer
        self.mode = "off"                 # off | capture | capture_clean | inject
        self.cache = {}                   # swapped stream: block_idx -> list[step]
        self.clean = {}                   # clean stream:   block_idx -> list[step]
        self.step = {}
        self.active = set()
        self.isolate = False
        self.handles = []
        for i, blk in enumerate(transformer.transformer_blocks):
            self.handles.append(blk.register_forward_pre_hook(self._mk(i), with_kwargs=True))

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
                self.cache.setdefault(i, []).append(ehs.detach().to("cpu", copy=True))
                return None
            if self.mode == "capture_clean":
                self.clean.setdefault(i, []).append(ehs.detach().to("cpu", copy=True))
                return None
            if self.mode == "inject":
                idx = self.step.get(i, 0); self.step[i] = idx + 1
                src = None
                if i in self.active:
                    src = self.cache[i][idx]
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

    def begin_capture(self):
        self.mode = "capture"; self.cache = {}; self.step = {}

    def begin_capture_clean(self):
        self.mode = "capture_clean"; self.clean = {}; self.step = {}

    def begin_inject(self, active, isolate=False):
        self.mode = "inject"; self.active = set(active); self.isolate = isolate; self.step = {}

    def off(self):
        self.mode = "off"; self.step = {}

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def main():
    t0 = time.time(); os.makedirs("results", exist_ok=True)
    coarse = json.load(open("results/sd3_coarse.json"))
    qids = coarse["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== SD3.5 STREAM injection: {len(qids)} pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    PREFIX_B = [0, 3, 6, 9, 12, 15, 18, 21]
    iso_flip = {b: [] for b in range(NBLOCKS)}     # ISOLATED single-block (propagation removed)
    prop_flip = {b: [] for b in range(NBLOCKS)}    # naive single-block (with propagation)
    prefix_flip = {b: [] for b in PREFIX_B}
    allswap = []

    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        # capture swapped and clean raw streams (same seed)
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        # reference: inject swapped at ALL blocks -> should reproduce swapped binding
        hooks.begin_inject(set(range(NBLOCKS))); imgA = gen(pipe, p["clean"], seed); hooks.off()
        allswap.append(flip2(qg, imgA, p))
        # ISOLATED single-block: swapped at b, CLEAN pinned everywhere else (no propagation)
        for b in range(NBLOCKS):
            hooks.begin_inject({b}, isolate=True); img = gen(pipe, p["clean"], seed); hooks.off()
            iso_flip[b].append(flip2(qg, img, p))
        # naive single-block (with propagation) -- kept for the contrast
        for b in range(NBLOCKS):
            hooks.begin_inject({b}, isolate=False); img = gen(pipe, p["clean"], seed); hooks.off()
            prop_flip[b].append(flip2(qg, img, p))
        # PREFIX injection b..23 (clean depth curve)
        for b in PREFIX_B:
            hooks.begin_inject(set(range(b, NBLOCKS))); img = gen(pipe, p["clean"], seed); hooks.off()
            prefix_flip[b].append(flip2(qg, img, p))
        hooks.cache = {}; hooks.clean = {}
        im = {b: float(np.mean(iso_flip[b])) for b in range(NBLOCKS)}
        top = sorted(im, key=im.get, reverse=True)[:5]
        log(f"pair{pid:2d} allswap={allswap[-1]:.2f} | top ISOLATED single-blocks: "
            + ", ".join(f"b{b}={im[b]:.2f}" for b in top))

    iso_mean = {b: float(np.mean(iso_flip[b])) for b in range(NBLOCKS)}
    prop_mean = {b: float(np.mean(prop_flip[b])) for b in range(NBLOCKS)}
    prefix_mean = {b: float(np.mean(prefix_flip[b])) for b in PREFIX_B}
    ranked = sorted(range(NBLOCKS), key=lambda b: iso_mean[b], reverse=True)
    log(f"\n--- ALL-block stream inject (ref): flip={np.mean(allswap):.3f} ---")
    log(f"--- ISOLATED single-block (propagation killed; vs K/V coarse ceiling 0.114) ---")
    for b in ranked[:12]:
        log(f"  block {b:2d}: iso={iso_mean[b]:.3f}  (naive-with-propagation={prop_mean[b]:.3f})")
    log(f"--- PREFIX injection (swapped text from depth b onward) ---")
    for b in PREFIX_B:
        log(f"  from block {b:2d}: {prefix_mean[b]:.3f}")

    best_iso = max(iso_mean.values())
    nz_iso = sum(1 for v in iso_mean.values() if v > 0.10)
    verdict = ("STREAM-LOCALIZED: an ISOLATED single block's text-stream injection flips binding "
               f"(max {best_iso:.3f} >> 0.114 K/V ceiling; {nz_iso} blocks >0.10) -> binding is accessible to intervention "
               "in the text stream/MLP at specific depth, not image-facing K/V (hypothesis SUPPORTED)"
               if best_iso >= 0.35 else
               f"STREAM-DISTRIBUTED-IN-DEPTH: isolated single-block low (max {best_iso:.3f}); binding "
               "built incrementally -- see prefix depth curve for where it locks in")
    log(f"\n  VERDICT: {verdict}")

    json.dump(dict(qualified=qids, allswap_ref=float(np.mean(allswap)),
                   iso_mean=iso_mean, prop_mean=prop_mean, prefix_mean=prefix_mean,
                   ranked_blocks=ranked, verdict=verdict),
              open("results/sd3_stream.json", "w"), indent=2)
    log(f"\nresults -> results/sd3_stream.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
