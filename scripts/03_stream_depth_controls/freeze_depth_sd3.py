"""Freeze-DEPTH sweep. The block-0 freeze (freeze_sd3.py) is partly off-distribution (object
presence also drops), so its binding loss could reflect general degradation. Here we freeze the
SD3.5 text stream at its block-b value FROM block b onward (blocks <b evolve normally), sweeping b.

Key test of binding-specificity: if there is a depth b where object PRESENCE is restored
(on-distribution, both objects render) yet correct BINDING is still degraded, the freeze
effect is binding-specific, not general degradation -- a cleaner within-backbone causal
control than the block-0 freeze. Reports correct-bind and object-presence vs b. One GPU.

Note: "presence" here is a proxy -- an object counts as present when the grader returns a
recognised color word for it. freeze_presence.py re-runs the same sweep and measures presence
with explicit yes/no questions. With those, presence is not restored at the depths where binding
breaks, so (as the paper reports) this control does not separate binding from object loss; the
'verdict' string stored in results/sd3_freeze_depth.json is a run-time heuristic.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "10"))
BS = [0, 3, 6, 9, 11, 13, 15, 18, 21]
LOG = open("results/sd3_freeze_depth.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def grade(qg, img, p):
    k1 = canon(qg._ask_color(img, p["o1"])); k2 = canon(qg._ask_color(img, p["o2"]))
    present = int(k1 != "other") + int(k2 != "other")
    correct = int(k1 == p["c1"] and k2 == p["c2"])
    return present, correct


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    qids = json.load(open("results/sd3_coarse.json"))["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== SD3.5 FREEZE-DEPTH sweep: {len(qids)} pairs, freeze-from b in {BS} =====")
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    present = {b: [] for b in BS}; correct = {b: [] for b in BS}
    norm_present, norm_correct = [], []
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        hooks.off(); imgN = gen(pipe, p["clean"], seed); pN, cN = grade(qg, imgN, p)
        norm_present.append(pN); norm_correct.append(cN)
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        nsteps = len(hooks.clean[0])
        for b in BS:
            # freeze from block b onward: every block j>=b gets block-b's value held constant
            fake = {j: [hooks.clean[b][s] for s in range(nsteps)] for j in range(b, NBLOCKS)}
            saved = hooks.cache; hooks.cache = fake
            hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
            img = gen(pipe, p["clean"], seed); hooks.off(); hooks.cache = saved
            pr, co = grade(qg, img, p); present[b].append(pr); correct[b].append(co)
        hooks.clean = {}
        log(f"pair{pid:2d} normal(pres={pN},bind={cN}) | "
            + " ".join(f"b{b}:{np.mean(present[b][-1:]):.0f}/{correct[b][-1]}" for b in BS))

    log(f"\n=== NORMAL: presence {np.mean(norm_present):.2f}/2  bind {np.mean(norm_correct):.2f} ===")
    log("freeze-from-b : object-presence (/2)   correct-bind")
    best = None
    for b in BS:
        pr = float(np.mean(present[b])); co = float(np.mean(correct[b]))
        log(f"  b={b:2d} : presence {pr:.2f}   bind {co:.2f}")
        # binding-specific signature: presence high (>=1.7) but bind degraded (<0.5)
        if pr >= 1.7 and co < 0.5 and best is None:
            best = b
    verdict = (f"BINDING-SPECIFIC: at freeze-from-b={best}, objects still render (presence>=1.7) "
               f"but binding is degraded (<0.5) -> the freeze effect is not general degradation."
               if best is not None else
               "No clean on-distribution binding break found; binding loss co-occurs with presence loss "
               "(freeze remains partly confounded).")
    log("VERDICT: " + verdict)
    json.dump(dict(bs=BS, subset=qids, normal_presence=float(np.mean(norm_present)),
                   normal_bind=float(np.mean(norm_correct)),
                   presence={b: float(np.mean(present[b])) for b in BS},
                   bind={b: float(np.mean(correct[b])) for b in BS}, verdict=verdict),
              open("results/sd3_freeze_depth.json", "w"), indent=2)


if __name__ == "__main__":
    main()
