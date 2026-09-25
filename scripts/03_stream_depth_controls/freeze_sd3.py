"""SD3.5 FREEZE-TEXT within-backbone control.

Tests whether attribute binding is causally built by the text-stream EVOLUTION (text
self-attention + context MLP across blocks), rather than being a static property. We freeze
the text stream to its FIRST-BLOCK value at every block (removing the contextualization that
distinguishes joint attention from the read-only isolated-cross-attention case) and ask:
does the model still bind the two objects' colors correctly?

Interpretation:
  - objects still render but binding DEGRADES under freeze -> text-stream evolution is
    causally necessary for binding -> supports "binding is computed in the evolving text
    stream" WITHIN SD3.5 (converts the cross-model migration toward a mechanism).
  - everything degrades (objects gone) -> freeze is too off-distribution -> inconclusive.
  - binding survives freeze -> evolution NOT necessary -> weakens the causal reading.

For each qualified pair we compare NORMAL vs FROZEN clean generation on: object-presence
(does the grader name a real color for each object) and correct binding (o1=c1, o2=c2).
The paper reports the freeze as a failed control: it does not separate binding loss from object
loss (freeze_presence.py); the 'verdict' string this script stores is a run-time heuristic. One GPU.
"""
import torch, json, time, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "10"))
LOG = open("results/sd3_freeze.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade(qg, img, p):
    a1 = qg._ask_color(img, p["o1"]); a2 = qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    present = int(k1 != "other") + int(k2 != "other")          # 0/1/2 objects gradeable
    correct = int(k1 == p["c1"] and k2 == p["c2"])             # clean binding correct
    return present, correct, (a1, a2)


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
    log(f"\n===== SD3.5 FREEZE-TEXT control: {len(qids)} pairs =====")

    pipe = load_pipe(device=DEVICE)
    hooks = StreamHooks(pipe.transformer)
    qg = QwenGrader(device=DEVICE)

    norm_present, norm_correct = [], []
    froz_present, froz_correct = [], []
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        # NORMAL clean
        hooks.off(); imgN = gen(pipe, p["clean"], seed)
        pN, cN, aN = grade(qg, imgN, p)
        # capture clean stream, then FREEZE: every block gets block-0's stream value
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        n_steps = len(hooks.clean[0])
        freeze_cache = {b: [hooks.clean[0][s] for s in range(n_steps)] for b in range(NBLOCKS)}
        hooks.cache = freeze_cache
        hooks.begin_inject(set(range(NBLOCKS)), isolate=False)   # inject block-0 value at all blocks
        imgF = gen(pipe, p["clean"], seed); hooks.off()
        pF, cF, aF = grade(qg, imgF, p)
        hooks.cache = {}; hooks.clean = {}
        norm_present.append(pN); norm_correct.append(cN)
        froz_present.append(pF); froz_correct.append(cF)
        log(f"pair{pid:2d} NORMAL present={pN}/2 bind={cN} ({aN[0]},{aN[1]}) | "
            f"FROZEN present={pF}/2 bind={cF} ({aF[0]},{aF[1]})")

    nP, nC = np.mean(norm_present), np.mean(norm_correct)
    fP, fC = np.mean(froz_present), np.mean(froz_correct)
    log(f"\n=== NORMAL: present {nP:.2f}/2, correct-bind {nC:.2f} | "
        f"FROZEN: present {fP:.2f}/2, correct-bind {fC:.2f} ===")
    if fP >= 0.75 * nP and fC < 0.5 * max(nC, 1e-6):
        verdict = ("FREEZE breaks BINDING while objects survive -> text-stream evolution is "
                   "causally necessary for binding (supports the mechanism).")
    elif fP < 0.5 * nP:
        verdict = ("FREEZE degrades everything (objects gone) -> too off-distribution -> "
                   "inconclusive; rely on the scoped cross-architecture claim.")
    else:
        verdict = ("FREEZE leaves binding largely intact -> evolution not strictly necessary "
                   "-> weakens the strong causal reading; report honestly.")
    log("VERDICT: " + verdict)
    json.dump(dict(qualified=qids, normal_present=norm_present, normal_correct=norm_correct,
                   frozen_present=froz_present, frozen_correct=froz_correct, verdict=verdict),
              open("results/sd3_freeze.json", "w"), indent=2)
    log(f"results -> results/sd3_freeze.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
