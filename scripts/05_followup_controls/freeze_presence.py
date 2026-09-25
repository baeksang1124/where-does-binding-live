"""Re-measure object presence for the freeze-from-b sweep. freeze_depth_sd3.py counts "object
presence" as the number of objects for which the grader returns a recognisable COLOUR word, which
is not the same as whether the object is in the image. This script re-runs the identical
freeze-from-b sweep on the identical 10 pairs and grades each image with (a) the original colour
questions (bracket: reproduces sd3_freeze_depth.json presence/bind means) and (b) explicit yes/no
PRESENCE questions per object, plus (c) a count of distinct objects, so a statement that both
objects render rests on a presence measurement.
One GPU. Writes results/freeze_presence.json (overwritten on re-run) and appends to results/freeze_presence.log.
"""
import torch, json, os, time, re
import numpy as np
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("FP_SUBSET", "10"))
BS = [0, 3, 6, 9, 11, 13, 15, 18, 21]
os.makedirs("results", exist_ok=True)
LOG = open("results/freeze_presence.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(a):
    a = (a or "").lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def yes(ans):
    a = (ans or "").strip().lower()
    return 1 if a.startswith("yes") else 0


def grade(qg, img, p):
    a1, a2 = qg._ask_color(img, p["o1"]), qg._ask_color(img, p["o2"])
    k1, k2 = canon(a1), canon(a2)
    colour_present = int(k1 != "other") + int(k2 != "other")          # colour-word "presence" as in freeze_depth_sd3.py
    bind = int(k1 == p["c1"] and k2 == p["c2"])
    y1 = qg._ask(img, f"Is there a {p['o1']} in this image? Answer yes or no.")
    y2 = qg._ask(img, f"Is there a {p['o2']} in this image? Answer yes or no.")
    present = yes(y1) + yes(y2)
    cnt = qg._ask(img, "How many distinct objects are in this image? Answer with a single number.")
    m = re.search(r"\d+", cnt or ""); count = int(m.group()) if m else -1
    return dict(colour_present=colour_present, bind=bind, present=present, count=count,
                raw=dict(c1=a1, c2=a2, y1=y1, y2=y2, cnt=cnt))


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]


def main():
    t0 = time.time()
    old = json.load(open("results/sd3_freeze_depth.json"))
    qids = old["subset"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== FREEZE-DEPTH presence re-measurement: {len(qids)} pairs, b in {BS} =====")
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    per = {"normal": []}; per.update({str(b): [] for b in BS})
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        hooks.off(); per["normal"].append(grade(qg, gen(pipe, p["clean"], seed), p))
        hooks.begin_capture_clean(); _ = gen(pipe, p["clean"], seed); hooks.off()
        nsteps = len(hooks.clean[0])
        for b in BS:
            fake = {j: [hooks.clean[b][s] for s in range(nsteps)] for j in range(b, NBLOCKS)}
            hooks.cache = fake; hooks.begin_inject(set(range(b, NBLOCKS)), isolate=False)
            img = gen(pipe, p["clean"], seed); hooks.off(); hooks.cache = {}
            per[str(b)].append(grade(qg, img, p))
        hooks.clean = {}
        log(f"pair{pid:2d} normal(pres={per['normal'][-1]['present']},cnt={per['normal'][-1]['count']},bind={per['normal'][-1]['bind']}) | "
            + " ".join(f"b{b}:p{per[str(b)][-1]['present']}/c{per[str(b)][-1]['colour_present']}/n{per[str(b)][-1]['count']}/b{per[str(b)][-1]['bind']}" for b in BS))

    def mean(k, f):
        return float(np.mean([r[f] for r in per[k]]))
    summ = {k: dict(colour_present=mean(k, "colour_present"), present=mean(k, "present"), bind=mean(k, "bind"),
                    count=mean(k, "count"), both_present=float(np.mean([r["present"] == 2 for r in per[k]])),
                    count_ge2=float(np.mean([r["count"] >= 2 for r in per[k]]))) for k in per}
    repro = {b: (abs(summ[str(b)]["colour_present"] - old["presence"][str(b)]) < 1e-9 and abs(summ[str(b)]["bind"] - old["bind"][str(b)]) < 1e-9) for b in BS}
    log(f"\nbracket: colour-presence/bind reproduce sd3_freeze_depth.json per depth: {sum(repro.values())}/{len(BS)}")
    log("depth   colour-pres(/2, submitted)  yes/no-presence(/2)  both-present  count>=2  bind")
    for k in ["normal"] + [str(b) for b in BS]:
        s = summ[k]
        log(f"  {k:6s}  {s['colour_present']:.2f}                        {s['present']:.2f}                 {s['both_present']:.2f}          {s['count_ge2']:.2f}      {s['bind']:.2f}")
    json.dump(dict(bs=BS, subset=qids, summary=summ, reproduces_submitted=repro, per=per),
              open("results/freeze_presence.json", "w"), indent=2)
    log(f"-> results/freeze_presence.json ; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
