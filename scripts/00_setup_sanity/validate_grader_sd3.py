"""Second-grader validation of the SD3.5 CHANNEL result (head-swap vs text-stream inject),
to close the mono-grader caveat. Generates+saves images once, then grades with Qwen2.5-VL-7B
AND LLaVA-1.5-7B (different family) sequentially (memory-safe: models never co-resident).
Reports each grader's head-flip and stream-flip means + per-image agreement. One GPU.
"""
import torch, json, os, gc, numpy as np, re
from PIL import Image
from prompts import make_pairs, COLORS

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0
SUBSET = int(os.environ.get("SD3_SUBSET", "8"))
IMGDIR = "results/validate_imgs"; os.makedirs(IMGDIR, exist_ok=True)


def canon(a):
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


# ---------- Phase A: generate + save images ----------
def gen_images():
    from binding_patch_sd3 import load_pipe, install, Controller, capture_swapped, generate
    from stream_sd3 import StreamHooks
    hs = json.load(open("results/sd3_headsweep.json")); b, h = hs["ranked"][0].split("|h"); top1 = [(int(b), int(h))]
    qids = json.load(open("results/sd3_coarse.json"))["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    pipe = load_pipe(device=DEVICE); ctrl = Controller(); install(pipe, ctrl)
    hooks = StreamHooks(pipe.transformer)
    meta = []
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        capture_swapped(pipe, ctrl, p["swapped"], seed, steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        head = generate(pipe, ctrl, p["clean"], seed, cells=top1, swap_k=True, swap_v=True,
                        steps=STEPS, guidance=GUID, height=H, width=W, device=DEVICE)
        ctrl.swap_ctx = {}
        hooks.begin_capture()
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        _ = pipe(p["swapped"], num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]
        hooks.off(); hooks.begin_inject(set(range(24)))
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        stream = pipe(p["clean"], num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0]
        hooks.off(); hooks.cache = {}
        head.save(f"{IMGDIR}/p{pid}_head.png"); stream.save(f"{IMGDIR}/p{pid}_stream.png")
        meta.append(dict(pid=pid, o1=p["o1"], o2=p["o2"], c1=p["c1"], c2=p["c2"]))
    json.dump(meta, open(f"{IMGDIR}/meta.json", "w"))
    del pipe, hooks; gc.collect(); torch.cuda.empty_cache()
    return meta


def flip2(grader, img, m):
    k1 = canon(grader._ask_color(img, m["o1"])); k2 = canon(grader._ask_color(img, m["o2"]))
    f = 0.0
    if k1 == m["c2"] and k1 != m["c1"]:
        f += 0.5
    if k2 == m["c1"] and k2 != m["c2"]:
        f += 0.5
    return f


def grade_all(grader, meta):
    out = {}
    for m in meta:
        head = Image.open(f"{IMGDIR}/p{m['pid']}_head.png")
        stream = Image.open(f"{IMGDIR}/p{m['pid']}_stream.png")
        out[m["pid"]] = dict(head=flip2(grader, head, m), stream=flip2(grader, stream, m))
    return out


def main():
    meta = gen_images()
    print(f"[gen] saved {len(meta)} pairs x 2 imgs")
    from grader import QwenGrader
    qg = QwenGrader(device=DEVICE); qres = grade_all(qg, meta)
    del qg; gc.collect(); torch.cuda.empty_cache()
    from grader import LlavaGrader
    lg = LlavaGrader(device=DEVICE); lres = grade_all(lg, meta)
    del lg; gc.collect(); torch.cuda.empty_cache()

    ph_q = [qres[m["pid"]]["head"] for m in meta]; ps_q = [qres[m["pid"]]["stream"] for m in meta]
    ph_l = [lres[m["pid"]]["head"] for m in meta]; ps_l = [lres[m["pid"]]["stream"] for m in meta]
    agree = np.mean([int(qres[m["pid"]]["stream"] == lres[m["pid"]]["stream"]) for m in meta] +
                    [int(qres[m["pid"]]["head"] == lres[m["pid"]]["head"]) for m in meta])
    print(f"\n=== SD3.5 CHANNEL, dual-grader (n={len(meta)}) ===")
    print(f"  Qwen : head-flip={np.mean(ph_q):.3f}  stream-flip={np.mean(ps_q):.3f}")
    print(f"  LLaVA: head-flip={np.mean(ph_l):.3f}  stream-flip={np.mean(ps_l):.3f}")
    print(f"  exact per-image agreement (both conds): {agree:.3f}")
    print(f"  VERDICT: {'CORROBORATED (both graders: stream>>head)' if np.mean(ps_l)-np.mean(ph_l)>0.4 and np.mean(ps_q)-np.mean(ph_q)>0.4 else 'DIVERGENT -- inspect'}")
    json.dump(dict(n=len(meta), qwen=dict(head=float(np.mean(ph_q)), stream=float(np.mean(ps_q))),
                   llava=dict(head=float(np.mean(ph_l)), stream=float(np.mean(ps_l))), agreement=float(agree)),
              open("results/validate_grader_sd3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
