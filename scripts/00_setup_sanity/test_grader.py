"""Validate that a grader separates the bracket: clean images vs swap-all images.
If CLIP separates cleanly, use it for the full sweep; else escalate to Qwen."""
import torch, numpy as np, json
from binding_patch import load_pipe, install, encode, generate, Controller
from prompts import make_pairs

DEVICE = "cuda"; STEPS = 30; N = 8

def main():
    pipe = load_pipe(device=DEVICE)
    ctrl = Controller(); install(pipe, ctrl)
    pairs = make_pairs()[:N]

    imgs = []
    for p in pairs:
        g = torch.Generator(device=DEVICE).manual_seed(p["seed"])
        lat = torch.randn((1,4,64,64), generator=g, device=DEVICE, dtype=torch.float16)
        se = encode(pipe, p["swapped"], DEVICE)
        ic = generate(pipe, ctrl, p["clean"], se, lat.clone(), cells=None, steps=STEPS)
        ia = generate(pipe, ctrl, p["clean"], se, lat.clone(), swap_all=True, steps=STEPS)
        ic.save(f"results/g_clean_{p['id']}.png"); ia.save(f"results/g_swapall_{p['id']}.png")
        imgs.append((p, ic, ia))
    del pipe; torch.cuda.empty_cache()

    from grader import CLIPGrader
    clip = CLIPGrader(device=DEVICE)
    print("\n=== CLIP flip_score (clean should be <0, swap-all >0) ===")
    cc, ca = [], []
    for p, ic, ia in imgs:
        fc = clip.flip_score(ic, p["clean"], p["swapped"])
        fa = clip.flip_score(ia, p["clean"], p["swapped"])
        cc.append(fc); ca.append(fa)
        print(f"  pair{p['id']:2d} clean={fc:+.4f}  swapall={fa:+.4f}  delta={fa-fc:+.4f}  | {p['clean']}")
    cc, ca = np.array(cc), np.array(ca)
    print(f"CLIP mean clean={cc.mean():+.4f} swapall={ca.mean():+.4f} sep={ca.mean()-cc.mean():+.4f}")
    print(f"CLIP per-pair delta>0 fraction = {(ca>cc).mean():.2f}")
    del clip; torch.cuda.empty_cache()

    from grader import QwenGrader
    qg = QwenGrader(device=DEVICE)
    print("\n=== Qwen binding_swap_score (clean~0, swap-all~1) ===")
    qc, qa = [], []
    for p, ic, ia in imgs:
        sc,_ = qg.binding_swap_score(ic, p); sa, ans = qg.binding_swap_score(ia, p)
        qc.append(sc); qa.append(sa)
        print(f"  pair{p['id']:2d} clean={sc:.2f} swapall={sa:.2f}  swapall_ans={ans} | {p['clean']}")
    qc, qa = np.array(qc), np.array(qa)
    print(f"Qwen mean clean={qc.mean():.3f} swapall={qa.mean():.3f} sep={qa.mean()-qc.mean():+.3f}")

if __name__ == "__main__":
    main()
