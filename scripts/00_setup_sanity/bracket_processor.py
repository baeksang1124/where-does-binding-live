"""2-pair bracket: is the clean generation bit-identical with vs without install(pipe, ctrl)
(off mode)? Checks for a processor asymmetry between the swap arm (block9_ext, ctrl
installed) and the placebo arm (window_placebo_paired, stock processor)."""
import torch, numpy as np
from prompts_ext import make_ext_pairs
from binding_patch_sd3 import load_pipe, install, Controller
DEVICE="cuda"; STEPS=28; GUID=7.0; H=W=512
P = {p["id"]: p for p in make_ext_pairs()}
pairs = [P[5000], P[5001]]
pipe = load_pipe(device=DEVICE)
def gen(prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return np.asarray(pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID, height=H, width=W, generator=g).images[0], dtype=np.int16)
stock = [gen(p["clean"], p["seed"]) for p in pairs]
ctrl = Controller(); install(pipe, ctrl)          # off mode: no swap cells set
inst = [gen(p["clean"], p["seed"]) for p in pairs]
for i, p in enumerate(pairs):
    d = np.abs(stock[i] - inst[i])
    print(f"pair{p['id']}: maxL1={d.max()} meanL1={d.mean():.4f} identical={bool((d==0).all())}")
