"""PixArt-Sigma tiles for the qualitative Fig. 5 (one GPU): clean | top-1 head K/V swap | swapped prompt, for the
pairs given in F5P_IDS (default 5029), rendered with the code path of
scripts/02_heldout_generalization/top1_pixart_ext.py (same latent(), constants, top-1 cell and grading rule) and
re-graded with QwenGrader. The all-head swap is also rendered and compared with the swapped-prompt image.
Writes results/figs/f5_candidates/<id>_pixart_<arm>.png and pixart_tiles.json there; reads results/*.json only.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/figures/f5_pixart_tiles.py
"""
import os, sys, json, gc, tempfile, contextlib
import numpy as np
import torch
from prompts_ext import make_ext_pairs
from grader import QwenGrader
from binding_patch_pixart import load_pipe, install, encode, generate, Controller

OUT = "results/figs/f5_candidates"
IDS = [int(x) for x in os.environ.get("F5P_IDS", "5029").split(",")]
ROOT = os.getcwd()


@contextlib.contextmanager
def _cwd(d):
    old = os.getcwd(); os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


# the measurement script opens results/top1_pixart_ext.log at import time -> import it from a scratch directory
_SANDBOX = tempfile.mkdtemp(prefix="f5p_import_"); os.makedirs(f"{_SANDBOX}/results")
sys.path.insert(0, os.path.join(ROOT, "scripts/02_heldout_generalization"))
with _cwd(_SANDBOX):
    import top1_pixart_ext as T


def main():
    os.makedirs(OUT, exist_ok=True)
    P = {p["id"]: p for p in make_ext_pairs()}
    sw = json.load(open("results/pixart_sweep.json")); ln, h = sw["ranked"][0].split("|h"); top1 = [(ln, int(h))]
    stored = dict(zip(*[json.load(open("results/top1_pixart_ext.json"))[k] for k in ("qualified_ids", "per_pair")]))
    pipe = load_pipe(device=T.DEVICE); ctrl = Controller(); install(pipe, ctrl)
    imgs = {}
    for pid in IDS:
        p = P[pid]; emb, _ = encode(pipe, p["swapped"], T.DEVICE); lat = T.latent(pipe, p["seed"])
        im = dict(clean=generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=None, steps=T.STEPS, guidance=T.GUID),
                  swapall=generate(pipe, ctrl, p["clean"], emb, lat.clone(), swap_all=True, steps=T.STEPS, guidance=T.GUID),
                  top1=generate(pipe, ctrl, p["clean"], emb, lat.clone(), cells=top1, swap_k=True, swap_v=True,
                                steps=T.STEPS, guidance=T.GUID),
                  target=generate(pipe, ctrl, p["swapped"], emb, lat.clone(), cells=None, steps=T.STEPS, guidance=T.GUID))
        for k, v in im.items():
            v.save(f"{OUT}/{pid}_pixart_{k}.png")
        imgs[pid] = im
    del pipe, ctrl; gc.collect(); torch.cuda.empty_cache()
    qg = QwenGrader(device=T.DEVICE); rec = {}
    for pid, im in imgs.items():
        p = P[pid]; g = {k: T.grade2(qg, v, p) for k, v in im.items()}
        k1, k2 = g["top1"]
        f = (int(k1 == p["c2"] and k1 != p["c1"]) + int(k2 == p["c1"] and k2 != p["c2"])) / 2.0
        rec[pid] = dict(clean=p["clean"], swapped=p["swapped"], answers={k: list(v) for k, v in g.items()},
                        qualified=(g["clean"] == (p["c1"], p["c2"]) and g["swapall"] == (p["c2"], p["c1"])),
                        top1_flip_regraded=f, top1_flip_stored=stored.get(pid),
                        target_vs_swapall_maxabs=int(np.abs(np.asarray(im["target"], np.int16)
                                                            - np.asarray(im["swapall"], np.int16)).max()))
        print(pid, rec[pid], flush=True)
    json.dump(dict(top1_cell=sw["ranked"][0], steps=T.STEPS, guidance=T.GUID, pairs=rec),
              open(f"{OUT}/pixart_tiles.json", "w"), indent=1)


if __name__ == "__main__":
    main()
