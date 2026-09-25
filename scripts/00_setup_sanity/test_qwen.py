"""Standalone Qwen grader validation on already-saved bracket images."""
import os, numpy as np
from PIL import Image
from prompts import make_pairs
from grader import QwenGrader

DEVICE = "cuda"; N = 8

def main():
    qg = QwenGrader(device=DEVICE)
    pairs = make_pairs()[:N]
    qc, qa = [], []
    print("=== Qwen binding_swap_score (clean~0, swap-all~1) ===")
    for p in pairs:
        ic = Image.open(f"results/g_clean_{p['id']}.png")
        ia = Image.open(f"results/g_swapall_{p['id']}.png")
        sc, ansc = qg.binding_swap_score(ic, p)
        sa, ansa = qg.binding_swap_score(ia, p)
        qc.append(sc); qa.append(sa)
        print(f"  pair{p['id']:2d} clean={sc:.2f}{str(ansc):>22} | swapall={sa:.2f}{str(ansa):>22} | {p['clean']}")
    qc, qa = np.array(qc), np.array(qa)
    print(f"\nQwen mean clean={qc.mean():.3f} swapall={qa.mean():.3f} sep={qa.mean()-qc.mean():+.3f}")
    print(f"per-pair swapall>clean fraction = {(qa>qc).mean():.2f}")

if __name__ == "__main__":
    main()
