"""SD1.5 tiles for the qualitative Fig. 5 (one GPU): clean | top-1 head K/V swap | swapped prompt (+ all-head swap
check) for the pairs in F5S_IDS (default 5072; base-set ids also work), rendered with the
code path of scripts/02_heldout_generalization/top1_sd15_ext.py via f5_candidates.gen_sd15 and re-graded with its
QwenGrader rule (o1 swap score). Writes results/figs/f5_candidates/<id>_sd15_<arm>.png and <id>_sd15_grades.json.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/figures/f5_sd15_tiles.py
"""
import os, sys, json
os.environ.setdefault("F5C_OUT", os.environ.get("F5S_OUT", "results/figs/f5_candidates"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts/figures"))
import f5_candidates as F
from prompts import make_pairs
from prompts_ext import make_ext_pairs

IDS = [int(x) for x in os.environ.get("F5S_IDS", "5072").split(",")]


def main():
    os.makedirs(F.OUT, exist_ok=True)
    P = {p["id"]: p for p in make_ext_pairs() + make_pairs()}
    brackets = {}
    F.gen_sd15(IDS, P, brackets)
    G = F.grade_all({"sd15": IDS, "sd3": []}, P)
    st = json.load(open("results/top1_sd15_ext.json")); stored = dict(zip(st["ids"], st["per_pair"]))
    for pid in IDS:
        g = G[pid]["sd15"]
        rec = dict(pair=pid, clean=P[pid]["clean"], swapped=P[pid]["swapped"], grades=g, brackets=brackets.get(pid),
                   stored_top1=stored.get(pid), regraded_top1=g["top1"]["o1_swap_score"],
                   qualified=(g["clean"]["o1_swap_score"] == 0 and g["swapall"]["o1_swap_score"] == 1))
        json.dump(rec, open(f"{F.OUT}/{pid}_sd15_grades.json", "w"), indent=1)
        print(pid, {k: rec[k] for k in ("stored_top1", "regraded_top1", "qualified")}, flush=True)


if __name__ == "__main__":
    main()
