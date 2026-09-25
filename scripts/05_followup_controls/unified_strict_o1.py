"""Unified strict o1 metric (Table 1, 'unified strict o1' rows) recomputed from the raw grader answers.

Metric: 1 iff the grader's canonicalized answer for o1 is the swapped color c2, else 0 -- identical for all
three models, on the same held-out images as the per-model rows (results/dual_grade_full.json 'raw').

CIs:
  * single-model means: EXACT percentile bootstrap. For a mean of 0/1 scores the bootstrap distribution
    is exactly Binomial(n, p_hat)/n, so its 2.5/97.5 percentiles are computed from the binomial CDF
    (smallest k with CDF >= q). This removes Monte-Carlo noise, which can put PixArt's upper bound at
    7/26=0.269 instead of 6/26=0.231 (P(X<=6)=0.9754, right at 0.975).
  * between-model differences: independent-means percentile bootstrap, 1e5 resamples, seed 1234
    (as in dual_grade_full.py), plus P(diff>0).

CPU only. Run from the repo root:
    PYTHONPATH=. $PY scripts/05_followup_controls/unified_strict_o1.py
Writes results/unified_strict_o1.json.
"""
import json, re
import numpy as np
from scipy.stats import binom
from prompts import make_pairs, COLORS
from prompts_ext import make_ext_pairs

RAW = "results/dual_grade_full.json"
OUT = "results/unified_strict_o1.json"
MODELS = ("sd15", "pixart", "sd3")
NBOOT, SEED = 100000, 1234


def canon(a):                       # same rule as dual_grade_full.py
    a = a.lower(); h = [c for c in COLORS if re.search(rf"\b{c}", a)]
    return h[0] if len(h) == 1 else "other"


def exact_boot_ci(x):
    n, k = len(x), int(sum(x)); p = k / n
    lo, hi = binom.ppf(0.025, n, p) / n, binom.ppf(0.975, n, p) / n
    return [p, float(lo), float(hi)]


def diff_ci(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = rng.choice(a, (NBOOT, len(a))).mean(1) - rng.choice(b, (NBOOT, len(b))).mean(1)
    return [float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float((d > 0).mean())]


def main():
    P = {p["id"]: p for p in make_pairs()}; P.update({p["id"]: p for p in make_ext_pairs()})
    raw = json.load(open(RAW))["raw"]
    out = {"note": "unified strict o1 (1 iff o1 reads the swapped color c2, else 0), recomputed from "
                   "dual_grade_full.json raw answers by scripts/05_followup_controls/unified_strict_o1.py; "
                   "single-model CIs are exact percentile-bootstrap (binomial), differences 1e5-resample "
                   "percentile bootstrap (seed 1234) with P(diff>0). Source for Table 1 unified rows."}
    S = {}
    for g in ("qwen", "llava"):
        S[g] = {}
        for m in MODELS:
            keys = sorted(k for k in raw if k.startswith(m + "_"))
            S[g][m] = [float(canon(raw[k][g]["a1"]) == P[int(k.split("_")[1])]["c2"]) for k in keys]
            out.setdefault(m, {})[f"{g}_strict"] = exact_boot_ci(S[g][m]); out[m]["n"] = len(keys)
    for g in ("qwen", "llava"):
        rng = np.random.default_rng(SEED)
        for a, b in (("sd15", "sd3"), ("sd15", "pixart"), ("pixart", "sd3")):
            out[f"{g}_diff_{a}_{b}"] = diff_ci(S[g][a], S[g][b], rng)
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in out.items():
        if k != "note":
            print(k, v)


if __name__ == "__main__":
    main()
