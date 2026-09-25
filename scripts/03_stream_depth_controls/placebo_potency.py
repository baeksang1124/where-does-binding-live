"""Placebo POTENCY control + whole-word-canon re-verification.

The third-color placebo scores 0.00 on the c1/c2 flip metric partly BY CONSTRUCTION (it
injects colors c3/c4 disjoint from the scored c1/c2). To show the placebo injection is
nonetheless POTENT (a real, effective intervention that simply does not carry the swap),
we grade the placebo-injected image against the placebo's OWN binding: does o1 read c3 and
o2 read c4? High potency + zero flip = the text-stream effect is binding-specific, not an
artifact of "any injection disrupts binding".

Also re-measures the matched all-block flip and placebo flip on the same pairs with the
whole-word canon() (regex word boundary; an earlier parser matched color substrings and gave
0.983 / 0.000), checking that the result is unchanged under the stricter grader parsing.
One GPU. Writes results/placebo_potency.{json,log}.
"""
import torch, json, os, numpy as np, re
from prompts import make_pairs, COLORS
from binding_patch_sd3 import load_pipe
from stream_sd3 import StreamHooks
from grader import QwenGrader

DEVICE = "cuda"; STEPS = 28; H = W = 512; GUID = 7.0; NBLOCKS = 24
SUBSET = int(os.environ.get("SD3_SUBSET", "30"))
LOG = open("results/placebo_potency.log", "a")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def canon(ans):
    ans = ans.lower(); hit = [c for c in COLORS if re.search(rf"\b{c}", ans)]
    return hit[0] if len(hit) == 1 else "other"


def grade2(qg, img, p):
    return canon(qg._ask_color(img, p["o1"])), canon(qg._ask_color(img, p["o2"]))


def flip2_vs(k1, k2, c1, c2):
    f = 0.0
    if k1 == c2 and k1 != c1:
        f += 0.5
    if k2 == c1 and k2 != c2:
        f += 0.5
    return f


@torch.no_grad()
def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
                height=H, width=W, generator=g).images[0]


def main():
    os.makedirs("results", exist_ok=True)
    qids = json.load(open("results/sd3_coarse.json"))["qualified"][:SUBSET]
    pairs = {p["id"]: p for p in make_pairs()}
    log(f"\n===== PLACEBO POTENCY + whole-word-canon reverify: {len(qids)} pairs =====")
    pipe = load_pipe(device=DEVICE); hooks = StreamHooks(pipe.transformer); qg = QwenGrader(device=DEVICE)

    matched_flip, placebo_flip, potency = [], [], []   # potency: /2 objects reading c3/c4
    for pid in qids:
        p = pairs[pid]; seed = p["seed"]
        others = [c for c in COLORS if c not in (p["c1"], p["c2"])]
        c3, c4 = (others + COLORS)[:2]
        placebo_prompt = f"a {c3} {p['o1']} next to a {c4} {p['o2']}"

        # matched all-block inject (fixed canon re-verify)
        hooks.begin_capture(); _ = gen(pipe, p["swapped"], seed); hooks.off()
        hooks.begin_inject(set(range(NBLOCKS)))
        k1, k2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        matched_flip.append(flip2_vs(k1, k2, p["c1"], p["c2"]))

        # placebo all-block inject: flip vs c1/c2 (expect 0) AND potency vs c3/c4
        hooks.begin_capture(); _ = gen(pipe, placebo_prompt, seed); hooks.off()
        hooks.begin_inject(set(range(NBLOCKS)))
        q1, q2 = grade2(qg, gen(pipe, p["clean"], seed), p); hooks.off()
        placebo_flip.append(flip2_vs(q1, q2, p["c1"], p["c2"]))
        potency.append(int(q1 == c3) + int(q2 == c4))
        hooks.cache = {}; hooks.clean = {}
        log(f"pair{pid:2d} matched={matched_flip[-1]:.2f} placebo={placebo_flip[-1]:.2f} "
            f"potency={potency[-1]}/2 (read {q1}/{q2}, want {c3}/{c4})")

    def ci(x):
        x = np.array(x, float); rng = np.random.default_rng(1234)
        bt = x[rng.integers(0, len(x), (100000, len(x)))].mean(1)
        return float(x.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))

    m, ml, mh = ci(matched_flip); pl, pll, plh = ci(placebo_flip)
    pot = np.array(potency, float) / 2.0; pm, pml, pmh = ci(pot)
    log(f"\nMATCHED  flip (whole-word canon): {m:.3f} [{ml:.3f},{mh:.3f}]  (was 0.983 substring)")
    log(f"PLACEBO  flip (whole-word canon): {pl:.3f} [{pll:.3f},{plh:.3f}]  (was 0.000)")
    log(f"PLACEBO POTENCY (own c3/c4 bind): {pm:.3f} [{pml:.3f},{pmh:.3f}]  per-object; "
        f"full 2/2 on {sum(1 for x in potency if x == 2)}/{len(potency)} pairs")
    json.dump(dict(qualified=qids, matched_flip=matched_flip, placebo_flip=placebo_flip,
                   potency_per_pair=potency,
                   matched_ci=[m, ml, mh], placebo_ci=[pl, pll, plh], potency_ci=[pm, pml, pmh]),
              open("results/placebo_potency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
