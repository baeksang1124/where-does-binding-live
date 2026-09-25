"""Extended color-binding candidate pairs (larger pool for the held-out / top-1 measurements).
Systematic combinations of two distinct colors x two distinct objects, swapped binding.
Distinct id/seed space (offset 5000) so it never collides with prompts.py pairs.
"""
from itertools import permutations

COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["cube", "sphere", "car", "cup", "book", "ball", "bottle", "box"]


def make_ext_pairs(limit=0):
    pairs = []
    i = 0
    # every ordered distinct color pair x a rotating set of distinct object pairs
    obj_pairs = [(o1, o2) for o1, o2 in permutations(OBJECTS, 2)]
    col_pairs = [(c1, c2) for c1, c2 in permutations(COLORS, 2)]
    # interleave to maximize diversity, cap by limit
    for k, (c1, c2) in enumerate(col_pairs):
        for j in range(len(OBJECTS) - 1):
            o1, o2 = obj_pairs[(k * 3 + j) % len(obj_pairs)]
            clean = f"a {c1} {o1} next to a {c2} {o2}"
            swapped = f"a {c2} {o1} next to a {c1} {o2}"
            pairs.append(dict(id=5000 + i, seed=5000 + i, clean=clean, swapped=swapped,
                              c1=c1, o1=o1, c2=c2, o2=o2))
            i += 1
    # de-dup by (clean) text
    seen = set(); uniq = []
    for p in pairs:
        if p["clean"] in seen:
            continue
        seen.add(p["clean"]); uniq.append(p)
    if limit:
        uniq = uniq[:limit]
    return uniq


if __name__ == "__main__":
    ps = make_ext_pairs()
    print(f"{len(ps)} extended candidate pairs")
    for p in ps[:5]:
        print("  ", p["clean"], "||", p["swapped"])
