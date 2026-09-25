"""Non-color attribute binding pairs (MATERIAL, SIZE) to test whether the binding
migration + depth lock-in generalize beyond color (DifFRACT was color-only too).

Same structure as prompts.py: each pair shares two objects and two attribute values;
only the binding (which attribute attaches to which object) is swapped. Fixed seed/pair.
"""

OBJECTS = ["cube", "sphere", "car", "cup", "book"]

ATTRS = {
    "material": {
        "values": ["wooden", "metal", "glass", "plastic"],
        # canonical substrings the VQA answer may contain -> base value
        "canon": {"wood": "wooden", "metal": "metal", "glass": "glass", "plastic": "plastic",
                  "metallic": "metal", "wooden": "wooden"},
        "question": "What material is the {obj} made of? Answer with a single word "
                    "(wooden, metal, glass, or plastic).",
        "tmpl": "a {a} {o}",   # "a wooden cube"
    },
    "size": {
        "values": ["large", "small"],
        "canon": {"large": "large", "small": "small", "big": "large", "tiny": "small",
                  "huge": "large", "little": "small"},
        "question": "Is the {obj} large or small? Answer with a single word (large or small).",
        "tmpl": "a {a} {o}",
    },
}

# Hand-built object pairs (o1, o2); attribute values assigned per pair from the attr's set.
_OBJ_PAIRS = [
    ("cube", "sphere"), ("car", "cup"), ("book", "cube"), ("sphere", "car"),
    ("cup", "book"), ("cube", "car"), ("sphere", "book"), ("car", "cube"),
    ("book", "sphere"), ("cup", "cube"), ("cube", "cup"), ("sphere", "cup"),
]


def make_pairs(attr):
    """Return list of dicts for the given attribute. clean: a1->o1, a2->o2 ;
    swapped: a2->o1, a1->o2. Fields: id, seed, clean, swapped, o1, o2, a1, a2."""
    cfg = ATTRS[attr]
    vals = cfg["values"]
    tmpl = cfg["tmpl"]
    pairs = []
    for i, (o1, o2) in enumerate(_OBJ_PAIRS):
        # adjacent rotation guarantees a1 != a2 for any value-set size (incl. 2)
        a1 = vals[i % len(vals)]
        a2 = vals[(i + 1) % len(vals)]
        clean = f"{tmpl.format(a=a1, o=o1)} next to {tmpl.format(a=a2, o=o2)}"
        swapped = f"{tmpl.format(a=a2, o=o1)} next to {tmpl.format(a=a1, o=o2)}"
        pairs.append(dict(id=i, seed=2000 + i, clean=clean, swapped=swapped,
                          o1=o1, o2=o2, a1=a1, a2=a2))
    return pairs


if __name__ == "__main__":
    for attr in ATTRS:
        ps = make_pairs(attr)
        print(f"=== {attr}: {len(ps)} pairs ===")
        for p in ps[:3]:
            print("  ", p["clean"], "  ||  ", p["swapped"])
