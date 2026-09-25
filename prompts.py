"""Minimal two-object color-binding prompt pairs.

Each pair shares the SAME two colors and SAME two objects; only the binding (which
color attaches to which object) is swapped. Fixed seed per pair => the ONLY variable
across a sweep is the patched head.
"""

COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["cube", "sphere", "car", "cup", "book"]

# Hand-built diverse set of 36 pairs (color1 obj1 / color2 obj2) -> swapped binding.
_RAW = [
    ("red", "cube", "blue", "sphere"),
    ("blue", "cube", "red", "sphere"),
    ("green", "cube", "yellow", "sphere"),
    ("yellow", "cube", "green", "sphere"),
    ("red", "car", "blue", "cup"),
    ("blue", "car", "green", "cup"),
    ("yellow", "car", "red", "cup"),
    ("green", "car", "yellow", "cup"),
    ("red", "book", "blue", "cube"),
    ("blue", "book", "yellow", "cube"),
    ("green", "book", "red", "cube"),
    ("yellow", "book", "green", "cube"),
    ("red", "sphere", "green", "car"),
    ("blue", "sphere", "yellow", "car"),
    ("green", "sphere", "red", "car"),
    ("yellow", "sphere", "blue", "car"),
    ("red", "cup", "blue", "book"),
    ("blue", "cup", "green", "book"),
    ("green", "cup", "yellow", "book"),
    ("yellow", "cup", "red", "book"),
    ("red", "cube", "green", "car"),
    ("blue", "cube", "yellow", "car"),
    ("green", "cube", "red", "cup"),
    ("yellow", "cube", "blue", "cup"),
    ("red", "sphere", "blue", "book"),
    ("blue", "sphere", "green", "book"),
    ("green", "sphere", "yellow", "cup"),
    ("yellow", "sphere", "red", "cube"),
    ("red", "car", "green", "book"),
    ("blue", "car", "yellow", "sphere"),
    ("green", "car", "red", "cube"),
    ("yellow", "car", "blue", "book"),
    ("red", "book", "green", "sphere"),
    ("blue", "book", "yellow", "cup"),
    ("green", "cup", "red", "sphere"),
    ("yellow", "cup", "blue", "cube"),
]


def _tmpl(c1, o1, c2, o2):
    return f"a {c1} {o1} next to a {c2} {o2}"


def make_pairs():
    """Return list of dicts: clean prompt, swapped prompt, per-object captions, seed."""
    pairs = []
    for i, (c1, o1, c2, o2) in enumerate(_RAW):
        pairs.append({
            "id": i,
            "seed": 1000 + i,
            "clean": _tmpl(c1, o1, c2, o2),       # c1->o1, c2->o2
            "swapped": _tmpl(c2, o1, c1, o2),     # c2->o1, c1->o2 (colors swapped between objects)
            "c1": c1, "o1": o1, "c2": c2, "o2": o2,
        })
    return pairs


if __name__ == "__main__":
    ps = make_pairs()
    print(f"{len(ps)} pairs")
    for p in ps[:4]:
        print(p["clean"], "  ||  ", p["swapped"])
