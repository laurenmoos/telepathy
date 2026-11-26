"""The practice cells: a fixed set of sudoku cells that every program is run on, with feedback.

    Cells(live, boards, n, rng)   n random empty cells from the boards
    cells.run(code)               (feedback text, cells right): cells right against no program,
                                  cells fixed and broken, where answers moved, what each line of
                                  the program did, and example cells with their traces
"""
import numpy as np

from ..sudoku.prompt import prompt27


class Cells:
    def __init__(self, live, boards, n, rng):
        cells = [(b["board"], r, c, b["solution"][r][c]) for b in boards for r in range(9) for c in range(9)
                 if b["board"][r][c] == 0]
        self.live, self.cells = live, rng.sample(cells, min(n, len(cells)))
        self.prompts = [prompt27(*c[:3]) for c in self.cells]
        self.digits = [live.token_id(str(d)) for d in range(1, 10)]
        self.plain = [int(np.argmax(lp)) + 1 for lp, _, _ in live.run(self.prompts, "", self.digits, key="practice")]

    def run(self, code, examples=1):
        out = self.live.run(self.prompts, code, self.digits, key="practice")
        digits = [int(np.argmax(lp)) + 1 for lp, _, _ in out]
        right = [c[3] for c in self.cells]
        n_right = sum(d == r for d, r in zip(digits, right))
        plain = sum(p == r for p, r in zip(self.plain, right))
        fixed = [i for i, (d, p, r) in enumerate(zip(digits, self.plain, right)) if d == r != p]
        broke = [i for i, (d, p, r) in enumerate(zip(digits, self.plain, right)) if p == r != d]
        moved = {}
        for d, p in zip(digits, self.plain):
            if d != p:
                moved[d] = moved.get(d, 0) + 1
        row = lambda i: (f"cell {i}: answered {digits[i]}, right {right[i]}, with no program {self.plain[i]}"
                         + (f"; {'; '.join(out[i][1])}" if out[i][1] else ""))
        lines = [f"{n_right} of {len(self.cells)} cells right (no program: {plain}). Fixed {len(fixed)}, broke {len(broke)}.",
                 f"Answers changed on {sum(moved.values())} cells" + (": " + ", ".join(
                     f"to {d}: {k}" for d, k in sorted(moved.items(), key=lambda t: -t[1])) if moved else "") + "."]
        if not any(o[2] for o in out):
            lines.append("The program made no peek or push calls.")
        lines += profile(code, [o[2] for o in out], digits, self.plain, right)
        if fixed:
            lines += ["Fixed, for example:"] + [row(i) for i in fixed[:examples]]
        if broke:
            lines += ["Broke, for example:"] + [row(i) for i in broke[:examples]]
        if not fixed and not broke:
            lines += ["For example:"] + [row(0)]
        return "\n".join(lines), n_right


def profile(code, events, digits, plain, right):
    """What each line of the program did, over the cells: a peek line, how many times it ran and the
    range of its values; a push line, on how many cells it ran, where the answers moved on those
    cells (against no program), and cells fixed and broken; an error, on how many cells."""
    src = code.splitlines()
    by = {}
    for cell, ev in enumerate(events):
        for line, kind, value in ev:
            by.setdefault((line, kind), []).append((cell, value))
    out = []
    for (line, kind), hits in sorted(by.items(), key=lambda t: (t[0][0] or 0, t[0][1])):
        cells = sorted({c for c, _ in hits})
        text = src[line - 1].strip() if line and line - 1 < len(src) else ""
        if kind == "peek":
            vals = [v for _, v in hits]
            what = f"peeked {len(vals)} times on {len(cells)} cells (values {min(vals):+.2f} to {max(vals):+.2f})"
        elif kind == "push":
            moved = {}
            for c in cells:
                if digits[c] != plain[c]:
                    moved[digits[c]] = moved.get(digits[c], 0) + 1
            fixed = sum(digits[c] == right[c] != plain[c] for c in cells)
            broke = sum(plain[c] == right[c] != digits[c] for c in cells)
            what = (f"pushed on {len(cells)} cells; answers changed on {sum(moved.values())}"
                    + (" (" + ", ".join(f"to {d}: {k}" for d, k in sorted(moved.items(), key=lambda t: -t[1])) + ")"
                       if moved else "") + f"; fixed {fixed}, broke {broke}")
        else:
            what = f"error on {len(cells)} cells: {hits[0][1]}"
        out.append(f"line {line}  {text[:60]:<60}  {what}")
    return (["What each line did:"] + out) if out else []
