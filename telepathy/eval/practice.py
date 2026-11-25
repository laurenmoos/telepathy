"""The practice cells: a fixed set of cells that programs are scored on, one forward pass per cell.

    Practice   for the writing loop (loop.py): digits per cell, and one-line comparisons
    Cells      for the agent (agent.py): Python programs, raw rows per cell, with the trace
"""
from ..programs import lang
from ..programs.python import compile_python
from ..programs.run import run
from ..sudoku.prompt import prompt27


class Practice:
    """A fixed set of practice cells. Every action is scored on them, one forward pass per cell."""

    def __init__(self, m, boards, n, rng):
        cells = [(b["board"], r, c, b["solution"][r][c]) for b in boards for r in range(9) for c in range(9)
                 if b["board"][r][c] == 0]
        self.m = m
        self.cells = rng.sample(cells, min(n, len(cells)))
        m.prepare([c[:3] for c in self.cells])
        self.plain = [int(m.cache[prompt27(*c[:3])].argmax()) + 1 for c in self.cells]   # no program
        self.values = m.word_values([c[:3] for c in self.cells])                           # every word, no program

    def run(self, statements):
        """The model's digit on each cell with the program."""
        if not statements:
            return list(self.plain)
        prog = lang.compile_program(statements)
        if not lang.valid(prog):
            raise ValueError("the program is too long")
        return [d for d, _, _ in run(self.m, [(*c[:3], prog) for c in self.cells])]

    def right(self, digits):
        return sum(d == c[3] for d, c in zip(digits, self.cells))

    def compare(self, digits, before):
        """What changed from `before` to `digits`, in one line."""
        changed = [d for d, b in zip(digits, before) if d != b]
        to = {}
        for d in changed:
            to[d] = to.get(d, 0) + 1
        fixed = sum(d == c[3] != b for d, b, c in zip(digits, before, self.cells))
        broke = sum(b == c[3] != d for d, b, c in zip(digits, before, self.cells))
        moved = ", ".join(f"to {d}: {k}" for d, k in sorted(to.items(), key=lambda t: -t[1])[:4])
        return (f"cells right {self.right(before)} -> {self.right(digits)} of {len(self.cells)}; "
                f"{len(changed)} answers changed" + (f" ({moved})" if moved else "") + f"; fixed {fixed}, broke {broke}.")

    def word_report(self, name):
        """A word's average value on the cells, by the right answer and by the model's own answer."""
        v = self.values[:, int(lang.NAME_TO_WORD[name][1:]) - 1]
        def by(labels):
            return {d: float(v[[i for i, x in enumerate(labels) if x == d]].mean())
                    for d in range(1, 10) if d in labels}
        a, q = by([c[3] for c in self.cells]), by(self.plain)
        return (f"average value by the right answer: " + " ".join(f"{d}:{x:+.1f}" for d, x in a.items())
                + f" (highest when the answer is {max(a, key=a.get)}); by the model's own answer: "
                + " ".join(f"{d}:{x:+.1f}" for d, x in q.items()) + ".")

class Cells:
    """The practice cells, and running code on them with the full trace."""

    def __init__(self, m, boards, n, rng):
        cells = [(b["board"], r, c, b["solution"][r][c]) for b in boards for r in range(9) for c in range(9)
                 if b["board"][r][c] == 0]
        self.m, self.cells = m, rng.sample(cells, min(n, len(cells)))
        m.prepare([c[:3] for c in self.cells])
        self.plain = [int(m.cache[prompt27(*c[:3])].argmax()) + 1 for c in self.cells]   # no program

    def run(self, code, first=0, count=8, examples=3):
        """Run Python code (programs/python.py) on every cell. Returns (text, cells right).
        The text compares with no program: cells right, fixed and broken, where answers moved,
        a few fixed and broken cells with their traces, then count rows from row first."""
        prog = compile_python(code)
        if lang.count(prog.stmts) > 5000:
            raise ValueError("the program is too long (over 5000 instructions once loops are unrolled)")
        out = run(self.m, [(*c[:3], prog) for c in self.cells], events=True)
        digits = [d for d, _, _, _ in out]
        right = sum(d == c[3] for d, c in zip(digits, self.cells))
        plain = sum(p == c[3] for p, c in zip(self.plain, self.cells))
        fixed = [i for i, (d, p, c) in enumerate(zip(digits, self.plain, self.cells)) if d == c[3] != p]
        broke = [i for i, (d, p, c) in enumerate(zip(digits, self.plain, self.cells)) if p == c[3] != d]
        moved = {}
        for d, p in zip(digits, self.plain):
            if d != p:
                moved[d] = moved.get(d, 0) + 1
        row = lambda i: (f"cell {i}: answered {digits[i]}, right {self.cells[i][3]}, with no program {self.plain[i]}"
                         + (f"; {'; '.join(out[i][1])}" if out[i][1] else ""))
        lines = [f"{right} of {len(self.cells)} cells right (no program: {plain}). Fixed {len(fixed)}, broke {len(broke)}.",
                 f"Answers changed on {sum(moved.values())} cells" + (": " + ", ".join(
                     f"to {d}: {k}" for d, k in sorted(moved.items(), key=lambda t: -t[1])) if moved else "") + "."]
        lines += profile(code, [o[3] for o in out], digits, self.plain, [c[3] for c in self.cells],
                         lang.schedule(prog)[1])
        if fixed:
            lines += ["Fixed, for example:"] + [row(i) for i in fixed[:examples]]
        if broke:
            lines += ["Broke, for example:"] + [row(i) for i in broke[:examples]]
        if count > 0:
            lines += [f"Rows {first} to {min(first + count, len(self.cells)) - 1}:"] + [
                row(i) for i in range(first, min(first + count, len(self.cells)))]
        return "\n".join(lines), right


def profile(code, events, digits, plain, right, skipped=()):
    """What each line of the program did on this run, measured over the cells:
    an if: on how many cells it was true, and the range of the value it compared;
    an argmax: how often each word was the largest;
    a write: on how many cells it ran, which digits the answers moved to on those cells
    (against no program), and how many of those cells it fixed and broke;
    a write that never ran because the model had passed its layer: which layers."""
    src = code.splitlines()
    by = {}
    for cell, ev in enumerate(events):
        for line, kind, info in ev:
            by.setdefault(line, (kind, []))[1].append((cell, info))
    out = []
    for line in sorted(by):
        kind, hits = by[line]
        cells = sorted({c for c, _ in hits})
        text = src[line - 1].strip() if line - 1 < len(src) else ""
        if kind == "if":
            vals = [v for _, (v, _) in hits]
            true = sum(t for _, (_, t) in hits)
            what = f"true on {true} of {len(hits)} cells (values {min(vals):+.2f} to {max(vals):+.2f})"
        elif kind == "argmax":
            n = {}
            for _, b in hits:
                n[b] = n.get(b, 0) + 1
            what = "largest: " + ", ".join(f"word {b}: {k}" for b, k in sorted(n.items())) + " cells"
        else:
            moved = {}
            for c in cells:
                if digits[c] != plain[c]:
                    moved[digits[c]] = moved.get(digits[c], 0) + 1
            fixed = sum(digits[c] == right[c] != plain[c] for c in cells)
            broke = sum(plain[c] == right[c] != digits[c] for c in cells)
            what = (f"ran on {len(cells)} cells; answers changed on {sum(moved.values())}"
                    + (" (" + ", ".join(f"to {d}: {k}" for d, k in sorted(moved.items(), key=lambda t: -t[1])) + ")"
                       if moved else "") + f"; fixed {fixed}, broke {broke}")
        out.append((line, f"line {line}  {text[:60]:<60}  {what}"))
    for op, w, line, L, at in skipped:
        if line is not None and line not in by:
            text = src[line - 1].strip() if line - 1 < len(src) else ""
            out.append((line, f"line {line}  {text[:60]:<60}  never ran: {lang.WORD_TO_NAME.get(w, w)} is at layer {L}, "
                              f"and the program had already reached layer {at}"))
    out = [t for _, t in sorted(set(out))]
    return (["What each line did:"] + out) if out else []
