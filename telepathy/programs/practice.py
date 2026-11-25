"""The practice cells: a fixed set of cells that every action is scored on, one forward pass per cell."""
from . import lang
from .run import run
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
