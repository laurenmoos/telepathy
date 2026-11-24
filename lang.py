"""The program language: read, for, write.

A program decides one cell. It reads the model's state at the end of the cell's row,
column and box lines, collects evidence about which digits are already there, then
pushes the model toward a digit when it answers.

    (seq S ...)                  run statements in order
    (for v S ...)                run the statements 9 times, with v = 1, 2, ..., 9
    (read UNIT D LAYER)          UNIT is ROW, COL or BOX. D is a digit 1-9, or v inside a for.
                                 Reads the gate "is D in this cell's UNIT?" at the end of that
                                 line, at LAYER. Adds the answer to D's evidence: positive means
                                 "D is there", negative means "D is missing".
    (write STRENGTH)             when the model answers, push it toward the digit with the
                                 lowest evidence (the one that looks most missing). Does nothing
                                 if nothing has been read yet.

"""
import copy, random
from dataclasses import dataclass, field
from typing import List, Union

UNITS = ("ROW", "COL", "BOX")


@dataclass
class Read:
    unit: str
    digit: Union[int, str]      # 1-9, or "v"
    layer: int


@dataclass
class Write:
    strength: float


@dataclass
class For:
    body: List["Stmt"] = field(default_factory=list)


@dataclass
class Prog:
    stmts: List["Stmt"] = field(default_factory=list)


Stmt = Union[Read, Write, For]


# ---------------------------------------------------------------- text format

def show(p):
    def stmt(s):
        if isinstance(s, Read):
            return f"(read {s.unit} {s.digit} {s.layer})"
        if isinstance(s, Write):
            return f"(write {s.strength:.2f})"
        return "(for v" + "".join(" " + stmt(x) for x in s.body) + ")"
    return "(seq" + "".join(" " + stmt(s) for s in p.stmts) + ")"


def parse(text):
    """Parse a program. Raises ValueError on anything malformed."""
    toks = text.lower().replace("(", " ( ").replace(")", " ) ").split()   # any capitalization
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end")
        pos += 1
        return toks[pos - 1]

    def expect(t):
        got = nxt()
        if got != t:
            raise ValueError(f"expected {t!r}, got {got!r}")

    def body():
        out = []
        while pos < len(toks) and toks[pos] != ")":
            out.append(stmt())
        expect(")")
        return out

    def stmt():
        expect("(")
        head = nxt()
        if head == "read":
            unit, d, layer = nxt().upper(), nxt(), nxt()
            if unit not in UNITS:
                raise ValueError(f"unknown unit {unit}")
            d = "v" if d == "v" else int(d)
            s = Read(unit, d, int(layer))
            expect(")")
            return s
        if head == "write":
            s = Write(float(nxt()))
            expect(")")
            return s
        if head == "for":
            expect("v")
            if toks[pos:pos + 2] == ["1", "9"]:       # also accept (for v 1 9 ...)
                nxt(); nxt()
            return For(body())
        if head == "seq":
            return Prog(body())
        raise ValueError(f"unknown instruction {head!r}")

    expect("(")
    if nxt() != "seq":
        raise ValueError("a program starts with (seq")
    p = Prog(body())
    if pos != len(toks):
        raise ValueError("text after the program")
    # flatten nested (seq ...) into their parent
    def flat(stmts):
        out = []
        for s in stmts:
            if isinstance(s, Prog):
                out += flat(s.stmts)
            elif isinstance(s, For):
                out.append(For(flat(s.body)))
            else:
                out.append(s)
        return out
    return Prog(flat(p.stmts))


# ---------------------------------------------------------------- checks

def size(p):
    """Instructions as written."""
    def n(stmts):
        return sum(1 + n(s.body) if isinstance(s, For) else 1 for s in stmts)
    return n(p.stmts)


def executed(p):
    """Instructions actually run, with each for counted 9 times."""
    def n(stmts):
        return sum(9 * n(s.body) if isinstance(s, For) else 1 for s in stmts)
    return n(p.stmts)


def valid(p, cfg):
    lim, layers = cfg["limits"], cfg["read_layers"]
    lo, hi = cfg["strength_range"]

    def ok(stmts, in_for):
        for s in stmts:
            if isinstance(s, Read):
                if s.layer not in layers or s.unit not in UNITS:
                    return False
                if s.digit == "v" and not in_for:
                    return False
                if s.digit != "v" and not 1 <= s.digit <= 9:
                    return False
            elif isinstance(s, Write):
                if not lo <= s.strength <= hi:
                    return False
            else:
                if in_for or not s.body or not ok(s.body, True):   # no for inside a for
                    return False
        return True

    return ok(p.stmts, False) and size(p) <= lim["max_size"] and executed(p) <= lim["max_executed"]


# ---------------------------------------------------------------- running

def run(p, read_gate, trace=None):
    """Run a program for one cell. read_gate(unit, digit, layer) -> float (>0 means present).
    Returns the list of (digit, strength) pushes to apply when the model answers.
    If trace is a dict, it is filled with the evidence per digit, the reads and the pushes."""
    evidence = [0.0] * 10
    read_any = False
    pushes = []

    def go(stmts, v):
        nonlocal read_any
        for s in stmts:
            if isinstance(s, Read):
                d = v if s.digit == "v" else s.digit
                evidence[d] += read_gate(s.unit, d, s.layer)
                read_any = True
            elif isinstance(s, Write):
                if read_any:
                    d = min(range(1, 10), key=lambda k: evidence[k])
                    pushes.append((d, s.strength))
            else:
                for value in range(1, 10):
                    go(s.body, value)

    go(p.stmts, None)
    if trace is not None:
        trace.update(evidence=evidence[1:], read_any=read_any, pushes=list(pushes))
    return pushes


# ---------------------------------------------------------------- variation

def random_read(cfg, rng, in_for=False):
    d = "v" if in_for and rng.random() < 0.8 else rng.randint(1, 9)
    return Read(rng.choice(UNITS), d, rng.choice(cfg["read_layers"]))


def random_write(cfg, rng):
    lo, hi = cfg["strength_range"]
    return Write(round(rng.uniform(lo, hi), 2))


def random_program(cfg, rng):
    stmts = []
    for _ in range(rng.randint(1, 4)):
        r = rng.random()
        if r < 0.4:
            stmts.append(random_read(cfg, rng))
        elif r < 0.7:
            stmts.append(For([random_read(cfg, rng, True) for _ in range(rng.randint(1, 2))]))
        else:
            stmts.append(random_write(cfg, rng))
    return Prog(stmts)


def mutate(p, cfg, rng):
    p = copy.deepcopy(p)
    blocks = [(p.stmts, False)] + [(s.body, True) for s in p.stmts if isinstance(s, For)]
    stmts, in_for = rng.choice(blocks)
    op = rng.choice(["insert_read", "insert_write", "insert_for", "delete", "unit", "digit",
                     "layer", "strength", "swap"])
    if op == "insert_read" or not stmts:
        stmts.insert(rng.randint(0, len(stmts)), random_read(cfg, rng, in_for))
        return p
    if op == "insert_write":
        stmts.insert(rng.randint(0, len(stmts)), random_write(cfg, rng))
        return p
    if op == "insert_for" and not in_for:
        stmts.insert(rng.randint(0, len(stmts)), For([random_read(cfg, rng, True)]))
        return p
    i = rng.randrange(len(stmts))
    s = stmts[i]
    if op == "delete":
        stmts.pop(i)
    elif op == "unit" and isinstance(s, Read):
        s.unit = rng.choice(UNITS)
    elif op == "digit" and isinstance(s, Read):
        s.digit = "v" if in_for and rng.random() < 0.5 else rng.randint(1, 9)
    elif op == "layer" and isinstance(s, Read):
        s.layer = rng.choice(cfg["read_layers"])
    elif op == "strength" and isinstance(s, Write):
        lo, hi = cfg["strength_range"]
        s.strength = round(min(hi, max(lo, s.strength * rng.choice([0.5, 2.0]) + rng.gauss(0, 0.02))), 2)
    elif op == "swap" and len(stmts) > 1:
        j = rng.randrange(len(stmts))
        stmts[i], stmts[j] = stmts[j], stmts[i]
    return p


def crossover(a, b, rng):
    i, j = rng.randint(0, len(a.stmts)), rng.randint(0, len(b.stmts))
    return Prog(copy.deepcopy(a.stmts[:i] + b.stmts[j:]))
