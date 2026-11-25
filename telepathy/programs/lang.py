"""The program language: a small Python subset that reads and writes words inside Qwen.

Words. A word is one direction in Qwen's activations at one token position and one layer.
    row_end[i], col_end[i], box_end[i]   at the end of the target cell's row, column and box line
                                         (words/board.py): they show what is in that line
    answer[i]                            at the answer position (words/answer.py): pushing them changes
                                         Qwen's answer; numbered from the earliest layer
A word's value on a board: where Qwen's state sits along it, standardized over real boards
(0 is typical, +1 or -1 is one spread away). Any word can be read and written.

Statements.
    set(W, EXPR)                 set word W to EXPR
    clear(W)                     set word W back to 0
    if EXPR > X:                 run the indented lines if EXPR > X on this board,
        ...                      otherwise the lines under else (optional); X is -2 to 2
    else:
        ...
    best = argmax(W1, W2, ...)   which of 2 to 9 words is largest on this board,
    if best == 0:                then run the lines under its number
        ...
    elif best == 1:
        ...
EXPR is a number, or a sum of words each with an optional factor: 0.5 * row_end[3] - col_end[1].
Indented lines are set(...) or clear(...).

Running. A program runs inside one forward pass of Qwen. Each statement acts at the layer of its
words, and the pass only moves forward: a statement that needs a layer the pass has already
left is skipped. Internally, statements compile to a few instructions on one working value,
acc: load, add, scale, const, store, clear, if, switch.

Actions. A writer changes a program one action at a time:
    ADD / REPLACE k / DELETE k   change the program (statements numbered from 1)
    TEST                         run statements on their own, once, and see what they did
    TEST W                       see word W's average value on the practice cells
Each action is the keyword, then its statements, then END.
"""
import re
from dataclasses import dataclass, field
from typing import List

PLACE_NAME = {"ROW": "row_end", "COL": "col_end", "BOX": "box_end", "ANSWER": "answer"}
NAME_TO_WORD = {}               # "row_end[17]" -> "w90" (the interpreter's word id); set by setup()
WORD_TO_NAME = {}
WORD_LAYERS = {}                # "w90" -> its layer
GROUP_SIZE = {}                 # "row_end" -> 32
SCALE_RANGE = (-2.0, 2.0)
MAX_NUMBER = 8.0
MAX_INSTRUCTIONS = 120


def setup(words):
    """Name the words. words: the model's word list, in order (w1 ... wN)."""
    for d in (NAME_TO_WORD, WORD_TO_NAME, WORD_LAYERS, GROUP_SIZE):
        d.clear()
    for k, p in enumerate(words, 1):
        group = PLACE_NAME[p["place"]]
        i = GROUP_SIZE.get(group, 0)
        GROUP_SIZE[group] = i + 1
        NAME_TO_WORD[f"{group}[{i}]"] = f"w{k}"
        WORD_TO_NAME[f"w{k}"] = f"{group}[{i}]"
        WORD_LAYERS[f"w{k}"] = p["layer"]


# ---------------------------------------------------------------- instructions (what statements compile to)

@dataclass
class Op:
    op: str                     # load, add, store, clear
    word: str                   # "w90"


@dataclass
class Scale:
    c: float


@dataclass
class Const:
    c: float


@dataclass
class If:
    c: float
    then: List = field(default_factory=list)
    other: List = field(default_factory=list)


@dataclass
class Switch:
    words: List[str] = field(default_factory=list)
    branches: List[List] = field(default_factory=list)


@dataclass
class Prog:
    stmts: List = field(default_factory=list)


def count(stmts):
    return sum(1 + sum(count(b) for b in ([s.then, s.other] if isinstance(s, If) else
                                         s.branches if isinstance(s, Switch) else [])) for s in stmts)


def valid(prog):
    return count(prog.stmts) <= MAX_INSTRUCTIONS


def schedule(prog):
    """Where each instruction runs in the forward pass.
    steps is {layer: [instruction, ...]} in run order. Each instruction is a tuple whose third item
    is its guards, ((decision id, branch), ...): it runs only on boards where those decisions
    chose those branches.
        (op, word, guards)                 load, add, store, clear
        ("scale", C, guards), ("const", C, guards)
        ("if", C, guards, id)              decision id: branch 0 if acc > C, else branch 1
        ("note", word, guards, (id, i))    keep the word's value as switch id's i-th candidate
        ("switch", n, guards, id)          decision id: the branch of the largest candidate
        ("record", word, ())               keep the word's value now, for a later read of it
    A read (load, add, note) of a word whose layer the pass has already left uses the value
    recorded at that layer: later writes cannot change an earlier layer. Its tuple has a 4th
    item, "recorded". skipped lists (op, word, source line, its layer, the layer the pass was at)
    for writes at a layer the pass had already left.
    lines maps id(instruction tuple) -> its source line, for instructions compiled from Python
    (programs/python.py tags them); it is empty for the old language.
    Returns (steps, skipped, lines)."""
    steps, skipped, ids, lines = {}, [], [0], {}

    def add(L, ins, src=None):
        steps.setdefault(L, []).append(ins)
        if getattr(src, "line", None) is not None:
            lines[id(ins)] = src.line

    def go(stmts, guards, at):
        for s in stmts:
            if isinstance(s, Op):
                L = WORD_LAYERS[s.word]
                if L < at and s.op in ("load", "add"):
                    add(L, ("record", s.word, ()))
                    add(at, (s.op, s.word, guards, "recorded"))
                    continue
                if L < at:
                    skipped.append((s.op, s.word, getattr(s, "line", None), L, at))
                    continue
                at = L
                add(L, (s.op, s.word, guards), s)
            elif isinstance(s, Scale):
                add(at, ("scale", s.c, guards))
            elif isinstance(s, Const):
                add(at, ("const", s.c, guards))
            elif isinstance(s, If):
                i = ids[0]; ids[0] += 1
                add(at, ("if", s.c, guards, i), s)
                at = max(go(s.then, guards + ((i, 0),), at), go(s.other, guards + ((i, 1),), at))
            elif isinstance(s, Switch):
                Ls = [WORD_LAYERS[w] for w in s.words]
                i = ids[0]; ids[0] += 1
                for j, w in enumerate(s.words):
                    if WORD_LAYERS[w] < at:
                        add(WORD_LAYERS[w], ("record", w, ()))
                        add(at, ("note", w, guards, (i, j), "recorded"))
                    else:
                        add(WORD_LAYERS[w], ("note", w, guards, (i, j)))
                at = max(at, max(Ls))
                add(at, ("switch", len(s.words), guards, i), s)
                at = max(go(b, guards + ((i, j),), at) for j, b in enumerate(s.branches))
        return at

    go(prog.stmts, (), 0)
    for L in steps:                       # within a layer: records, notes, the switch, then the rest in order
        steps[L].sort(key=lambda ins: {"record": 0, "note": 1, "switch": 2}.get(ins[0], 3))
    return steps, skipped, lines


def writes(prog):
    """True if the program changes Qwen at all (a store or clear that runs)."""
    return any(ins[0] in ("store", "clear") for lst in schedule(prog)[0].values() for ins in lst)


# ---------------------------------------------------------------- Python statements -> instructions

WORD_RX = r"(?:row_end|col_end|box_end|answer)\[\d+\]"
NUM_RX = r"-?\d+(?:\.\d+)?"


def word(name):
    name = name.strip()
    if name not in NAME_TO_WORD:
        raise ValueError(f"no word named {name}")
    return NAME_TO_WORD[name]


def split_statements(text):
    """Group lines into statements: a top-level line plus its indented, else and elif lines. A
    'best = argmax(...)' line and the if/elif block after it form one statement."""
    out, cur = [], None
    for line in [l.rstrip() for l in text.strip("\n").split("\n") if l.strip()]:
        joins = line.startswith(("else:", "elif ")) or (
            cur is not None and len(cur) == 1 and cur[0].startswith("best =") and line.startswith("if best"))
        if not line.startswith(" ") and not joins:
            cur = [line]
            out.append(cur)
        elif cur is None:
            raise ValueError(f"indented line with no statement before it: {line!r}")
        else:
            cur.append(line)
    return ["\n".join(s) for s in out]


def factors(x):
    """A factor as scale steps within SCALE_RANGE."""
    if x == 0 or abs(x) > 64:
        raise ValueError(f"factor {x} out of range")
    steps = []
    while abs(x) > SCALE_RANGE[1]:
        steps.append(SCALE_RANGE[1])
        x /= SCALE_RANGE[1]
    return [Scale(round(s, 4)) for s in steps + [x]]


def compile_expr(text):
    """Instructions that leave the expression's value in acc."""
    t = text.replace(" ", "")
    if re.fullmatch(NUM_RX, t):
        c, n = float(t), 0
        if abs(c) > MAX_NUMBER:
            raise ValueError(f"number {c} out of range")
        while abs(c) > SCALE_RANGE[1]:
            c /= SCALE_RANGE[1]
            n += 1
        return [Const(round(c, 4))] + [Scale(SCALE_RANGE[1])] * n
    terms = re.findall(r"([+-]?)(?:(\d+(?:\.\d+)?)\*)?(" + WORD_RX + r")", t)
    if not terms or "".join(s + (c + "*" if c else "") + w for s, c, w in terms) != t:
        raise ValueError(f"not an expression of words: {text!r}")
    terms = [((-1 if s == "-" else 1) * (float(c) if c else 1.0), word(w)) for s, c, w in terms]
    terms = sorted([x for x in terms if x[0] != 0], key=lambda x: WORD_LAYERS[x[1]])   # read earlier layers first
    if not terms:
        return [Const(0.0)]
    out = [Op("load", terms[0][1])]                       # Horner form: ((w1 * c1/c2 + w2) * c2/c3 + w3) ...
    for (a, _), (b, w) in zip(terms, terms[1:]):
        out += factors(a / b) + [Op("add", w)]
    if terms[-1][0] != 1:
        out += factors(terms[-1][0])
    return out


def compile_line(line):
    t = line.strip().replace(" ", "")
    m = re.fullmatch(r"set\((" + WORD_RX + r"),(.+)\)", t)
    if m:
        return compile_expr(m.group(2)) + [Op("store", word(m.group(1)))]
    m = re.fullmatch(r"clear\((" + WORD_RX + r")\)", t)
    if m:
        return [Op("clear", word(m.group(1)))]
    raise ValueError(f"not set(...) or clear(...): {line.strip()!r}")


def compile_statement(text):
    lines = text.split("\n")
    head = lines[0].strip()
    if head.startswith(("set(", "clear(")):
        if len(lines) > 1:
            raise ValueError("set or clear with indented lines under it")
        return compile_line(head)
    m = re.fullmatch(r"if (.+) > (" + NUM_RX + r"):", head)
    if m:
        thr = float(m.group(2))
        if not SCALE_RANGE[0] <= thr <= SCALE_RANGE[1]:
            raise ValueError("an if threshold must be between -2 and 2")
        then, other = [], []
        branch = then
        for l in lines[1:]:
            if l.strip() == "else:":
                branch = other
            elif l.startswith(" "):
                branch.extend(compile_line(l))
            else:
                raise ValueError(f"unexpected line in an if: {l!r}")
        if not then and not other:
            raise ValueError("an if with no lines under it")
        return compile_expr(m.group(1)) + [If(thr, then, other)]
    m = re.fullmatch(r"best = argmax\((.+)\)", head)
    if m:
        ws = [word(x) for x in m.group(1).split(",")]
        if not 2 <= len(ws) <= 9 or len(set(ws)) != len(ws):
            raise ValueError("argmax takes 2 to 9 different words")
        branches, cur = [[] for _ in ws], None
        for l in lines[1:]:
            mm = re.fullmatch(r"(?:if|elif) best == (\d+):", l.strip())
            if mm:
                cur = int(mm.group(1))
                if cur >= len(ws):
                    raise ValueError(f"best == {cur}, but argmax has only {len(ws)} words")
            elif l.startswith(" ") and cur is not None:
                branches[cur].extend(compile_line(l))
            else:
                raise ValueError(f"unexpected line after argmax: {l!r}")
        return [Switch(ws, branches)]
    raise ValueError(f"not a statement: {head!r}")


def compile_program(statements):
    """A list of statement texts -> Prog."""
    out = []
    for s in statements:
        out += compile_statement(s)
    return Prog(out)


def show(statements):
    """A program as writers see it, statements numbered."""
    if not statements:
        return "(empty program)"
    return "\n".join(f"# {i}\n{s}" for i, s in enumerate(statements, 1))


# ---------------------------------------------------------------- actions

def parse_action(text):
    """Returns (kind, k, statements): kind is ADD, REPLACE, DELETE, TEST, or TESTWORD (statements
    is then [the word's name])."""
    t = text.strip()
    if not t.endswith("END"):
        raise ValueError("an action ends with END")
    head, _, rest = t[:-3].rstrip("\n").partition("\n")
    m = re.fullmatch(r"TEST (" + WORD_RX + r")", head.strip())
    if m:
        word(m.group(1))
        return "TESTWORD", 0, [m.group(1)]
    m = re.fullmatch(r"(ADD|TEST|REPLACE (\d+)|DELETE (\d+))", head.strip())
    if not m:
        raise ValueError(f"an action starts with ADD, REPLACE k, DELETE k or TEST, not {head!r}")
    kind = head.split()[0]
    stmts = split_statements(rest) if rest.strip() else []
    if kind != "DELETE" and not stmts:
        raise ValueError(f"{kind} with no statements")
    return kind, int(m.group(2) or m.group(3) or 0), stmts


def apply_action(statements, action):
    """The program after an ADD, REPLACE or DELETE."""
    kind, k, new = action
    if kind == "ADD":
        return list(statements) + new
    if not 1 <= k <= len(statements):
        raise ValueError(f"there is no statement {k}")
    if kind == "DELETE":
        return statements[:k - 1] + statements[k:]
    return statements[:k - 1] + new + statements[k:]


# ---------------------------------------------------------------- grammar, for constrained writing

GRAMMAR_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[](),.*+-=:> \n")


def grammar(actions=1):
    """A regular expression for a reply: one action, or (actions > 1) up to that many actions
    followed by a line DONE. It matches exactly what parse_action accepts."""
    idx = lambda n: "(?:" + "|".join(str(i) for i in range(n - 1, -1, -1)) + ")"
    w = "(?:" + "|".join(rf"{g}\[{idx(n)}\]" for g, n in GROUP_SIZE.items()) + ")"
    num = r"-?[0-9]\.[0-9]{1,2}"
    thr = r"(?:-?[01]\.[0-9]{1,2}|-?2\.00?)"
    term = rf"(?:{num} \* )?{w}"
    expr = rf"(?:{num}|{term}(?: [+-] {term}){{0,5}})"
    line = rf"(?:set\({w}, {expr}\)|clear\({w}\))"
    cond = rf"if {expr} > {thr}:\n(?:    {line}\n)+(?:else:\n(?:    {line}\n)+)?"
    pick = (rf"best = argmax\({w}(?:, {w}){{1,8}}\)\nif best == 0:\n(?:    {line}\n)+"
            rf"(?:elif best == [1-8]:\n(?:    {line}\n)+){{0,8}}")
    stmt = rf"(?:{line}\n|{cond}|{pick})"
    k = r"[1-9][0-9]?"
    one = rf"(?:ADD\n{stmt}{{1,3}}|TEST\n{stmt}{{1,3}}|TEST {w}\n|REPLACE {k}\n{stmt}{{1,3}}|DELETE {k}\n)END"
    return one if actions == 1 else rf"(?:{one}\n){{1,{actions}}}DONE"


def split_reply(text, actions=1):
    """The actions in a reply."""
    if actions == 1:
        return [text.strip()]
    return [x.strip() + "\nEND" for x in re.split(r"\nEND\b", text.replace("DONE", "")) if x.strip()][:actions]
