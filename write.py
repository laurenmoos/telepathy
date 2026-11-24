"""How the model writes programs.

Several sessions run side by side. In each, the model sees the program language, two examples, the best programs so far with their
scores, and the trace of the best one. It writes a program. The program is run on a few
practice boards, and the model sees that program's score and trace. It revises, for a
limited number of turns. Whatever doesn't parse or breaks the limits is not run.
"""
import lang

SYSTEM = """You write tiny programs that steer a language model, the same kind of model as you,
while it fills in one cell of a sudoku board. The model sees the board's 9 rows, 9 columns
and 9 boxes. A program can read what the model has seen, then push it toward an answer.

Instructions:
  (seq S ...)              run statements in order. Every program is one (seq ...).
  (for v S ...)            run the statements for v = 1, 2, ..., 9
  (read UNIT D LAYER)      UNIT is ROW, COL or BOX (the cell's row, column or box).
                           D is a digit 1-9, or v inside a for. LAYER is one of {layers}.
                           Checks whether digit D is already in that unit. Evidence is kept per digit.
  (write STRENGTH)         push the model toward the digit that looks most missing so far.
                           STRENGTH is between {lo} and {hi}.

Examples:
  (seq (read ROW 5 2) (read ROW 7 2) (write 0.10))
  (seq (for v (read BOX v 3)))

A program is scored on many boards: boards fully solved, cells right, sudoku rules kept.
Shorter programs score a little higher. Reply with exactly one program on one line, nothing else.
After each program you will be shown how it did, cell by cell. Use that to write a better one."""


def extract(text):
    """The first balanced (seq ...) in the text, any capitalization, or None."""
    i = text.lower().find("(seq")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "("
        depth -= text[j] == ")"
        if depth == 0:
            return text[i:j + 1]
    return None


def sessions(model, cfg, best, best_trace, n, turns, evaluate, log=print, state=None, on_turn=None):
    """n writing sessions, side by side: each turn, every session writes a program at once
    (one batched generation), all the drafts are run together, and each session sees its
    own draft's score and trace before revising.
    best: list of (score, program text), best first. best_trace: the trace of the best one.
    evaluate(programs) -> list of (score, trace text).
    state (optional): a dict holding the conversations so far. It is updated after every
    turn and passed to on_turn(), so a stopped run can continue where it left off.
    Returns, per session, the list of (score, program) it wrote."""
    lo, hi = cfg["strength_range"]
    system = SYSTEM.format(layers=", ".join(map(str, cfg["read_layers"])), lo=lo, hi=hi)
    if best:
        shown = "\n".join(f"  score {s:.3f}: {t}" for s, t in best)
        first = f"Programs so far, best first:\n{shown}\n"
        if best_trace:
            first += f"\nHow the best one did:\n{best_trace}\n"
        first += "\nWrite one new program that should score higher than all of these."
    else:
        first = "Write one program."
    state = state if state is not None else {}
    if not state:
        state.update(msgs=[[{"role": "system", "content": system}, {"role": "user", "content": first}] for _ in range(n)],
                     written=[[] for _ in range(n)], turn=0)
    for turn in range(state["turn"], turns):
        replies = model.generate(state["msgs"])
        progs, errs = [], []
        for msgs, reply in zip(state["msgs"], replies):
            msgs.append({"role": "assistant", "content": reply})
            t = extract(reply)
            try:
                prog = lang.parse(t) if t else None
                err = None if prog else "no (seq ...) found"
            except (ValueError, IndexError) as e:
                prog, err = None, f"it does not parse: {e}"
            if prog is not None and not lang.valid(prog, cfg):
                prog, err = None, "it breaks the limits (layers, strength, size, or a for inside a for)"
            progs.append(prog); errs.append(err)
        ok = [i for i, p in enumerate(progs) if p is not None]
        results = dict(zip(ok, evaluate([progs[i] for i in ok]))) if ok else {}
        for i, msgs in enumerate(state["msgs"]):
            if progs[i] is None:
                log(f"    session {i + 1} turn {turn + 1}: thrown out, {errs[i]}: {replies[i].strip()[:80]!r}")
                msgs.append({"role": "user", "content": f"That program could not be run: {errs[i]}. Write one valid program."})
            else:
                score, tr = results[i]
                state["written"][i].append((score, lang.show(progs[i])))
                log(f"    session {i + 1} turn {turn + 1}: score {score:.3f}  {lang.show(progs[i])}")
                msgs.append({"role": "user", "content": f"Score {score:.3f}.\n{tr}\n\nWrite an improved program."})
        state["turn"] = turn + 1
        if on_turn:
            on_turn()
    return [[(sc, lang.parse(t)) for sc, t in w] for w in state["written"]]
