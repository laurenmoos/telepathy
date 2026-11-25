"""Writing a program by trying things and seeing what happens: what the writer is shown, and
what one action does.

A writer builds a program for the steered model one step at a time. Several writing sessions
run side by side (loop.py), and each one is a conversation:

    1. The writer sees the words it can use (and what each goes up with on real boards), the
       current program, and what its last actions did.
    2. It replies with one action, or with up to `actions` actions followed by DONE:
           ADD / REPLACE k / DELETE k    change the program
           TEST                         run statements on their own, once
           TEST W                       see word W's average value on the practice cells
    3. Each action runs at once on the same practice cells (one forward pass per cell).
       A change is kept unless it lowers the number of cells right. A test changes nothing.
    4. It sees one line per action saying what happened, and replies again.
"""
from . import lang



# ---------------------------------------------------------------- one action

def act(session, text, practice):
    """Run one action for a session. Returns one line saying what happened."""
    head = text.splitlines()[0]
    try:
        kind, k, stmts = lang.parse_action(text)
        if kind == "TESTWORD":
            return f"{head}: {practice.word_report(stmts[0])}"
        if kind == "TEST":
            return f"{head} {' / '.join(s.strip() for s in stmts)}: {practice.compare(practice.run(stmts), practice.plain)}"
        if kind == "ADD" and any(s in session["program"] for s in stmts):
            raise ValueError("that statement is already in the program")
        new = lang.apply_action(session["program"], (kind, k, stmts))
        digits = practice.run(new)
        line = practice.compare(digits, session["digits"])
        if practice.right(digits) < practice.right(session["digits"]):
            return f"{head}: made it worse, undone: {line}"
        session["program"], session["digits"] = new, digits
        if practice.right(digits) > practice.right(session["best_digits"]):
            session["best"], session["best_digits"] = list(new), digits
        return f"{head}: done: {line}"
    except (ValueError, KeyError, IndexError) as e:
        return f"{head}: could not be run: {e}."


# ---------------------------------------------------------------- the words shown

def word_table(m, keep=None):
    out = []
    for k, p in enumerate(m.words, 1):
        name = lang.WORD_TO_NAME[f"w{k}"]
        if keep is not None and name not in keep:
            continue
        shown = [(n, c) for n, c in p.get("facts", []) if abs(c) > p.get("chance", 0.0)][:2]
        what = "; ".join(f"{n} ({c:+.2f})" for n, c in shown) if shown else "no clear meaning"
        out.append(f"  {name}" + (f", layer {p['layer']}" if p["place"] == "ANSWER" else "") + f": {what}")
    return "\n".join(out)
