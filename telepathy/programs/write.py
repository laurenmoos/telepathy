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


INSTRUCTIONS = """You write a small Python program that runs inside a language model, the same kind of
model as you, while it fills in one empty cell of a sudoku board. The model reads the board's
rows, columns and boxes, one line each, and then names the digit for the cell.

Your program reads and changes a few numbers inside the model, called words. On each board a
word has a value: 0 is typical, +1 or -1 is unusual.
  row_end[i], col_end[i], box_end[i]: at the end of the cell's row, column and box line, at the
      model's first layer. They show what is in that line.
  answer[i]: where the model gives its answer, at later layers. Changing them changes the answer.
The words, and what each one's value goes up with on real boards:
{words}

Statements:
  set(W, EXPR)                set word W to EXPR. EXPR is a number like 2.00, or words with
                              factors, like 0.50 * row_end[3] - col_end[7].
  clear(W)                    set word W back to 0.
  if EXPR > X:                run the indented lines if EXPR > X on this board (X from -2 to 2),
      set(...) or clear(...)
  else:                       otherwise these (optional).
      set(...) or clear(...)
  best = argmax(W1, W2, ...)  which of these 2 to 9 words is largest on this board,
  if best == 0:               then run the lines under its number.
      set(...) or clear(...)
  elif best == 1:
      set(...) or clear(...)
The model runs forward through its layers, and so does your program: a statement can only read
words at the same layer as, or an earlier layer than, the words it writes. The line words come
first, the answer words later.

You build the program one step at a time. {per_reply}
  ADD                 add statements at the end
  <statements>
  END
  REPLACE k           replace statement number k
  <statements>
  END
  DELETE k            delete statement number k
  END
  TEST                run statements on their own, once, without changing the program
  <statements>
  END
  TEST W              see word W's average value on the practice cells
  END
After each reply you see what each step did on the practice cells. A change that makes the
program worse is undone. Use TEST to learn what words do, then ADD what helps.
The goal is to get as many cells right as possible."""


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
