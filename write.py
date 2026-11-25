"""Write a program by trying things and seeing what happens.

A writer builds a program for the steered model one step at a time. Several writing sessions
run side by side, and each one is a conversation:

    1. The writer sees the words it can use (and what each goes up with on real boards), the
       current program, and what its last actions did.
    2. It replies with one action, or with up to `actions` actions followed by DONE:
           ADD / REPLACE k / DELETE k    change the program
           TEST                         run statements on their own, once
           TEST W                       see word W's average value on the practice cells
    3. Each action runs at once on the same practice cells (one forward pass per cell).
       A change is kept unless it lowers the number of cells right. A test changes nothing.
    4. It sees one line per action saying what happened, and replies again.

At the end, check.py runs the best program from all sessions on the practice and held-out boards.

Who writes (write.writer in config.yaml):
    null        the steered model writes its own programs
    qwen3-8b-4bit  another model writes (on MLX); the steered model still runs them
    files       someone else writes: each prompt is saved as <out>/s<session>_<step>.prompt.txt,
                and the loop waits for <out>/s<session>_<step>.reply.txt

    python write.py [--set write.sessions=2] [--set write.writer=qwen3-8b-4bit] [--out runs/write]
"""
import json, os, random, re, time

import check, core, lang
from model import prompt27

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


# ---------------------------------------------------------------- the practice cells

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
        return [d for d, _, _ in self.m.run([(*c[:3], prog) for c in self.cells])]

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


# ---------------------------------------------------------------- writers

class ModelWriter:
    """A language model writes the replies, with constrained decoding so every reply is valid text."""

    def __init__(self, m, w):
        self.m, self.w = m, w
        self.grammar = lang.grammar(w["actions"])
        self.vocab = token_pieces(m)
        end = "END" if w["actions"] == 1 else "DONE"
        self.stop = lambda t: t.rstrip().endswith(end)

    def reply(self, conversations, step):
        return self.m.generate(conversations, stop=self.stop, temperature=self.w["temperature"],
                               max_new_tokens=300 * self.w["actions"], grammar=self.grammar, vocab=self.vocab,
                               repetition_penalty=self.w["repetition_penalty"])


class MLXWriter:
    """A quantized model on MLX writes the replies (for writers too big for memory in bf16).
    Same constrained decoding as ModelWriter, one conversation at a time."""

    def __init__(self, path, w):
        import types
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        self.w = w
        self.model, self.wrap = load(path)
        self.tok = self.wrap._tokenizer
        self.rx = __import__("regex").compile(lang.grammar(w["actions"]))
        self.index = {}                                       # candidates grouped by first character
        for tid in token_pieces(types.SimpleNamespace(tok=self.tok)):
            t = self.tok.decode([tid])
            if t:
                self.index.setdefault(t[0], []).append((tid, t))
        self.ends = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]
        self.sampler = make_sampler(temp=w["temperature"], top_p=0.95)
        self.penalty = make_logits_processors(repetition_penalty=w["repetition_penalty"])
        end = "END" if w["actions"] == 1 else "DONE"
        self.stop = lambda t: t.rstrip().endswith(end)

    def _mask(self, start):
        import mlx.core as mx
        import numpy as np

        def proc(tokens, logits):
            text = self.tok.decode(np.array(tokens[start:]).tolist(), skip_special_tokens=True)
            if self.rx.fullmatch(text):
                ok = self.ends
            else:
                ok = [tid for ch, toks in self.index.items() if self.rx.fullmatch(text + ch, partial=True)
                      for tid, t in toks if self.rx.fullmatch(text + t, partial=True)] or self.ends
            allow = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
            allow[ok] = 0.0
            return logits + mx.array(allow)
        return proc

    def reply(self, conversations, step):
        from mlx_lm import stream_generate
        out = []
        for conv in conversations:
            prompt = self.tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
            if not isinstance(prompt, list):
                prompt = prompt["input_ids"]
            text = ""
            for r in stream_generate(self.model, self.wrap, prompt, max_tokens=300 * self.w["actions"],
                                     sampler=self.sampler, logits_processors=self.penalty + [self._mask(len(prompt))]):
                text += r.text
                if self.stop(text):
                    break
            out.append(text)
        return out


class FileWriter:
    """Someone else writes the replies, into files."""

    def __init__(self, out):
        self.out = out

    def reply(self, conversations, step):
        for i, conv in enumerate(conversations):
            open(f"{self.out}/s{i + 1}_{step}.prompt.txt", "w").write(
                "\n\n".join(f"=== {m['role']} ===\n{m['content']}" for m in conv))
        paths = [f"{self.out}/s{i + 1}_{step}.reply.txt" for i in range(len(conversations))]
        core.log(f"waiting for {', '.join(paths)}")
        while not all(os.path.exists(p) for p in paths):
            time.sleep(2)
        time.sleep(0.5)
        return [open(p).read() for p in paths]


def token_pieces(m):
    """The token ids that programs are made of: tokenize keywords, numbers, every word name and
    typical statements, and add every single character, so constrained writing never reaches a
    dead end. A short candidate list makes constrained writing fast."""
    names = list(lang.NAME_TO_WORD)
    texts = ["ADD\n", "REPLACE 12\n", "DELETE 3\n", "TEST\n", "END\n", "DONE", "else:\n", "elif best == 8:\n",
             "if best == 0:\n", "    "] + [str(x) for x in range(100)] + [f"{x / 100:.2f}" for x in range(-999, 1000)]
    for n in names:
        texts += [f"set({n}, 1.00)\n", f"    set({n}, -2.50)\n", f"clear({n})\n", f"    clear({n})\n",
                  f"if {n} > 0.50:\n", f"best = argmax({n}, {n})\n", f"TEST {n}\nEND", f"0.50 * {n} + {n} - {n}"]
    ids = set()
    for t in texts:
        ids.update(m.tok(t, add_special_tokens=False).input_ids)
    for ch in lang.GRAMMAR_CHARS:
        one = m.tok(ch, add_special_tokens=False).input_ids
        if len(one) == 1:
            ids.add(one[0])
    return sorted(i for i in ids if set(m.tok.decode([i])) <= lang.GRAMMAR_CHARS)


# ---------------------------------------------------------------- the loop

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


def main(argv=None):
    cfg, a = core.settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/write"))
    w = cfg["write"]
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(w["seed"])
    m = core.load_model(cfg, a)
    core.load_words(m, cfg)
    practice_boards, test_boards = core.make_boards(cfg)
    practice = Practice(m, practice_boards, w["cells"], rng)
    core.log(f"no program: {practice.right(practice.plain)} of {len(practice.cells)} practice cells right")

    if w["writer"] == "files":
        writer = FileWriter(a.out)
    elif str(cfg["models"].get(w["writer"], w["writer"])).startswith("mlx-community/"):
        writer = MLXWriter(cfg["models"].get(w["writer"], w["writer"]), w)
    elif w["writer"]:
        writer = ModelWriter(core.load_model(cfg, a, w["writer"]), w)
    else:
        writer = ModelWriter(m, w)

    per_reply = ("Each reply is exactly one step:" if w["actions"] == 1 else
                 f"Each reply is up to {w['actions']} steps, one after another, then a line DONE. They run in order:")
    board_names = [n for n in lang.NAME_TO_WORD if not n.startswith("answer")]
    sessions = []
    for i in range(w["sessions"]):
        keep = set(rng.sample(board_names, min(w["show_words"], len(board_names)))) | {
            n for n in lang.NAME_TO_WORD if n.startswith("answer")}
        system = INSTRUCTIONS.format(words=word_table(m, keep), per_reply=per_reply)
        first = (f"The program is empty. With no program, {practice.right(practice.plain)} of "
                 f"{len(practice.cells)} practice cells are right.\nTake your first step.")
        sessions.append(dict(chat=[{"role": "system", "content": system}, {"role": "user", "content": first}],
                             program=[], digits=practice.plain, best=[], best_digits=practice.plain, log=[]))

    for step in range(1, w["steps"] + 1):
        replies = writer.reply([s["chat"] for s in sessions], step)
        for s, text in zip(sessions, replies):
            text = text.strip()
            results = [act(s, x, practice) for x in lang.split_reply(text, w["actions"])]
            feedback = "\n".join(results) + f"\n\nCurrent program:\n{lang.show(s['program'])}\n\nNext step."
            s["chat"] += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback}]
            s["log"].append(dict(step=step, reply=text, results=results))
        core.log(f"step {step}: cells right per session " + ", ".join(str(practice.right(s["digits"])) for s in sessions)
                 + f" (no program {practice.right(practice.plain)})")
        for i, s in enumerate(sessions, 1):
            core.log(f"    session {i}: " + " | ".join(r[:100] for r in s["log"][-1]["results"]))
        json.dump([dict(program=s["program"], best=s["best"], log=s["log"]) for s in sessions],
                  open(f"{a.out}/sessions.json", "w"), indent=1)

    best = max(sessions, key=lambda s: practice.right(s["best_digits"]))["best"]
    core.log("best program:\n" + lang.show(best))
    open(f"{a.out}/best_program.txt", "w").write("\n".join(best) + "\n")
    if best:
        core.log("check:\n" + check.text(check.check(m, best, practice_boards, test_boards)))


if __name__ == "__main__":
    main()
