"""A model writes a program in a standard coding loop: generate, run, verify, read the trace, improve.

    1. Search: the writer generates fresh programs until one gets more practice cells right
       than no program (or agent.fresh_tries run out).
    2. Improve: the writer sees the kept program and what happened when it ran (cells right,
       what each line did, a fixed and a broken cell with their traces), and its recent attempts
       that were not better, and writes an improved version. A program already tried is not run
       again. A new one is run and kept only if it gets more cells right. Repeat agent.steps times.
At the end the kept program is checked on the practice and held-out boards.

Every text the writer sees is in config.yaml under prompts.agent. Today the steered model is
Qwen3-0.6B and the writer is another model; later the writer will steer itself.

    telepathy/run.sh agent [--set agent.steps=20] [--out runs/agent]
"""
import json, os, random, re

from .eval import check
from .eval.practice import Cells
from .programs.python import compile_python
from .setup.log import log
from .setup.settings import settings, load_model, load_words
from .sudoku.boards import make_boards


class Writer:
    """An MLX model that replies to one prompt with a program."""

    def __init__(self, path, a):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok = load(path)
        self.sampler = make_sampler(temp=a["temperature"], top_p=0.95, top_k=20)
        self.a = a

    def program(self, prompt):
        from mlx_lm import generate
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           enable_thinking=self.a["think"])
        out = generate(self.model, self.tok, ids, max_tokens=self.a["max_tokens"], sampler=self.sampler)
        out = out.split("</think>")[-1]
        m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
        return (m.group(1) if m else out).strip()


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/agent"))
    ac, p = cfg["agent"], cfg["prompts"]["agent"]
    os.makedirs(a.out, exist_ok=True)
    m = load_model(cfg, a)
    load_words(m, cfg)
    practice_boards, test_boards = make_boards(cfg, log=log)
    cells = Cells(m, practice_boards, ac["cells"], random.Random(ac["seed"]))
    plain = cells.run("")[1]
    n = len(cells.cells)
    log(f"no program: {plain} of {n} practice cells right")
    writer = Writer(cfg["models"].get(ac["writer"], ac["writer"]), ac)
    history, tried = [], {}                    # tried: code -> cells right (None if it does not compile)

    def attempt(stage, prompt):
        code = writer.program(prompt)
        if code in tried:
            history.append(dict(stage=stage, prompt=prompt, code=code, right=tried[code], feedback="already tried"))
            return code, "tried", ""
        try:
            feedback, right = cells.run(code, count=0, examples=ac["examples"])
        except ValueError as e:
            feedback, right = f"It does not compile: {e}", None
        tried[code] = right
        history.append(dict(stage=stage, prompt=prompt, code=code, right=right, feedback=feedback))
        json.dump(history, open(f"{a.out}/session.json", "w"), indent=1)
        return code, right, feedback

    kept, kept_right, kept_feedback = None, plain, ""
    for i in range(1, ac["fresh_tries"] + 1):
        code, right, feedback = attempt("search", p["base"] + "\n\n" + p["fresh"])
        log(f"search {i}: {right if right is not None else 'does not compile'} right (no program {plain})")
        if right not in (None, "tried") and right > kept_right:
            kept, kept_right, kept_feedback = code, right, feedback
            break
    if kept is None:
        log("no fresh program beat no program")
        return
    log(f"kept, {kept_right} right:\n" + kept)

    failed = []                                # recent attempts that were not better: (code, right)
    for i in range(1, ac["steps"] + 1):
        recent = "\n".join(p["failed_one"].format(right=r if r is not None else "does not compile", code=c)
                           for c, r in failed[-ac["show_failed"]:])
        prompt = p["base"] + "\n\n" + p["improve"].format(right=kept_right, cells=n, plain=plain, code=kept,
                                                         feedback=kept_feedback,
                                                         failed=(p["failed"] + "\n" + recent) if recent else "")
        code, right, feedback = attempt("improve", prompt)
        if right == "tried":
            log(f"improve {i}: a program already tried, not run")
            continue
        keep = right is not None and right > kept_right
        log(f"improve {i}: {right if right is not None else 'does not compile'} right, "
            f"{'kept' if keep else 'not kept'} (kept {max(kept_right, right or 0)})")
        if keep:
            kept, kept_right, kept_feedback = code, right, feedback
            log("kept:\n" + kept)
        else:
            failed.append((code, right))

    open(f"{a.out}/program.py", "w").write(kept + "\n")
    log(f"kept program, {kept_right} of {n} practice cells right:\n" + kept)
    log("check:\n" + check.text(check.check(m, compile_python(kept), practice_boards, test_boards)))


if __name__ == "__main__":
    main()
