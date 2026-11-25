"""A model writes a program in a standard coding loop: generate, run, verify, read the trace, improve.

    1. Search: the writer generates fresh programs until one gets more practice cells right
       than no program.
    2. Improve: the writer sees the kept program and what happened when it ran (cells right,
       what each line did, a fixed and a broken cell with their traces), and its recent attempts
       that were not better, and writes an improved version. A program already tried is not run
       again. Each step samples agent.samples programs and keeps the best if it gets more cells
       right. After agent.restart_after steps with no gain, start again from step 1. The best
       program over all restarts is the result. agent.steps steps in all.
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

    best, best_right = None, plain             # the best program over all restarts
    kept, kept_right, kept_feedback, failed, stall = None, plain, "", [], 0
    for step in range(1, ac["steps"] + 1):
        if kept is None:                       # search: fresh programs until one beats no program
            code, right, feedback = attempt("search", p["base"] + "\n\n" + p["fresh"])
            log(f"step {step}, search: {right if right is not None else 'does not compile'} right (no program {plain})")
            if right not in (None, "tried") and right > plain:
                kept, kept_right, kept_feedback, failed, stall = code, right, feedback, [], 0
                log(f"kept, {kept_right} right:\n" + kept)
        else:                                  # improve: several samples from one prompt; keep the best if better
            recent = "\n".join(p["failed_one"].format(right=r if r is not None else "does not compile", code=c)
                               for c, r in failed[-ac["show_failed"]:])
            prompt = p["base"] + "\n\n" + p["improve"].format(right=kept_right, cells=n, plain=plain, code=kept,
                                                             feedback=kept_feedback,
                                                             failed=(p["failed"] + "\n" + recent) if recent else "")
            got = [attempt("improve", prompt) for _ in range(ac["samples"])]
            new = [(c, r, f) for c, r, f in got if r not in (None, "tried")]
            top = max(new, key=lambda t: t[1], default=None)
            log(f"step {step}, improve: " + ", ".join(str(r) if r is not None else "error" for _, r, _ in got)
                + f" (kept {kept_right})")
            if top and top[1] > kept_right:
                kept, kept_right, kept_feedback, stall = top[0], top[1], top[2], 0
                log(f"kept, {kept_right} right:\n" + kept)
            else:
                failed += [(c, r) for c, r, _ in got if r != "tried"]
                stall += 1
                if stall >= ac["restart_after"]:
                    log(f"no gain in {stall} steps: restart from a fresh program")
                    kept = None
        if kept is not None and kept_right > best_right:
            best, best_right = kept, kept_right
    if best is None:
        log("no program beat no program")
        return
    kept, kept_right = best, best_right

    open(f"{a.out}/program.py", "w").write(kept + "\n")
    log(f"kept program, {kept_right} of {n} practice cells right:\n" + kept)
    log("check:\n" + check.text(check.check(m, compile_python(kept), practice_boards, test_boards)))


if __name__ == "__main__":
    main()
