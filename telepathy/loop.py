"""The outer loop: the model writes a program for itself in a standard coding loop.

    1. Search: the model writes fresh programs until one gets more practice cells right than no
       program.
    2. Improve: it sees the kept program, what happened when it ran (cells right, what each line
       did, a fixed and a broken cell with their traces) and its recent attempts that were not
       better, and writes a different program. A program already tried is not run again. Each
       step samples loop.samples programs and keeps the best if it gets more cells right. After
       loop.restart_after steps with no gain, start again from 1. The best program over all
       restarts is the result.
At the end the best program is checked on the practice and held-out boards.

The same model writes the programs and runs them (model/live.py). Every text it sees is in
config.yaml under prompts.

    telepathy/run.sh [--set loop.steps=20] [--out runs/loop]
"""
import json, os, random, re

from .eval import check
from .eval.practice import Cells
from .setup.log import log
from .setup.settings import settings, load_live
from .sudoku.boards import make_boards
from .sudoku.prompt import prompt27


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/loop"))
    lc, p = cfg["loop"], cfg["prompts"]
    os.makedirs(a.out, exist_ok=True)
    live = load_live(cfg)
    practice, test = make_boards(cfg, log=log)
    rng = random.Random(lc["seed"])
    cells = Cells(live, practice, lc["cells"], rng)
    n = len(cells.cells)
    plain = sum(d == c[3] for d, c in zip(cells.plain, cells.cells))
    log(f"no program: {plain} of {n} practice cells right")
    example = cells.prompts[0]
    base = p["base"].format(example=example, layers=len(live.layers), last=len(live.tok.encode(example)) - 1)
    history, tried = [], {}                    # tried: code -> cells right (None if it did not run)

    def write(prompt):
        out = live.generate(prompt, lc["max_tokens"], lc["temperature"], lc["think"]).split("</think>")[-1]
        m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
        return (m.group(1) if m else out).strip()

    def attempt(stage, prompt):
        code = write(prompt)
        if code in tried:
            history.append(dict(stage=stage, code=code, right=tried[code], feedback="already tried"))
            return code, "tried", ""
        try:
            compile(code, "<program>", "exec")
            feedback, right = cells.run(code, examples=lc["examples"])
        except SyntaxError as e:
            feedback, right = f"It does not run: line {e.lineno}: {e.msg}", None
        tried[code] = right
        history.append(dict(stage=stage, prompt=prompt, code=code, right=right, feedback=feedback))
        json.dump(history, open(f"{a.out}/session.json", "w"), indent=1)
        return code, right, feedback

    best, best_right = None, plain
    kept, kept_right, kept_feedback, failed, stall = None, plain, "", [], 0
    for step in range(1, lc["steps"] + 1):
        if kept is None:                       # search
            code, right, feedback = attempt("search", base + "\n\n" + p["fresh"])
            log(f"step {step}, search: {right if right is not None else 'error'} right (no program {plain})")
            if right not in (None, "tried") and right > plain:
                kept, kept_right, kept_feedback, failed, stall = code, right, feedback, [], 0
                log(f"kept, {kept_right} right:\n" + kept)
        else:                                  # improve
            recent = "\n".join(p["failed_one"].format(right=r if r is not None else "does not run", code=c)
                               for c, r in failed[-lc["show_failed"]:])
            prompt = base + "\n\n" + p["improve"].format(right=kept_right, cells=n, plain=plain, code=kept,
                                                         feedback=kept_feedback,
                                                         failed=(p["failed"] + "\n" + recent) if recent else "")
            got = [attempt("improve", prompt) for _ in range(lc["samples"])]
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
                if stall >= lc["restart_after"]:
                    log(f"no gain in {stall} steps: restart from a fresh program")
                    kept = None
        if kept is not None and kept_right > best_right:
            best, best_right = kept, kept_right
    if best is None:
        log("no program beat no program")
        return
    open(f"{a.out}/program.py", "w").write(best + "\n")
    log(f"best program, {best_right} of {n} practice cells right:\n" + best)
    log("check:\n" + check.text(check.check(live, best, practice, test)))


if __name__ == "__main__":
    main()
