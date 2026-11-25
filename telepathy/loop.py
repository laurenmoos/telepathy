"""The outer loop: writing sessions side by side, each building a program step by step.

    ./run.sh [--set write.sessions=2] [--set write.writer=qwen3-8b-4bit] [--out runs/write]

At the end, the best program from all sessions is checked on the practice and held-out boards
(programs/check.py). What one step does is in programs/write.py; who writes is in
programs/writers.py.
"""
import json, os, random

from .programs import check, lang
from .programs.practice import Practice
from .programs.write import act, word_table
from .programs.writers import FileWriter, MLXWriter, ModelWriter
from .setup.log import log
from .setup.settings import settings, load_model, load_words
from .sudoku.boards import make_boards


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/write"))
    w = cfg["write"]
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(w["seed"])
    m = load_model(cfg, a)
    load_words(m, cfg)
    practice_boards, test_boards = make_boards(cfg, log=log)
    practice = Practice(m, practice_boards, w["cells"], rng)
    log(f"no program: {practice.right(practice.plain)} of {len(practice.cells)} practice cells right")

    if w["writer"] == "files":
        writer = FileWriter(a.out)
    elif str(cfg["models"].get(w["writer"], w["writer"])).startswith("mlx-community/"):
        writer = MLXWriter(cfg["models"].get(w["writer"], w["writer"]), w)
    elif w["writer"]:
        writer = ModelWriter(load_model(cfg, a, w["writer"]), w)
    else:
        writer = ModelWriter(m, w)

    p = cfg["prompts"]["loop"]
    per_reply = p["one_action"] if w["actions"] == 1 else p["many_actions"].format(actions=w["actions"])
    board_names = [n for n in lang.NAME_TO_WORD if not n.startswith("answer")]
    sessions = []
    for i in range(w["sessions"]):
        keep = set(rng.sample(board_names, min(w["show_words"], len(board_names)))) | {
            n for n in lang.NAME_TO_WORD if n.startswith("answer")}
        system = p["system"].format(words=word_table(m, keep), per_reply=per_reply).rstrip()
        first = p["first"].format(right=practice.right(practice.plain), cells=len(practice.cells))
        sessions.append(dict(chat=[{"role": "system", "content": system}, {"role": "user", "content": first}],
                             program=[], digits=practice.plain, best=[], best_digits=practice.plain, log=[]))

    for step in range(1, w["steps"] + 1):
        replies = writer.reply([s["chat"] for s in sessions], step)
        for s, text in zip(sessions, replies):
            text = text.strip()
            results = [act(s, x, practice) for x in lang.split_reply(text, w["actions"])]
            feedback = p["next"].format(results="\n".join(results), program=lang.show(s["program"]))
            s["chat"] += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback}]
            s["log"].append(dict(step=step, reply=text, results=results))
        log(f"step {step}: cells right per session " + ", ".join(str(practice.right(s["digits"])) for s in sessions)
            + f" (no program {practice.right(practice.plain)})")
        for i, s in enumerate(sessions, 1):
            log(f"    session {i}: " + " | ".join(r[:100] for r in s["log"][-1]["results"]))
        json.dump([dict(program=s["program"], best=s["best"], log=s["log"]) for s in sessions],
                  open(f"{a.out}/sessions.json", "w"), indent=1)

    best = max(sessions, key=lambda s: practice.right(s["best_digits"]))["best"]
    log("best program:\n" + lang.show(best))
    open(f"{a.out}/best_program.txt", "w").write("\n".join(best) + "\n")
    if best:
        log("check:\n" + check.text(check.check(m, best, practice_boards, test_boards)))


if __name__ == "__main__":
    main()
