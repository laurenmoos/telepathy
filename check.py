"""Check a program: run it on the practice boards and on the held-out boards, filling every empty
cell in order (wrong digits stay in), and compare with no program.

    python check.py programs/row_argmax.txt [--set boards.blanks=[2,5]]

A program file holds statements in the program language (lang.py).
"""
import sys

import core, lang


def check(m, statements, practice, test):
    """Cells right and boards solved, with no program and with the program, on both board sets."""
    progs = [lang.Prog([]), lang.compile_program(statements)]
    out = {}
    for name, boards in (("practice", practice), ("held-out", test)):
        none, prog = (core.summarize(mv) for mv in core.play(progs, boards, m))
        out[name] = dict(no_program=none, program=prog, boards=len(boards))
    return out


def text(result):
    return "\n".join(f"{name} ({r['boards']} boards): cells right {r['no_program']['cells_right']:.0%} -> "
                     f"{r['program']['cells_right']:.0%}; boards solved {r['no_program']['solved']:.0%} -> "
                     f"{r['program']['solved']:.0%}" for name, r in result.items())


if __name__ == "__main__":
    cfg, a = core.settings(sys.argv[2:])
    statements = lang.split_statements(open(sys.argv[1]).read())
    m = core.load_model(cfg, a)
    core.load_words(m, cfg)
    practice, test = core.make_boards(cfg)
    print(lang.show(statements))
    print(text(check(m, statements, practice, test)))
