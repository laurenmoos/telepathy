"""Check a program: run it on the practice boards and on the held-out boards, filling every empty
cell in order (wrong digits stay in), and compare with no program.

    ./run.sh check programs/row_argmax.txt [--set boards.blanks=[2,5]]

A program file holds statements in the program language (lang.py).
"""
import sys

from ..programs import lang
from .play import play, summarize


def check(m, statements, practice, test):
    """Cells right and boards solved, with no program and with the program, on both board sets."""
    progs = [lang.Prog([]), lang.compile_program(statements)]
    out = {}
    for name, boards in (("practice", practice), ("held-out", test)):
        none, prog = (summarize(mv) for mv in play(progs, boards, m))
        out[name] = dict(no_program=none, program=prog, boards=len(boards))
    return out


def text(result):
    return "\n".join(f"{name} ({r['boards']} boards): cells right {r['no_program']['cells_right']:.0%} -> "
                     f"{r['program']['cells_right']:.0%}; boards solved {r['no_program']['solved']:.0%} -> "
                     f"{r['program']['solved']:.0%}" for name, r in result.items())


def main(argv):
    from ..setup.log import log
    from ..setup.settings import settings, load_model, load_words
    from ..sudoku.boards import make_boards
    cfg, a = settings(argv[1:])
    statements = lang.split_statements(open(argv[0]).read())
    m = load_model(cfg, a)
    load_words(m, cfg)
    practice, test = make_boards(cfg, log=log)
    print(lang.show(statements))
    print(text(check(m, statements, practice, test)))


if __name__ == "__main__":
    main(sys.argv[1:])
