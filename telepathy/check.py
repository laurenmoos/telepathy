"""Check a program file on the practice and held-out boards.

    telepathy/run.sh check runs/loop/program.py [--set boards.blanks=[2,5]]
"""
import sys

from .eval.check import check, text
from .setup.log import log
from .setup.settings import settings, load_live
from .sudoku.boards import make_boards


def main(argv):
    cfg, a = settings(argv[1:])
    code = open(argv[0]).read()
    live = load_live(cfg)
    practice, test = make_boards(cfg, log=log)
    print(code)
    print(text(check(live, code, practice, test)))


if __name__ == "__main__":
    main(sys.argv[1:])
