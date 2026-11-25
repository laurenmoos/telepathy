"""Check a program file on the practice and held-out boards.

    telepathy/run.sh check programs/row_argmax.txt [--set boards.blanks=[2,5]]

A program file holds statements in the program language (programs/lang.py).
"""
import sys

from .eval.check import check, text
from .programs import lang
from .setup.log import log
from .setup.settings import settings, load_model, load_words
from .sudoku.boards import make_boards


def main(argv):
    cfg, a = settings(argv[1:])
    statements = lang.split_statements(open(argv[0]).read())
    m = load_model(cfg, a)
    load_words(m, cfg)
    practice, test = make_boards(cfg, log=log)
    print(lang.show(statements))
    print(text(check(m, statements, practice, test)))


if __name__ == "__main__":
    main(sys.argv[1:])
