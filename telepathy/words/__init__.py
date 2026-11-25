"""Finding the words, with no labels: answer words (answer.py) and board words (board.py).

    load(m, cfg)    both kinds for the model, built once and cached in cache/; board words first
                    (layer 0), then answer words, earliest layer first
    telepathy/run.sh words [--set model=...]    build (or load) both kinds and write their reports
"""
from ..setup.log import log


def load(m, cfg, log=log):
    from . import answer, board
    ws = list(answer.load(m, cfg, log=log)["primitives"]) + list(board.load(m, cfg, log=log)["reads"])
    return sorted(ws, key=lambda p: (p["layer"], p["place"], p.get("rank", 0)))
