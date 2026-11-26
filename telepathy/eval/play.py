"""Playing whole boards with a program.

    play(live, code, boards)   the program fills every board, cell by cell, in reading order;
                               wrong digits stay in. Returns moves[board].
    summarize(moves)           boards solved and cells right
"""
import numpy as np

from ..sudoku.prompt import prompt27


def play(live, code, boards):
    digits = [live.token_id(str(d)) for d in range(1, 10)]
    grids = [[row[:] for row in b["board"]] for b in boards]
    todo = [[(r, c) for r in range(9) for c in range(9) if b["board"][r][c] == 0] for b in boards]
    moves = [[] for _ in boards]
    for step in range(max(len(t) for t in todo)):
        active = [b for b in range(len(boards)) if step < len(todo[b])]
        out = live.run([prompt27(grids[b], *todo[b][step]) for b in active], code, digits)
        for b, (lp, trace, _) in zip(active, out):
            r, c = todo[b][step]
            d = int(np.argmax(lp)) + 1
            grids[b][r][c] = d
            moves[b].append(dict(r=r, c=c, digit=d, correct=d == boards[b]["solution"][r][c],
                                 answer=boards[b]["solution"][r][c], trace=trace))
    return moves


def summarize(moves):
    n = len(moves)
    return dict(solved=sum(all(x["correct"] for x in mv) for mv in moves) / n,
                cells_right=sum(sum(x["correct"] for x in mv) / len(mv) for mv in moves) / n)
