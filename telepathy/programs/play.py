"""Playing whole boards with programs.

    play(progs, boards, m)   every program fills every board, cell by cell; wrong digits stay in
    summarize(moves)         boards solved and cells right
"""
from .run import run


def play(progs, boards, m):
    """Every program fills every board's empty cells in reading order. Wrong digits stay in.
    All (program, board) pairs move one cell at a time together. Returns moves[program][board]."""
    pairs = [(i, b) for i in range(len(progs)) for b in range(len(boards))]
    grids = [[row[:] for row in boards[b]["board"]] for _, b in pairs]
    todo = [[(r, c) for r in range(9) for c in range(9) if boards[b]["board"][r][c] == 0] for _, b in pairs]
    moves = [[] for _ in pairs]
    for step in range(max(len(t) for t in todo)):
        active = [p for p in range(len(pairs)) if step < len(todo[p])]
        m.prepare([(grids[p], *todo[p][step]) for p in active])
        ran = run(m, [(grids[p], *todo[p][step], progs[pairs[p][0]]) for p in active])
        for p, (digit, tr, _) in zip(active, ran):
            i, b = pairs[p]
            r, c = todo[p][step]
            grids[p][r][c] = digit
            moves[p].append(dict(r=r, c=c, digit=digit, correct=digit == boards[b]["solution"][r][c],
                                 answer=boards[b]["solution"][r][c], trace=tr))
    out = [[None] * len(boards) for _ in progs]
    for p, (i, b) in enumerate(pairs):
        out[i][b] = moves[p]
    return out


def summarize(moves):
    n = len(moves)
    return dict(solved=sum(all(x["correct"] for x in mv) for mv in moves) / n,
                cells_right=sum(sum(x["correct"] for x in mv) / len(mv) for mv in moves) / n)
