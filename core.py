"""Shared setup: settings, the model and its words, the boards, and playing boards with a program.

    settings(argv)        config.yaml, with --set key.sub=value overrides
    load_model(cfg, a)    the model that is steered (cfg["model"])
    load_words(m, cfg)    the words: answer words (words.py) and board words (reads.py), built
                          once and cached in word_cache; named for programs (lang.setup)
    make_boards(cfg)      practice and held-out boards, saved in boards/ so every run uses the same
    play(progs, boards)   every program fills every board, cell by cell; wrong digits stay in
    summarize(moves)      boards solved and cells right
"""
import argparse, time
import torch, yaml

import lang, sudoku
from model import Model

T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


def settings(argv=None, extra=None):
    """Read config.yaml. extra: a function that adds more arguments to the parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", action="append", default=[], help="override a setting, e.g. boards.blanks=[2,5]")
    if extra:
        extra(ap)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    cfg["model_path"] = cfg["models"].get(cfg["model"], cfg["model"])
    return cfg, a


def load_model(cfg, a, name=None):
    path = cfg["models"].get(name, name) if name else cfg["model_path"]
    m = Model(path, device=a.device)
    log(f"model {path} on {m.device} in {m.model.dtype}")
    return m


def load_words(m, cfg):
    """Give the model its words and name them for programs. Board words come first (layer 0),
    then answer words, earliest layer first."""
    import reads, words
    ws = list(words.load(m, cfg, log=log)["primitives"]) + list(reads.load(m, cfg, log=log)["reads"])
    ws = sorted(ws, key=lambda p: (p["layer"], p["place"], p.get("rank", 0)))
    m.set_words(ws)
    lang.setup(ws)
    log(f"{len(ws)} words: " + ", ".join(f"{g} {n}" for g, n in lang.GROUP_SIZE.items()))


def make_boards(cfg):
    b = cfg["boards"]
    lo, hi = b["blanks"]
    practice = sudoku.board_set(f"{b['dir']}/tune_{b['practice']}_blanks{lo}-{hi}_seed{b['seed']}.json",
                                b["practice"], (lo, hi), b["seed"])
    test = sudoku.board_set(f"{b['dir']}/test_{b['test']}_blanks{lo}-{hi}_seed{b['seed'] + 1}.json",
                            b["test"], (lo, hi), b["seed"] + 1)
    log(f"boards: {len(practice)} practice, {len(test)} held-out ({lo}-{hi} empty cells)")
    return practice, test


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
        ran = m.run([(grids[p], *todo[p][step], progs[pairs[p][0]]) for p in active])
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
