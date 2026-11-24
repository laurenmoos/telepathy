"""The loop. One Qwen model writes programs that steer itself through sudoku boards.

    model writes programs

    repeat:
        run each program on practice boards, score it
        keep the best
        model writes new ones from the best, their scores and traces
        recombine and mutate the best

    test the best on new boards

Each line is one function below. The model's weights never change. Programs act on its
internal state while it answers (model.py); the model also writes them (write.py).

The loop saves a checkpoint as it goes. If it stops (for example at --max-seconds),
running the same command again continues where it stopped.

Usage:
    python loop.py
    python loop.py --model qwen3-8b --device mps
"""
import argparse, copy, json, os, random, sys, time
import torch, yaml

import lang, sudoku, write
from model import Model

T0 = time.time()
ARGS = None


def log(*x):
    print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)


class OutOfTime(Exception):
    pass


def check_time():
    if ARGS and ARGS.max_seconds and time.time() - T0 > ARGS.max_seconds:
        raise OutOfTime()


# ---------------------------------------------------------------- setup

def settings():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", help="a name under `models` in the config, or a path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    ap.add_argument("--generations", type=int)
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    ap.add_argument("--max-seconds", type=float, default=None, help="save and stop after this long")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ev = cfg["evolution"]
    if a.generations is not None:
        ev["generations"] = a.generations
    for new, old in [("writing_sessions", "from_writer"), ("turns", "writer_turns"), ("show_best", "writer_shows")]:
        ev.setdefault(new, ev.get(old))                    # older run configs
    name = a.model or cfg.get("model") or cfg.get("solver_model")
    cfg["model_path"] = cfg["models"].get(name, name)
    return cfg, a


def load_model(cfg, a):
    m = Model(cfg["model_path"], cfg["read_layers"], device=a.device,
              dtype=getattr(torch, a.dtype) if a.dtype else None)
    log(f"model {cfg['model_path']} on {m.device} in {m.model.dtype}")
    return m


def fit_gates(m, cfg):
    """One gate per (unit, digit, layer): "is this digit in this line?". Fitted once per
    model on boards never used for scoring, and saved."""
    path = f"gates_cache/{os.path.basename(cfg['model_path'].rstrip('/'))}.pt"
    recipe = dict(layers=cfg["read_layers"], n=cfg["gate_boards"])
    if os.path.exists(path):
        saved = torch.load(path)
        if saved["recipe"] == recipe:
            m.load_state(saved["state"])
            log(f"gates loaded from {path}")
            return
    log("fitting gates")
    for line in m.fit_gates(cfg["gate_boards"], seed=12345, log=log):
        log(" ", line)
    os.makedirs("gates_cache", exist_ok=True)
    torch.save(dict(recipe=recipe, state=m.state()), path)


def make_boards(cfg):
    """Boards a plain solver can finish, saved to files so every run uses the same ones."""
    b = cfg["boards"]
    lo, hi = b["blanks"]
    practice = sudoku.board_set(f"{b['dir']}/tune_{b['tune']}_blanks{lo}-{hi}_seed{b['seed']}.json", b["tune"], (lo, hi), b["seed"])
    test = sudoku.board_set(f"{b['dir']}/test_{b['test']}_blanks{lo}-{hi}_seed{b['seed'] + 1}.json", b["test"], (lo, hi), b["seed"] + 1)
    hard = None
    if "hard" in b:
        hlo, hhi = b["hard"]["blanks"]
        hard = sudoku.board_set(f"{b['dir']}/hard_{b['hard']['n']}_blanks{hlo}-{hhi}_seed{b['seed'] + 2}.json",
                                b["hard"]["n"], (hlo, hhi), b["seed"] + 2)
    log(f"boards: {len(practice)} practice, {len(test)} test ({lo}-{hi} empty cells)"
        + (f", {len(hard)} harder test ({hlo}-{hhi} empty cells)" if hard else ""))
    return practice, test, hard


# ---------------------------------------------------------------- running and scoring

def run_boards(progs, boards, m, save=None):
    """Every program fills every board's empty cells in reading order. Wrong digits stay in.
    All (program, board) pairs move one cell at a time together, so each new board state
    gets one model pass, shared by every pair that reaches it. Returns moves[program][board]."""
    pairs = [(i, b) for i in range(len(progs)) for b in range(len(boards))]
    grids = [[row[:] for row in boards[b]["board"]] for _, b in pairs]
    todo = [[(r, c) for r in range(9) for c in range(9) if boards[b]["board"][r][c] == 0] for _, b in pairs]
    moves = [[] for _ in pairs]
    for step in range(max(len(t) for t in todo)):
        active = [p for p in range(len(pairs)) if step < len(todo[p])]
        items = [(grids[p], *todo[p][step]) for p in active]
        for k in range(0, len(items), 16):                  # one batch at a time, saved as it goes
            check_time()
            m.prepare(items[k:k + 16])
            if save:
                save()
        for p in active:
            i, b = pairs[p]
            r, c = todo[p][step]
            tr = {}
            digit = m.answer(lambda read: lang.run(progs[i], read, tr), grids[p], r, c)
            legal = digit in sudoku.all_candidates(grids[p], r, c)
            grids[p][r][c] = digit
            moves[p].append(dict(r=r, c=c, digit=digit, correct=digit == boards[b]["solution"][r][c],
                                 legal=legal, answer=boards[b]["solution"][r][c], trace=tr))
    out = [[None] * len(boards) for _ in progs]
    for p, (i, b) in enumerate(pairs):
        out[i][b] = moves[p]
    return out


def summarize(all_moves):
    n = len(all_moves)
    return dict(solved=sum(all(x["correct"] for x in mv) for mv in all_moves) / n,
                cells_right=sum(sum(x["correct"] for x in mv) / len(mv) for mv in all_moves) / n,
                legal=sum(sum(x["legal"] for x in mv) / len(mv) for mv in all_moves) / n)


def fitness(s, prog, cfg):
    w = cfg["fitness"]
    return (w["solved"] * s["solved"] + w["cells_right"] * s["cells_right"] + w["legal"] * s["legal"]
            - w["per_instruction"] * lang.size(prog))


def trace(all_moves, max_cells=6):
    """What the model is shown about a program: its results, then one line per cell."""
    lines = []
    for b, mv in enumerate(all_moves):
        for x in mv:
            if len(lines) >= max_cells:
                break
            tr = x["trace"]
            if not tr.get("read_any"):
                how = "nothing read, no push"
            else:
                ev = tr["evidence"]
                low = sorted(range(9), key=lambda k: ev[k])[:3]
                how = ("lowest evidence (looks missing): " + ", ".join(f"{k + 1} ({ev[k]:+.1f})" for k in low)
                       + f"; evidence for the right digit {x['answer']}: {ev[x['answer'] - 1]:+.1f}"
                       + ("; pushed toward " + ", ".join(str(d) for d, _ in tr["pushes"]) if tr["pushes"] else "; no write"))
            verdict = "right" if x["correct"] else ("wrong, but allowed" if x["legal"] else "wrong, breaks a rule")
            lines.append(f"board {b + 1} cell r{x['r'] + 1}c{x['c'] + 1}: wrote {x['digit']}, answer {x['answer']}, {verdict}. {how}")
    s = summarize(all_moves)
    return (f"On {len(all_moves)} practice boards: solved {s['solved']:.0%}, cells right {s['cells_right']:.0%}, "
            f"moves allowed by the rules {s['legal']:.0%}.\n" + "\n".join(lines))


# ---------------------------------------------------------------- the steps of the loop

def model_writes_programs(m, cfg, practice, best=(), best_trace="", progress=None, save=None):
    """Writing sessions. In each, the model sees the best programs, their scores and the
    best one's trace; writes a program; sees it run on a few practice boards; revises.
    Keeps each session's best draft. progress holds finished sessions, so a stopped run
    picks up at the next session (or the next turn)."""
    ev = cfg["evolution"]
    trial = practice[:ev["trial_boards"]]
    progress = progress if progress is not None else {}
    if "programs" in progress and "sessions" not in progress:      # an older checkpoint: keep what it wrote
        return [lang.parse(t) for t in progress["programs"]]
    progress.setdefault("sessions", {})

    def evaluate(progs):                                           # all of a turn's drafts run together
        moves = run_boards(progs, trial, m)
        return [(fitness(summarize(mv), p, cfg), trace(mv)) for p, mv in zip(progs, moves)]

    def on_turn():
        if save:
            save()
        check_time()

    check_time()
    log(f"  {ev['writing_sessions']} writing sessions, side by side")
    written = write.sessions(m, cfg, list(best), best_trace, ev["writing_sessions"], ev["turns"], evaluate,
                             log=log, state=progress["sessions"], on_turn=on_turn)
    progress["transcripts"] = progress["sessions"].get("msgs", [])
    return [max(w, key=lambda x: x[0])[1] for w in written if w]   # each session's best draft


def run_and_score(programs, practice, m, cfg, scores, save):
    """Run each program not scored before on every practice board, and score it."""
    new = [p for p in programs if lang.show(p) not in scores]
    if new:
        for p, mv in zip(new, run_boards(new, practice, m, save)):
            s = summarize(mv)
            scores[lang.show(p)] = (fitness(s, p, cfg), s)
        save()
    return sorted(((scores[lang.show(p)], p) for p in programs), key=lambda x: -x[0][0])


def keep_best(scored, cfg):
    return [p for _, p in scored[:cfg["evolution"]["keep"]]]


def recombine_and_mutate(scored, cfg, rng):
    """Pick parents by tournament, cross them (front of one, back of the other), mutate."""
    ev = cfg["evolution"]

    def parent():
        return max(rng.sample(scored, min(ev["tournament"], len(scored))), key=lambda x: x[0][0])[1]

    out, tries = [], 0
    while len(out) < ev["population"] and tries < 200:
        tries += 1
        child = lang.crossover(parent(), parent(), rng) if rng.random() < ev["crossover_rate"] else copy.deepcopy(parent())
        if rng.random() < ev["mutation_rate"]:
            child = lang.mutate(child, cfg, rng)
        if lang.valid(child, cfg) and child.stmts:
            out.append(child)
    return out


def unique_valid(progs, cfg):
    seen, out = set(), []
    for p in progs:
        k = lang.show(p)
        if k not in seen and lang.valid(p, cfg):
            seen.add(k); out.append(p)
    return out


def run_boards_by_rule(boards, m):
    """Baseline with no program: the model is told the sudoku rule in words and fills each
    board's empty cells in reading order, the same way programs do."""
    grids = [[row[:] for row in b["board"]] for b in boards]
    todo = [[(r, c) for r in range(9) for c in range(9) if b["board"][r][c] == 0] for b in boards]
    moves = [[] for _ in boards]
    for step in range(max(len(t) for t in todo)):
        check_time()
        active = [i for i in range(len(boards)) if step < len(todo[i])]
        digits = m.answer_by_rule([(grids[i], *todo[i][step]) for i in active])
        for i, d in zip(active, digits):
            r, c = todo[i][step]
            legal = d in sudoku.all_candidates(grids[i], r, c)
            grids[i][r][c] = d
            sol = boards[i]["solution"][r][c]
            moves[i].append(dict(r=r, c=c, digit=d, correct=d == sol, legal=legal, answer=sol, trace={}))
    return moves


def test_on_new_boards(best, boards, m, label, save=None):
    """The best program on boards it has never seen, next to baselines."""
    programs = {
        "no program": lang.parse("(seq)"),
        "hand-written: row only": lang.parse("(seq (for v (read ROW v 2)) (write 0.10))"),
        "hand-written: row, col, box": lang.parse("(seq (for v (read ROW v 2) (read COL v 2) (read BOX v 2)) (write 0.10))"),
        "best found": best,
    }
    report = {}
    log(f"{label}: share solved / share of empty cells right / share of moves legal")
    s = summarize(run_boards_by_rule(boards, m))
    report["rule in words (no program)"] = dict(program=None, **s)
    log(f"  {'rule in words (no program)':30s} {s['solved']:.2f} / {s['cells_right']:.2f} / {s['legal']:.2f}")
    for (name, p), mv in zip(programs.items(), run_boards(list(programs.values()), boards, m, save)):
        s = summarize(mv)
        report[name] = dict(program=lang.show(p), **s)
        log(f"  {name:30s} {s['solved']:.2f} / {s['cells_right']:.2f} / {s['legal']:.2f}   {lang.show(p)}")
    return report


# ---------------------------------------------------------------- the loop

def main():
    global ARGS
    cfg, a = settings()
    ARGS = a
    ev = cfg["evolution"]
    m = load_model(cfg, a)
    fit_gates(m, cfg)
    practice, test, hard = make_boards(cfg)

    ck = None if a.fresh or not os.path.exists(a.checkpoint) else torch.load(a.checkpoint, weights_only=False)
    if ck:
        gen, scores, history = ck["gen"], ck["scores"], ck["history"]
        programs = [lang.parse(t) for t in ck["population"]]
        rng = random.Random(); rng.setstate(ck["rng"])
        m.cache = ck.get("cache", ck.get("solver_cache", {}))
        progress = ck.get("writing", ck.get("writer_progress", {}))
        origin, sessions_log = ck.get("origin", {}), ck.get("sessions_log", [])
        log(f"resuming at generation {gen} ({len(scores)} programs already scored)")
    else:
        rng = random.Random(ev["seed"]); torch.manual_seed(ev["seed"])
        gen, scores, history, progress, origin, sessions_log = 0, {}, [], {}, {}, []
        programs = None

    def save():
        torch.save(dict(gen=gen, population=[lang.show(p) for p in programs or []], scores=scores,
                        history=history, rng=rng.getstate(), cache=m.cache, writing=progress,
                        origin=origin, sessions_log=sessions_log), a.checkpoint)

    try:
        # model writes programs
        if programs is None:
            log("model writes the first programs")
            programs = model_writes_programs(m, cfg, practice, progress=progress, save=save)
            for p in programs:
                origin.setdefault(lang.show(p), "written by the model, first programs")
            while len(programs) < ev["population"]:
                q = lang.random_program(cfg, rng)
                origin.setdefault(lang.show(q), "random")
                programs.append(q)
            programs = unique_valid(programs, cfg)[:ev["population"]]
            progress = {}
            save()

        while gen < ev["generations"]:
            # run each program on practice boards, score it
            scored = run_and_score(programs, practice, m, cfg, scores, save)
            (f, s), top = scored[0]
            if not history or history[-1]["gen"] != gen:
                history.append(dict(gen=gen, fitness=f, **s, program=lang.show(top), origin=origin.get(lang.show(top), "?"),
                                    population=[dict(fitness=sc[0], **sc[1], program=lang.show(q),
                                                     origin=origin.get(lang.show(q), "?")) for sc, q in scored]))
            log(f"gen {gen}  fitness {f:.3f}  solved {s['solved']:.2f}  cells right {s['cells_right']:.2f}  "
                f"legal {s['legal']:.2f}   {lang.show(top)}")

            # keep the best
            best = keep_best(scored, cfg)

            # model writes new ones from the best, their scores and traces
            shown = [(sc[0], lang.show(p)) for sc, p in scored[:ev["show_best"]]]
            top_trace = trace(run_boards([top], practice[:ev["trial_boards"]], m)[0])
            new = model_writes_programs(m, cfg, practice, shown, top_trace, progress, save)
            for q in new:
                origin.setdefault(lang.show(q), f"written by the model, generation {gen}")

            # recombine and mutate the best
            mixed = recombine_and_mutate(scored, cfg, rng)
            for q in mixed:
                origin.setdefault(lang.show(q), f"recombined and mutated, generation {gen}")

            programs = unique_valid(best + new + mixed, cfg)[:ev["population"]]
            sessions_log.append(dict(gen=gen, shown=shown, sessions=progress.get("transcripts", [])))
            gen += 1
            progress = {}
            save()

        # test the best on new boards
        best = run_and_score(programs, practice, m, cfg, scores, save)[0][1]
        report = dict(test=test_on_new_boards(best, test, m, "test boards", save))
        if hard:
            report["hard"] = test_on_new_boards(best, hard, m, "harder test boards", save)
    except OutOfTime:
        save()
        log("stopped at --max-seconds; run the same command again to continue")
        sys.exit(3)

    json.dump(dict(config=cfg, history=history, best=lang.show(best), best_origin=origin.get(lang.show(best), "?"),
                   results=report, origin=origin, writing_sessions=sessions_log,
                   all_scored=scores), open(a.out, "w"), indent=1, default=str)
    log("saved", a.out)


if __name__ == "__main__":
    main()
