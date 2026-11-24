"""Boards, board families, prompts and the checker."""
import random

DIGITS = set(range(1, 10))


def solved_grid(rng):
    """A random valid solved sudoku, by shuffling a base pattern."""
    def pattern(r, c):
        return (3 * (r % 3) + r // 3 + c) % 9
    rb = list(range(3)); rng.shuffle(rb)
    rows = [g * 3 + r for g in rb for r in rng.sample(range(3), 3)]
    cb = list(range(3)); rng.shuffle(cb)
    cols = [g * 3 + c for g in cb for c in rng.sample(range(3), 3)]
    nums = list(range(1, 10)); rng.shuffle(nums)
    g = [[nums[pattern(r, c)] for c in cols] for r in rows]
    if rng.random() < 0.5:
        g = [list(x) for x in zip(*g)]
    return g


def unit_cells(r, c, unit):
    if unit == "row":
        return [(r, j) for j in range(9)]
    if unit == "col":
        return [(i, c) for i in range(9)]
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return [(br + i, bc + j) for i in range(3) for j in range(3)]


def candidates(board, r, c, unit):
    """Digits the target could still be, looking at one unit only."""
    seen = {board[i][j] for i, j in unit_cells(r, c, unit) if board[i][j]}
    return DIGITS - seen


def all_candidates(board, r, c):
    return candidates(board, r, c, "row") & candidates(board, r, c, "col") & candidates(board, r, c, "box")


def matches(board, r, c, spec):
    for unit, want in spec.items():
        n = len(candidates(board, r, c, unit))
        if want == "decisive" and n != 1:
            return False
        if want == "open" and n < 2:
            return False
    return len(all_candidates(board, r, c)) == 1   # always solvable from the three units


def make_board(rng, spec, tries=2000):
    """Blank cells until the target's units match the family spec."""
    for _ in range(tries):
        sol = solved_grid(rng)
        r, c = rng.randrange(9), rng.randrange(9)
        board = [row[:] for row in sol]
        board[r][c] = 0
        k = rng.randint(4, 30)
        for _ in range(k):
            i, j = rng.randrange(9), rng.randrange(9)
            if (i, j) != (r, c):
                board[i][j] = 0
        if matches(board, r, c, spec):
            return dict(board=board, r=r, c=c, answer=sol[r][c], solution=sol)
    raise RuntimeError(f"could not make a board for {spec}")


def prompt(b):
    """9 lines of 9 digits (0 = blank), then the target cell. The program's slot
    tokens follow, then '=' and the model's digit."""
    return board_prompt(b["board"], b["r"], b["c"])


def board_prompt(grid, r, c):
    rows = "\n".join("".join(str(x) for x in row) for row in grid)
    return f"{rows}\nr{r + 1}c{c + 1}"


def check(b, digit):
    """correct, legal, and which units the digit breaks. units_ok (0-3) is the graded
    signal: on these boards a legal digit is always the answer, so legal alone can't
    separate a near miss from a wild guess."""
    broken = [u for u in ("row", "col", "box")
              if digit not in candidates(b["board"], b["r"], b["c"], u)]
    return dict(correct=digit == b["answer"], legal=not broken, broken=broken, units_ok=3 - len(broken))


# ---------------------------------------------------------------- whole boards

def solve_passes(grid):
    """A plain solver. Each pass fills every empty cell where exactly one digit fits,
    looking at its row, column and box. Returns the boards: the start, then the board
    after each pass. Stops when the board is full or a pass fills nothing."""
    boards = [[row[:] for row in grid]]
    while True:
        g = boards[-1]
        nxt = [row[:] for row in g]
        filled = 0
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cand = all_candidates(g, r, c)
                    if len(cand) == 1:
                        nxt[r][c] = cand.pop()
                        filled += 1
        if filled == 0:
            return boards
        boards.append(nxt)
        if all(all(row) for row in nxt):
            return boards


def finishable(grid):
    """True if solve_passes fills the whole board."""
    return all(all(row) for row in solve_passes(grid)[-1])


def make_finishable_board(rng, blanks):
    """Blank cells of a solved grid one at a time, at random, keeping only blanks that
    leave the board finishable by solve_passes. Stops at `blanks` empty cells."""
    while True:
        sol = solved_grid(rng)
        grid = [row[:] for row in sol]
        cells = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(cells)
        n = 0
        for r, c in cells:
            keep = grid[r][c]
            grid[r][c] = 0
            if finishable(grid):
                n += 1
                if n == blanks:
                    break
            else:
                grid[r][c] = keep
        if n == blanks:
            return dict(board=grid, solution=sol, blanks=blanks,
                        passes=len(solve_passes(grid)) - 1)


def board_set(path, n, blanks_range, seed):
    """Load a saved set of finishable boards, or make it and save it.
    The same file is reused by every run, so runs compare on the same boards."""
    import json, os
    if os.path.exists(path):
        return json.load(open(path))
    rng = random.Random(seed)
    lo, hi = blanks_range
    boards = [make_finishable_board(rng, rng.randint(lo, hi)) for _ in range(n)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(boards, open(path, "w"))
    return boards
