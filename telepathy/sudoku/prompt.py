"""The board prompt: 27 lines of 9 digits (0 = empty), the 9 rows, then the 9 columns, then the
9 boxes, then the target cell, like r4c7=. The places a word can sit at are the end of the
target cell's row, column and box line, and the answer position '='."""

PLACES = ("ANSWER", "ROW", "COL", "BOX")
PLACE_WORDS = {"ANSWER": "answer position", "ROW": "row line end", "COL": "column line end", "BOX": "box line end"}


def box_of(r, c):
    return 3 * (r // 3) + c // 3


def lines(grid):
    rows = [grid[r] for r in range(9)]
    cols = [[grid[r][c] for r in range(9)] for c in range(9)]
    boxes = [[grid[3 * (b // 3) + i][3 * (b % 3) + j] for i in range(3) for j in range(3)] for b in range(9)]
    return rows + cols + boxes


def prompt27(grid, r, c):
    return "\n".join("".join(str(x) for x in ln) for ln in lines(grid)) + f"\nr{r + 1}c{c + 1}="


def line_of(place, r, c):
    """Which line of its kind a place belongs to (0 for the answer position)."""
    return {"ANSWER": 0, "ROW": r, "COL": c, "BOX": box_of(r, c)}[place]
