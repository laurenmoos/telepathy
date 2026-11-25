# telepathy

A small frozen language model (Qwen3-0.6B) fills in sudoku cells. A short program runs inside the model while it does, reading some of its internal numbers and changing others. The goal is for a model to write that program itself, by trying things and seeing what happens.

## How it works

**1. The model reads the board.** The prompt lists the board's 9 rows, 9 columns and 9 boxes, one line each (0 means empty), and then names the target cell, like `r4c7=`. The model's next character is its answer.

**2. Words: numbers inside the model that a program can use.** A word is one direction in the model's internal state, at one place in the prompt and one layer. Its value on a board says where the model's state sits along that direction (0 is typical, +1 or -1 is unusual). A program can read a word's value or set it. All words are found with no labels.

- **Board words** (`reads.py`) sit at the end of the target cell's row, column and box lines, where the model has just read that line. They come from independent component analysis of the model's state there, 32 per line. Many of them go up when a particular digit is missing from the line.
- **Answer words** (`words.py`) sit where the model gives its answer. They are the 8 directions that give the most varied control over the answer across many boards. Setting one pushes the answer toward certain digits.

**3. The program language** (`lang.py`) is a small Python subset:

```python
best = argmax(row_end[25], row_end[15], row_end[6])   # which word is largest on this board
if best == 0:
    set(answer[5], 3.00)                              # push this answer word
elif best == 1:
    set(answer[7], 3.00)
```

It also has `clear(W)`, sums like `0.5 * row_end[3] - col_end[1]`, and `if EXPR > X:` with an optional `else:`.

**4. Running a program** (`model.py`). The program runs inside one forward pass. Each statement acts at the layer of its words, and the pass only moves forward, so a statement can read only words at or before the layer it writes.

**5. Writing a program** (`write.py`). A writer builds the program one step at a time. Each step is an action: add, replace or delete a statement, or test statements or a word. Each action runs at once on 96 practice cells, and the writer sees what changed: cells right, answers changed, cells fixed and broken. A change that lowers the number of cells right is undone. The writer can be the steered model itself, another model, or a person (or assistant) writing replies into files.

**6. Checking a program** (`check.py`). The program fills every empty cell of the practice boards and of held-out boards, and is compared with no program.

## What we know so far

- The model records which digits are in each row, column and box at the end of each line, but that information never reaches the place where it answers.
- A program that reads the board words and pushes the matching answer word closes that gap: written step by step in the loop (`programs/row_argmax.txt`), it takes the model from 10% to 43% of held-out cells right (6% to 25% of boards solved) on boards with 1-2 empty cells.
- The model can only be pushed toward some digits: 1, 2, 4, 5, 6 and 7, not 3 or 8, and only weakly toward 9.
- Qwen3-0.6B cannot yet write such a program itself. Qwen3-8B chose sensible board words in a short test, but the two models together do not fit in this Mac's memory at full precision.

## Run

```
python words.py                            # answer words (cached in word_cache)
python reads.py                            # board words (cached in word_cache)
python write.py --out runs/write           # the writing loop; the best program is checked at the end
python write.py --set write.writer=qwen3-8b --set write.sessions=2 --set write.actions=8
python write.py --set write.writer=files   # write the replies yourself
python check.py programs/row_argmax.txt    # check a program
```

All settings are in `config.yaml`; override any of them with `--set key.sub=value` (quote values with brackets in zsh).

## Files

| File | What it does |
|---|---|
| `config.yaml` | Settings |
| `sudoku.py` | Boards |
| `words.py` | Answer words |
| `reads.py` | Board words |
| `lang.py` | The program language |
| `model.py` | The model: runs programs, writes replies |
| `core.py` | Shared setup: settings, loading, boards, playing boards |
| `write.py` | The writing loop |
| `check.py` | Checking a program |
| `programs/` | Programs worth keeping |
