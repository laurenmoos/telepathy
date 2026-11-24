# telepathy

One Qwen model writes small programs that steer itself through sudoku boards. A program reads the model's internal state while it answers, and pushes its answer toward a digit. The model sees how its programs did, cell by cell, and writes better ones. The weights never change, and the prompt has no instructions in words.

## Run

```
./run.sh                    # run, continuing from the last checkpoint if there is one
./run.sh --fresh            # start over
./run.sh --model qwen3-8b   # the bigger model (~16 GB of memory)
```

The first time, `run.sh` sets up Python in `.venv` and installs `requirements.txt`. It uses the Mac's GPU automatically and loads models from the local Hugging Face cache. Progress prints as it goes; the checkpoint is `checkpoint.pt` and the final results are `results.json`.

## The loop

```
model writes programs

repeat:
    run each program on practice boards, score it
    keep the best
    model writes new ones from the best, their scores and traces
    recombine and mutate the best

test the best on new boards
```

Each line is one function in `loop.py`.

**Writing.** Several sessions run side by side. In each, the model sees the best programs, their scores and the best one's trace. It writes a program, sees it run on a few practice boards with a cell-by-cell trace, and revises, for a few turns. Each session's best draft joins the population.

**Running a program on a board.** Every empty cell is filled in reading order. Each cell's prompt is the current board, and the model's digit is written in before the next cell. Wrong digits stay in.

**Scoring.** Boards fully solved, then empty cells right, then moves that broke no sudoku rule, minus a small cost per instruction.

**Testing.** The best program on held-out boards, and on harder boards with 10-20 empty cells, next to two baselines: the model told the sudoku rule in words with no program, and no program at all. Every program in the loop is written by the model or comes from recombining and mutating its programs (plus a few random ones at the start). None are written by hand.

## How the model sees a board

The prompt has 27 lines of 9 digits (0 = empty): the 9 rows, then the 9 columns, then the 9 boxes. Then the target cell, like `r4c7=`. Every row, column and box has its own line, so each has a line end where the model's state can be read.

## The program language

```
(seq S ...)              run statements in order
(for v S ...)            run the statements for v = 1, 2, ..., 9
(read UNIT D LAYER)      UNIT is ROW, COL or BOX. D is a digit 1-9, or v inside a for.
                         Reads the gate "is D in this cell's UNIT?" at the end of that line.
(write STRENGTH)         push the model toward the digit that looks most missing so far
```

A gate is a fitted direction plus a threshold in the model's state, fitted once per model on separate boards. The write push follows the model's own output weights for that digit, added at its last layer.

## Files

| File | What it is |
|---|---|
| `run.sh` | Sets up and runs everything. |
| `loop.py` | The loop above. Saves a checkpoint as it goes; running again continues where it stopped. |
| `model.py` | The model, loaded once. Runs programs on boards (one model pass per board state, cached) and replies to chats. |
| `write.py` | How the model writes programs: what it is shown, and the write, run, see the trace, revise sessions. |
| `lang.py` | The program language: parse, print, check limits, run, mutate, recombine. |
| `config.yaml` | All settings: model, layers, limits, boards, loop, fitness weights. |
| `sudoku.py` | Boards, a plain solver (fills every cell where only one digit fits), board sets saved to files. |
| `probes.py` | Device choice and gate fitting. |

## Results so far (Qwen3-0.6B on the Mac's GPU, `--generations 0`)

This run only wrote the first programs and tested the best one, with no generations of improvement.

| | Test boards (2-5 empty cells): solved / cells right | Harder boards (10-20 empty): solved / cells right |
|---|---|---|
| Rule in words, no program | 0% / 11% | 0% / 14% |
| No program | 0% / 13% | 0% / 8% |
| Best of the first programs | 83% / 95% | 0% / 59% |

The gates read "is this digit in this line?" at 93-98% accuracy at layers 1 and 2, for rows, columns and boxes alike.
