# Next: a bigger model writes the program

Read `README.md` first. This file lists what to do next and what to watch for.

## The question

Can a bigger model, using the writing loop (`write.py`), write a program that helps Qwen3-0.6B? Qwen3-0.6B stays the model that is steered and scored. The writer only proposes actions.

The bar to beat: `programs/row_argmax.txt`, written step by step in the same loop by Claude in 23 actions, gives 10% -> 43% of held-out cells right (6% -> 25% of boards solved). Qwen3-0.6B as the writer reached 5% held-out, worse than no program.

## Memory: pick a writer that fits

This Mac has about 24 GB of memory.

- **Qwen3-8B in bf16 (16 GB) next to Qwen3-0.6B swaps badly.** A step took 12 minutes, then the run stalled. Don't use it as is.
- **Qwen3-4B in bf16 (about 8 GB) should fit.** It is in `config.yaml` as `qwen3-4b` but not downloaded yet; the first run downloads it. This is the easiest option.
- **Qwen3-8B in 4-bit** is downloaded for MLX (`mlx-community/Qwen3-8B-4bit`), but the code can't use it yet. It would need `pip install mlx-lm`, plus a writer class in `write.py` that does constrained decoding with MLX.

## Run

```
python write.py --set write.writer=qwen3-4b --set write.sessions=2 --set write.steps=8 --set write.actions=8 --out runs/write_4b
```

- `write.actions=8` lets the writer run a batch of tests in one reply. Claude needed this to learn the answer words quickly.
- At the end, the best program is checked on the practice and held-out boards. It is saved as `runs/write_4b/best_program.txt`, with the full conversations in `sessions.json`.
- To watch it: `grep "step\|session" ` on the run's output. Each step prints every session's score and one line per action.

## What to look for

1. Does the writer test the answer words (for example `TEST` with `set(answer[k], 3.00)`) before using them?
2. Does it pick board words whose descriptions match what it wants (for example the row word that goes up when a digit is missing)?
3. Does it build an `argmax` over several board words with a different answer word in each branch?
4. Does the program hold up on held-out boards? A practice gain that falls on held-out boards is noise (the 0.6B writer's best did this).

## Known traps

- **Layer order.** A statement can read only words at or before the layer it writes. Board words are at layer 0, so they can always be read; answer words are at layers 16-27.
- **Repetition.** Small writers repeat lines. Adding a line already in the program is refused, and generation has a repetition penalty (`write.repetition_penalty`).
- **Speed.** Each action runs on 96 practice cells, about 10 seconds. With 2 sessions of up to 8 actions, a step's scoring takes about 3 minutes, plus the writer's generation.

## Rules for this project

- No labels and no sudoku or digit priors in finding words. Never add per-digit fixes, such as hunting for a word that pushes toward 3 or 8.
- Hand-written or Claude-written programs are for checking that a good program exists, not the result itself. The result is a model writing the program.

## After that

- If the 4B writer succeeds: run Qwen3-0.6B writing with the same settings, to measure the gap, and try harder boards (`--set boards.blanks=[2,5]`).
- If it fails: run with `write.writer=files` and read its prompts. The writer sees exactly that text, so check whether it has what it needs.
- Not scheduled: a writer program that steers the writer while it writes, as a limited form of self-improvement.
