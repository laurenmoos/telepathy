# telepathy

<p align="center">
  <img src="splash.jpg" alt="An engraving: a child-faced moon leans on a cloud of cushions above a lake, looking down at her reflection." width="420">
  <br>
  <sub>J. J. Grandville, <i>La lune peinte par elle-même</i>, from <i>Un autre monde</i> (1844)</sub>
</p>

A language model (Qwen3-8B, 4-bit, on MLX) writes a short program for itself. The program runs inside the model while it reads a task prompt: it reads directions in the model's internal state and moves the state along them. Then the model attempts the task, and a judge says whether it passed. The first task is Lean proofs (miniF2F), with Lean as the judge.

The full design is in `docs/lean_loop.md`.

## The loop

`telepathy/loop.py` takes one problem at a time. Nothing is computed over all problems in advance.

0. **Read.** One prompt pass over the plain task prompt, with no program. The states at this problem's tokens give noisy candidate read instructions: at each layer, the directions along which the tokens vary most.
1. **Match.** A candidate close to a cached read instruction adds to that instruction's count. Any other candidate enters the cache as new.
2. **Write.** The model writes an `llm_program` from two instructions, `read` and `write`.
3. **Trace.** The program runs on the prompt pass only, with no generation. The model sees each line that ran, what it read or wrote, and how the read values moved against the plain states. Steps 2 and 3 repeat `write.rounds` times.
4. **Check.** The task's attempt runs with no program and with the program. A program is tried only if it runs: it compiles, raises no error, and makes at least one write. After a failed check, the model edits the program, and the edit is traced and checked again, up to `check.retries` times. The best program is retained.
5. **Cache.** The best program is kept if it passed where the plain attempt failed. Read instructions are kept if they came back on later problems or a kept program used them. The others are discarded.

## The instruction set

- **Read instruction.** One direction in the residual stream at one layer. `read(name, pos)` gives how far the state at token `pos` sits along it, in units of its spread over the problems seen so far: 0 is typical, +2 or -2 is unusual.
- **Write instruction.** `write(name, pos, amount)` moves the state at `pos`, at the instruction's layer, so that the read value there goes up by `amount`. The write happens in the prompt pass, so the KV cache carries it into every token of the reply.

Reads and writes must go in layer order, because the program runs with the pass.

## Layout

```
telepathy/
  run.sh, loop.py, config.yaml   the entry point, the loop, the settings (and every prompt the loop uses)
  model/live.py                  the model: prompt pass with a hook after each layer, generation
  instructions/read.py           step 0: plain states and candidate read instructions
  instructions/cache.py          steps 1 and 5: the cache of read instructions
  programs/runtime.py            the llm_program: read and write inside the prompt pass
  programs/trace.py              step 3: the trace
  programs/write.py              steps 2 and 3: write, trace, write again
  tasks/attempt.py               the task's attempt alone, with no program
  tasks/lean/task.py, judge.py   the Lean task: TASK_PROMPT, the attempt, and Lean as the judge
```

Only `tasks/` knows about the task. A new task is a module with `problems`, `prompt` and `attempt`.

## Run

From the repo root (Lean and miniF2F must be built in `lean/miniF2F`):

```
telepathy/run.sh --set task.sample=20 --out runs/loop      # the loop
telepathy/run.sh attempt --set task.sample=5               # the attempt alone, with no program
```

Settings are in `telepathy/config.yaml`; any of them can be changed with `--set key.sub=value`. Each run saves `results.json` (per problem: the programs, their traces, both attempts) and the cache (`cache.json`, `cache.npz`) in `--out`, and a rerun goes on from there.
