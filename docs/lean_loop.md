# The Lean outer loop

## Instructions

- **`TASK_PROMPT`**: the problem prompt, with the Lean 4 rules and tactic hints first and the problem's statement last (`telepathy/tasks/lean/task.py`).
- **`llm_program`**: the program the model writes for itself. It uses two kinds of instruction: read and write.
- **Read instruction**: one direction in the residual stream at one layer. `read(name, pos)` gives how far the state at token `pos` sits along that direction. It replaces `peek`.
- **Write instruction**: `write(name, pos, amount)` moves the state at `pos`, at the read instruction's layer, so that its value there goes up by `amount`. It replaces `push`.
- **Plain state**: the residual stream state at one token after one layer, when the model reads `TASK_PROMPT` with no `llm_program` running.
- **The cache**: the read instructions kept from earlier problems, each with a count of how often it came back.

## The loop

The loop (`telepathy/loop.py`) is online: it takes one problem at a time, and nothing is computed over all problems in advance. The model is not good at reflecting on itself, so the loop has no step where it names what it lacks.

0. **Read.** Gather noisy plain states from the problem at hand and find candidate read instructions in them.
1. **Match.** Compare each candidate with the cache. A candidate close to a cached read instruction adds to that instruction's count. Any other candidate goes into the cache as new, with a count of 1.
2. **Write.** The model writes an `llm_program` for this problem. It sees the problem's tokens with their positions, and what each shown read instruction does on them.
3. **Trace.** Run the `llm_program` on the prompt pass only, with no generation, and show the model a trace. Steps 2 and 3 repeat `write.rounds` times.
4. **Check.** Run the task's attempt with no program and with the program. The judge says passed or failed.
5. **Cache.** Keep the `llm_program` if it passed where the plain attempt failed. Keep read instructions that come back on later problems or that a kept program used. Discard the others.
6. Go to the next problem, at step 0.

## Step 0: read

Step 0 (`telepathy/instructions/read.py`) looks only at the problem at hand. It uses no labels and no judge.

- **Input.** `TASK_PROMPT` with this problem's statement, in the chat form the attempt uses. No `llm_program` runs and nothing is generated.
- **Tokens.** From the problem's theorem line to the last prompt token. Two parts of the prompt are left out because they are the same on every problem, and the model reads causally, so their states say nothing about this problem: the fixed instructions, and the statement's header (imports, options, open namespaces).
- **Layers.** `read.layers`: 0, 6, 12, 18, 24, 30, 35.
- **Candidates.** At each layer, the 8 directions along which the problem's tokens vary most (PCA). That gives 56 candidates per problem.

The candidates are noisy: they come from one problem, they have no labels, and the model is 4-bit quantized. Step 0 does not try to remove this noise. The cache does, over time: a read instruction that is real comes back on later problems (step 1), and one that is noise does not (step 5).

## Steps 1 and 5: the cache

The cache is in `telepathy/instructions/cache.py`.

- **Match.** A candidate matches a cached instruction at the same layer if their |cosine| is at least `cache.match` (0.5). Each cached instruction matches at most one candidate per problem. On a match, the instruction's direction moves toward the candidate (a running mean, weighted by the count).
- **Values.** Each instruction keeps a running mean and spread of its values over the tokens of every problem since it entered the cache. Read values are in units of that spread.
- **Discard.** An instruction is discarded if it has not come back for `cache.patience` problems (5), unless it came back `cache.persist` times (3) or a kept program used it.

## Step 3: trace

Step 3 (`telepathy/programs/trace.py`) should read like a debugger's trace. It has two parts:

- **Line by line.** Each line of the program that ran, in order, with what it read or wrote: the instruction, the token position and its text, the layer, and the value. A line that ran several times in a row is one entry, with its count, its range of positions and values, and where the value was highest. Errors show at the line that raised them.
- **How the read values moved.** For each read instruction the program used, and each one shown to the model: its value with the program minus its plain value, at the written tokens, over the problem's tokens, and at the last token. A write at layer L shows its exact amount at layer L; later layers show how far it carried.

## Step 4: check

A program is tried only if it runs: it compiles, raises no error, and makes at least one write. Otherwise the attempt with the program would be a plain attempt.

The attempt is self-contained (`attempt` in `telepathy/tasks/lean/task.py`): the model thinks, writes a proof, and repairs it from Lean's errors, up to `attempt.rounds` times. With a program, the program runs in the prompt pass of every round. Its writes are in the KV cache, so every token of the reply sees them.

## Open questions

- One try per side is noise. In the first run, a program that did not compile "passed" while the plain attempt failed. How many tries per side, and what keep rule?
- How close must a candidate be to match (`cache.match`)? At 0.5, 41 of 56 candidates on the second problem matched.
- How many problems before a read instruction that does not come back is discarded?
