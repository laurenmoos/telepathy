# Plan: the word set assembles itself

Today the words are fixed before any program is written. This plan lets the word set grow and shrink by
search, and later by the writer itself.

## Where words come from now

| Kind | Method | Places | Layers | Count |
|---|---|---|---|---|
| Board words (`words/board.py`) | independent components (ICA) of the state, "what varies" | row, column and box line ends | 0 only | 96 |
| Answer words (`words/answer.py`) | directions that move the answer most, by empowerment, "what acts" | all 4 places searched | all 28 searched | 8, all chosen at the answer position, layers 16-27 |

The answer-word method already searches every place and layer (112 place-layer pairs, 2,688
candidates), but keeps only the global top 8. The board-word method looks at layer 0 only. Programs
can already read and write any word; the limit is which words exist.

## Phase 1: one word table, both methods, every place, a grid of layers

1. A word is (place, layer, direction, method). The two methods stay: "varies" (ICA) and "acts"
   (empowerment). They are two ways to find directions, not two kinds of word.
2. Run both methods at all 4 places (row, column and box line ends, and the answer position) on a
   grid of layers, for example 0, 4, 8, ..., 24, 27.
3. Keep a few words per (place, layer, method) cell instead of a global top 8, so every cell has
   words to offer.
4. Name words by where they sit, for example `row_end.L8.acts[2]` or `answer.L20.varies[0]`. The
   writer can read and write any of them, subject to the layer order.
5. One entry point builds the whole table (`telepathy/run.sh words`); the cache holds the full
   table, and each run picks its active set from it.

Check: `row_argmax` still works when its words are taken from the full table; the full table loads in
seconds from the cache.

## Phase 2: a search that adds and prunes words

The active set is a small subset of the table. Two tools change it:

    add_words(place, layer, method, k)   add k words from that cell of the table
    drop_words(names)                    remove words from the active set

**2a. The harness runs the search (no writer).** A loop proposes a change to the active set, scores
the new set, and keeps the change if the score goes up. The score of a set is the best program a fast
search can build from it on the practice cells, checked on held-out boards. The fast search is the
table search: every write is measured alone once per cell, and argmax-then-write programs are scored
by lookup, without running the model.

**2b. The writer calls the tools.** The same two tools are given to the writer. It decides which
cells of the table to draw words from, based on what its experiments showed it was missing.
Compared with 2a, this shows whether the writer chooses words better than blind search does.

## Later: new words, not only words from the table

A tool that extracts new words on demand, JEPA style: fit a predictor from the states at one place and
layer to the states at another, with no labels, and add the source directions that predict best.
This is how the writer would extend its own instruction set, not only choose from a fixed table.

## Open questions

- Which layers in the grid? Every layer is 28 x 4 places x 2 methods; a grid of 8 layers keeps the
  table small enough to show the writer.
- Words per cell: 2 to 4?
- The 2a score uses the right answers on practice cells (it is the verifier). The words themselves
  are still found with no labels.
