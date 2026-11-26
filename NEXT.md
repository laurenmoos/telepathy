# Next

Read `README.md` first, then `docs/lean_loop.md`.

## Where it stands

The loop runs end to end on one problem (about 2.5 minutes with one attempt round). Two things showed up in the first runs:

- **One try per side is noise.** In one run the program did not compile, so the attempt "with the program" was a plain attempt. It passed while the plain attempt failed. A program is now tried only if it runs, but a single pass against a single fail still says little. The check needs several tries per side.
- **The model writes poor programs.** It used token texts as instruction names. After an example was added, it read layer 0 after layer 35, which is an error. The trace shows these errors, and the program is not tried.

- **An edit can be the same program.** In the first run with retries, the model's edit was identical to the failed program, and it passed. The plain attempt passed too, after failing in two earlier runs.

## To do

1. Several tries per side in the check (step 4), and a keep rule that compares pass counts.
2. Tune `cache.match`. At 0.5, 41 of 56 candidates on the second problem matched the cache, so the top directions may be mostly shared across problems.
3. Refuse an edit that is the same program as one already tried, and ask again.
4. A run over 20 problems, to see whether read instructions come back and whether any program is kept.
