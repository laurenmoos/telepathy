"""Running programs inside the model's forward pass.

At each layer, after Qwen's own layer, the program's instructions for that layer run, so later
layers see the writes.
"""
import torch

from . import lang
from ..sudoku.prompt import prompt27


@torch.no_grad()
def run(m, items, batch=16):
    """Run programs inside Qwen's forward pass. items: (grid, r, c, program).
    Returns (digit, trace, scores) per item: trace lists what each instruction did; scores
    are the 9 digit scores. A program that changes nothing gets Qwen's plain answer
    (m.prepare() first). If and argmax are decided per board: an instruction runs only where
    its branch was taken."""
    out = [None] * len(items)
    todo = []
    for j, (g, r, c, prog) in enumerate(items):
        steps, skipped = lang.schedule(prog)
        note = [f"skipped {op} {lang.WORD_TO_NAME.get(w, w)} (its layer had already run)" for op, w in skipped]
        if any(ins[0] in ("store", "clear") for lst in steps.values() for ins in lst):
            todo.append((j, steps, note))
        else:
            plain = m.cache[prompt27(g, r, c)]
            out[j] = (int(plain.argmax()) + 1, note + ["nothing is written, so Qwen is unchanged"], plain)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        enc, pos = m.encode([items[j][:3] for j, _, _ in chunk])
        acc = [0.0] * len(chunk)
        decided = [{} for _ in chunk]                # if / argmax id -> the branch this board took
        notes = [{} for _ in chunk]                  # argmax id -> {candidate: (word, value)}
        recorded = [{} for _ in chunk]               # word -> its value when its layer ran
        trace = [list(note) for _, _, note in chunk]
        layers = sorted({L for _, steps, _ in chunk for L in steps})

        def at_layer(L):
            def hook(mod, inp, o):
                t = (o[0] if isinstance(o, tuple) else o).clone()
                for n, (j, steps, _) in enumerate(chunk):
                    g, r, c, _ = items[j]
                    now = lambda w: m.value(m.word(w), t[n, pos[n][m.word(w)["place"]]], r, c)
                    value = lambda w, ins=None: recorded[n][w] if ins is not None and ins[-1] == "recorded" else now(w)
                    name = lambda w: lang.WORD_TO_NAME.get(w, w)
                    for ins in steps.get(L, []):
                        if any(decided[n].get(d) != b for d, b in ins[2]):
                            continue                                  # in a branch this board did not take
                        op = ins[0]
                        if op == "record":
                            recorded[n][ins[1]] = now(ins[1])
                        elif op == "scale":
                            acc[n] *= ins[1]
                        elif op == "const":
                            acc[n] = ins[1]
                        elif op == "if":
                            decided[n][ins[3]] = 0 if acc[n] > ins[1] else 1
                            trace[n].append(f"if: {acc[n]:+.2f} > {ins[1]:.2f} is "
                                            + ("true" if decided[n][ins[3]] == 0 else "false"))
                        elif op == "note":
                            notes[n].setdefault(ins[3][0], {})[ins[3][1]] = (ins[1], value(ins[1], ins))
                        elif op == "switch":
                            got = notes[n].get(ins[3], {})
                            best = max(got, key=lambda i: got[i][1])
                            decided[n][ins[3]] = best
                            trace[n].append("argmax: " + ", ".join(f"{name(w)} {v:+.2f}" for _, (w, v) in sorted(got.items()))
                                            + f"; best == {best}")
                        elif op == "load":
                            acc[n] = value(ins[1], ins)
                        elif op == "add":
                            acc[n] += value(ins[1], ins)
                        else:                                         # store, clear
                            p = m.word(ins[1])
                            v = value(ins[1])
                            target = acc[n] if op == "store" else 0.0
                            t[n, pos[n][p["place"]]] += ((target - v) * p["sd"] * p["u"].to(t.device, torch.float32)).to(t.dtype)
                            trace[n].append(f"{'set' if op == 'store' else 'clear'} {name(ins[1])}: {v:+.2f} -> {target:+.2f}")
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return hook
        hooks = [m.layers[L].register_forward_hook(at_layer(L)) for L in layers]
        try:
            lg = m.model(**enc, logits_to_keep=1).logits[:, -1][:, m.digit_ids].float().cpu()
        finally:
            for h in hooks:
                h.remove()
        for n, (j, _, _) in enumerate(chunk):
            out[j] = (int(lg[n].argmax()) + 1, trace[n], lg[n])
    return out
