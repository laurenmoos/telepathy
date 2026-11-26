"""Step 3, trace: run an llm_program on the prompt pass only (no generation) and show what it did.

The trace reads like a debugger's: each line of the program that ran, in order, with what it read
or wrote; then how the writes moved each read instruction's values, against the plain states.


    trace(live, code, view, first, last, cache, names, plain, plain_lp)  (text, program events)
"""
from ..instructions.read import gather
from .runtime import Program


def trace(live, code, view, first, last, cache, names, plain, plain_lp, show_lines=30):
    positions = list(range(first, last + 1))
    table = cache.table(cache.ranked())
    prog = Program(code, view, first, last, table, len(live.layers))
    states, _ = gather(live, view.ids[:last + 1], sorted(plain), positions, prog)
    prog.finish()
    src = code.splitlines()
    tok = lambda p: repr(view.tokens[p])
    out = ["The program, line by line (a line that ran several times in a row is one entry):"]
    groups = []
    for e in prog.events:
        if groups and groups[-1][0][:3] == e[:3] and e[1] != "error":
            groups[-1].append(e)
        else:
            groups.append([e])
    for g in groups[:show_lines]:
        line, kind, name = g[0][:3]
        text = src[line - 1].strip() if line and line - 1 < len(src) else ""
        if kind == "error":
            what = g[0][4]
        elif len(g) == 1:
            pos, value = g[0][3], g[0][4]
            what = f"{kind} {name} at {pos} {tok(pos)}, layer {table[name][0]}: {value:+.2f}"
        else:
            ps, vs = [e[3] for e in g], [e[4] for e in g]
            hi = g[max(range(len(g)), key=lambda j: vs[j])]
            what = (f"{kind} {name} {len(g)} times, at {min(ps)} to {max(ps)}, layer {table[name][0]}: "
                    f"{min(vs):+.2f} to {max(vs):+.2f} (highest at {hi[3]} {tok(hi[3])})")
        out.append(f"  line {line}  {text[:48]:<48}  {what}")
    if len(groups) > show_lines:
        out.append(f"  ... and {len(groups) - show_lines} more entries")
    if not prog.events:
        out.append("  the program made no read or write")
    out += [f"  {t}" for t in prog.trace if "error" not in t]

    written = sorted({pos for _, kind, _, pos, _ in prog.events if kind == "write"})
    shown = list(dict.fromkeys([e[2] for e in prog.events if e[1] in ("read", "write")] + list(names)))
    out += ["How the read values moved, against the plain states (program minus plain):",
            f"  {'instruction':<12} {'layer':>5}  {'at written tokens':>17}  {'over the problem':>16}  {'at last':>7}"]
    for n in shown:
        L = cache.items[n]["layer"]
        if L not in states:
            continue
        d = cache.values(n, states[L]) - cache.values(n, plain[L])
        at_w = f"{d[[p - first for p in written if p >= first]].mean():+.2f}" if any(p >= first for p in written) else "-"
        out.append(f"  {n:<12} {L:>5}  {at_w:>17}  {d.mean():>+16.2f}  {d[-1]:>+7.2f}")

    return "\n".join(out), prog.events
