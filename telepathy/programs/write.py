"""Steps 2 and 3, write and trace: the model writes an llm_program and sees its trace, in one
conversation that goes on. Every text the model sees is in config.yaml under prompts, and none of it
is about the task.

    w = Writer(live, cfg, prompts, view, first, last, cache, names, plain, kept)
    w.round()          the first program, with its trace
    w.round(note)      a new program after `note`, a message to the model (prompts.retrace after a
                       trace, prompts.retry after a failed check)
    -> dict(code, trace, writes, errors, used)

A program runs if it compiles, raises no error, and makes at least one write (runs(round)).
"""
import re

import numpy as np

from .trace import trace


def describe(view, first, last, cache, names, plain):
    """The problem's tokens with their positions, and what each shown read instruction does on them."""
    toks = " ".join(f"{p}:{view.tokens[p]!r}" for p in range(first, last + 1))
    rows = []
    for n in names:
        it = cache.items[n]
        v = cache.values(n, plain[it["layer"]])
        hi, lo = np.argsort(-v)[:3], np.argsort(v)[:2]
        at = lambda js: ", ".join(f"{first + j} {view.tokens[first + j]!r} {v[j]:+.1f}" for j in js)
        rows.append(f"  {n:<10} layer {it['layer']:>2}, came back on {it['count']} problem(s); "
                    f"highest at {at(hi)}; lowest at {at(lo)}")
    return toks, "\n".join(rows)


def runs(r):
    return r["writes"] > 0 and not r["errors"]


def code_of(reply):
    m = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    return (m[-1] if m else reply).strip()


class Writer:
    def __init__(self, live, cfg, prompts, view, first, last, cache, names, plain, kept):
        self.live, self.cfg, self.view, self.first, self.last = live, cfg, view, first, last
        self.cache, self.names, self.plain = cache, names, plain
        toks, rows = describe(view, first, last, cache, names, plain)
        before = prompts["kept"].format(code=kept[-1]) if kept else ""
        self.messages = [{"role": "user", "content": prompts["write"].format(
            tokens=toks, first=first, last=last, layers=len(live.layers), instructions=rows, kept=before,
            a=names[0], b=names[-1])}]

    def round(self, note=None):
        if note is not None:
            self.messages.append({"role": "user", "content": note})
        c = self.cfg
        reply = self.live.generate(self.messages, c["max_tokens"], c["temperature"], c["think"],
                                   c.get("think_tokens")).split("</think>")[-1]
        self.messages.append({"role": "assistant", "content": reply.strip()})
        code = code_of(reply)
        try:
            compile(code, "<program>", "exec")
            text, events = trace(self.live, code, self.view, self.first, self.last, self.cache, self.names,
                                 self.plain)
        except SyntaxError as e:
            text, events = f"It does not run: line {e.lineno}: {e.msg}", [(e.lineno, "error", None, None, e.msg)]
        return dict(code=code, trace=text, writes=sum(e[1] == "write" for e in events),
                    errors=[e[4] for e in events if e[1] == "error"],
                    used=sorted({e[2] for e in events if e[1] in ("read", "write")}))
