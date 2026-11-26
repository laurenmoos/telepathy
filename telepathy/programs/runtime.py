"""The llm_program: ordinary Python that runs inside the model's prompt pass.

It gets the text and tokens of the task prompt and two instructions:

    read(name, pos)            the value of read instruction `name` at token position `pos`, at the
                               instruction's layer: 0 is typical, +2 or -2 is unusual
    write(name, pos, amount)   move the state at `pos`, at the instruction's layer, along the
                               instruction's direction, so that its value there goes up by `amount`

The pass runs layer by layer, and so does the program: reads and writes must go in layer order. The
program runs in its own thread and pauses at each read until the pass reaches that layer (MLX runs
only on the main thread). Program(...) is a hook for live.prefill; one Program runs once.
"""
import math, sys, threading

import numpy as np

SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "max", "min",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "ValueError")}


class Program:
    def __init__(self, code, view, first, last, table, layers, timeout=10.0):
        """view: the task prompt as the model reads it (model.live.View); first, last: the first and
        last positions of the problem's own tokens; table: name -> (layer, direction, mean, spread)."""
        self.code, self.view, self.first, self.last = code, view, first, last
        self.table, self.n_layers, self.timeout = table, layers, timeout
        self.T = last + 1
        self.writes, self.trace, self.events = [], [], []    # events: (line, kind, name, pos, value)
        self.waiting = None                  # (layer, pos, name) while waiting for a read
        self.reply = None
        self.go, self.yielded = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.started = False

    # ------------------------------------------------------------ the hook (main thread)

    def __call__(self, L, h):
        import mlx.core as mx
        if not self.started:
            self.started = True
            self.thread.start()
            self._wait()
        while True:
            changed = False
            due = [w for w in self.writes if w[0] <= L]
            self.writes = [w for w in self.writes if w[0] > L]
            adds = []
            for wl, pos, name, amount, line in due:
                if wl < L:
                    self._note(line, f"write at layer {wl} skipped: the pass had passed it (now at layer {L})")
                else:
                    adds.append((pos, name, amount))
            if adds:
                delta = np.zeros(h.shape[1:], dtype=np.float32)
                for pos, name, amount in adds:
                    _, d, _, spread = self.table[name]
                    delta[pos] += amount * spread * d
                h = h + mx.array(delta)[None].astype(h.dtype)
                changed = True
            if self.waiting is not None and self.waiting[0] <= L:
                wl, pos, name = self.waiting
                if wl < L:
                    self._answer(error=f"read at layer {wl}: the pass had passed it (now at layer {L}); "
                                       f"reads and writes must go in layer order")
                else:
                    _, d, mean, spread = self.table[name]
                    x = np.array(h[0, pos].astype(mx.float32))
                    self._answer(value=(float(x @ d) - mean) / spread)
                self._wait()
                changed = True
            if not changed:
                break
        if L == self.n_layers - 1:
            self.finish()
        return h

    def finish(self):
        if self.waiting is not None:
            self._answer(error="the program asked for a layer after the last one")
            self._wait()
        self.thread.join(0.1)

    def _wait(self):
        if not self.yielded.wait(self.timeout):
            self.trace.append(f"stopped: the program ran for more than {self.timeout:.0f} seconds")
            self.waiting = None
            return
        self.yielded.clear()

    def _answer(self, value=None, error=None):
        self.reply, self.waiting = (value, error), None
        self.go.set()

    def _note(self, line, text):
        self.trace.append(f"line {line}: {text}" if line else text)

    # ------------------------------------------------------------ inside the program's thread

    @staticmethod
    def _line():
        f = sys._getframe(2)
        return f.f_lineno if f.f_code.co_filename == "<program>" else None

    def _pos(self, pos):
        if isinstance(pos, bool) or not isinstance(pos, int) or not -self.T <= pos < self.T:
            raise ValueError(f"position {pos!r} is not a token position (0 to {self.T - 1})")
        return pos % self.T

    def _name(self, name):
        if name not in self.table:
            raise ValueError(f"{name!r} is not a read instruction")
        return name

    def layer_of(self, name):
        return self.table[self._name(name)][0]

    def read(self, name, pos):
        line = self._line()
        name, pos = self._name(name), self._pos(pos)
        self.waiting = (self.table[name][0], pos, name)
        self.yielded.set()
        self.go.wait()
        self.go.clear()
        value, error = self.reply
        if error:
            raise ValueError(error)
        self.events.append((line, "read", name, pos, value))
        return value

    def write(self, name, pos, amount):
        line = self._line()
        name, pos = self._name(name), self._pos(pos)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError(f"amount {amount!r} is not a number")
        self.writes.append((self.table[name][0], pos, name, float(amount), line))
        self.events.append((line, "write", name, pos, float(amount)))

    def _main(self):
        v = self.view
        env = dict(__builtins__=SAFE, prompt=v.text, tokens=v.tokens[:self.T], position_of=v.position_of,
                   first=self.first, last=self.last, layers=self.n_layers, instructions=sorted(self.table),
                   layer_of=self.layer_of, read=self.read, write=self.write)
        try:
            exec(compile(self.code, "<program>", "exec"), env)
        except Exception as e:
            tb, line = sys.exc_info()[2], None
            while tb is not None:
                if tb.tb_frame.f_code.co_filename == "<program>":
                    line = tb.tb_lineno
                tb = tb.tb_next
            self._note(line, f"error: {type(e).__name__}: {e}")
            self.events.append((line, "error", None, None, f"{type(e).__name__}: {e}"))
        self.waiting = None
        self.yielded.set()
