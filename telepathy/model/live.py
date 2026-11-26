"""One MLX model that writes programs and runs them inside its own forward pass.

A program is ordinary Python. It gets the prompt text, finds places in it with ordinary string
code, and calls two tools that need nothing learned, only the model's own output layer:

    peek(pos, layer, token)          how strongly the model leans toward `token` at token position
                                     `pos` after layer `layer`: the log-probability of `token` when
                                     the state there is read through the output layer
    push(pos, layer, token, amount)  add to the state at `pos` after `layer` the output direction of
                                     `token`, scaled to `amount` times the size of the state there

It also gets: prompt (the text), tokens (the text of each token), position_of(i) (the token
position holding character i of the prompt), last (the last token position) and layers.

The forward pass runs layer by layer. After each layer, the pushes for that layer are applied and
the peeks for that layer are answered, until every program waits for a later layer or is done.
Peeks and pushes must go in layer order. MLX runs only on the main thread; each prompt's program
runs in its own thread and pauses at each peek.

Speed: run(..., key=k) saves the plain states of those prompts at a few layers (keep), and later
runs with the same key start from the last saved layer before the program's first peek or push.
"""
import math, sys, threading

import numpy as np

SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "max", "min",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "ValueError")}


class Live:
    def __init__(self, path, keep=(0, 6, 12, 18, 24, 30)):
        from mlx_lm import load
        self.model, self.wrap = load(path)
        self.tok = self.wrap._tokenizer
        self.layers = self.model.model.layers
        self.keep = sorted(keep)
        self._dirs, self._saved = {}, {}

    # ------------------------------------------------------------ writing

    def generate(self, prompt, max_tokens=700, temperature=0.7, think=False):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        ids = self.wrap.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            enable_thinking=think)
        return generate(self.model, self.wrap, ids, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temperature, top_p=0.95, top_k=20))

    # ------------------------------------------------------------ the model's own read and write

    def token_id(self, s):
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not one token")
        return ids[0]

    def _lens(self, h):
        """Log-probabilities over the vocabulary for states h [n, d]."""
        import mlx.core as mx
        logits = self.model.lm_head(self.model.model.norm(h)).astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _direction(self, tid):
        """The unit output direction of a token: its row of the output layer, as numpy."""
        import mlx.core as mx
        if tid not in self._dirs:
            head = self.model.lm_head
            if hasattr(head, "scales"):
                w = mx.dequantize(head.weight[tid:tid + 1], head.scales[tid:tid + 1],
                                  head.biases[tid:tid + 1] if head.biases is not None else None,
                                  head.group_size, head.bits)[0]
            else:
                w = head.weight[tid]
            w = np.array(w.astype(mx.float32))
            self._dirs[tid] = w / np.linalg.norm(w)
        return self._dirs[tid]

    # ------------------------------------------------------------ running programs

    def run(self, prompts, code, candidates=None, key=None, batch=16, timeout=10.0):
        """Run `code` once per prompt inside the forward pass. Returns per prompt (log-probabilities
        at the last position, of the candidate token ids or of the whole vocabulary, as numpy;
        trace lines; events (source line, "peek" or "push", value))."""
        out = []
        for i in range(0, len(prompts), batch):
            out += self._run(prompts[i:i + batch], code, candidates, None if key is None else (key, i), timeout)
        return out

    def _run(self, prompts, code, candidates, key, timeout):
        import mlx.core as mx
        enc = [self.tok(p, return_offsets_mapping=True, add_special_tokens=False) for p in prompts]
        if len({len(e["input_ids"]) for e in enc}) != 1:
            raise ValueError("prompts in one batch must have the same number of tokens")
        T = len(enc[0]["input_ids"])
        if key is not None and key not in self._saved:            # save the plain states once
            self._saved[key] = {}
            self._pass([_Worker(self, p, e, "", T, timeout) for p, e in zip(prompts, enc)], enc, key, None, save=True)
        workers = [_Worker(self, p, e, code, T, timeout) for p, e in zip(prompts, enc)]
        return self._pass(workers, enc, key, candidates)

    def _pass(self, workers, enc, key, candidates, save=False):
        import mlx.core as mx
        for w in workers:
            w.start()
        for w in workers:
            w.wait_yield()
        first = min([w.waiting[0] for w in workers if w.waiting is not None] +
                    [p[0] for w in workers for p in w.pushes] + [len(self.layers)])
        saved = self._saved.get(key, {}) if key is not None else {}
        start = max([L for L in saved if L <= first], default=None)
        if start is None:                    # from the start
            h, todo = self.model.model.embed_tokens(mx.array([e["input_ids"] for e in enc])), range(len(self.layers))
        else:                                # from the plain state saved after layer `start`
            h, todo = self._serve(workers, saved[start], start), range(start + 1, len(self.layers))
        for L in todo:
            h = self.layers[L](h, "causal", None)
            if save and L in self.keep:
                mx.eval(h)
                self._saved[key][L] = h
            h = self._serve(workers, h, L)
        lp = np.array(self._lens(h[:, -1]))
        out = []
        for n, w in enumerate(workers):
            if w.waiting is not None:
                w.answer(error="the program asked for a layer after the last one")
                w.wait_yield()
            w.finish()
            out.append((lp[n][candidates] if candidates is not None else lp[n], w.trace, w.events))
        return out

    def _serve(self, workers, h, L):
        """After layer L: apply the pushes for L and answer the peeks for L, until nothing changes."""
        import mlx.core as mx
        while True:
            changed, adds = False, []
            for n, w in enumerate(workers):
                keep = []
                for (pl, pos, tid, amount, line) in w.pushes:
                    if pl < L:
                        w.note(line, f"push at layer {pl} ignored: the model had passed it (now at layer {L})")
                    elif pl > L:
                        keep.append((pl, pos, tid, amount, line))
                    else:
                        adds.append((n, pos, tid, amount))
                w.pushes = keep
            if adds:
                mx.eval(h)
                delta = np.zeros(h.shape, dtype=np.float32)
                for n, pos, tid, amount in adds:
                    size = float(mx.linalg.norm(h[n, pos].astype(mx.float32)))
                    delta[n, pos] += amount * size * self._direction(tid)
                h = h + mx.array(delta).astype(h.dtype)
                changed = True
            asking = [(n, w) for n, w in enumerate(workers) if w.waiting is not None and w.waiting[0] <= L]
            if asking:
                lp = np.array(self._lens(mx.stack([h[n, w.waiting[1]] for n, w in asking])))
                for (n, w), row in zip(asking, lp):
                    pl, pos, tid = w.waiting
                    if pl < L:
                        w.answer(error=f"peek at layer {pl}: the model had passed it (now at layer {L}); "
                                       f"peeks and pushes must go in layer order")
                    else:
                        w.answer(value=float(row[tid]))
                for _, w in asking:
                    w.wait_yield()
                changed = True
            if not changed:
                return h


class _Worker:
    """One prompt's program, in its own thread."""

    def __init__(self, live, prompt, enc, code, T, timeout):
        self.live, self.code, self.timeout = live, code, timeout
        self.offsets = enc["offset_mapping"]
        self.tokens = [prompt[s:e] for s, e in self.offsets]
        self.prompt, self.T = prompt, T
        self.pushes, self.trace, self.events = [], [], []
        self.waiting = None                  # (layer, pos, token id) while waiting for a peek
        self.reply = None
        self.go, self.yielded = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._main, daemon=True)

    def start(self):
        self.thread.start()

    def wait_yield(self):
        if not self.yielded.wait(self.timeout):
            self.trace.append(f"stopped: the program ran for more than {self.timeout:.0f} seconds")
            self.waiting = None
            return
        self.yielded.clear()

    def answer(self, value=None, error=None):
        self.reply, self.waiting = (value, error), None
        self.go.set()

    def finish(self):
        self.thread.join(0.1)

    def note(self, line, text):
        self.trace.append(f"line {line}: {text}" if line else text)

    # ---- inside the program's thread

    @staticmethod
    def _line():
        f = sys._getframe(2)
        return f.f_lineno if f.f_code.co_filename == "<program>" else None

    def _pos(self, pos):
        if isinstance(pos, bool) or not isinstance(pos, int) or not -self.T <= pos < self.T:
            raise ValueError(f"position {pos!r} is not a token position (0 to {self.T - 1})")
        return pos % self.T

    def _layer(self, layer):
        n = len(self.live.layers)
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < n:
            raise ValueError(f"layer {layer!r} is not a layer (0 to {n - 1})")
        return layer

    def peek(self, pos, layer, token):
        line = self._line()
        pos, layer, tid = self._pos(pos), self._layer(layer), self.live.token_id(str(token))
        self.waiting = (layer, pos, tid)
        self.yielded.set()
        self.go.wait()
        self.go.clear()
        value, error = self.reply
        if error:
            raise ValueError(error)
        self.note(line, f"peek({pos}, {layer}, {token!r}) = {value:.2f}")
        self.events.append((line, "peek", value))
        return value

    def push(self, pos, layer, token, amount):
        line = self._line()
        pos, layer, tid = self._pos(pos), self._layer(layer), self.live.token_id(str(token))
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError(f"amount {amount!r} is not a number")
        self.pushes.append((layer, pos, tid, float(amount), line))
        self.note(line, f"push({pos}, {layer}, {token!r}, {amount:.2f})")
        self.events.append((line, "push", float(amount)))

    def position_of(self, i):
        for t, (s, e) in enumerate(self.offsets):
            if s <= i < e:
                return t
        raise ValueError(f"character {i!r} is not in the prompt")

    def _main(self):
        env = dict(__builtins__=SAFE, prompt=self.prompt, tokens=self.tokens, position_of=self.position_of,
                   last=self.T - 1, layers=len(self.live.layers), peek=self.peek, push=self.push)
        try:
            exec(compile(self.code, "<program>", "exec"), env)
        except Exception as e:
            tb = sys.exc_info()[2]
            line = None
            while tb is not None:
                if tb.tb_frame.f_code.co_filename == "<program>":
                    line = tb.tb_lineno
                tb = tb.tb_next
            self.note(line, f"error: {type(e).__name__}: {e}")
            self.events.append((line, "error", f"{type(e).__name__}: {e}"))
        self.waiting = None
        self.yielded.set()
