"""Programs that run live inside one forward pass of an MLX model, using basic tools that need no
learned words. A program is ordinary Python. It gets the prompt text, finds places in it with
ordinary string code, and calls:

    peek(pos, layer, token)          how strongly the model leans toward `token` at token position
                                     `pos` after layer `layer`: the log-probability of `token` when
                                     the state there is read through the model's own output layer
                                     (the "logit lens")
    push(pos, layer, token, amount)  add to the state at `pos` after `layer` the model's own output
                                     direction for `token`, scaled to `amount` times the state's size

It also gets: prompt (the text), tokens (the text of each token), token_at(i) (the token position
holding character i of the prompt), last (the last token position) and layers (how many layers).

The forward pass runs layer by layer. After each layer, the pushes for that layer are applied and
the peeks for that layer are answered, until every program waits for a later layer or is done. So
peeks and pushes must go in layer order: one for a layer already passed is an error.

MLX runs only on the main thread. Each cell's program runs in its own thread, and pauses at
each peek until the main thread answers it.
"""
import math, threading

import numpy as np

SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "max", "min",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "ValueError", "print")}


class Live:
    def __init__(self, path):
        from mlx_lm import load
        self.model, self.wrap = load(path)
        self.tok = self.wrap._tokenizer
        self.layers = self.model.model.layers
        self._dirs = {}

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
        """The unit output direction of a token: its row of the output layer."""
        import mlx.core as mx
        if tid not in self._dirs:
            head = self.model.lm_head
            if hasattr(head, "scales"):
                w = mx.dequantize(head.weight[tid:tid + 1], head.scales[tid:tid + 1],
                                  head.biases[tid:tid + 1] if head.biases is not None else None,
                                  head.group_size, head.bits)[0]
            else:
                w = head.weight[tid]
            w = w.astype(mx.float32)
            self._dirs[tid] = w / mx.linalg.norm(w)
        return self._dirs[tid]

    # ------------------------------------------------------------ running programs

    def run(self, prompts, code, candidates=None, timeout=10.0):
        """Run `code` once per prompt inside the forward pass. Prompts of one call are run as one batch,
        so they must have the same number of tokens. Returns per prompt (log-probabilities of the
        candidate tokens at the last position, or of the whole vocabulary, as numpy; trace lines)."""
        import mlx.core as mx
        enc = [self.tok(p, return_offsets_mapping=True, add_special_tokens=False) for p in prompts]
        n_tok = {len(e["input_ids"]) for e in enc}
        if len(n_tok) != 1:
            raise ValueError("prompts in one run must have the same number of tokens")
        T = n_tok.pop()
        workers = [_Worker(self, p, e, code, T, timeout) for p, e in zip(prompts, enc)]
        for w in workers:
            w.start()
        for w in workers:
            w.wait_yield()

        h = self.model.model.embed_tokens(mx.array([e["input_ids"] for e in enc]))
        mask = "causal"
        for L, layer in enumerate(self.layers):
            h = layer(h, mask, None)
            while True:
                changed = False
                adds = []
                for n, w in enumerate(workers):
                    for (pl, pos, tid, amount) in w.take_pushes():
                        if pl < L:
                            w.trace.append(f"push at layer {pl} ignored: the model had passed it")
                            continue
                        if pl > L:
                            w.later.append((pl, pos, tid, amount))
                            continue
                        adds.append((n, pos, tid, amount))
                    w.pushes, w.later = w.later, []
                if adds:
                    mx.eval(h)
                    delta = np.zeros(h.shape, dtype=np.float32)
                    for n, pos, tid, amount in adds:
                        size = float(mx.linalg.norm(h[n, pos].astype(mx.float32)))
                        delta[n, pos] += amount * size * np.array(self._direction(tid))
                    h = h + mx.array(delta).astype(h.dtype)
                    changed = True
                asking = [(n, w) for n, w in enumerate(workers) if w.waiting is not None and w.waiting[0] <= L]
                if asking:
                    rows = mx.stack([h[n, w.waiting[1]] for n, w in asking])
                    lp = np.array(self._lens(rows))
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
                    break
        logits = self._lens(h[:, -1])
        lp = np.array(logits)
        out = []
        for n, w in enumerate(workers):
            if w.waiting is not None:
                w.answer(error="the program asked for a layer after the last one")
                w.wait_yield()
            w.finish()
            row = lp[n][candidates] if candidates is not None else lp[n]
            out.append((row, w.trace))
        return out


class _Worker:
    """One cell's program, in its own thread."""

    def __init__(self, live, prompt, enc, code, T, timeout):
        self.live, self.code, self.timeout = live, code, timeout
        self.offsets = enc["offset_mapping"]
        self.tokens = [prompt[s:e] for s, e in self.offsets]
        self.prompt, self.T = prompt, T
        self.pushes, self.later, self.trace = [], [], []
        self.waiting = None                  # (layer, pos, token id) while waiting for a peek
        self.reply = None
        self.go, self.yielded = threading.Event(), threading.Event()
        self.done = False
        self.thread = threading.Thread(target=self._main, daemon=True)

    def start(self):
        self.thread.start()

    def wait_yield(self):
        if not self.yielded.wait(self.timeout):
            self.trace.append(f"stopped: the program ran for more than {self.timeout:.0f} seconds")
            self.done, self.waiting = True, None
            return
        self.yielded.clear()

    def take_pushes(self):
        return list(self.pushes)

    def answer(self, value=None, error=None):
        self.reply = (value, error)
        self.waiting = None
        self.go.set()

    def finish(self):
        self.thread.join(0.1)

    # ---- inside the program's thread

    def _pos(self, pos):
        if not isinstance(pos, int) or not -self.T <= pos < self.T:
            raise ValueError(f"position {pos!r} is not a token position (0 to {self.T - 1})")
        return pos % self.T

    def _layer(self, layer):
        n = len(self.live.layers)
        if not isinstance(layer, int) or not 0 <= layer < n:
            raise ValueError(f"layer {layer!r} is not a layer (0 to {n - 1})")
        return layer

    def peek(self, pos, layer, token):
        pos, layer, tid = self._pos(pos), self._layer(layer), self.live.token_id(token)
        self.waiting = (layer, pos, tid)
        self.yielded.set()
        self.go.wait()
        self.go.clear()
        value, error = self.reply
        if error:
            raise ValueError(error)
        self.trace.append(f"peek({pos}, {layer}, {token!r}) = {value:.2f}")
        return value

    def push(self, pos, layer, token, amount):
        pos, layer, tid = self._pos(pos), self._layer(layer), self.live.token_id(token)
        if not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError(f"amount {amount!r} is not a number")
        self.pushes.append((layer, pos, tid, float(amount)))
        self.trace.append(f"push({pos}, {layer}, {token!r}, {amount:.2f})")

    def token_at(self, i):
        for t, (s, e) in enumerate(self.offsets):
            if s <= i < e:
                return t
        raise ValueError(f"character {i!r} is not in the prompt")

    def _main(self):
        env = dict(__builtins__=SAFE, prompt=self.prompt, tokens=self.tokens, token_at=self.token_at,
                   last=self.T - 1, layers=len(self.live.layers), peek=self.peek, push=self.push)
        try:
            exec(compile(self.code, "<program>", "exec"), env)
        except Exception as e:
            self.trace.append(f"error: {type(e).__name__}: {e}")
        self.done = True
        self.waiting = None
        self.yielded.set()
