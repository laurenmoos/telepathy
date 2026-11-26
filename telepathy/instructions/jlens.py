"""Step 0 by the J-lens (Gurnee et al. 2026): candidate instructions that steer what the model says.

The J-lens at layer L maps a change in the state at layer L to the change in the final state
(after the last norm, which the unembedding reads): J_L = d h_final / d h_L. The lens vector of
token t is J_L^T u_t (u_t: the unembedding row of t). Pushing along it makes the model more ready
to say t.

The paper averages J over positions and a corpus. Here the write is at the last prompt token
only, so J is taken exactly there, on the problem at hand: the earlier tokens are fixed in the KV
cache, and J v (JVP) or J^T u (VJP) is one pass of one token through the later layers.

    forward(live, ids, hook)        the final states [B, T, d] after the last norm
    states(live, ids, layers)       the state at the last token at each layer
    LastToken(live, ids)            .jvp(L, v) = J_L v, .vjp(L, u) = J_L^T u, at the last token
    candidates(live, lt, diffs, k)  for each state difference and layer, the k tokens the
                                    difference most raises in the lens, and their lens vectors

The state differences come from three prompt passes: empty, the task's instructions only, and the
instructions with the problem. Their last tokens are the same text, so their states compare.
"""
import contextlib

import numpy as np


def forward(live, ids, hook=None):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    m = live.model.model
    h = m.embed_tokens(mx.array([ids]) if isinstance(ids, list) else ids)
    mask = create_attention_mask(h, None)
    for L, layer in enumerate(live.layers):
        h = layer(h, mask, None)
        if hook is not None:
            h = hook(L, h)
    return m.norm(h)


def states(live, ids, layers):
    """{layer: the state at the last token [d]}."""
    import mlx.core as mx
    out = {}

    def hook(L, h):
        if L in layers:
            out[L] = np.array(h[0, -1].astype(mx.float32))
        return h
    mx.eval(forward(live, ids, hook))
    return out


@contextlib.contextmanager
def plain_attention(live):
    """MLX's fused attention for one query has no gradient. Inside this, attention is written out."""
    import mlx.core as mx
    mod = __import__(type(live.layers[0]).__module__, fromlist=["x"])
    fused = mod.scaled_dot_product_attention

    def sdpa(q, k, v, cache=None, scale=1.0, mask=None, **kw):
        rep = q.shape[1] // k.shape[1]
        if rep > 1:
            k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        s = (q * scale) @ k.transpose(0, 1, 3, 2)
        return mx.softmax(s.astype(mx.float32), axis=-1).astype(v.dtype) @ v
    mod.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        mod.scaled_dot_product_attention = fused


class LastToken:
    """The exact J-lens at the last prompt token, on the problem at hand."""

    def __init__(self, live, ids, layers):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        self.live, self.n = live, len(ids) - 1
        self.cache = make_prompt_cache(live.model)
        live.prefill(ids[:-1], cache=self.cache)
        st = {}

        def hook(L, h):
            if L in layers:
                st[L] = h[:, -1:]
            return h
        forward(live, ids, hook)
        mx.eval(st, [c.state for c in self.cache])
        self.x = st

    def _f(self, L):
        m = self.live.model.model

        def f(x):
            h = x
            for l in range(L + 1, len(self.live.layers)):
                h = self.live.layers[l](h, None, self.cache[l])
            return m.norm(h)
        return f

    def _run(self, fn):
        import mlx.core as mx
        with plain_attention(self.live):
            out = fn()
            mx.eval(out)
        for c in self.cache:
            if c.offset > self.n:
                c.trim(c.offset - self.n)
        return out

    def jvp(self, L, v):
        import mlx.core as mx
        x = self.x[L]
        _, (t,) = self._run(lambda: mx.jvp(self._f(L), [x], [mx.array(v).reshape(x.shape).astype(x.dtype)]))
        return np.array(t.astype(mx.float32)).reshape(-1)

    def vjp(self, L, u):
        import mlx.core as mx
        x = self.x[L]
        _, (g,) = self._run(lambda: mx.vjp(self._f(L), [x], [mx.array(u).reshape(x.shape).astype(x.dtype)]))
        return np.array(g.astype(mx.float32)).reshape(-1)


def unembed(live, tokens):
    """The unembedding rows of tokens, [len(tokens), d]."""
    import mlx.core as mx
    head = live.model.lm_head
    t = mx.array(list(tokens))
    if hasattr(head, "scales"):
        w = mx.dequantize(head.weight[t], head.scales[t], head.biases[t], head.group_size, head.bits)
    else:
        w = head.weight[t]
    return np.array(w.astype(mx.float32))


def lens(live, y):
    """Logits of a change y in the final state, [vocab]."""
    import mlx.core as mx
    return np.array(live.model.lm_head(mx.array(y[None], dtype=mx.float32)).astype(mx.float32))[0]


def vocabulary(live, ids, n, top, temperature, seed):
    """The tokens a reply uses: the union of the top tokens at each position of one sampled reply
    of n tokens. Without it, the lens favours rare tokens, whose rise moves almost no probability."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(seed)
    cache = make_prompt_cache(live.model)
    logits = live.model(mx.array([ids]), cache=cache)[:, -1]
    sample, vocab = make_sampler(temp=temperature, top_p=0.95, top_k=20), set()
    for _ in range(n):
        vocab |= set(np.argsort(-np.array(logits.astype(mx.float32))[0])[:top].tolist())
        y = sample(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item()
        logits = live.model(mx.array([[y]]), cache=cache)[:, -1]
    return np.array(sorted(vocab))


def candidates(live, lt, diffs, k, vocab=None):
    """[dict(layer, source, token, text, direction)]: for each difference (source -> {layer: vector})
    and layer, the k tokens (from vocab, if given) whose lens logits the difference raises most,
    with unit lens vectors."""
    out, seen = [], set()
    for source, by_layer in diffs.items():
        for L, dx in sorted(by_layer.items()):
            logits = lens(live, lt.jvp(L, dx))
            pool = np.arange(len(logits)) if vocab is None else vocab
            top = [int(t) for t in pool[np.argsort(-logits[pool])[:k]]]
            for t, u in zip(top, unembed(live, top)):
                if (L, t) in seen:
                    continue
                seen.add((L, t))
                v = lt.vjp(L, u)
                out.append(dict(layer=L, source=source, token=t, text=live.tok.decode([t]),
                                direction=v / np.linalg.norm(v)))
    return out
