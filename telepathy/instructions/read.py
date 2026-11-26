"""Step 0, read: the plain states of the problem at hand, and candidate read instructions in them.

    record(layers, positions)   a hook that saves the residual stream at those layers and positions
    gather(live, ids, ...)      the plain states: one prompt pass with no program, as {layer: [n, d]}
    candidates(states, k)       at each layer, the k directions along which the tokens vary most

Only this problem's tokens are used (from its first token to the last prompt token). The model
reads causally, so the tokens before them are the same on every problem and say nothing about it.
The candidates are noisy: they come from one problem. The cache (cache.py) keeps the ones that come
back on later problems.
"""
import numpy as np


class Record:
    def __init__(self, layers, positions, inner=None):
        self.layers, self.positions, self.inner = set(layers), list(positions), inner
        self.states = {}

    def __call__(self, L, h):
        import mlx.core as mx
        if self.inner is not None:
            h = self.inner(L, h)
        if L in self.layers:
            self.states[L] = np.array(h[0, mx.array(self.positions)].astype(mx.float32))
        return h


def gather(live, ids, layers, positions, program=None):
    """The states at `layers` and `positions` after one prompt pass, and the next-token
    log-probabilities. With `program` (a hook), the program runs in the same pass."""
    rec = Record(layers, positions, program)
    lp = live.logprobs(live.prefill(ids, rec))
    return rec.states, lp


def candidates(states, k):
    """[(layer, unit direction)]: at each layer, the top k principal directions of the tokens."""
    out = []
    for L, x in sorted(states.items()):
        x = x - x.mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        out += [(L, vt[j] / np.linalg.norm(vt[j])) for j in range(min(k, len(vt)))]
    return out
