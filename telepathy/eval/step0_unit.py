"""Unit test of one part of step 0: do the J-lens directions steer the model more than random ones?

One problem, the plain prompt, nothing else from the pipeline. The prompt ends with the start of
a reply, "<think>\nOkay, so I need to", because the first reply tokens are always the same and
the next word here is not.

  1. At each layer, read the last token's state through the J-lens: the words the model is
     ready to say. Take the top control.tokens words, and their lens vectors as directions.
  2. Push the state at the last token along each direction, and along as many random directions,
     by the same amount: a fraction of the state's length at that layer.
  3. Measure how far the next-token distribution moves: KL(plain || pushed).

It passes if the lens directions move the distribution clearly more than the random ones.

    telepathy/run.sh step0-unit [--set control.layers=[0,12,24] ...]
"""
import random

import numpy as np

from ..instructions import jlens
from ..setup.log import log
from ..setup.settings import settings, load_live
from .. import tasks

SIZES = (0.1, 0.25, 0.5, 1.0)
PREFIX = "<think>\nOkay, so I need to"   # the reply always starts "Okay"; here the next word is open


def main(argv=None):
    cfg, a = settings(argv)
    c, layers = cfg["control"], cfg["control"]["layers"]
    task = tasks.load(cfg["task"]["name"])
    probs = task.problems(cfg["task"]["split"])
    name = sorted(random.Random(cfg["task"]["seed"]).sample(sorted(probs), c["problems"]))[0]
    live = load_live(cfg)
    view = live.view([{"role": "user", "content": task.prompt(probs[name])[0]}])
    ids = view.ids + live.tok.encode(PREFIX, add_special_tokens=False)
    lt = jlens.LastToken(live, ids, layers)

    def probs_after(L, push):
        """The next-token distribution with push added to the last token's state at layer L."""
        import mlx.core as mx
        x = lt.x[L]
        h = lt._run(lambda: lt._f(L)(x + mx.array(push).reshape(x.shape).astype(x.dtype)))
        z = np.array(live.model.lm_head(h).astype(mx.float32)).reshape(-1)
        z = z - z.max()
        return np.exp(z) / np.exp(z).sum()

    p = probs_after(layers[0], np.zeros(lt.x[layers[0]].size))
    top = np.argsort(-p)[:5]
    log(f"{name}: entropy {-(p * np.log(p + 1e-12)).sum():.2f} nats; plain next token: " + ", ".join(f"{live.tok.decode([t])!r} {p[t]:.2f}" for t in top))

    def kl(q):
        return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())

    rng = np.random.default_rng(0)
    for L in layers:
        x = np.array(lt.x[L].astype(__import__("mlx.core").core.float32)).reshape(-1)
        words = [int(t) for t in np.argsort(-jlens.lens(live, lt.jvp(L, x)))[:c["tokens"]]]
        found = [lt.vjp(L, u) for u in jlens.unembed(live, words)]
        found = [v / np.linalg.norm(v) for v in found]
        rand = [v / np.linalg.norm(v) for v in rng.standard_normal((len(found), x.size))]
        log(f"layer {L}: lens words " + ", ".join(repr(live.tok.decode([int(t)])) for t in words))
        for s in SIZES:
            amount = s * np.linalg.norm(x)
            kf = [kl(probs_after(L, amount * v)) for v in found]
            kr = [kl(probs_after(L, amount * v)) for v in rand]
            log(f"   push {s:4.2f} x length: KL lens median {np.median(kf):7.3f} max {max(kf):7.3f}   "
                f"random median {np.median(kr):7.3f} max {max(kr):7.3f}   ratio {np.median(kf) / max(np.median(kr), 1e-9):6.1f}")


if __name__ == "__main__":
    main()
