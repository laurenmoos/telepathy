"""The control experiment for step 0: how much diverse control over the reply its candidates give.

The parts are checked one at a time, and each round reruns only the part being changed.

  The metric, fixed per problem and saved in <out>/metric/<problem>.npz on the first run:
    - a plain reply of control.reply tokens, sampled once, which the pushed model reads
      teacher-forced
    - the push size at each layer, set so that random directions give |e|^2 = control.target on
      average: a layer cannot score only by pushing harder
    - a pool of random directions at each layer, with their effects. The random floor takes as
      many of them at each layer as there are candidates there.
  A push is added at the last prompt token, at its layer. Its effect vector is 2(sqrt q - sqrt p)
  over the reply positions, where p and q are the plain and pushed next-token distributions over
  the plain top control.top tokens plus one bucket for the rest. |e|^2 is 8 times the squared
  Hellinger distance, which is the Fisher metric for small pushes. The score of a set of pushes
  is their empowerment together: 0.5 log det(I + E E^T / noise). Pushes that do the same thing
  add little.

  The lens check: each candidate has a token, and the lens says its direction makes the model
  more likely to say that token. The check pushes along the direction and sees whether the
  chance of that token as the first reply token goes up.

  Step 0 (timed), the part being changed: the candidates.

    telepathy/run.sh control [--set control.problems=3 ...] [--out runs/control]
"""
import json, os, random, time

import numpy as np

from ..instructions import jlens
from ..setup.log import log
from ..setup.settings import settings, load_live
from .. import tasks


def reply(live, ids, n, temperature, seed):
    """n sampled reply tokens after the prompt ids."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(seed)
    cache = make_prompt_cache(live.model)
    logits = live.model(mx.array([ids]), cache=cache)[:, -1]
    sample, out = make_sampler(temp=temperature, top_p=0.95, top_k=20), []
    for _ in range(n):
        y = sample(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item()
        out.append(y)
        logits = live.model(mx.array([[y]]), cache=cache)[:, -1]
    return out


class Effects:
    """Effect vectors of pushes at the last prompt token, read over a fixed plain reply. The tokens
    before the last prompt token are the same for every push, so they stay in a KV cache, and a
    push runs only the last prompt token and the reply."""

    def __init__(self, live, prompt_ids, reply_ids, top):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        self.live, self.R = live, len(reply_ids)
        self.cache = make_prompt_cache(live.model)
        live.prefill(prompt_ids[:-1], cache=self.cache)
        mx.eval([c.state for c in self.cache])
        self.n = len(prompt_ids) - 1
        self.ids = mx.array([prompt_ids[-1:] + reply_ids])
        self.d = live.model.model.embed_tokens(self.ids[:, :1]).shape[-1]
        p = self._probs(None)
        self.top = np.argsort(-p, axis=1)[:, :top]
        self.p, self.p0 = self._bucket(p), p[0]

    def _probs(self, push):
        """[R, vocab]: the next-token distributions over the reply, with push = (layer, vector)."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        m = self.live.model.model
        h = m.embed_tokens(self.ids)
        mask = create_attention_mask(h, self.cache[0])
        for L, layer in enumerate(self.live.layers):
            h = layer(h, mask, self.cache[L])
            if push is not None and push[0] == L:
                delta = np.zeros(h.shape[1:], dtype=np.float32)
                delta[0] = push[1]
                h = h + mx.array(delta)[None].astype(h.dtype)
        logits = self.live.model.lm_head(m.norm(h[:, :self.R])).astype(mx.float32)
        p = np.array(mx.softmax(logits, axis=-1))[0]
        for c in self.cache:
            c.trim(c.offset - self.n)
        return p

    def _bucket(self, p):
        q = np.take_along_axis(p, self.top, axis=-1)
        rest = np.clip(1 - q.sum(-1, keepdims=True), 0, 1)
        return np.concatenate([q, rest], -1)

    def __call__(self, pushes, tokens=None):
        """pushes: [(layer, vector [d])]. Returns E [len(pushes), R * (top + 1)], and with tokens
        (one per push) also the rise in log p(token) at the first reply position."""
        E, rise = [], []
        for i, push in enumerate(pushes):
            full = self._probs(push)
            if tokens is not None:
                t = tokens[i]
                rise.append(np.log(full[0, t] + 1e-12) - np.log(self.p0[t] + 1e-12))
            E.append((2 * (np.sqrt(self._bucket(full)) - np.sqrt(self.p))).reshape(-1))
        E = np.array(E)
        return (E, np.array(rise)) if tokens is not None else E


def empowerment(E, noise):
    if len(E) == 0:
        return 0.0
    return 0.5 * float(np.linalg.slogdet(np.eye(len(E)) + E @ E.T / noise)[1])


def calibrate(eff, layers, norms, c, rounds=3, seed=1):
    """{layer: push size} at which a random direction moves the reply by |e|^2 = control.target
    on average. |e|^2 grows about as the square of the push for small pushes."""
    rng = np.random.default_rng(seed)
    dirs = {L: [v / np.linalg.norm(v) for v in rng.standard_normal((c["calibrate"], eff.d))] for L in layers}
    size = {L: c["push"] * norms[L] for L in layers}
    for _ in range(rounds):
        e2 = (eff([(L, size[L] * v) for L in layers for v in dirs[L]]) ** 2).sum(1).reshape(len(layers), -1).mean(1)
        size = {L: size[L] * float(np.clip(np.sqrt(c["target"] / max(m, 1e-9)), 0.25, 4)) for L, m in zip(layers, e2)}
    return size


def metric(live, name, full, layers, c, seed, out):
    """The fixed metric for one problem: computed on the first run, loaded after.
    Returns the effects, the push sizes, and the random pool {layer: effect vectors}."""
    path = f"{out}/metric/{name}.npz"
    if os.path.exists(path):
        z = np.load(path)
        eff = Effects(live, full, z["reply"].tolist(), c["top"])
        return eff, dict(zip(layers, z["size"].tolist())), {L: z[f"pool{L}"] for L in layers}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = reply(live, full, c["reply"], c["temperature"], seed)
    eff = Effects(live, full, rep, c["top"])
    norms = {L: float(np.linalg.norm(v)) for L, v in jlens.states(live, full, layers).items()}
    size = calibrate(eff, layers, norms, c)
    rng = np.random.default_rng(0)
    pool = {L: eff([(L, size[L] * v / np.linalg.norm(v)) for v in rng.standard_normal((c["pool"], eff.d))])
            for L in layers}
    np.savez(path, reply=np.array(rep), size=np.array([size[L] for L in layers]),
             **{f"pool{L}": pool[L] for L in layers})
    return eff, size, pool


def floor(pool, cands, noise):
    """The random floor: as many random directions at each layer as there are candidates there."""
    n = {L: sum(x["layer"] == L for x in cands) for L in pool}
    if any(n[L] > len(pool[L]) for L in pool):
        raise ValueError(f"more candidates at a layer than the random pool holds ({n})")
    Er = np.concatenate([pool[L][:n[L]] for L in pool])
    return empowerment(Er, noise), (Er ** 2).sum(1)


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/control"))
    c = cfg["control"]
    os.makedirs(a.out, exist_ok=True)
    task = tasks.load(cfg["task"]["name"])
    probs = task.problems(cfg["task"]["split"])
    names = sorted(random.Random(cfg["task"]["seed"]).sample(sorted(probs), c["problems"]))
    live = load_live(cfg)
    layers, seed = c["layers"], cfg["task"]["seed"]
    results = {}

    for name in names:
        full = live.view([{"role": "user", "content": task.prompt(probs[name])[0]}]).ids
        instr = live.view([{"role": "user", "content": task.TASK_PROMPT.format(statement="")}]).ids
        empty = live.view([{"role": "user", "content": ""}]).ids
        eff, size, pool = metric(live, name, full, layers, c, seed, a.out)

        # step 0, timed
        t0 = time.time()
        s_full, s_instr, s_empty = (jlens.states(live, ids, layers) for ids in (full, instr, empty))
        lt = jlens.LastToken(live, full, layers)
        diffs = {"instructions": {L: s_instr[L] - s_empty[L] for L in layers},
                 "problem": {L: s_full[L] - s_instr[L] for L in layers}}
        vocab = (jlens.vocabulary(live, full, c["vocab_reply"], c["vocab_top"], c["temperature"], seed + 1000)
                 if c["vocab_reply"] else None)
        cands = jlens.candidates(live, lt, diffs, c["tokens"], vocab)
        step0 = time.time() - t0

        # the score, and the lens check
        t1 = time.time()
        E, rise = eff([(x["layer"], size[x["layer"]] * x["direction"]) for x in cands], [x["token"] for x in cands])
        score, e2 = empowerment(E, c["noise"]), (E ** 2).sum(1)
        low, low_e2 = floor(pool, cands, c["noise"])
        sat = int((e2 > c["saturated"]).sum())
        log(f"{name}: step 0 {step0:.0f}s, scoring {time.time() - t1:.0f}s; {len(cands)} candidates"
            f"{'' if vocab is None else f' from a vocabulary of {len(vocab)}'}; empowerment {score:.2f} "
            f"(random floor {low:.2f}); mean |e|^2 {e2.mean():.3f} (random {low_e2.mean():.3f}); {sat} saturated")
        log(f"   lens check: the push made its token more likely as the first reply token in "
            f"{int((rise > 0).sum())} of {len(rise)}; median rise in log p {np.median(rise):.2f}")
        for x, n2, r in sorted(zip(cands, e2, rise), key=lambda z: -z[1])[:c["show"]]:
            log(f"   layer {x['layer']:2d}  {x['source']:12s}  {x['text']!r:16}  |e|^2 {n2:.3f}  log p rise {r:+.2f}")
        results[name] = dict(step0_seconds=step0, empowerment=score, random=low, saturated=sat,
                             candidates=[dict(layer=x["layer"], source=x["source"], text=x["text"],
                                              effect=float(n2), rise=float(r)) for x, n2, r in zip(cands, e2, rise)])
        json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)

    log(f"mean over {len(results)} problems: empowerment {np.mean([r['empowerment'] for r in results.values()]):.2f}, "
        f"random floor {np.mean([r['random'] for r in results.values()]):.2f}, "
        f"step 0 {np.mean([r['step0_seconds'] for r in results.values()]):.0f}s")


if __name__ == "__main__":
    main()
