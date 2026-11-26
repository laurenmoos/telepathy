"""The cache of read instructions (steps 1 and 5 of the loop).

A read instruction is one direction in the residual stream at one layer. Its value at a token is
how far the state there sits along the direction, in units of its spread: 0 is typical, +2 or -2
is unusual. The spread is a running mean and spread over the tokens of every problem seen since the
instruction entered the cache.

    cache.match(cands, i, states)   step 1: a candidate close to a cached instruction (|cosine| at
                                    least `match`) adds to that instruction's count and moves its
                                    direction toward the candidate; any other enters as new
    cache.values(name, x)           an instruction's values on states x [n, d] of its layer
    cache.prune(i)                  step 5: discard instructions that did not come back and were not
                                    used by a kept program within `patience` problems, unless they
                                    have come back `persist` times
"""
import json, os

import numpy as np


class Cache:
    def __init__(self, match=0.5, persist=3, patience=5):
        self.match_at, self.persist, self.patience = match, persist, patience
        self.items, self.serial = {}, 0          # name -> dict(layer, dir, count, born, seen, used, n, mean, m2)

    # ------------------------------------------------------------ step 1

    def match(self, cands, i, states):
        """Returns (names that came back, names that are new)."""
        back, new, taken = [], [], set()
        for L, d in cands:
            best, cos = None, 0.0
            for name, it in self.items.items():
                if it["layer"] == L and name not in taken:
                    c = float(it["dir"] @ d)
                    if abs(c) > abs(cos):
                        best, cos = name, c
            if best is not None and abs(cos) >= self.match_at:
                it = self.items[best]
                d = d if cos > 0 else -d
                it["dir"] = it["dir"] * it["count"] + d
                it["dir"] /= np.linalg.norm(it["dir"])
                it["count"] += 1
                it["seen"] = i
                taken.add(best)
                back.append(best)
            else:
                name = f"L{L}_{self.serial}"
                self.serial += 1
                self.items[name] = dict(layer=L, dir=d, count=1, born=i, seen=i, used=0, n=0, mean=0.0, m2=0.0)
                taken.add(name)
                new.append(name)
        for it in self.items.values():                        # the running mean and spread
            if it["layer"] in states:
                self._add(it, states[it["layer"]] @ it["dir"])
        return back, new

    @staticmethod
    def _add(it, v):
        n, mean, m2 = len(v), float(v.mean()), float(((v - v.mean()) ** 2).sum())
        tot = it["n"] + n
        delta = mean - it["mean"]
        it["mean"] += delta * n / tot
        it["m2"] += m2 + delta ** 2 * it["n"] * n / tot
        it["n"] = tot

    # ------------------------------------------------------------ using instructions

    def spread(self, name):
        it = self.items[name]
        return max((it["m2"] / max(it["n"] - 1, 1)) ** 0.5, 1e-6)

    def values(self, name, x):
        it = self.items[name]
        return (x @ it["dir"] - it["mean"]) / self.spread(name)

    def table(self, names):
        """What a program needs: name -> (layer, direction, mean, spread)."""
        return {n: (self.items[n]["layer"], self.items[n]["dir"], self.items[n]["mean"], self.spread(n)) for n in names}

    def ranked(self, among=None):
        """Names, the most often come back first."""
        names = list(self.items) if among is None else [n for n in among if n in self.items]
        return sorted(names, key=lambda n: (-self.items[n]["used"], -self.items[n]["count"], self.items[n]["born"]))

    def pick(self, k):
        """k names to show: the best ranked at each layer in turn, so every layer is shown."""
        by = {}
        for n in self.ranked():
            by.setdefault(self.items[n]["layer"], []).append(n)
        out, j = [], 0
        while len(out) < k and any(j < len(v) for v in by.values()):
            out += [v[j] for _, v in sorted(by.items()) if j < len(v)][:k - len(out)]
            j += 1
        return out

    # ------------------------------------------------------------ step 5

    def use(self, names):
        for n in names:
            if n in self.items:
                self.items[n]["used"] += 1

    def prune(self, i):
        """Discard the instructions that did not come back. Returns their names."""
        gone = [n for n, it in self.items.items()
                if it["used"] == 0 and it["count"] < self.persist and i - it["seen"] >= self.patience]
        for n in gone:
            del self.items[n]
        return gone

    # ------------------------------------------------------------ saving

    def save(self, d):
        meta = {n: {k: v for k, v in it.items() if k != "dir"} for n, it in self.items.items()}
        json.dump(dict(serial=self.serial, items=meta), open(f"{d}/cache.json", "w"), indent=1)
        np.savez(f"{d}/cache.npz", **{n: it["dir"] for n, it in self.items.items()})

    def load(self, d):
        if not os.path.exists(f"{d}/cache.json"):
            return self
        meta = json.load(open(f"{d}/cache.json"))
        dirs = np.load(f"{d}/cache.npz")
        self.serial = meta["serial"]
        self.items = {n: dict(it, dir=dirs[n]) for n, it in meta["items"].items()}
        return self
