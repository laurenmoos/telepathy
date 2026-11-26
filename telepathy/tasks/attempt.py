"""The task's attempt alone, with no program, on a sample of problems (task.* in config.yaml).

    telepathy/run.sh attempt [--set task.sample=5] [--out runs/attempt]

Results are saved after every problem to <out>/results.json, and a rerun skips problems done.
"""
import json, os, random, time

from . import load
from ..setup.log import log
from ..setup.settings import settings, load_live


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/attempt"))
    tc = cfg["task"]
    task = load(tc["name"])
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    probs = task.problems(tc["split"])
    if tc["sample"]:
        probs = {n: probs[n] for n in sorted(random.Random(tc["seed"]).sample(sorted(probs), tc["sample"]))}
    live = load_live(cfg)
    t0 = time.time()
    for i, (name, problem) in enumerate(probs.items(), 1):
        if name in results:
            continue
        results[name] = task.attempt(live, problem, cfg["attempt"])
        json.dump(results, open(path, "w"), indent=1)
        log(f"{i}/{len(probs)} {name}: {'passed' if results[name]['passed'] else 'failed'}; "
            f"{sum(r['passed'] for r in results.values())} of {len(results)} passed ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
