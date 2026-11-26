"""The loop. It takes one problem at a time; nothing is computed over all problems in advance.

    0. read    one prompt pass over the plain task prompt; the states at this problem's tokens give
               noisy candidate read instructions
    1. match   a candidate close to a cached read instruction adds to its count; others enter as new
    2. write   the model writes an llm_program from read and write instructions
    3. trace   the program runs on the prompt pass only, and the model sees each line it ran and how
               the read values moved (2 and 3 repeat write.rounds times)
    4. check   the task's attempt, with no program and with the program (if it runs: it compiles,
               raises no error and makes a write); the judge says passed or failed. After a failed
               check the model edits the program, which is traced and checked again, up to
               check.retries times; the best program is kept (tasks.score)
    5. cache   keep the best program if it passed where the plain attempt failed; keep the read
               instructions that came back or that a kept program used; discard the rest

    telepathy/run.sh [--set task.sample=20] [--out runs/loop]

Everything is saved after each problem in <out>, and a rerun goes on from there.
"""
import json, os, random

from .instructions.cache import Cache
from .instructions.read import gather, candidates
from .programs.runtime import Program
from .programs.write import Writer, runs
from .setup.log import log
from .setup.settings import settings, load_live
from . import tasks
from .tasks import score


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/loop"))
    tc, rc, cc, wc, ac, p = cfg["task"], cfg["read"], cfg["cache"], cfg["write"], cfg["attempt"], cfg["prompts"]
    cc_retries = cfg["check"]["retries"]
    os.makedirs(a.out, exist_ok=True)
    task = tasks.load(tc["name"])
    probs = task.problems(tc["split"])
    if tc["sample"]:
        probs = {n: probs[n] for n in sorted(random.Random(tc["seed"]).sample(sorted(probs), tc["sample"]))}
    path = f"{a.out}/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    kept = [r["program"] for r in results.values() if r["kept"]]
    cache = Cache(cc["match"], cc["persist"], cc["patience"]).load(a.out)
    live = load_live(cfg)

    for i, (name, problem) in enumerate(probs.items()):
        if name in results:
            continue
        task_prompt, part = task.prompt(problem)
        view = live.view([{"role": "user", "content": task_prompt}])
        first, last = view.position_of(view.text.index(part)), len(view.ids) - 1

        # 0. read
        plain, _ = gather(live, view.ids, rc["layers"], range(first, last + 1))
        # 1. match
        back, new = cache.match(candidates(plain, rc["candidates"]), i, plain)
        log(f"{name}: {len(back)} read instructions came back, {len(new)} new, {len(cache.items)} in the cache")
        # 2, 3. write and trace
        w = Writer(live, wc, p, view, first, last, cache, cache.pick(wc["show"]), plain, kept)
        r = w.round()
        for _ in range(wc["rounds"] - 1):
            r = w.round(p["retrace"].format(trace=r["trace"]))
        # 4. check, with retries: after a failed check the model edits the program; the best is kept
        table = cache.table(cache.ranked())
        plain_try = task.attempt(live, problem, ac)
        tries, best = [], None
        for k in range(cc_retries + 1):
            got = task.attempt(live, problem, ac, program=lambda code=r["code"]: Program(
                code, view, first, last, table, len(live.layers))) if runs(r) else None
            tries.append(dict(r, attempt=got))
            log(f"{name}: program {k + 1}: " + ("did not run" if got is None else "passed" if got["passed"]
                                              else "failed") + f"\n{r['code']}\n{r['trace']}")
            if best is None or score(got) > score(best["attempt"]):
                best = tries[-1]
            if got is not None and got["passed"]:
                break
            if k < cc_retries:
                r = w.round(p["retry"].format(result="did not run" if got is None else "failed", trace=r["trace"]))
        program_try = best["attempt"]
        # 5. cache
        keep = bool(program_try and program_try["passed"] and not plain_try["passed"])
        if keep:
            kept.append(best["code"])
            cache.use(best["used"])
        gone = cache.prune(i)
        results[name] = dict(program=best["code"], kept=keep, plain=plain_try, with_program=program_try, tries=tries,
                             came_back=back, new=new, discarded=gone)
        json.dump(results, open(path, "w"), indent=1)
        cache.save(a.out)
        verdict = "did not run" if program_try is None else "passed" if program_try["passed"] else "failed"
        log(f"{name}: plain {'passed' if plain_try['passed'] else 'failed'}, with the program {verdict}; "
            f"{'kept' if keep else 'not kept'}; "
            f"{len(gone)} read instructions discarded; {sum(r['kept'] for r in results.values())} programs kept")


if __name__ == "__main__":
    main()
