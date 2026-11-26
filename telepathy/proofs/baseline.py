"""Baseline: can the model prove each miniF2F problem on its own, with no program?

For each problem, up to `tries` attempts (stopping at the first that Lean accepts). Results are
saved after every problem to <out>/results.json, and a rerun skips problems already done.

    python -m telepathy.proofs.baseline [--split Valid] [--tries 4] [--sample 12] [--out runs/baseline]
"""
import argparse, json, os, random, re, time

from .judge import problems, check
from ..setup.log import log

PROMPT = """Complete this Lean 4 proof (Mathlib is imported). Reply with only the tactics that come
after ":= by", in one ```lean block.

```lean
{statement}
```"""


def tactics(reply):
    """The proof from a reply: the last ```lean block, and only what follows the statement's ":= by"
    if the reply repeats the statement (a theorem, lemma or example line)."""
    blocks = re.findall(r"```(?:lean4?)?\n(.*?)```", reply, re.S)
    if blocks:
        text = blocks[-1]
    elif "```" in reply:                       # a block cut off at the token limit
        text = reply.rsplit("```", 1)[1].split("\n", 1)[-1]
    else:
        text = reply
    if re.search(r"^\s*(theorem|lemma|example)\b", text, re.M) and ":= by" in text:
        text = text.split(":= by", 1)[1]          # the reply repeated the statement
    lines = [l for l in text.strip("\n").splitlines() if l.strip()]
    if not lines:
        return ""
    indent = min(len(l) - len(l.lstrip()) for l in lines)
    return "\n".join(l[indent:] for l in lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Valid")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--sample", type=int, default=0, help="only this many random problems (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/baseline")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    from ..model.live import Live
    live = Live(a.model)
    probs = problems(a.split)
    if a.sample:
        keep = random.Random(a.seed).sample(sorted(probs), a.sample)
        probs = {n: probs[n] for n in sorted(keep)}
    t0 = time.time()
    for i, (name, statement) in enumerate(probs.items(), 1):
        if name in results:
            continue
        attempts = []
        for k in range(a.tries):
            reply = live.generate(PROMPT.format(statement=statement), a.max_tokens, 0.7, False,
                                  repetition_penalty=1.1).split("</think>")[-1]
            proof = tactics(reply)
            ok, msg = check(statement, proof) if proof else (False, "no proof in the reply")
            attempts.append(dict(proof=proof, passed=ok, lean=msg[:2000]))
            if ok:
                break
        results[name] = dict(solved=any(x["passed"] for x in attempts), attempts=attempts)
        json.dump(results, open(path, "w"), indent=1)
        done = sum(r["solved"] for r in results.values())
        log(f"{i}/{len(probs)} {name}: {'solved' if results[name]['solved'] else 'not solved'} in {len(attempts)} "
            f"tries; {done} of {len(results)} solved so far ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
