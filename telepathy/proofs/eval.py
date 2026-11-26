"""The proofs eval: on each problem the model thinks, writes a proof, and repairs it from Lean's
errors, up to `rounds` times. A problem is solved if Lean accepts any round's proof.

Results are saved after every problem to <out>/results.json (every round's proof and Lean's
messages), and a rerun skips problems already done.

    python -m telepathy.proofs.eval [--split Valid] [--sample 5] [--seed 2] [--rounds 4] [--out runs/eval]
"""
import argparse, json, os, random, time

from .baseline import PROMPT, tactics
from .judge import problems, check
from ..setup.log import log

REPAIR = "Lean says:\n{err}\n\n{hint}Fix the proof. Reply with only the tactics, in one ```lean block."
UNKNOWN = ("A name Lean says is unknown does not exist in Mathlib. Do not guess another name: write exact? at "
           "that step, and Lean searches for a lemma that closes the goal.\n")


def attempt(live, statement, rounds, think_tokens, max_tokens):
    """Think, write, and repair from Lean's errors. Returns the rounds: proof, passed, Lean's messages."""
    chat = [{"role": "user", "content": PROMPT.format(statement=statement)}]
    out = []
    for _ in range(rounds):
        reply = live.generate(None, max_tokens, 0.6, True, 1.05, think_tokens=think_tokens, messages=chat)
        answer = reply.split("</think>")[-1]
        proof = tactics(answer)
        ok, msg = check(statement, proof) if proof else (False, "There is no ```lean block in your reply.")
        out.append(dict(proof=proof, passed=ok, lean=msg[:2000]))
        if ok:
            break
        hint = UNKNOWN if "unknown" in msg.lower() and ("identifier" in msg or "constant" in msg) else ""
        chat += [{"role": "assistant", "content": answer.strip()},
                 {"role": "user", "content": REPAIR.format(err="\n".join(msg.splitlines()[:12]), hint=hint)}]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Valid")
    ap.add_argument("--sample", type=int, default=5, help="random problems (0: all)")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--think_tokens", type=int, default=1000)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--out", default="runs/eval")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    probs = problems(a.split)
    if a.sample:
        probs = {n: probs[n] for n in sorted(random.Random(a.seed).sample(sorted(probs), a.sample))}
    from ..model.live import Live
    live = Live(a.model)
    t0 = time.time()
    for i, (name, statement) in enumerate(probs.items(), 1):
        if name in results:
            continue
        rounds = attempt(live, statement, a.rounds, a.think_tokens, a.max_tokens)
        results[name] = dict(solved=any(r["passed"] for r in rounds), rounds=rounds)
        json.dump(results, open(path, "w"), indent=1)
        last = next((l for l in rounds[-1]["lean"].splitlines() if "error" in l), "")
        log(f"{i}/{len(probs)} {name}: " + (f"solved in round {len(rounds)}" if results[name]["solved"]
                                            else f"not solved; last error: {last[:100]}")
            + f"; {sum(r['solved'] for r in results.values())} of {len(results)} solved ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
