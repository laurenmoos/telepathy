"""The Lean task: prove miniF2F statements; Lean is the judge. Only this module knows about Lean.

A task gives the loop four things:

    problems(split)               {name: problem}
    prompt(problem)               (TASK_PROMPT, the problem's own part of it)
    attempt(live, problem, cfg, program)
                                  the self-contained attempt: think, write a proof, and repair it
                                  from Lean's errors, up to cfg["rounds"] times. program() makes a
                                  fresh hook for each prompt pass (or is None). Returns
                                  dict(passed, rounds), each round's proof and Lean's messages.
"""
import re

from .judge import problems, check

TASK_PROMPT = """Complete this Lean 4 proof (Mathlib is imported). Think about the math first. Then reply
with only the tactics that come after ":= by", in one ```lean block.

Write Lean 4, not Lean 3:
- one tactic per line; no commas between tactics
- obtain ⟨a, b⟩ := h  (not: cases h with a b)
- fun x => ...  (not: λ x, ...)
- have h : P := by tactic
- sums are written ∑ k ∈ Finset.range n, f k  (not: ∑ k in ...)
Use the hypothesis names given in the statement (like h₀, h₁).
Prefer tactics that do the algebra themselves: norm_num, linarith, nlinarith [sq_nonneg (x - y)],
omega, ring_nf, field_simp, positivity, decide, simp. Use rw only with an equation whose left side
appears exactly in the goal.
If you need a Mathlib lemma but are not sure of its exact name, write exact? at that step:
Lean searches for a lemma that closes the goal.

```lean
{statement}
```"""

REPAIR = "Lean says:\n{err}\n\n{hint}Fix the proof. Reply with only the tactics, in one ```lean block."
UNKNOWN = ("A name Lean says is unknown does not exist in Mathlib. Do not guess another name: write exact? at "
           "that step, and Lean searches for a lemma that closes the goal.\n")


def prompt(statement):
    """The problem's own part starts at its theorem line: the header before it (imports, options,
    open namespaces) is the same on every problem."""
    return TASK_PROMPT.format(statement=statement), statement[statement.index("theorem"):]


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


def attempt(live, statement, cfg, program=None):
    chat = [{"role": "user", "content": prompt(statement)[0]}]
    out = []
    for _ in range(cfg["rounds"]):
        reply = live.generate(chat, cfg["max_tokens"], cfg["temperature"], True, cfg["think_tokens"],
                              cfg["repetition_penalty"], hook=program() if program else None)
        answer = reply.split("</think>")[-1]
        proof = tactics(answer)
        ok, msg = check(statement, proof) if proof else (False, "There is no ```lean block in your reply.")
        out.append(dict(proof=proof, passed=ok, lean=msg[:2000]))
        if ok:
            break
        hint = UNKNOWN if "unknown" in msg.lower() and ("identifier" in msg or "constant" in msg) else ""
        chat += [{"role": "assistant", "content": answer.strip()},
                 {"role": "user", "content": REPAIR.format(err="\n".join(msg.splitlines()[:12]), hint=hint)}]
    return dict(passed=any(r["passed"] for r in out), rounds=out)

