"""Lean as the judge, on miniF2F (lean/miniF2F, with Mathlib).

    problems(split)        {name: statement} for split "Valid" or "Test"; each statement ends
                           with ":= by", and a proof is the tactics that follow
    check(name, proof)     (passed, Lean's messages): Lean checks the statement with the proof,
                           and it passes only with no error and no sorry
"""
import os, re, subprocess, tempfile

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lean", "miniF2F")


def problems(split="Valid"):
    out = {}
    d = os.path.join(ROOT, "MiniF2F", split)
    for f in sorted(os.listdir(d)):
        if f.endswith(".lean"):
            text = open(os.path.join(d, f)).read()
            out[f[:-5]] = re.sub(r"\s*sorry\s*$", "", text.rstrip())       # ends with ":= by"
    return out


def check(statement, proof, timeout=120):
    """Is `proof` (tactics) a proof of `statement` (ending ":= by")?"""
    body = "\n".join("  " + line for line in proof.strip("\n").splitlines())
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=ROOT, delete=False) as f:
        f.write(statement + "\n" + body + "\n")
        path = f.name
    try:
        r = subprocess.run(["lake", "env", "lean", path], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        msg = (r.stdout + r.stderr).replace(path, "proof.lean").strip()
        ok = r.returncode == 0 and "error" not in msg and "sorry" not in msg
    except subprocess.TimeoutExpired:
        ok, msg = False, f"Lean took more than {timeout} seconds"
    finally:
        os.remove(path)
    return ok, msg
