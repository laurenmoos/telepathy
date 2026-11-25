"""Check a program: run it on the practice boards and on the held-out boards, filling every empty
cell in order (wrong digits stay in), and compare with no program. Used by loop.py and agent.py
at the end of a run; telepathy/check.py checks a program file.
"""

from ..programs import lang
from .play import play, summarize


def check(m, program, practice, test):
    """Cells right and boards solved, with no program and with the program, on both board sets.
    program: a compiled lang.Prog, or statements in the old language."""
    progs = [lang.Prog([]), program if isinstance(program, lang.Prog) else lang.compile_program(program)]
    out = {}
    for name, boards in (("practice", practice), ("held-out", test)):
        none, prog = (summarize(mv) for mv in play(progs, boards, m))
        out[name] = dict(no_program=none, program=prog, boards=len(boards))
    return out


def text(result):
    return "\n".join(f"{name} ({r['boards']} boards): cells right {r['no_program']['cells_right']:.0%} -> "
                     f"{r['program']['cells_right']:.0%}; boards solved {r['no_program']['solved']:.0%} -> "
                     f"{r['program']['solved']:.0%}" for name, r in result.items())

