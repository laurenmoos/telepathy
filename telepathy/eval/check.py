"""Check a program: it fills every empty cell of the practice boards and of the held-out boards, in
order (wrong digits stay in), and is compared with no program. Used at the end of loop.py, and by
check.py to check a program file.
"""
from .play import play, summarize


def check(live, code, practice, test):
    """Cells right and boards solved, with no program and with the program, on both board sets."""
    out = {}
    for name, boards in (("practice", practice), ("held-out", test)):
        out[name] = dict(no_program=summarize(play(live, "", boards)), program=summarize(play(live, code, boards)),
                         boards=len(boards))
    return out


def text(result):
    return "\n".join(f"{name} ({r['boards']} boards): cells right {r['no_program']['cells_right']:.0%} -> "
                     f"{r['program']['cells_right']:.0%}; boards solved {r['no_program']['solved']:.0%} -> "
                     f"{r['program']['solved']:.0%}" for name, r in result.items())
