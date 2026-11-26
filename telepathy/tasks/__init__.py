"""Tasks. A task is a module with problems, prompt and attempt (see tasks/lean/task.py).

    load(name)      the task module
    score(result)   how good an attempt was, for comparing programs: a passed attempt beats a
                    failed one, and fewer rounds beat more; no attempt (None) is worst
"""
import importlib


def load(name):
    return importlib.import_module(f".{name}.task", __name__)


def score(result):
    if result is None:
        return (-1, 0)
    return (int(result["passed"]), -len(result["rounds"]))
