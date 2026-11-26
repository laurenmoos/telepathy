"""Tasks. A task is a module with problems, prompt and attempt (see tasks/lean/task.py)."""
import importlib


def load(name):
    return importlib.import_module(f".{name}.task", __name__)
