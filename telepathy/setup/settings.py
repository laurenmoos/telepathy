"""Shared setup: settings, the model and its words.

    settings(argv)        config.yaml, with --set key.sub=value overrides
    load_model(cfg, a)    the model that is steered (cfg["model"])
    load_words(m, cfg)    the words: answer words (words/answer.py) and board words (words/board.py),
                          built once and cached in cache/; named for programs (lang.setup)
"""
import argparse, os
import yaml

from .log import log

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def settings(argv=None, extra=None):
    """Read config.yaml. extra: a function that adds more arguments to the parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", action="append", default=[], help="override a setting, e.g. boards.blanks=[2,5]")
    if extra:
        extra(ap)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    cfg["model_path"] = cfg["models"].get(cfg["model"], cfg["model"])
    return cfg, a


def load_model(cfg, a, name=None):
    from ..model import Model
    path = cfg["models"].get(name, name) if name else cfg["model_path"]
    m = Model(path, device=a.device)
    log(f"model {path} on {m.device} in {m.model.dtype}")
    return m


def load_words(m, cfg):
    """Give the model its words and name them for programs. Board words come first (layer 0),
    then answer words, earliest layer first."""
    from ..programs import lang
    from ..words import load
    ws = load(m, cfg, log=log)
    m.set_words(ws)
    lang.setup(ws)
    log(f"{len(ws)} words: " + ", ".join(f"{g} {n}" for g, n in lang.GROUP_SIZE.items()))
