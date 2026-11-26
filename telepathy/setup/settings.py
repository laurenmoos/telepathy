"""Settings and the model.

    settings(argv)     config.yaml, with --set key.sub=value overrides
    load_live(cfg)     the model (cfg["model"]): it writes programs and runs them on itself
"""
import argparse, os
import yaml

from .log import log

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def settings(argv=None, extra=None):
    """Read config.yaml. extra: a function that adds more arguments to the parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
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


def load_live(cfg):
    from ..model.live import Live
    live = Live(cfg["model_path"], keep=cfg["live"]["keep"])
    log(f"model {cfg['model_path']}: {len(live.layers)} layers")
    return live
