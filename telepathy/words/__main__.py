"""Build (or load) both kinds of words for the model in config.yaml: python -m telepathy.words"""
import sys

from . import load
from ..setup.log import log
from ..setup.settings import settings, load_model

cfg, a = settings(sys.argv[1:])
ws = load(load_model(cfg, a), cfg)
log(f"{len(ws)} words: " + ", ".join(f"{k} {sum(w['place'] == k for w in ws)}" for k in ("ROW", "COL", "BOX", "ANSWER")))
