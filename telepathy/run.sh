#!/bin/bash
# Run from the repo root, so boards/ and runs/ sit there.
#
#   telepathy/run.sh [options]                  the loop (loop.py)
#   telepathy/run.sh check <program file>       check a program (check.py)
#
# Options go through to Python, e.g. --set loop.steps=10 --out runs/loop.
set -e
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export HF_HUB_OFFLINE=1               # load the model from the local Hugging Face cache only

case "$1" in
  check) shift; exec $PY -m telepathy.check "$@" ;;
  *)     exec $PY -m telepathy.loop "$@" ;;
esac
