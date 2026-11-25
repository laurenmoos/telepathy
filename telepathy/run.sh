#!/bin/bash
# Run from the repo root, so boards/, cache/ and runs/ sit there.
#
#   telepathy/run.sh [options]                  the writing loop (loop.py)
#   telepathy/run.sh agent [options]            the tool-calling writer (agent.py)
#   telepathy/run.sh check <program file>       check a program (check.py)
#   telepathy/run.sh words [options]            build or load both kinds of words (words/)
#
# Options go through to Python, e.g. --set write.sessions=2 --out runs/write.
set -e
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1   # anything the Mac's GPU can't do runs on the CPU instead

case "$1" in
  agent)        shift; exec $PY -m telepathy.agent "$@" ;;
  check)        shift; exec $PY -m telepathy.check "$@" ;;
  words)        shift; exec $PY -m telepathy.words "$@" ;;
  *)            exec $PY -m telepathy.loop "$@" ;;
esac
