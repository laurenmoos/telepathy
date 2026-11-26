#!/bin/bash
# Run from anywhere; runs/ sits at the repo root.
#
#   telepathy/run.sh [options]            the loop (loop.py)
#   telepathy/run.sh attempt [options]    the task's attempt alone, with no program
#   telepathy/run.sh control [options]    the step 0 control experiment (eval/control.py)
#
# Options go through to Python, e.g. --set task.sample=10 --out runs/loop.
set -e
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export HF_HUB_OFFLINE=1               # load the model from the local Hugging Face cache only

case "$1" in
  attempt) shift; exec $PY -m telepathy.tasks.attempt "$@" ;;
  control) shift; exec $PY -m telepathy.eval.control "$@" ;;
  step0-unit) shift; exec $PY -m telepathy.eval.step0_unit "$@" ;;
  *)       exec $PY -m telepathy.loop "$@" ;;
esac
