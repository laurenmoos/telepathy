#!/bin/bash
# Run the loop on this Mac. The first time, it sets up Python and installs what it needs.
#
#   ./run.sh                    run, continuing from the last checkpoint if there is one
#   ./run.sh --fresh            start over
#   ./run.sh --model qwen3-8b   use the bigger model
#
# Any other options are passed to loop.py (see: ./run.sh --help).
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up Python in .venv (takes a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1   # anything the Mac's GPU can't do runs on the CPU instead
export HF_HUB_OFFLINE=1               # load models from the local Hugging Face cache only

exec .venv/bin/python loop.py "$@"
