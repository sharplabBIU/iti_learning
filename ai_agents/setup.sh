#!/bin/bash
# One-time setup for the AI-agent version of the bandit study.
# The Python environment lives OUTSIDE this (Synology-synced) folder, in ~/.venvs/ai_bandit, so the sync does not copy
# thousands of package files; Playwright's browser goes to its own cache (~/Library/Caches/ms-playwright).
set -e
VENV="$HOME/.venvs/ai_bandit"
PY=${PYTHON:-/opt/miniconda3/bin/python3}
if [ ! -x "$VENV/bin/python" ]; then
  echo ">>> creating $VENV"
  mkdir -p "$HOME/.venvs"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install -r "$(dirname "$0")/requirements.txt"
"$VENV/bin/python" -m playwright install chromium
echo
echo "Setup done. Run the agents with:"
echo "  $VENV/bin/python \"$(cd "$(dirname "$0")" && pwd)/run_agents.py\" --help"
