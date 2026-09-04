#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12 nicht gefunden: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv nicht gefunden: $UV_BIN" >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$ROOT/.venv"
fi
"$UV_BIN" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements-apple-silicon.txt"
echo "Python-Umgebung bereit: $ROOT/.venv"
