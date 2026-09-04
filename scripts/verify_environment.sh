#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
PROJECTATLAS_BIN="${PROJECTATLAS_BIN:-/Users/tobiasburandt/.local/bin/projectatlas}"

echo "== Xcode =="
xcodebuild -version
xcodebuild -checkFirstLaunchStatus

echo "== ProjectAtlas =="
"$PROJECTATLAS_BIN" --format json runtime-info | jq '{project,version,executable,capabilities}'

echo "== Python / MLX / Metal / MPS =="
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python-Umgebung fehlt: $PYTHON_BIN" >&2
  echo "Zuerst ./scripts/bootstrap_apple.sh ausführen." >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

import mlx.core as mx
import torch

print(f"mlx={version('mlx')}")
print(f"metal={mx.metal.is_available()}")
print(f"device={mx.default_device()}")
print(f"device_info={mx.device_info()}")
print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")
if not mx.metal.is_available() or not torch.backends.mps.is_available():
    raise SystemExit("MLX Metal oder PyTorch MPS ist nicht verfügbar")

a = torch.ones((2, 2), device="mps")
assert torch.equal((a + a).cpu(), torch.full((2, 2), 2.0))
print("basic_mps_correctness=ok")
PY

echo "== Generated MCP configs =="
for config in \
  "$ROOT/.projectatlas/projectatlas.mcp.json" \
  "$ROOT/.projectatlas/projectatlas.claude.mcp.json" \
  "$ROOT/.projectatlas/projectatlas.opencode.json"; do
  [[ -f "$config" ]] || { echo "Fehlt: $config" >&2; exit 1; }
  jq empty "$config"
  echo "ok: $config"
done
