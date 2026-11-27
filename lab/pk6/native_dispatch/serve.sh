#!/usr/bin/env bash
# Run from a disposable checkout prepared using native_dispatch.prepare.
set -euo pipefail
out=$(realpath -m "$1")
shift
mkdir -p "$(dirname "$out")"
telemetry="$(dirname "$out")/$(basename "$(dirname "$out")").csv"
export TZ=UTC
export IRONMULE_COMPARE_LOCAL_KEY=ironmule-local-bench
# The runner also holds this lock internally. The sampler is passive and can cover multiple serial arms.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 >> "$telemetry" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
python -m lab.pk6.native_dispatch.run --out "$out" "$@"
