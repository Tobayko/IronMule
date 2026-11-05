"""TP0: what a Kaggle TPU session offers IronMule, measured before anything is built for it (BACKLOG TP1).

MLX, IronMule's runtime, has no TPU backend. Before a TPU path is worth building, this probe records, on one
Kaggle TPU session: the devices JAX sees, the host CPU and memory, and the time of bf16 matrix products at the
shapes one decode step of `Qwen3-1.7B` needs (hidden 2048, MLP 6144, 28 layers), on one TPU chip with JAX and on
the host CPU with MLX's CPU build. It changes no model and serves nothing: the numbers are a ceiling for the
matrix part of decode, not tokens per second of any path. Descriptive only.

    python tools/tpu_probe.py --smoke --out report.json      # tiny sizes: checks this tool, measures nothing
    python tools/tpu_probe.py --analyse report.json
    python tools/tpu_probe.py --write-kernel DIR --owner tobayko --commit <full sha>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA = "ironmule.tpu_probe.v1"
HIDDEN, MLP, LAYERS = 2048, 6144, 28                     # Qwen3-1.7B
# One decode step's matrix products per layer: q, k, v, o (GQA: 16 query heads, 8 kv heads of 128), gate, up, down.
LAYER_SHAPES = ((HIDDEN, 2048), (HIDDEN, 1024), (HIDDEN, 1024), (2048, HIDDEN),
                (HIDDEN, MLP), (HIDDEN, MLP), (MLP, HIDDEN))
ROWS = (1, 8, 64)                                         # one request, the T4's best stacked width, a wide batch
T4_REFERENCE = {"float16_single_tok_s": 57.7, "dense_width8_tok_s": 290.9,
                "source": "BACKLOG AG4-Q run 2 (Kaggle T4, Qwen3-1.7B bf16)"}


def _median_s(run: Callable[[], Any], repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine(),
                           "host_cpus": os.cpu_count()}
    try:
        with open("/proc/meminfo") as handle:
            env["host_mem_gib"] = round(int(handle.readline().split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        env["host_mem_gib"] = None
    try:
        import jax

        devices = jax.devices()
        env.update(jax=jax.__version__, backend=jax.default_backend(), devices=len(devices),
                   device_kind=devices[0].device_kind if devices else None)
    except Exception as error:  # noqa: BLE001 - the probe records what is missing instead of failing
        env["jax_error"] = f"{type(error).__name__}: {error}"
    try:
        import mlx.core as mx

        env["mlx"] = getattr(mx, "__version__", "unknown")
    except Exception as error:  # noqa: BLE001
        env["mlx_error"] = f"{type(error).__name__}: {error}"
    return env


def jax_products(shapes: Iterable[tuple[int, int]], rows: Iterable[int], repeats: int, warmup: int) -> list[dict]:
    """bf16 `x @ w` on JAX's default device (one TPU chip on a TPU session), jitted, timed to completion."""
    import jax
    import jax.numpy as jnp

    out = []
    key = jax.random.PRNGKey(0)
    for k, n in shapes:
        w = jax.random.normal(key, (k, n), dtype=jnp.bfloat16)
        for m in rows:
            x = jax.random.normal(key, (m, k), dtype=jnp.bfloat16)
            product = jax.jit(lambda a, b: a @ b)
            seconds = _median_s(lambda: product(x, w).block_until_ready(), repeats, warmup)
            out.append({"k": k, "n": n, "rows": m, "seconds": seconds})
    return out


def mlx_products(shapes: Iterable[tuple[int, int]], rows: Iterable[int], repeats: int, warmup: int) -> list[dict]:
    """bf16 `x @ w` with MLX on the host CPU (MLX's CPU build), evaluated to completion."""
    import mlx.core as mx

    out = []
    for k, n in shapes:
        w = mx.random.normal((k, n)).astype(mx.bfloat16)
        for m in rows:
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            mx.eval(w, x)
            seconds = _median_s(lambda: mx.eval(x @ w), repeats, warmup)
            out.append({"k": k, "n": n, "rows": m, "seconds": seconds})
    return out


def probe(*, smoke: bool = False) -> dict[str, Any]:
    shapes = ((64, 64), (64, 128)) if smoke else LAYER_SHAPES
    rows = (1, 2) if smoke else ROWS
    repeats, warmup = (3, 1) if smoke else (20, 3)
    report: dict[str, Any] = {"schema": SCHEMA, "smoke": smoke, "env": environment(),
                              "shapes": [list(s) for s in shapes], "rows": list(rows), "layers": LAYERS,
                              "t4_reference": T4_REFERENCE}
    for name, run in (("jax", jax_products), ("mlx_cpu", mlx_products)):
        try:
            report[name] = run(shapes, rows, repeats, warmup)
        except Exception as error:  # noqa: BLE001 - a missing runtime is a finding, not a crash
            report[name] = {"error": f"{type(error).__name__}: {error}"}
    return report


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """Per runtime and row count: one decode step's matrix time (all layers' products, nothing else), the tokens
    per second that time alone would allow, and the weight bytes streamed per second. A ceiling, not a speed."""
    layers = report.get("layers", LAYERS)
    out: dict[str, Any] = {"schema": SCHEMA + ".analysis", "env": report.get("env"), "smoke": report.get("smoke"),
                           "t4_reference": report.get("t4_reference"), "runtimes": {}}
    for name in ("jax", "mlx_cpu"):
        rows = report.get(name)
        if not isinstance(rows, list):
            out["runtimes"][name] = rows
            continue
        per_rows: dict[str, Any] = {}
        for m in sorted({r["rows"] for r in rows}):
            cells = [r for r in rows if r["rows"] == m]
            step_s = layers * sum(r["seconds"] for r in cells)
            weight_bytes = layers * sum(2 * r["k"] * r["n"] for r in cells)
            per_rows[str(m)] = {"matrix_step_s": step_s,
                                "ceiling_tok_s": (m / step_s) if step_s > 0 else None,
                                "weight_gb_s": (weight_bytes / step_s / 1e9) if step_s > 0 else None}
        out["runtimes"][name] = per_rows
    return out


_BOOTSTRAP = '''"""IronMule TP0 on this Kaggle TPU session; written by tools/tpu_probe.py at {commit}."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlx[cpu]"], check=False)  # best effort: a finding
'''


def write_kernel(directory: Path, owner: str, commit: str) -> Path:
    """A private TPU script kernel that runs this probe; the report lands as `tpu_probe_report.json`."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    entry = f"\nsys.exit(main({['--out', '/kaggle/working/tpu_probe_report.json']!r}))\n"
    (directory / "run.py").write_text(_BOOTSTRAP.format(commit=commit) + body + entry)
    name = "ironmule-tpu-probe"
    metadata = {"id": f"{owner}/{name}", "title": name, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
                "enable_internet": True, "machine_shape": "TpuV5E8", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the report here")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--analyse", type=Path, metavar="REPORT", help="print the analysis of a finished report")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR", help="write a Kaggle TPU kernel folder instead")
    parser.add_argument("--owner")
    parser.add_argument("--commit")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1, sort_keys=True))
        return 0
    if args.write_kernel:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(write_kernel(args.write_kernel, args.owner, args.commit))
        return 0
    report = probe(smoke=args.smoke)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps(analyse(report), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
