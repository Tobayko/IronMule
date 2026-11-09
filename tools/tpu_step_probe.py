"""TP0b: one whole decode step of `Qwen3-1.7B`'s shapes, compiled as one program, on one TPU chip (BACKLOG TP1).

TP0 timed each matrix product alone, jitted and synced on its own, and every product took 0.20-0.29 ms whatever its
size: a per-call floor, not the chip's speed. This probe compiles a whole decode step instead, every layer in one
program: embedding, RMSNorm, q/k/v with Qwen 3's per-head q/k norms, rotary positions, the step's key and value
written into a fixed KV cache, attention over that cache, o, the SwiGLU MLP, the final norm and the tied head with
its argmax. It times that step called once per step (`per_call`, host dispatch included) and `CHAIN` steps run
inside one compiled loop, each fed the previous step's tokens (`chained`, the cost a decode path pays when the host
reads back only every `CHAIN` steps).

The weights are random bf16 of the model's shapes: no checkpoint is downloaded and no answer is produced, so this is
a speed ceiling of the step's arithmetic on the chip, not tokens per second of any serving path. Descriptive only.

    python tools/tpu_step_probe.py --smoke --out report.json     # tiny sizes on any JAX backend: checks the tool
    python tools/tpu_step_probe.py --analyse report.json
    python tools/tpu_step_probe.py --write-kernel DIR --owner tobayko --commit <full sha>
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
from typing import Any, Iterable

SCHEMA = "ironmule.tpu_step_probe.v1"
# Qwen3-1.7B (config.json): hidden 2048, 28 layers, 16 query heads and 8 KV heads of 128, MLP 6144, vocab 151,936,
# tied embeddings, RoPE theta 1,000,000, RMSNorm eps 1e-6.
QWEN3_1_7B = {"hidden": 2048, "layers": 28, "heads": 16, "kv_heads": 8, "head_dim": 128, "mlp": 6144,
              "vocab": 151936, "theta": 1_000_000.0, "eps": 1e-6}
SMOKE = {"hidden": 64, "layers": 2, "heads": 4, "kv_heads": 2, "head_dim": 16, "mlp": 128, "vocab": 256,
         "theta": 10_000.0, "eps": 1e-6}
# (rows, KV capacity, position): one request and the T4's best stacked width at a 1,024-token context in a
# 2,048-token cache, and a wide batch in a 512-token cache (64 rows of 2,048 would not fit one chip's 16 GiB).
CASES = ((1, 2048, 1024), (8, 2048, 1024), (64, 512, 256))
SMOKE_CASES = ((1, 32, 8), (2, 32, 8))
CHAIN = 32
T4_REFERENCE = {"float16_single_tok_s": 57.7, "dense_width8_tok_s": 290.9,
                "source": "BACKLOG AG4-Q run 2 (Kaggle T4, Qwen3-1.7B bf16)"}


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine(),
                           "host_cpus": os.cpu_count()}
    try:
        import jax

        devices = jax.devices()
        env.update(jax=jax.__version__, backend=jax.default_backend(), devices=len(devices),
                   device_kind=devices[0].device_kind if devices else None)
    except Exception as error:  # noqa: BLE001 - the probe records what is missing instead of failing
        env["jax_error"] = f"{type(error).__name__}: {error}"
    return env


def weights(config: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Random bf16 weights of the model's shapes, layers stacked along the first axis (scanned in the step)."""
    import jax
    import jax.numpy as jnp

    h, layers, d = config["hidden"], config["layers"], config["head_dim"]
    q, kv, mlp = config["heads"] * d, config["kv_heads"] * d, config["mlp"]
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 16))

    def normal(shape: tuple[int, ...], scale: float) -> Any:
        return (jax.random.normal(next(keys), shape, dtype=jnp.float32) * scale).astype(jnp.bfloat16)

    return {"embed": normal((config["vocab"], h), 0.02),
            "wq": normal((layers, h, q), h ** -0.5), "wk": normal((layers, h, kv), h ** -0.5),
            "wv": normal((layers, h, kv), h ** -0.5), "wo": normal((layers, q, h), q ** -0.5),
            "wg": normal((layers, h, mlp), h ** -0.5), "wu": normal((layers, h, mlp), h ** -0.5),
            "wd": normal((layers, mlp, h), mlp ** -0.5),
            "n_attn": jnp.ones((layers, h), jnp.bfloat16), "n_mlp": jnp.ones((layers, h), jnp.bfloat16),
            "n_q": jnp.ones((layers, d), jnp.bfloat16), "n_k": jnp.ones((layers, d), jnp.bfloat16),
            "n_out": jnp.ones((h,), jnp.bfloat16)}


def cache(config: dict[str, Any], rows: int, capacity: int) -> tuple[Any, Any]:
    import jax.numpy as jnp

    shape = (config["layers"], rows, config["kv_heads"], capacity, config["head_dim"])
    return jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16)


def make_step(config: dict[str, Any], logits: bool = False):
    """`step(params, tokens, keys, values, position) -> (next tokens, keys, values)`: one decode step of every row
    at `position` (the same for all rows), every layer inside one `lax.scan`. With `logits` it returns the float32
    logits in place of the tokens (the tests check them against a plain NumPy step)."""
    import jax
    import jax.numpy as jnp

    heads, kv_heads, d, eps = config["heads"], config["kv_heads"], config["head_dim"], config["eps"]
    group = heads // kv_heads
    inv = 1.0 / (config["theta"] ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))

    def norm(x: Any, weight: Any) -> Any:
        x32 = x.astype(jnp.float32)
        return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)).astype(x.dtype) * weight

    def rope(x: Any, position: Any) -> Any:  # (rows, heads, d), the same position for every row
        angle = position.astype(jnp.float32) * inv
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        x32 = x.astype(jnp.float32)
        first, second = x32[..., : d // 2], x32[..., d // 2:]
        return jnp.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1).astype(x.dtype)

    def layer(carry: tuple[Any, Any], per_layer: tuple[Any, ...]) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        x, position = carry
        wq, wk, wv, wo, wg, wu, wd, n_attn, n_mlp, n_q, n_k, keys, values = per_layer
        rows = x.shape[0]
        h = norm(x, n_attn)
        q = rope(norm((h @ wq).reshape(rows, heads, d), n_q), position)
        k = rope(norm((h @ wk).reshape(rows, kv_heads, d), n_k), position)
        v = (h @ wv).reshape(rows, kv_heads, d)
        keys = jax.lax.dynamic_update_slice(keys, k[:, :, None, :], (0, 0, position, 0))
        values = jax.lax.dynamic_update_slice(values, v[:, :, None, :], (0, 0, position, 0))
        qg = q.reshape(rows, kv_heads, group, d)
        scores = jnp.einsum("bkgd,bkcd->bkgc", qg, keys, preferred_element_type=jnp.float32) * d ** -0.5
        visible = jnp.arange(keys.shape[2]) <= position
        scores = jnp.where(visible[None, None, None, :], scores, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1).astype(values.dtype)
        attn = jnp.einsum("bkgc,bkcd->bkgd", probs, values).reshape(rows, heads * d)
        x = x + attn @ wo
        h = norm(x, n_mlp)
        x = x + (jax.nn.silu(h @ wg) * (h @ wu)) @ wd
        return (x, position), (keys, values)

    def step(params: dict[str, Any], tokens: Any, keys: Any, values: Any, position: Any) -> tuple[Any, Any, Any]:
        x = params["embed"][tokens]
        stacked = (params["wq"], params["wk"], params["wv"], params["wo"], params["wg"], params["wu"], params["wd"],
                   params["n_attn"], params["n_mlp"], params["n_q"], params["n_k"], keys, values)
        (x, _), (keys, values) = jax.lax.scan(layer, (x, position), stacked)
        scores = (norm(x, params["n_out"]) @ params["embed"].T).astype(jnp.float32)
        return (scores if logits else jnp.argmax(scores, axis=-1).astype(tokens.dtype)), keys, values

    return step


def make_chain(step: Any, count: int):
    """`count` steps inside one compiled loop, each fed the previous tokens, the position advancing."""
    import jax

    def chain(params: dict[str, Any], tokens: Any, keys: Any, values: Any, position: Any) -> tuple[Any, Any, Any]:
        def body(i: Any, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            tokens, keys, values = carry
            return step(params, tokens, keys, values, position + i)
        return jax.lax.fori_loop(0, count, body, (tokens, keys, values))

    return chain


def _median_s(run: Any, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def time_case(config: dict[str, Any], params: dict[str, Any], rows: int, capacity: int, position: int,
              repeats: int, warmup: int, chain: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    step = jax.jit(make_step(config), donate_argnums=(2, 3))
    chained = jax.jit(make_chain(make_step(config), chain), donate_argnums=(2, 3))
    tokens = jnp.zeros((rows,), jnp.int32)
    state = {"keys": None, "values": None}
    state["keys"], state["values"] = cache(config, rows, capacity)
    at = jnp.int32(position)

    def once() -> None:
        out, state["keys"], state["values"] = step(params, tokens, state["keys"], state["values"], at)
        out.block_until_ready()

    def many() -> None:
        out, state["keys"], state["values"] = chained(params, tokens, state["keys"], state["values"], at)
        out.block_until_ready()

    started = time.perf_counter()
    once()
    compile_s = time.perf_counter() - started
    per_call = _median_s(once, repeats, warmup)
    started = time.perf_counter()
    many()
    chain_compile_s = time.perf_counter() - started
    per_chain = _median_s(many, max(3, repeats // 4), max(1, warmup // 2))
    return {"rows": rows, "capacity": capacity, "position": position, "per_call_s": per_call,
            "chained_step_s": per_chain / chain, "chain": chain, "compile_s": compile_s,
            "chain_compile_s": chain_compile_s}


def probe(*, smoke: bool = False) -> dict[str, Any]:
    config = SMOKE if smoke else QWEN3_1_7B
    cases = SMOKE_CASES if smoke else CASES
    repeats, warmup, chain = (3, 1, 4) if smoke else (20, 3, CHAIN)
    report: dict[str, Any] = {"schema": SCHEMA, "smoke": smoke, "env": environment(), "config": config,
                              "cases": [list(c) for c in cases], "t4_reference": T4_REFERENCE, "results": []}
    try:
        params = weights(config)
    except Exception as error:  # noqa: BLE001 - a missing runtime is a finding, not a crash
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    for rows, capacity, position in cases:
        try:
            report["results"].append(time_case(config, params, rows, capacity, position, repeats, warmup, chain))
        except Exception as error:  # noqa: BLE001 - one case that does not fit is recorded, the others still run
            report["results"].append({"rows": rows, "capacity": capacity, "position": position,
                                      "error": f"{type(error).__name__}: {error}"[:500]})
    return report


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """Per case: the tokens per second each timing allows (rows over seconds per step), and the decision TP1 fixed:
    TP1 reopens only if the chained step beats both T4 references, at 1 row and at 8 rows."""
    out: dict[str, Any] = {"schema": SCHEMA + ".analysis", "env": report.get("env"), "smoke": report.get("smoke"),
                           "t4_reference": report.get("t4_reference"), "cases": {}}
    for row in report.get("results") or []:
        name = f"{row['rows']} rows, {row['capacity']} cache"
        if "error" in row:
            out["cases"][name] = {"error": row["error"]}
            continue
        out["cases"][name] = {"per_call_tok_s": row["rows"] / row["per_call_s"],
                              "chained_tok_s": row["rows"] / row["chained_step_s"],
                              "chained_step_ms": 1e3 * row["chained_step_s"], "per_call_ms": 1e3 * row["per_call_s"]}
    if report.get("smoke"):
        out["decision"] = "smoke: checks the tool, measures nothing"
        return out
    reference = report.get("t4_reference") or T4_REFERENCE
    one = out["cases"].get("1 rows, 2048 cache", {}).get("chained_tok_s")
    eight = out["cases"].get("8 rows, 2048 cache", {}).get("chained_tok_s")
    if one is None or eight is None:
        out["decision"] = "no decision: a case did not run"
    elif one > reference["float16_single_tok_s"] and eight > reference["dense_width8_tok_s"]:
        out["decision"] = "TP1 reopens: the next step is a JAX decode path, fixed in the backlog before it is built"
    else:
        out["decision"] = "TP1 stays closed: the chained step does not beat both T4 references"
    return out


_BOOTSTRAP = '''"""IronMule TP0b on this Kaggle TPU session; written by tools/tpu_step_probe.py at {commit}."""
from __future__ import annotations
'''


def write_kernel(directory: Path, owner: str, commit: str) -> Path:
    """A private TPU script kernel that runs this probe; the report lands as `tpu_step_probe_report.json`. The
    session's own JAX is used (TP0 ran JAX 0.11.0 there)."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    entry = f"\nsys.exit(main({['--out', '/kaggle/working/tpu_step_probe_report.json']!r}))\n"
    (directory / "run.py").write_text(_BOOTSTRAP.format(commit=commit) + body + entry)
    name = "ironmule-tpu-step-probe"
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
