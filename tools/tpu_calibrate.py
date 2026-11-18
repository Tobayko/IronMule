"""Select a JAX decode-loop length for a synthetic TP0b shape on this machine.

    python tools/tpu_calibrate.py --profile-dir /tmp/tpu-plans --rows 1 --budget-seconds 240
    python tools/tpu_calibrate.py --profile-dir /tmp/tpu-plans --smoke --budget-seconds 60

A worker warms the real compiled loops, checks their decoded sequences and both KV tensors,
then alternates their timings. One matching cached decision avoids repeating calibration;
the selected executable still runs. Each invocation is bounded, including compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import tpu_step_probe as probe

SCHEMA = "ironmule.tpu_calibration.v1"


def valid_timings(pairs):
    return (isinstance(pairs, list) and len(pairs) == 4 and all(
        isinstance(pair, dict) and all(type(pair.get(mode)) in (int, float)
                                       and math.isfinite(pair[mode]) and pair[mode] > 0
                                       for mode in ("single", "chained")) for pair in pairs))


def choose(pairs: list[dict], matched: bool) -> str:
    """Each of four pairs must save at least 5% for a chained loop to win."""
    if matched is not True or not valid_timings(pairs):
        return "single"
    return "chained" if all(p["chained"] <= 0.95 * p["single"] for p in pairs) else "single"


def binding(jax, device, config, shape, steps, smoke):
    import jaxlib

    driver = getattr(device.client, "platform_version", "")
    # Host identity prevents a copy of the profile directory from licensing an unmeasured TPU.
    code = hashlib.sha256(Path(__file__).read_bytes() + Path(probe.__file__).read_bytes()).hexdigest()
    return {"schema": SCHEMA, "host": platform.node(), "platform": device.platform, "model": device.device_kind,
            "device": str(device), "jax": jax.__version__, "jaxlib": jaxlib.__version__, "driver": str(driver),
            "local_devices": jax.local_device_count(), "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "libtpu_flags": os.environ.get("LIBTPU_INIT_ARGS", ""), "config": config, "dtype": "bfloat16",
            "shape": list(shape), "steps": steps, "synthetic": True, "smoke": smoke, "code": code}


def restore(path: Path, identity: dict):
    try:
        saved = json.loads(path.read_text())
        if saved["binding"] != identity or type(saved["matched"]) is not bool:
            return None
        pairs = saved["pairs"]
        if not (valid_timings(pairs) if saved["matched"] else pairs == []):
            return None
        return saved if saved["choice"] == choose(pairs, saved["matched"]) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    import jax
    import jax.numpy as jnp

    device = jax.local_devices()[0]
    if not args.smoke and device.platform != "tpu":
        raise ValueError("full calibration requires a TPU; use --smoke for a tiny local check")
    config = probe.SMOKE if args.smoke else probe.QWEN3_1_7B
    cases = probe.SMOKE_CASES if args.smoke else probe.CASES
    shape = next((case for case in cases if case[0] == args.rows), None)
    if shape is None:
        raise ValueError(f"supported row counts on this protocol: {[case[0] for case in cases]}")
    rows, capacity, position = shape
    identity = binding(jax, device, config, shape, args.steps, args.smoke)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = args.profile_dir / (key + ".json")
    previous = None if args.recheck else restore(path, identity)
    with jax.default_device(device):
        params = probe.weights(config)
        keys, values = probe.cache(config, rows, capacity)
        # Earlier positions contain existing context so cause/effect in the KV update is visible.
        keys = keys.at[:, :, :, position - 1].set(jnp.asarray(0.125, jnp.bfloat16))
        values = values.at[:, :, :, position - 1].set(jnp.asarray(-0.25, jnp.bfloat16))
        initial = jnp.arange(rows, dtype=jnp.int32) + 1
        at = jnp.int32(position)
        positions = tuple(jax.block_until_ready(at + offset) for offset in range(args.steps))
        plain = probe.make_step(config)
        one = jax.jit(plain)

        def loop(params, tokens, keys, values, at):
            def body(carry, offset):
                state = plain(params, *carry, at + offset)
                return state, state[0]
            state, trajectory = jax.lax.scan(body, (tokens, keys, values), jnp.arange(args.steps))
            return trajectory, state[1], state[2]

        many = jax.jit(loop)

        def execute(choice, tokens):
            if choice == "chained":
                result = many(params, tokens, keys, values, at)
            else:
                k, v = keys, values
                trajectory = []
                for pos in positions:
                    tokens, k, v = one(params, tokens, k, v, pos)
                    tokens.block_until_ready()
                    trajectory.append(tokens)
                result = (jnp.stack(trajectory), k, v)
            return jax.block_until_ready(result)

        def duration(choice, tokens):
            start = time.perf_counter()
            execute(choice, tokens)
            return time.perf_counter() - start

        if previous is None:
            a = execute("single", initial)
            b = execute("chained", initial)
            matched = all(bool(jnp.array_equal(x, y)) for x, y in zip(a, b))
            del a, b
            pairs = []
            if matched:
                for order in (("single", "chained"), ("chained", "single")) * 2:
                    pairs.append({mode: duration(mode, initial) for mode in order})
            record = {"binding": identity, "pairs": pairs, "matched": matched, "choice": choose(pairs, matched)}
        else:
            record = previous
        # Verify the selected program actually executes on fresh inputs. For the chained candidate check
        # the alternate trajectory before caching its decision and whenever restoring it on a subsequent run.
        held = initial + 19
        selected = record["choice"]
        start = time.perf_counter()
        output = execute(selected, held)
        elapsed = time.perf_counter() - start
        if selected == "chained":
            reference = execute("single", held)
            if not all(bool(jnp.array_equal(x, y)) for x, y in zip(output, reference)):
                record = {**record, "choice": "single", "matched": False, "pairs": [],
                          "reason": "held-out trajectory or KV differed"}
                selected = "single"
                elapsed = duration(selected, held)
        if previous is None or record != previous:
            save(path, record)
    return {"profile": str(path), "restored": previous is not None, "selected": selected, "matched": record["matched"],
            "pairs": record["pairs"], "selected_run_s_including_compilation": elapsed, "smoke": args.smoke,
            "backend": device.platform, "hardware": device.device_kind, "serving": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--steps", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--budget-seconds", type=int, default=240)
    parser.add_argument("--recheck", action="store_true", help="repeat a cached comparison")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.budget_seconds < 1:
        parser.error("--budget-seconds must be positive")
    if args.worker:
        try:
            print(json.dumps(run(args), sort_keys=True))
        except (ImportError, ValueError, RuntimeError, OSError) as error:
            print(json.dumps({"status": "unavailable", "error": str(error)}))
            return 1
        return 0
    try:
        return subprocess.run([sys.executable, __file__, *argv, "--worker"], timeout=args.budget_seconds).returncode
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "timeout", "budget_seconds": args.budget_seconds}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
