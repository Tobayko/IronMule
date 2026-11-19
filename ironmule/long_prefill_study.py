"""Evidence contract for the opt-in split-KV 2/4 × 444 prompt experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

SCHEMA = "ironmule.long-prefill-native.v1"
SHA = re.compile(r"[0-9a-f]{64}")
WIDTHS = (8, 32)


def protocol_hash():
    from .ggml_engine import file_sha256

    return file_sha256(str(Path(__file__).with_name("long_prefill_gate.py")))


def uuid_hash(value):
    return hashlib.sha256(
        value.lower().removeprefix("gpu-").replace("-", "").encode()
    ).hexdigest()


def local_identity(engine=None):
    from ironmule_devices import ggml_devices, runtime_binding
    from .compare_learning import digest, implementation
    from .ggml_binary import identity
    from .kernel_gen import _visible_cuda_devices

    objects = identity().get("sha256")
    devices = _visible_cuda_devices()
    if not objects or len(devices) != 1:
        raise ValueError("missing native or physical device identity")
    return {
        "cuda_uuid": devices[0]["uuid"],
        "native_runtime_sha256": digest(
            [objects, runtime_binding(ggml_devices(engine.L if engine else None))]
        ),
        "implementation_sha256": implementation(),
    }


def qualified_receipt(
    receipt,
    *,
    gguf_sha256=None,
    cuda_uuid=None,
    native_runtime_sha256=None,
    implementation_sha256=None,
):
    """Recompute acceptance from paired observations and check local identities when supplied."""
    try:
        r = receipt
        if (
            r["schema"] != SCHEMA
            or r["status"] != "measured"
            or r["provider"] != "cuda"
            or (r["seqs"], r["seq_ctx"], r["ubatch"]) != (32, 2048, 512)
            or r["settings"].get("flash_attn") is not True
            or r["settings"].get("kv_unified") is not False
            or r["offloaded_layers"] != [65, 65]
            or r["protocol_sha256"] != protocol_hash()
            or any(
                not SHA.fullmatch(r[k])
                for k in (
                    "gguf_sha256",
                    "gpu_uuid_sha256",
                    "native_runtime_sha256",
                    "implementation_sha256",
                )
            )
        ):
            return False
        fields = {
            "gguf_sha256": gguf_sha256,
            "gpu_uuid_sha256": uuid_hash(cuda_uuid) if cuda_uuid else None,
            "native_runtime_sha256": native_runtime_sha256,
            "implementation_sha256": implementation_sha256,
        }
        if any(actual is not None and r[k] != actual for k, actual in fields.items()):
            return False
        gate = r["long_gate"]
        if (
            gate["protocol"] != "ironmule.long-prefill.v1"
            or gate["matches"] is not True
            or set(gate["shapes"]) != {"2x444", "4x444"}
        ):
            return False
        for width in (2, 4):
            v = gate["shapes"][f"{width}x444"]
            neutral = [{"calls": 0, "prompts": 0, "rows": 0}] * (8 // width)
            if any(v["witnesses"][arm] != neutral for arm in ("reference", "repeat")):
                return False
            cases = v["cases"]
            if (
                len({case["prompt_sha256"] for case in cases}) != 8
                or any(
                    case["prompt_tokens"] != 444
                    or not SHA.fullmatch(case["prompt_sha256"])
                    or case["reference"] != case["repeat"]
                    or case["reference"] != case["candidate"]
                    or not case["reference"]["tokens"]
                    or case["reference"]["stop"] not in ("length", "eos")
                    for case in cases
                )
                or any(
                    not any(
                        case["category"] == category
                        and case["reference"]["correct"] is True
                        for case in cases
                    )
                    for category in ("facts", "arithmetic", "extraction")
                )
            ):
                return False
            if (
                v["matches"] is not True
                or v["reasons"]
                or len(v["cases"]) != 8
                or v["witnesses"]["candidate"]
                != [{"calls": 1, "prompts": width, "rows": 444 * width}] * (8 // width)
            ):
                return False
        numerical = r["nll"]
        if (
            numerical["chunks"] < 4
            or not 0 < numerical["interval"][0] <= numerical["interval"][1] <= 1.005
            or not all(math.isfinite(x) for x in numerical["interval"])
            or r["factual"]["passes"] is not True
        ):
            return False
        base = {(s["repeat"], s["width"]): s for s in r["waves"] if s["mode"] == "off"}
        trial = {
            (s["repeat"], s["width"]): s for s in r["waves"] if s["mode"] == "long-444"
        }
        if (
            len(r["waves"]) != 12
            or set(base) != set(trial)
            or set(base) != {(i, w) for i in range(3) for w in WIDTHS}
        ):
            return False
        ratios = {w: [] for w in WIDTHS}
        for pair, a in base.items():
            b = trial[pair]
            if (
                not SHA.fullmatch(a["outputs_sha256"])
                or not SHA.fullmatch(a["prompts_sha256"])
                or a["outputs_sha256"] != b["outputs_sha256"]
                or a["prompts_sha256"] != b["prompts_sha256"]
                or a["output_tokens"] != b["output_tokens"]
                or a["output_tokens"] <= 0
                or any(a["hits"].values())
                or set(b["hits"]) != {"2", "4"}
                or any(type(x) is not int or x < 0 for x in b["hits"].values())
                or not any(b["hits"].values())
                or not all(
                    math.isfinite(s["seconds"]) and s["seconds"] > 0 for s in (a, b)
                )
            ):
                return False
            ratios[pair[1]].append(a["seconds"] / b["seconds"])
        med = [statistics.median(v) for v in ratios.values()]
        return min(med) >= 1.0 and max(med) >= 1.03
    except (TypeError, KeyError, IndexError, ValueError, AttributeError, OSError, OverflowError):
        return False


def measured_shapes(gate, waves=None):
    """Read only the rectangles admitted and actually hit by the current long-only protocol."""
    shapes = {}
    for name, evidence in gate["shapes"].items():
        width, length = map(int, name.split("x"))
        hit = waves is None or any(
            row["mode"] == "long-444" and row["hits"][str(width)] > 0 for row in waves
        )
        if evidence["matches"] and evidence["calls"] > 0 and hit:
            shapes.setdefault(length, []).append(width)
    return {length: tuple(sorted(widths)) for length, widths in shapes.items()}


def enable_worker(engine, tokenizer, gguf_sha256, gate):
    """Enable only on the received native profile's original split context after an in-worker cold holdout."""
    path = os.environ.get("IRONMULE_GGML_PREFILL_RECEIPT")
    if not (
        path
        and gate
        and gate.get("passes") is True
        and gate.get("streams_serving")
        and engine.prefill_batch_long
        and not engine.shared
    ):
        return False
    try:
        with open(path, "rb") as f:
            data = f.read(1_000_001)
        if len(data) > 1_000_000:
            return False
        receipt = json.loads(data)
        if (
            not qualified_receipt(
                receipt, gguf_sha256=gguf_sha256, **local_identity(engine)
            )
            or (engine.seqs, engine.seq_ctx, engine.threads)
            != (receipt["seqs"], receipt["seq_ctx"], receipt["threads"])
            or engine.settings != receipt["settings"]
            or engine.L.llama_n_ubatch(engine.ctx) != receipt["ubatch"]
        ):
            return False
        from .long_prefill_gate import check

        engine.prefill_batch = True
        passed = False
        try:
            passed = check(engine, tokenizer)["matches"] is True
            if passed:
                engine.prefill_long_shapes = measured_shapes(
                    receipt["long_gate"], receipt["waves"]
                )
            return passed
        finally:
            engine.prefill_batch = passed
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return False
