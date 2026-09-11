"""Child process for the isolated stock ``mlx_lm`` reference backend."""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import queue
import resource
import secrets
import sys
import threading
import time
from typing import Any
from types import SimpleNamespace


if __package__ in (None, ""):
    _PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, _PACKAGE_ROOT)

from ironmule_product.model_policy import ModelPolicyError, validate_model_config


PROTOCOL_VERSION = 1
MAX_LINE = 1024 * 1024
CONTEXT_LIMIT = 8192
WORKER_VARIANTS = frozenset(("reference", "bounded_prefetch", "prefix_reuse", "current_engine", "automatic"))
_PROTOCOL_OUT = sys.stdout


def _emit(value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_LINE:
        # This message itself is bounded and contains no request text.
        encoded = json.dumps({"type": "error", "code": "protocol_error", "message": "event too large"}, separators=(",", ":"))
    _PROTOCOL_OUT.write(encoded + "\n")
    _PROTOCOL_OUT.flush()


def _startup_telemetry(mx: Any, startup_started: float) -> dict[str, Any]:
    values: dict[str, Any] = {
        "startup_wall_seconds": time.monotonic() - startup_started,
        # Darwin reports ru_maxrss in bytes (unlike Linux's KiB convention).
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "recommended_working_set_bytes": mx.device_info().get("max_recommended_working_set_size"),
    }
    startup = values["startup_wall_seconds"]
    try:
        startup_float = float(startup)
    except (OverflowError, TypeError, ValueError):
        startup_float = math.inf
    if (isinstance(startup, bool) or not isinstance(startup, (int, float))
            or not math.isfinite(startup_float) or startup <= 0):
        raise RuntimeError("invalid startup telemetry: startup_wall_seconds")
    byte_values = {key: value for key, value in values.items() if key != "startup_wall_seconds"}
    if any(type(value) is not int or not 0 <= value <= 2**63 - 1 for value in byte_values.values()):
        raise RuntimeError("invalid startup telemetry byte field")
    if byte_values["process_peak_rss_bytes"] <= 0 or byte_values["recommended_working_set_bytes"] <= 0:
        raise RuntimeError("invalid startup telemetry byte field")
    if values["mlx_peak_bytes"] < values["mlx_active_bytes"]:
        raise RuntimeError("invalid startup telemetry memory ordering")
    return values


def _automatic_batch_qualified(evidence: list[dict[str, Any]], identity: dict[str, Any]) -> bool:
    """Report only current, independently qualified multi-request evidence."""
    from ironmule_product.selection import IntegrationEvidence, SelectionContractError

    for value in evidence:
        try:
            row = IntegrationEvidence.from_mapping(value)
        except (SelectionContractError, TypeError, AttributeError):
            continue
        if (row.proves_improvement and row.profile == "throughput"
                and row.model_id == identity["model_id"] and row.model_revision == identity["revision"]
                and row.model_sha256 == identity["model_sha256"]
                and row.hardware_sha256 == identity["hardware_sha256"]
                and row.environment_sha256 == identity["environment_sha256"]
                and row.code_sha256 == identity["code_sha256"]
                and row.source_manifest_sha256 == identity["code_sha256"]
                and row.max_session_requests >= 2 and any(width > 1 for width in row.group_widths)):
            return True
    return False


def _safe_error(code: str, request_id: str | None = None) -> None:
    event: dict[str, Any] = {"type": "error", "code": code, "message": "stock MLX backend unavailable"}
    if request_id is not None:
        event["request_id"] = request_id
    _emit(event)


def _safe_batch_error(code: str, batch_id: str | None = None) -> None:
    event: dict[str, Any] = {"type": "error", "code": code, "message": "grouped engine backend unavailable"}
    if batch_id is not None:
        event["batch_id"] = batch_id
    _emit(event)


def _enqueue_terminal(commands: queue.Queue[dict[str, Any] | None], value: dict[str, Any] | None) -> None:
    """Deliver EOF/shutdown without blocking on a full bounded queue."""
    try:
        commands.put_nowait(value)
    except queue.Full:
        # Shutdown/EOF supersedes queued work. Drop one bounded command so the
        # main loop can always observe the terminal marker and exit cleanly.
        try:
            commands.get_nowait()
        except queue.Empty:
            pass
        try:
            commands.put_nowait(value)
        except queue.Full:
            pass


def _load(spec: dict[str, Any]):
    path = Path(spec["snapshot_path"])
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError("local model snapshot is unavailable")
    expected_weights = spec.get("weight_bytes")
    if type(expected_weights) is not int or expected_weights <= 0:
        raise RuntimeError("model weight registration is invalid")
    weight_files = tuple(path.glob("model*.safetensors"))
    if not weight_files or any(not file.is_file() for file in weight_files):
        raise RuntimeError("local model weights are unavailable")
    actual_weights = sum(file.stat().st_size for file in weight_files)
    if actual_weights != expected_weights:
        raise RuntimeError("local model weight registration changed")
    model_config = validate_model_config(str(path), spec.get("revision"))
    # Opening the MLX device is intentionally the first native operation and
    # happens before importing mlx_lm.  A headless host may terminate here;
    # the parent converts that into BackendUnavailable.
    import mlx.core as mx

    # A compiled-in Metal backend is not proof that the current process may
    # open the device. Force that check before mlx_lm creates native streams.
    if not mx.metal.is_available():
        raise RuntimeError("a Metal GPU device is required")
    device_info = mx.device_info()
    max_working_set = device_info.get("max_recommended_working_set_size")
    if type(max_working_set) is not int or max_working_set <= 0:
        raise RuntimeError("device working-set telemetry is unavailable")
    # This is an Apple recommendation, not a hard hardware capacity. Observe
    # it without refusing a model on this heuristic; the OS retains its limits.
    device = mx.gpu
    mx.set_default_device(device)
    with redirect_stdout(sys.stderr):
        from mlx_lm import load, stream_generate

        model, tokenizer = load(
            str(path),
            tokenizer_config={"trust_remote_code": False},
            model_config=model_config,
            revision=spec["revision"],
        )
    return model, tokenizer, stream_generate, device


def _receiver(
    commands: queue.Queue[dict[str, Any] | None],
    cancellations: dict[str, threading.Event],
    pending_cancellations: set[str],
    lock: threading.Lock,
) -> None:
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            # EOF must eventually reach the main loop even when command
            # admission is momentarily full; this is a bounded wait on the
            # queue, not an unbounded command backlog.
            _enqueue_terminal(commands, None)
            return
        if len(line) > MAX_LINE:
            _enqueue_terminal(commands, {"type": "invalid"})
            _enqueue_terminal(commands, None)
            return
        try:
            command = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                commands.put_nowait({"type": "invalid"})
            except queue.Full:
                _safe_error("overloaded")
            continue
        if not isinstance(command, dict) or not isinstance(command.get("type"), str):
            try:
                commands.put_nowait({"type": "invalid"})
            except queue.Full:
                _safe_error("overloaded")
            continue
        if command.get("type") == "shutdown":
            _enqueue_terminal(commands, command)
            return
        if command.get("type") == "cancel":
            request_id = command.get("request_id")
            if isinstance(request_id, str) and request_id and len(request_id) <= 256:
                with lock:
                    event = cancellations.get(request_id)
                    if event is not None:
                        event.set()
                    elif len(pending_cancellations) < 64:
                        # The command may still be waiting in the bounded
                        # queue.  Retain the cancellation so the main loop
                        # can set the event before tokenization/generation.
                        pending_cancellations.add(request_id)
            continue
        try:
            commands.put_nowait(command)
        except queue.Full:
            _safe_error("overloaded")


def _generate(
    command: dict[str, Any],
    model_id: str,
    model: Any,
    tokenizer: Any,
    stream_generate: Any,
    cancel_event: threading.Event,
    variant: str,
    prefix_session: Any = None,
    engine_bridge: Any = None,
    automatic_runtime: Any = None,
) -> dict[str, Any] | None:
    request_id = command.get("request_id") if isinstance(command.get("request_id"), str) else None
    if request_id is None or not request_id or len(request_id) > 256:
        _safe_error("invalid_request")
        return None
    generated = None
    terminal: dict[str, Any] | None = None
    error_emitted = False
    engine_result = None
    selection_metadata = None

    # Request validation and chat-template rendering are user-input failures.
    # Keep the loaded worker alive when either rejects a request.
    try:
        if command.get("model") != model_id:
            raise ValueError
        if variant not in WORKER_VARIANTS or command.get("variant") != variant:
            raise ValueError
        if (command.get("temperature", 0.0) != 0.0
                or command.get("top_p", 1.0) != 1.0
                or type(command.get("stream", False)) is not bool):
            raise ValueError
        seed = command.get("seed")
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError
            import mlx.core as mx
            mx.random.seed(seed)
        messages = command.get("messages")
        max_tokens = command.get("max_tokens")
        if not isinstance(messages, list) or not messages or type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError
        if any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError
        # Tokenize the rendered template exactly once; stream_generate receives
        # IDs and therefore does not apply the chat template a second time.
        with redirect_stdout(sys.stderr):
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
            raise ValueError
        prompt_tokens = len(prompt_ids)
        if prompt_tokens + max_tokens > CONTEXT_LIMIT:
            raise ValueError
    except Exception:
        _safe_error("invalid_request", request_id)
        return None

    try:
        if cancel_event.is_set():
            terminal = {"type": "done", "request_id": request_id, "finish_reason": "cancelled",
                        "prompt_tokens": prompt_tokens, "completion_tokens": 0, "variant": variant}
        else:
            with redirect_stdout(sys.stderr):
                if variant == "prefix_reuse":
                    if prefix_session is None:
                        raise ValueError("prefix cache is not enabled")
                    generated = prefix_session.stream(list(prompt_ids), max_tokens=max_tokens)
                elif variant == "current_engine":
                    if engine_bridge is None:
                        raise ValueError("current engine is not enabled")
                    engine_result = engine_bridge.generate(list(prompt_ids), max_tokens)
                    # Engine.serve is completion-only. Buffer honestly; do not
                    # invent token streaming, TTFT, or log probabilities.
                    generated = (SimpleNamespace(
                        token=token, text=engine_result.text if i + 1 == engine_result.token_count else "",
                        prompt_tokens=prompt_tokens, generation_tokens=i + 1,
                        finish_reason=engine_result.finish_reason if i + 1 == engine_result.token_count else None,
                    ) for i, token in enumerate(engine_result.tokens))
                elif variant == "automatic":
                    if automatic_runtime is None:
                        raise ValueError("automatic runtime is not enabled")
                    decision = automatic_runtime.choose(list(prompt_ids), max_tokens, stream=False,
                                                        session_requests=1, group_width=1)
                    selected = decision.candidate_id
                    selection_metadata = {**(automatic_runtime.last_decision or {}),
                                          "actual_backend": selected, "actual_group_width": 1}
                    if selected == "reference":
                        generated = stream_generate(model, tokenizer, list(prompt_ids), max_tokens=max_tokens)
                    elif selected == "prefix_reuse":
                        generated = automatic_runtime.prefix_session.stream(list(prompt_ids), max_tokens=max_tokens)
                    else:
                        engine_result = automatic_runtime.engine_for(selected).generate(list(prompt_ids), max_tokens)
                        generated = (SimpleNamespace(
                            token=token, text=engine_result.text if i + 1 == engine_result.token_count else "",
                            prompt_tokens=prompt_tokens, generation_tokens=i + 1,
                            finish_reason=engine_result.finish_reason if i + 1 == engine_result.token_count else None,
                        ) for i, token in enumerate(engine_result.tokens))
                else:
                    generated = stream_generate(model, tokenizer, list(prompt_ids), max_tokens=max_tokens)
                for response in generated:
                    response_prompt = getattr(response, "prompt_tokens", prompt_tokens)
                    response_completion = getattr(response, "generation_tokens", None)
                    if (type(response_prompt) is not int or response_prompt != prompt_tokens
                            or type(response_completion) is not int or not 1 <= response_completion <= max_tokens):
                        raise ValueError
                    if cancel_event.is_set():
                        terminal = {
                            "type": "done", "request_id": request_id, "finish_reason": "cancelled",
                            "prompt_tokens": prompt_tokens, "completion_tokens": response_completion,
                            "variant": variant,
                        }
                        break
                    token = getattr(response, "token", None)
                    text = getattr(response, "text", "")
                    if type(token) is not int or token < 0 or not isinstance(text, str):
                        raise ValueError
                    event: dict[str, Any] = {
                        "type": "token", "request_id": request_id, "text": text, "token_id": token,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": response_completion,
                    }
                    _emit(event)
                    finish = getattr(response, "finish_reason", None)
                    if finish is not None:
                        if finish not in ("stop", "length"):
                            raise ValueError
                        metrics = {}
                        for key in ("prompt_tps", "generation_tps", "peak_memory"):
                            value = getattr(response, key, None)
                            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                                metrics[key] = float(value)
                        terminal = {
                            "type": "done", "request_id": request_id,
                            "finish_reason": finish,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": response_completion,
                            "variant": variant,
                            "metrics": metrics,
                        }
                        break
            if terminal is None:
                raise ValueError("generation ended without a completion frame")
    except Exception:
        terminal = None
        _safe_error("generation_error", request_id)
        error_emitted = True
    finally:
        if generated is not None:
            try:
                generated.close()
            except Exception:
                terminal = None
                if not error_emitted:
                    _safe_error("generation_error", request_id)
    if terminal is not None and engine_result is not None:
        terminal["engine"] = {**engine_result.metadata, "delivery": "buffered_completion",
                              "computed_tokens": engine_result.token_count}
    if terminal is not None and selection_metadata is not None:
        terminal["selection"] = selection_metadata
    if terminal is not None and command.get("trace_prompt_identity") is True:
        terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(list(prompt_ids),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return terminal


def _render_engine_request(command: dict[str, Any], model_id: str, tokenizer: Any) -> tuple[str, list[int], int]:
    """Apply exactly the single-request identity and rendering validation."""
    request_id = command.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise ValueError("invalid request id")
    if command.get("model") != model_id or command.get("temperature", 0.0) != 0.0 or command.get("top_p", 1.0) != 1.0:
        raise ValueError("invalid exact request")
    if command.get("stream", False) is not False:
        raise ValueError("batch streaming is unavailable")
    seed = command.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("invalid seed")
    messages, max_tokens = command.get("messages"), command.get("max_tokens")
    if not isinstance(messages, list) or not messages or type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ValueError("invalid request")
    if any(not isinstance(message, dict) or set(message) != {"role", "content"}
           or message["role"] not in {"system", "user", "assistant"}
           or not isinstance(message["content"], str) for message in messages):
        raise ValueError("invalid messages")
    with redirect_stdout(sys.stderr):
        raw_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("invalid prompt ids")
    ids = list(raw_ids)
    if len(ids) + max_tokens > CONTEXT_LIMIT:
        raise ValueError("context limit")
    return request_id, ids, max_tokens


def _run_engine_batch(command: dict[str, Any], model_id: str, tokenizer: Any,
                      engine_bridge: Any, cancellations: dict[str, threading.Event],
                      pending_cancellations: set[str], lock: threading.Lock,
                      *, rendered_override: list[tuple[str, list[int], int]] | None = None,
                      selection: dict[str, Any] | None = None) -> None:
    batch_id = command.get("batch_id")
    requests = command.get("requests")
    if (not isinstance(batch_id, str) or not 1 <= len(batch_id) <= 256
            or command.get("variant") not in {"current_engine", "automatic"} or not isinstance(requests, list)
            or not 2 <= len(requests) <= 32):
        _safe_batch_error("invalid_request", batch_id if isinstance(batch_id, str) else None)
        return
    registered: list[str] = []
    try:
        rendered = (rendered_override if rendered_override is not None else
                    [_render_engine_request(request, model_id, tokenizer) for request in requests])
        request_ids = [row[0] for row in rendered]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request id")
        with lock:
            for request_id in request_ids:
                event = threading.Event()
                if request_id in pending_cancellations:
                    event.set()
                    pending_cancellations.discard(request_id)
                cancellations[request_id] = event
                registered.append(request_id)
            pre_cancelled = {request_id for request_id in request_ids if cancellations[request_id].is_set()}
        active = [row for row in rendered if row[0] not in pre_cancelled]
        with redirect_stdout(sys.stderr):
            results = (engine_bridge.generate_many([row[1] for row in active], [row[2] for row in active])
                       if active else [])
        if not isinstance(results, list) or len(results) != len(active):
            raise RuntimeError("invalid grouped result count")
        result_by_id = {row[0]: result for row, result in zip(active, results)}
        for request_id, prompt_ids, limit in rendered:
            if request_id in pre_cancelled:
                terminal = {"type": "done", "batch_id": batch_id, "request_id": request_id,
                            "finish_reason": "cancelled", "prompt_tokens": len(prompt_ids),
                            "completion_tokens": 0, "variant": command["variant"], "metrics": {},
                            "engine": {**engine_bridge.metadata(), "delivery": "buffered_completion",
                                       "computed_tokens": 0, "skipped": "cancelled_before_generation"}}
                if command.get("trace_prompt_identity") is True:
                    terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(prompt_ids, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                if selection is not None:
                    terminal["selection"] = selection
                _emit(terminal)
                continue
            result = result_by_id[request_id]
            tokens = result.tokens
            if (not isinstance(tokens, (list, tuple)) or type(result.token_count) is not int
                    or result.token_count != len(tokens) or not 1 <= result.token_count <= limit
                    or result.finish_reason not in {"stop", "length"} or not isinstance(result.text, str)
                    or not isinstance(result.metadata, dict)):
                raise RuntimeError("invalid grouped engine result")
            emitted_tokens = 0
            for index, token in enumerate(tokens):
                if type(token) is not int or token < 0:
                    raise RuntimeError("invalid grouped token")
                with lock:
                    cancelled = cancellations[request_id].is_set()
                if cancelled:
                    break
                _emit({"type": "token", "batch_id": batch_id, "request_id": request_id,
                       "text": result.text if index + 1 == result.token_count else "", "token_id": token,
                       "prompt_tokens": len(prompt_ids), "completion_tokens": index + 1})
                emitted_tokens += 1
            with lock:
                cancelled = cancellations[request_id].is_set()
            terminal = {"type": "done", "batch_id": batch_id, "request_id": request_id,
                        "finish_reason": "cancelled" if cancelled else result.finish_reason,
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": emitted_tokens if cancelled else result.token_count,
                        "variant": command["variant"], "metrics": {},
                        "engine": {**result.metadata, "delivery": "buffered_completion",
                                   "computed_tokens": result.token_count}}
            if command.get("trace_prompt_identity") is True:
                terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(prompt_ids, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            if selection is not None:
                terminal["selection"] = selection
            _emit(terminal)
        _emit({"type": "batch_done", "batch_id": batch_id, "request_ids": request_ids})
    except ValueError:
        _safe_batch_error("invalid_request", batch_id)
    except Exception:
        _safe_batch_error("generation_error", batch_id)
    finally:
        with lock:
            for request_id in registered:
                cancellations.pop(request_id, None)


def _run_automatic_batch(command: dict[str, Any], model_id: str, model: Any, tokenizer: Any,
                         stream_generate: Any, automatic_runtime: Any,
                         cancellations: dict[str, threading.Event], pending_cancellations: set[str],
                         lock: threading.Lock) -> None:
    batch_id, requests = command.get("batch_id"), command.get("requests")
    if (not isinstance(batch_id, str) or not 1 <= len(batch_id) <= 256
            or command.get("variant") != "automatic" or not isinstance(requests, list)
            or not 2 <= len(requests) <= 32):
        _safe_batch_error("invalid_request", batch_id if isinstance(batch_id, str) else None)
        return
    registered: list[str] = []
    try:
        rendered = [_render_engine_request(request, model_id, tokenizer) for request in requests]
        request_ids = [row[0] for row in rendered]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request id")
        decision = automatic_runtime.choose_many(
            [row[1] for row in rendered], [row[2] for row in rendered], stream=False,
            session_requests=len(rendered), group_width=min(4, len(rendered)),
        )
        selected = decision.candidate_id
        actual_width = min(4, len(rendered)) if selected.endswith("throughput") else 1
        selection = {**(automatic_runtime.last_decision or {}), "actual_backend": selected,
                     "actual_group_width": actual_width}
        if selected not in {"reference", "prefix_reuse"}:
            bridge = automatic_runtime.engine_for(selected)
            _run_engine_batch(command, model_id, tokenizer, bridge, cancellations,
                              pending_cancellations, lock, rendered_override=rendered,
                              selection=selection)
            return
        with lock:
            for request_id in request_ids:
                event = threading.Event()
                if request_id in pending_cancellations:
                    event.set()
                    pending_cancellations.discard(request_id)
                cancellations[request_id] = event
                registered.append(request_id)
        for request_id, prompt_ids, limit in rendered:
            with lock:
                pre_cancelled = cancellations[request_id].is_set()
            if pre_cancelled:
                terminal = {"type": "done", "batch_id": batch_id, "request_id": request_id,
                            "finish_reason": "cancelled", "prompt_tokens": len(prompt_ids),
                            "completion_tokens": 0, "variant": "automatic", "metrics": {},
                            "engine": {"delivery": "buffered_completion", "computed_tokens": 0,
                                       "skipped": "cancelled_before_generation"}, "selection": selection}
            else:
                generated = None
                terminal = None
                emitted = 0
                try:
                    with redirect_stdout(sys.stderr):
                        generated = (automatic_runtime.prefix_session.stream(prompt_ids, max_tokens=limit)
                                     if selected == "prefix_reuse" else
                                     stream_generate(model, tokenizer, prompt_ids, max_tokens=limit))
                        for response in generated:
                            completion = getattr(response, "generation_tokens", None)
                            token, text = getattr(response, "token", None), getattr(response, "text", "")
                            if (type(completion) is not int or not 1 <= completion <= limit
                                    or type(token) is not int or token < 0 or not isinstance(text, str)):
                                raise RuntimeError("invalid automatic generation result")
                            with lock:
                                cancelled = cancellations[request_id].is_set()
                            if cancelled:
                                break
                            _emit({"type": "token", "batch_id": batch_id, "request_id": request_id,
                                   "text": text, "token_id": token, "prompt_tokens": len(prompt_ids),
                                   "completion_tokens": completion})
                            emitted = completion
                            finish = getattr(response, "finish_reason", None)
                            if finish is not None:
                                if finish not in {"stop", "length"}:
                                    raise RuntimeError("invalid automatic finish reason")
                                terminal = {"type": "done", "batch_id": batch_id, "request_id": request_id,
                                            "finish_reason": finish, "prompt_tokens": len(prompt_ids),
                                            "completion_tokens": emitted, "variant": "automatic", "metrics": {},
                                            "engine": {"delivery": "buffered_completion",
                                                       "computed_tokens": emitted}, "selection": selection}
                                break
                    with lock:
                        cancelled = cancellations[request_id].is_set()
                        if selected == "prefix_reuse":
                            prefix_metadata = asdict(automatic_runtime.prefix_session.finalize(
                                not cancelled and terminal is not None))
                    if cancelled:
                        terminal = {"type": "done", "batch_id": batch_id, "request_id": request_id,
                                    "finish_reason": "cancelled", "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": emitted, "variant": "automatic", "metrics": {},
                                    "engine": {"delivery": "buffered_completion",
                                               "computed_tokens": emitted}, "selection": selection}
                    if terminal is None:
                        raise RuntimeError("automatic generation ended without completion")
                    if selected == "prefix_reuse":
                        terminal["prefix_cache"] = prefix_metadata
                        terminal["engine"]["prefix_cache"] = prefix_metadata
                finally:
                    if generated is not None:
                        try:
                            generated.close()
                        except Exception:
                            pass
            if command.get("trace_prompt_identity") is True:
                terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(prompt_ids, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            _emit(terminal)
        _emit({"type": "batch_done", "batch_id": batch_id, "request_ids": request_ids})
    except ValueError:
        _safe_batch_error("invalid_request", batch_id)
    except Exception:
        _safe_batch_error("generation_error", batch_id)
    finally:
        with lock:
            for request_id in registered:
                cancellations.pop(request_id, None)


def _run_generation(
    command: dict[str, Any],
    model_id: str,
    model: Any,
    tokenizer: Any,
    stream_generate: Any,
    cancel_event: threading.Event,
    variant: str,
    trace_forwards: bool,
    prefix_session: Any = None,
    engine_bridge: Any = None,
    automatic_runtime: Any = None,
    cancellation_lock: Any = None,
) -> None:
    """Run one request, installing the candidate only inside this worker."""
    generate_module = None
    original_generate_step = None
    model_type = None
    original_model_call = None
    forward_count = 0
    terminal = None
    if variant == "bounded_prefetch":
        # worker.py is launched by absolute path, so its package directory is
        # sys.path[0] rather than the package root.  Add that root for the
        # optional candidate import; this mutation is isolated to this child.
        package_root = str(Path(__file__).resolve().parent.parent)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from ironmule_product.greedy import _check_stock_version, bounded_generate_step

        try:
            _check_stock_version()
        except Exception:
            _safe_error("variant_unavailable", command.get("request_id") if isinstance(command.get("request_id"), str) else None)
            return

        import importlib

        generate_module = importlib.import_module("mlx_lm.generate")
        original_generate_step = generate_module.generate_step
        generate_module.generate_step = bounded_generate_step

    if trace_forwards:
        model_type = type(model)
        original_model_call = model_type.__call__

        def counted_model_call(instance: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal forward_count
            if instance is model:
                forward_count += 1
            return original_model_call(instance, *args, **kwargs)

        model_type.__call__ = counted_model_call
    try:
        terminal = _generate(command, model_id, model, tokenizer, stream_generate, cancel_event, variant,
                             prefix_session, engine_bridge, automatic_runtime)
    finally:
        if generate_module is not None:
            generate_module.generate_step = original_generate_step
        if model_type is not None and original_model_call is not None:
            model_type.__call__ = original_model_call
    if terminal is not None:
        selected_prefix = (variant == "prefix_reuse" or variant == "automatic"
                           and terminal.get("selection", {}).get("candidate_id") == "prefix_reuse")
        if selected_prefix and prefix_session is not None:
            # The receiver sets cancellation under this same lock. Publishing
            # the cache and selecting normal/cancelled completion are one
            # decision; later cancels cannot relabel this model completion.
            with cancellation_lock if cancellation_lock is not None else nullcontext():
                if cancel_event.is_set():
                    terminal["finish_reason"] = "cancelled"
                accepted = terminal["finish_reason"] in ("stop", "length")
                terminal["prefix_cache"] = asdict(prefix_session.finalize(accepted))
        if trace_forwards:
            terminal["model_forward_invocations"] = forward_count
        _emit(terminal)
    elif (variant == "prefix_reuse" or variant == "automatic") and prefix_session is not None:
        prefix_session.finalize(False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--execution-variant", choices=sorted(WORKER_VARIANTS), default="reference")
    parser.add_argument("--engine-configuration", default="current_profile",
                        choices=("current_profile", "baseline_interactive", "core_interactive",
                                 "baseline_throughput", "core_throughput"))
    parser.add_argument("--prefix-cache-max-entries", type=int, default=4)
    parser.add_argument("--prefix-cache-max-bytes", type=int, default=1024**3)
    parser.add_argument("--selection-evidence", default="[]")
    args = parser.parse_args(argv)
    startup_started = time.monotonic()
    prefix_session = engine_bridge = automatic_runtime = None
    try:
        spec = json.loads(args.spec)
        selection_evidence = json.loads(args.selection_evidence)
        if not isinstance(spec, dict):
            raise ValueError
        if (not isinstance(selection_evidence, list) or len(selection_evidence) > 256
                or any(not isinstance(row, dict) for row in selection_evidence)
                or len(args.selection_evidence.encode("utf-8")) >= MAX_LINE):
            raise ValueError("invalid selection evidence")
        if any(not 1 <= value <= 2**63 - 1 for value in (args.prefix_cache_max_entries, args.prefix_cache_max_bytes)):
            raise ValueError("invalid cache capacity")
        identity = None
        if args.execution_variant in ("prefix_reuse", "current_engine", "automatic"):
            from friday_evidence.identity import assert_model_unchanged, runtime_identity
            identity = runtime_identity(spec)
        model, tokenizer, stream_generate, device = _load(spec)
        if identity is not None:
            assert_model_unchanged(spec, identity)
        if args.execution_variant == "prefix_reuse":
            from ironmule_product.prefix_session import WorkerPrefixSession
            prefix_session = WorkerPrefixSession(model, tokenizer,
                binding_sha256=identity["identity_sha256"], scope_nonce=secrets.token_hex(16),
                max_entries=args.prefix_cache_max_entries, max_bytes=args.prefix_cache_max_bytes)
        elif args.execution_variant == "current_engine":
            from ironmule_product.engine_bridge import CurrentEngineBridge
            from ironmule_product.types import ModelSpec
            engine_bridge = CurrentEngineBridge.from_loaded(model, tokenizer, ModelSpec.from_dict(spec),
                                                           configuration=args.engine_configuration)
        elif args.execution_variant == "automatic":
            from ironmule_product.automatic_runtime import AutomaticRuntime
            from ironmule_product.types import ModelSpec
            automatic_runtime = AutomaticRuntime(
                model, tokenizer, ModelSpec.from_dict(spec), identity, selection_evidence,
                prefix_cache_max_entries=args.prefix_cache_max_entries,
                prefix_cache_max_bytes=args.prefix_cache_max_bytes,
            )
            prefix_session = automatic_runtime.prefix_session
    except Exception as exc:
        if engine_bridge is not None:
            engine_bridge.close()
        _safe_error(getattr(exc, "code", "backend_unavailable") if isinstance(exc, ModelPolicyError) else "backend_unavailable")
        return 1
    import mlx.core as mx
    try:
        telemetry = _startup_telemetry(mx, startup_started)
    except Exception:
        _safe_error("backend_unavailable")
        return 1
    _emit({
        "type": "ready", "protocol_version": PROTOCOL_VERSION,
        "model_id": spec.get("model_id"), "revision": spec.get("revision"),
        "device": "gpu", "stop_handling": "parent", "context_limit": CONTEXT_LIMIT,
        "execution_variant": args.execution_variant,
        **({"automatic_batch_qualified": _automatic_batch_qualified(selection_evidence, identity)}
           if automatic_runtime is not None else {}),
        **({"engine": engine_bridge.metadata()} if engine_bridge is not None else {}),
        **telemetry,
    })
    commands: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=8)
    cancellations: dict[str, threading.Event] = {}
    pending_cancellations: set[str] = set()
    cancellation_lock = threading.Lock()
    reader = threading.Thread(
        target=_receiver,
        args=(commands, cancellations, pending_cancellations, cancellation_lock),
        daemon=True,
    )
    reader.start()
    allowed_variants = ({"automatic"} if automatic_runtime is not None else
                        {"current_engine"} if engine_bridge is not None else
                        {"reference", "prefix_reuse"} if prefix_session is not None else
                        {"reference", "bounded_prefetch"})
    try:
        while True:
            command = commands.get()
            if command is None or command.get("type") == "shutdown":
                reader.join(timeout=1.0)
                return 0
            if command.get("type") == "clear_prefix_cache":
                request_id = command.get("request_id")
                if prefix_session is None or not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
                    _safe_error("invalid_request", request_id if isinstance(request_id, str) else None)
                else:
                    _emit({"type": "prefix_cache_cleared", "request_id": request_id,
                           "prefix_cache": asdict(prefix_session.clear())})
                continue
            if command.get("type") == "generate_batch":
                if automatic_runtime is not None:
                    _run_automatic_batch(command, str(spec.get("model_id")), model, tokenizer,
                                         stream_generate, automatic_runtime, cancellations,
                                         pending_cancellations, cancellation_lock)
                elif (engine_bridge is None
                        or args.engine_configuration not in {"baseline_throughput", "core_throughput"}):
                    _safe_batch_error("invalid_request", command.get("batch_id") if isinstance(command.get("batch_id"), str) else None)
                else:
                    _run_engine_batch(command, str(spec.get("model_id")), tokenizer, engine_bridge,
                                      cancellations, pending_cancellations, cancellation_lock)
                continue
            if command.get("type") != "generate":
                _safe_error("protocol_error")
                continue
            request_id = command.get("request_id")
            variant = command.get("variant")
            trace_forwards = command.get("trace_forwards", False)
            if (not isinstance(request_id, str) or not isinstance(variant, str)
                    or variant not in allowed_variants or type(trace_forwards) is not bool):
                _safe_error("invalid_request", request_id if isinstance(request_id, str) else None)
                continue
            cancel_event = threading.Event()
            with cancellation_lock:
                if request_id in pending_cancellations:
                    cancel_event.set()
                    pending_cancellations.discard(request_id)
                cancellations[request_id] = cancel_event
            try:
                _run_generation(command, str(spec.get("model_id")), model, tokenizer, stream_generate,
                                cancel_event, variant, trace_forwards, prefix_session, engine_bridge,
                                automatic_runtime, cancellation_lock)
            finally:
                with cancellation_lock:
                    cancellations.pop(request_id, None)
    finally:
        if automatic_runtime is not None:
            automatic_runtime.close()
        elif prefix_session is not None:
            prefix_session.close()
        if engine_bridge is not None:
            engine_bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
