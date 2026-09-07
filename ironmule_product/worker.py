"""Child process for the isolated stock ``mlx_lm`` reference backend."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Any


if __package__ in (None, ""):
    _PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, _PACKAGE_ROOT)


PROTOCOL_VERSION = 1
MAX_LINE = 1024 * 1024
CONTEXT_LIMIT = 8192
WORKER_VARIANTS = frozenset(("reference", "bounded_prefetch"))
_PROTOCOL_OUT = sys.stdout


def _emit(value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_LINE:
        # This message itself is bounded and contains no request text.
        encoded = json.dumps({"type": "error", "code": "protocol_error", "message": "event too large"}, separators=(",", ":"))
    _PROTOCOL_OUT.write(encoded + "\n")
    _PROTOCOL_OUT.flush()


def _safe_error(code: str, request_id: str | None = None) -> None:
    event: dict[str, Any] = {"type": "error", "code": code, "message": "stock MLX backend unavailable"}
    if request_id is not None:
        event["request_id"] = request_id
    _emit(event)


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
    if type(max_working_set) is not int or max_working_set <= 0 or actual_weights >= max_working_set:
        raise RuntimeError("model weights exceed the device working-set limit")
    device = mx.gpu
    mx.set_default_device(device)
    with redirect_stdout(sys.stderr):
        from mlx_lm import load, stream_generate

        model, tokenizer = load(
            str(path),
            tokenizer_config={"trust_remote_code": False},
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
            commands.put(None)
            return
        if len(line) > MAX_LINE:
            try:
                commands.put_nowait({"type": "invalid"})
            except queue.Full:
                _safe_error("overloaded")
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
) -> dict[str, Any] | None:
    request_id = command.get("request_id") if isinstance(command.get("request_id"), str) else None
    if request_id is None or not request_id or len(request_id) > 256:
        _safe_error("invalid_request")
        return None
    generated = None
    terminal: dict[str, Any] | None = None
    error_emitted = False

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
    return terminal


def _run_generation(
    command: dict[str, Any],
    model_id: str,
    model: Any,
    tokenizer: Any,
    stream_generate: Any,
    cancel_event: threading.Event,
    variant: str,
    trace_forwards: bool,
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
        terminal = _generate(command, model_id, model, tokenizer, stream_generate, cancel_event, variant)
    finally:
        if generate_module is not None:
            generate_module.generate_step = original_generate_step
        if model_type is not None and original_model_call is not None:
            model_type.__call__ = original_model_call
    if terminal is not None:
        if trace_forwards:
            terminal["model_forward_invocations"] = forward_count
        _emit(terminal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec)
        if not isinstance(spec, dict):
            raise ValueError
        model, tokenizer, stream_generate, device = _load(spec)
    except Exception:
        _safe_error("backend_unavailable")
        return 1
    _emit({
        "type": "ready", "protocol_version": PROTOCOL_VERSION,
        "model_id": spec.get("model_id"), "revision": spec.get("revision"),
        "device": "gpu", "stop_handling": "parent", "context_limit": CONTEXT_LIMIT,
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
    while True:
        command = commands.get()
        if command is None or command.get("type") == "shutdown":
            return 0
        if command.get("type") != "generate":
            _safe_error("protocol_error")
            continue
        request_id = command.get("request_id")
        variant = command.get("variant")
        trace_forwards = command.get("trace_forwards", False)
        if (not isinstance(request_id, str) or not isinstance(variant, str)
                or variant not in WORKER_VARIANTS or type(trace_forwards) is not bool):
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
                            cancel_event, variant, trace_forwards)
        finally:
            with cancellation_lock:
                cancellations.pop(request_id, None)


if __name__ == "__main__":
    raise SystemExit(main())
