"""Child process for the isolated stock ``mlx_lm`` reference backend."""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import hashlib
import json
import math
import os
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

from ironmule_product.model_policy import ModelPolicyError, config_model_type, validate_model_config


PROTOCOL_VERSION = 1
MAX_LINE = 1024 * 1024
CONTEXT_LIMIT = 8192
WORKER_VARIANTS = frozenset(("reference", "bounded_prefetch", "prefix_reuse", "current_engine", "automatic"))
# Markers chat templates close a turn with (Gemma, Llama 3, ChatML/Qwen, Phi).
# ponytail: a known list; derive the marker from the chat template if a family is ever missed.
END_OF_TURN = ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<|end|>")
_PROTOCOL_OUT = sys.stdout


def _emit(value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_LINE:
        # This message itself is bounded and contains no request text.
        encoded = json.dumps({"type": "error", "code": "protocol_error", "message": "event too large"}, separators=(",", ":"))
    _PROTOCOL_OUT.write(encoded + "\n")
    _PROTOCOL_OUT.flush()


def _working_set_bytes(mx: Any) -> Any:
    # Metal reports Apple's recommended working set; MLX's CUDA backend only the
    # device's total memory (DATA3, Kaggle T4); its CPU backend nothing, so installed RAM (CPU4).
    if not _gpu_available(mx):
        from ironmule.hw import installed_memory_bytes
        return installed_memory_bytes()
    info = mx.device_info()
    return info.get("max_recommended_working_set_size", info.get("total_memory"))


def _gpu_available(mx: Any) -> bool:
    cuda = getattr(mx, "cuda", None)
    return bool(mx.metal.is_available() or (cuda is not None and cuda.is_available()))


def _startup_telemetry(mx: Any, startup_started: float) -> dict[str, Any]:
    values: dict[str, Any] = {
        "startup_wall_seconds": time.monotonic() - startup_started,
        # Darwin reports ru_maxrss in bytes, Linux in KiB.
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "recommended_working_set_bytes": _working_set_bytes(mx),
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


def _memory_counters() -> dict[str, int] | None:
    """MEM2: MLX's allocator counters at a group boundary: bytes in live arrays (active), freed
    buffers MLX keeps for reuse (cache) and the most ever active (peak). Memory outside MLX (driver
    context, library workspaces, instantiated CUDA graphs) is not in them."""
    try:
        import mlx.core as mx
        return {"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory()),
                "peak": int(mx.get_peak_memory())}
    except Exception:  # noqa: BLE001 - unknown stays unknown
        return None


CUDA_CACHE_LIMIT_ENV = "IRONMULE_CUDA_CACHE_LIMIT_MIB"
CUDA_CACHE_LIMIT_MIB = 1024  # MEM2: about one group's temporaries on the T4 (MEM2-D: peak - active ~0.86 GiB)


def _apply_cuda_cache_limit(mx: Any, environ: Any = os.environ) -> int | None:
    """MEM2: bound MLX's buffer cache on CUDA; returns the bytes applied, None when unchanged.

    MLX keeps freed buffers for reuse up to 0.95 of device memory by default, without counting the
    driver context, library state and the CUDA memory pool's own reserve; on a 15 GiB T4 the cache
    reached 11 GiB of freed buffers in many sizes and the device ran out (MEM2-D). Buffers beyond
    the bound go back to the CUDA memory pool, which reuses them for any size. Only the cache is
    bounded: live arrays, outputs and the plan are untouched. Metal keeps MLX's default (not
    measured there); `IRONMULE_CUDA_CACHE_LIMIT_MIB=0` keeps it on CUDA too."""
    cuda = getattr(mx, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    raw = environ.get(CUDA_CACHE_LIMIT_ENV, str(CUDA_CACHE_LIMIT_MIB))
    try:
        mib = int(raw)
    except ValueError:
        raise ValueError(f"{CUDA_CACHE_LIMIT_ENV} must be a whole number of MiB") from None
    if mib < 0:
        raise ValueError(f"{CUDA_CACHE_LIMIT_ENV} must not be negative")
    if mib == 0:
        return None
    mx.set_cache_limit(mib << 20)
    return mib << 20


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


def _note_failure(where: str) -> None:
    """STAB1: the handled exception's type, its frames and whether it reports exhausted memory, on
    stderr only (the parent keeps the tail when it retires this worker). Never its message: a library
    message can quote input text."""
    import traceback
    error = sys.exc_info()[1]
    if error is None:
        return
    frames = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(error.__traceback__)[-8:]]
    message = str(error).lower()
    try:
        sys.stderr.write(json.dumps({"worker_error": where, "type": type(error).__name__, "frames": frames,
                                     "out_of_memory": isinstance(error, MemoryError) or "out of memory" in message
                                     or "outofmemory" in message.replace(" ", ""),
                                     "illegal_memory_access": "illegal memory access" in message}) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - a diagnostic never changes the protocol
        pass


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


def _load(spec: dict[str, Any], lazy: bool = False):
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
    model_type = config_model_type(str(path), spec.get("revision"))
    # Opening the MLX device is intentionally the first native operation and
    # happens before importing mlx_lm.  A headless host may terminate here;
    # the parent converts that into BackendUnavailable.
    import mlx.core as mx

    from ironmule.hw import apply_cuda_graph_defaults
    apply_cuda_graph_defaults(model_type)  # before the first GPU operation reads them

    # A compiled-in GPU backend is not proof that the current process may
    # open the device. Force that check before mlx_lm creates native streams.
    # Without one, MLX's CPU device serves (CPU4); slow, but it answers.
    device = mx.gpu if _gpu_available(mx) else mx.cpu
    max_working_set = _working_set_bytes(mx)
    if type(max_working_set) is not int or max_working_set <= 0:
        raise RuntimeError("device working-set telemetry is unavailable")
    # This is an Apple recommendation, not a hard hardware capacity. Observe
    # it without refusing a model on this heuristic; the OS retains its limits.
    mx.set_default_device(device)
    with redirect_stdout(sys.stderr):
        from mlx_lm import load, stream_generate

        model, tokenizer = load(
            str(path),
            tokenizer_config={"trust_remote_code": False},
            model_config=model_config,
            revision=spec["revision"],
            lazy=lazy,  # RUST2-SRV: rust2 holds the weights; MLX keeps only the tokenizer's model unloaded
        )
    stop_at_end_of_turn(tokenizer)
    reuse_detokenizer(tokenizer)
    return model, tokenizer, stream_generate, device


def _sources_digest() -> str:
    from ironmule_product.code_identity import sources_digest
    return sources_digest()


def reuse_detokenizer(tokenizer: Any) -> None:
    """Give every request the same streaming detokenizer, reset, instead of a new one.

    mlx-lm builds a new one on each ``tokenizer.detokenizer`` access, walking the whole
    vocabulary: about 0.45 s of every request's first token for Gemma 3's 262k tokens on a T4
    (SERVE6b), in ``stream_generate`` as well. ``reset()`` clears all of its state, so the text
    is the same. ponytail: one instance per worker, which generates one request at a time; a
    worker that streams several at once needs one per stream.
    """
    cached = tokenizer.detokenizer

    class _Reused(type(tokenizer)):
        @property
        def detokenizer(self):
            cached.reset()
            return cached

    _Reused.__name__ = type(tokenizer).__name__
    _Reused.reuses_detokenizer = True
    tokenizer.__class__ = _Reused


def stop_at_end_of_turn(tokenizer: Any) -> None:
    """Treat the chat template's end-of-turn marker as end of sequence.

    Some checkpoints name only the plain end token in config.json: Gemma 3 lists `<eos>`
    but ends every chat turn with `<end_of_turn>`, so a served reply ran on to
    max_tokens repeating it. Adding the marker changes where generation stops, never a
    token before that point, and every serving path shares this tokenizer.
    """
    if not hasattr(tokenizer, "get_vocab") or not hasattr(tokenizer, "add_eos_token"):
        return  # no chat vocabulary to consult; the declared EOS set stays as loaded
    vocab = tokenizer.get_vocab()
    for marker in END_OF_TURN:
        if marker in vocab:
            tokenizer.add_eos_token(marker)


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
        if command.get("type") == "join_batch":
            JOINS.put(command)  # SERVE8: taken by the running batch, not queued behind it
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


CHAT_REUSE = False  # SERVE7: set once from --chat-reuse; a worker serves one configuration
THINKING = True  # CPU5: --no-thinking renders the chat template with thinking off


def _render(tokenizer: Any, messages: list[dict[str, Any]]) -> Any:
    """The chat template with a generation prompt, thinking off when the worker was told so.
    A template without that switch ignores it."""
    extra = {} if THINKING else {"enable_thinking": False}
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **extra)


def _turn_boundaries(tokenizer: Any, messages: list[dict[str, Any]], prompt_ids: Any) -> list[int]:
    """SERVE7: where each assistant message begins in the rendered prompt, found by rendering the
    conversation before it with the chat template; kept only where that is a true token prefix."""
    cuts = []
    for index, message in enumerate(messages):
        if index == 0 or message["role"] != "assistant":
            continue
        with redirect_stdout(sys.stderr):
            ids = list(_render(tokenizer, messages[:index]))
        if 0 < len(ids) < len(prompt_ids) and list(prompt_ids[:len(ids)]) == ids:
            cuts.append(len(ids))
    return cuts


def _engine_stream(engine_bridge: Any, tokenizer: Any, prompt_ids: list[int], max_tokens: int,
                   prompt_tokens: int, holder: dict[str, Any], consume: Any,
                   boundaries: list[int] | None = None) -> dict[str, Any] | None:
    """SERVE6: hand the engine's tokens to ``consume`` as it reads them back, detokenized the way
    mlx-lm streams, and return the terminal frame ``consume`` produced.

    The engine calls back on this thread (a second thread lost 0.5-0.7 s of the first token to the
    GIL on Gemma 3 1B, SERVE6b). Each token is sent as it is read back; the one that ends the
    answer (EOS or the output limit, the only ways the engine stops) carries the final text and
    the finish reason. Holding every token for the next one, as before, delayed the first by a
    whole chained readback (KP1 run 1: 56 ms at readback 4). After a cancellation the engine runs
    to its end and nothing more is sent. The streamed tokens must equal the engine's result.
    """
    eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
    detokenizer = tokenizer.detokenizer
    streamed: list[int] = []
    state: dict[str, Any] = {"terminal": None}

    def on_token(token: int) -> None:
        streamed.append(token)
        if state["terminal"] is not None or "error" in state:
            return
        try:
            last = token in eos or len(streamed) >= max_tokens
            text = ""
            if token not in eos:
                detokenizer.add_token(token)
                text = detokenizer.last_segment
            if last:
                detokenizer.finalize()
                text += detokenizer.last_segment
            state["terminal"] = consume(SimpleNamespace(
                token=token, text=text, prompt_tokens=prompt_tokens, generation_tokens=len(streamed),
                finish_reason=("stop" if token in eos else "length") if last else None))
        except Exception as exc:  # raised after the engine returns, never inside its executor
            state["error"] = exc

    result = (engine_bridge.generate(prompt_ids, max_tokens, on_token=on_token) if boundaries is None else
              engine_bridge.generate(prompt_ids, max_tokens, on_token=on_token, boundaries=boundaries))
    holder["result"] = result
    if "error" in state:
        raise state["error"]
    if not streamed or streamed != list(result.tokens):
        raise RuntimeError("streamed tokens differ from the engine result")
    terminal = state["terminal"]
    if terminal is None or (terminal["finish_reason"] != "cancelled"
                            and terminal["finish_reason"] != result.finish_reason):
        raise RuntimeError("the engine's answer ended other than its last streamed token")
    return terminal


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
    streamed: dict[str, Any] | None = None
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
            prompt_ids = _render(tokenizer, messages)
        if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
            raise ValueError
        prompt_tokens = len(prompt_ids)
        if prompt_tokens + max_tokens > CONTEXT_LIMIT:
            raise ValueError
    except Exception:
        _safe_error("invalid_request", request_id)
        return None

    def consume(response: Any) -> dict[str, Any] | None:
        """Emit one generated token; the terminal frame once generation must stop."""
        response_prompt = getattr(response, "prompt_tokens", prompt_tokens)
        response_completion = getattr(response, "generation_tokens", None)
        if (type(response_prompt) is not int or response_prompt != prompt_tokens
                or type(response_completion) is not int or not 1 <= response_completion <= max_tokens):
            raise ValueError
        if cancel_event.is_set():
            return {
                "type": "done", "request_id": request_id, "finish_reason": "cancelled",
                "prompt_tokens": prompt_tokens, "completion_tokens": response_completion,
                "variant": variant,
            }
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
            return {
                "type": "done", "request_id": request_id,
                "finish_reason": finish,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response_completion,
                "variant": variant,
                "metrics": metrics,
            }
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
                    streamed = {}
                    boundaries = _turn_boundaries(tokenizer, messages, prompt_ids) if CHAT_REUSE else None
                    terminal = _engine_stream(engine_bridge, tokenizer, list(prompt_ids), max_tokens,
                                              prompt_tokens, streamed, consume, boundaries)
                    streamed["plan"] = "chat_turns" if boundaries is not None else None
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
                for response in generated or ():
                    terminal = consume(response)
                    if terminal is not None:
                        break
            if terminal is None:
                raise ValueError("generation ended without a completion frame")
    except Exception:
        terminal = None
        _note_failure("generate")
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
    delivery = "buffered_completion"
    if streamed is not None and "result" in streamed:
        engine_result, delivery = streamed["result"], "streamed"
    if terminal is not None and engine_result is not None:
        terminal["engine"] = {**engine_result.metadata, "delivery": delivery,
                              "computed_tokens": engine_result.token_count}
        if streamed is not None and streamed.get("plan"):
            terminal["engine"]["plan"] = streamed["plan"]
    if terminal is not None and selection_metadata is not None:
        terminal["selection"] = selection_metadata
    if terminal is not None and command.get("trace_prompt_identity") is True:
        terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(list(prompt_ids),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return terminal


def _render_engine_request(command: dict[str, Any], model_id: str, tokenizer: Any,
                           allow_stream: bool = False) -> tuple[str, list[int], int]:
    """Apply exactly the single-request identity and rendering validation."""
    request_id = command.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise ValueError("invalid request id")
    if command.get("model") != model_id or command.get("temperature", 0.0) != 0.0 or command.get("top_p", 1.0) != 1.0:
        raise ValueError("invalid exact request")
    if command.get("stream", False) is not False and not (allow_stream and command.get("stream") is True):
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
        raw_ids = _render(tokenizer, messages)
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("invalid prompt ids")
    ids = list(raw_ids)
    if len(ids) + max_tokens > CONTEXT_LIMIT:
        raise ValueError("context limit")
    return request_id, ids, max_tokens


# Engine configurations whose worker accepts grouped requests: the historical throughput
# candidates (buffered) and the current profile's streaming groups (SERVE4).
BATCH_CONFIGURATIONS = frozenset({"baseline_throughput", "core_throughput", "current_profile", "dense_stacked"})
# The ones whose groups stream, take joins and stop on cancel (SERVE4, SERVE8): the client sends them lone
# streamed batches too (`MLXWorkerClient.streams_batches`), so the two lists must agree.
STREAMED_BATCH_CONFIGURATIONS = frozenset({"current_profile", "dense_stacked"})
_DETOKENIZER_POOLS: dict[int, "_DetokenizerPool"] = {}
JOINS: queue.Queue[dict[str, Any]] = queue.Queue()  # SERVE8: join_batch commands for the running batch


class _DetokenizerPool:
    """Reusable streaming detokenizers (building one walks the whole vocabulary, about 0.45 s for
    Gemma 3, SERVE6b). A stream takes one and gives it back when its last token is in, so the
    pool grows with the streams running at once, not with every request a long batch admits."""

    def __init__(self, tokenizer: Any) -> None:
        plain = (type(tokenizer).__mro__[1] if getattr(type(tokenizer), "reuses_detokenizer", False)
                 else type(tokenizer))
        self._build = lambda: plain.detokenizer.fget(tokenizer)
        self._free: list[Any] = []

    def acquire(self) -> Any:
        detokenizer = self._free.pop() if self._free else self._build()
        detokenizer.reset()
        return detokenizer

    def release(self, detokenizer: Any) -> None:
        self._free.append(detokenizer)


def _detokenizer_pool(tokenizer: Any) -> _DetokenizerPool:
    pool = _DETOKENIZER_POOLS.get(id(tokenizer))
    if pool is None:
        pool = _DETOKENIZER_POOLS[id(tokenizer)] = _DetokenizerPool(tokenizer)
    return pool


class _BatchStream:
    """One grouped request's tokens as the engine reads them back (SERVE4).

    Each token is sent as it is read back; the one that ends the answer (EOS or the output limit)
    carries the final text. Holding every token for the next, as before, delayed the first by a
    whole chained readback (KP1 run 1). A cancelled request stops sending while the group runs on.
    Errors are kept and raised after the engine returns, never inside its executor."""

    def __init__(self, batch_id: str, request_id: str, prompt_tokens: int, limit: int, pool: _DetokenizerPool,
                 cancellations: dict[str, threading.Event], lock: threading.Lock, tokenizer: Any):
        self.batch_id, self.request_id, self.prompt_tokens, self.limit = batch_id, request_id, prompt_tokens, limit
        self.pool, self.detokenizer = pool, pool.acquire()
        self.cancellations, self.lock = cancellations, lock
        self.eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
        self.tail = ""
        self.tokens: list[int] = []
        self.sent = 0
        self.stopped = False
        self.error: BaseException | None = None

    def on_token(self, token: int) -> None:
        self.tokens.append(token)
        if self.stopped or self.error is not None:
            return
        try:
            with self.lock:
                self.stopped = self.cancellations[self.request_id].is_set()
            if self.stopped:
                return
            text = ""
            if token not in self.eos:
                self.detokenizer.add_token(token)
                text = self.detokenizer.last_segment
            if token in self.eos or len(self.tokens) >= self.limit:
                self._release()  # its last token: the detokenizer serves the next admitted stream
                text += self.tail
                self.tail = ""
            _emit({"type": "token", "batch_id": self.batch_id, "request_id": self.request_id, "text": text,
                   "token_id": token, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.sent + 1})
            self.sent += 1
        except BaseException as exc:  # noqa: BLE001 - raised by finish()
            self.error = exc

    def _release(self) -> None:
        if self.detokenizer is not None:
            self.detokenizer.finalize()
            self.tail = self.detokenizer.last_segment
            self.pool.release(self.detokenizer)
            self.detokenizer = None

    def finish(self, result: Any) -> int:
        """Check the streamed tokens against the engine's result; returns how many were sent."""
        if self.error is not None:
            raise self.error
        with self.lock:
            self.stopped = self.stopped or self.cancellations[self.request_id].is_set()
        produced = list(result.tokens)
        # A request cancelled during a fallback re-run ends shorter than what it already streamed
        # (RSI3-LIVE-P2); its output is never delivered, so a matching prefix is enough for it.
        if self.tokens != produced and not (self.stopped and result.finish_reason == "cancelled"
                                             and self.tokens[:len(produced)] == produced):
            raise RuntimeError("streamed tokens differ from the engine result")
        self._release()
        return self.sent


def _run_engine_batch(command: dict[str, Any], model_id: str, tokenizer: Any,
                      engine_bridge: Any, cancellations: dict[str, threading.Event],
                      pending_cancellations: set[str], lock: threading.Lock,
                      *, rendered_override: list[tuple[str, list[int], int]] | None = None,
                      selection: dict[str, Any] | None = None) -> None:
    batch_id = command.get("batch_id")
    requests = command.get("requests")
    # SERVE4: the current-profile engine and the dense plan stream every grouped request's tokens as
    # they are read back; the historical throughput configurations keep their buffered completion.
    live = (rendered_override is None
            and engine_bridge.metadata().get("configuration") in STREAMED_BATCH_CONFIGURATIONS)
    if (not isinstance(batch_id, str) or not 1 <= len(batch_id) <= 256
            or command.get("variant") not in {"current_engine", "automatic"} or not isinstance(requests, list)
            or not (1 if live else 2) <= len(requests) <= 32):
        _safe_batch_error("invalid_request", batch_id if isinstance(batch_id, str) else None)
        return
    registered: list[str] = []
    try:
        rendered = (rendered_override if rendered_override is not None else
                    [_render_engine_request(request, model_id, tokenizer, allow_stream=live) for request in requests])
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
        pool = _detokenizer_pool(tokenizer)
        streams: dict[str, _BatchStream] = {}
        # SERVE7 in the grouped path: every request, joined ones too, keeps its chat-turn plan.
        messages_by_id = {row[0]: request.get("messages") for row, request in zip(rendered, requests)} \
            if rendered_override is None else {}

        def cuts_for(row: tuple[str, list[int], int]) -> list[int] | None:
            return (_turn_boundaries(tokenizer, messages_by_id[row[0]], row[1])
                    if live and CHAT_REUSE and messages_by_id.get(row[0]) else None)
        all_ids = list(request_ids)
        leftovers: list[tuple[str, list[int], int]] = []

        def drain_joins() -> list[tuple[str, list[int], int]]:
            """SERVE8: requests the client sent to join this batch, rendered and registered."""
            rows = []
            while True:
                try:
                    join = JOINS.get_nowait()
                except queue.Empty:
                    return rows
                if join.get("batch_id") != batch_id or not isinstance(join.get("requests"), list):
                    raise ValueError("join for another batch")
                for request in join["requests"]:
                    row = _render_engine_request(request, model_id, tokenizer, allow_stream=True)
                    messages_by_id[row[0]] = request.get("messages")
                    if row[0] in all_ids:
                        raise ValueError("duplicate request id")
                    with lock:
                        event = threading.Event()
                        if row[0] in pending_cancellations:
                            event.set()
                            pending_cancellations.discard(row[0])
                        cancellations[row[0]] = event
                        registered.append(row[0])
                    all_ids.append(row[0])
                    rows.append(row)

        report = rendered
        while report:
            with lock:
                skipped = {row[0] for row in report if cancellations[row[0]].is_set()}
            active = [row for row in report if row[0] not in skipped]
            for row in active if live else ():
                streams[row[0]] = _BatchStream(batch_id, row[0], len(row[1]), row[2], pool, cancellations, lock,
                                               tokenizer)
            joined: list[tuple[str, list[int], int]] = []
            waiting: list[tuple[str, list[int], int]] = []  # KB1: joiners waiting for a slot of this run

            def admit(capacity: int) -> list[tuple[list[int], int, Any]]:
                fresh = []
                slots = getattr(engine_bridge, "free_slots", lambda: None)()  # KB1: the dense plan's free slots
                candidates, waiting[:] = waiting + drain_joins(), []
                for row in candidates:
                    if len(row[1]) + row[2] + 8 > capacity:
                        leftovers.append(row)  # too long for this run's cache; it runs right after
                        continue
                    if slots is not None and len(fresh) >= slots:
                        waiting.append(row)  # no free slot now; it takes the first one a member frees
                        continue
                    streams[row[0]] = _BatchStream(batch_id, row[0], len(row[1]), row[2], pool, cancellations,
                                                   lock, tokenizer)
                    joined.append(row)
                    fresh.append((row[1], row[2], streams[row[0]].on_token, cuts_for(row)))
                return fresh

            def cancelled_for(request_id: str):
                def check() -> bool:
                    with lock:
                        return cancellations[request_id].is_set()
                return check

            with redirect_stdout(sys.stderr):
                turns = [cuts_for(row) for row in active]
                results = (engine_bridge.generate_many([row[1] for row in active], [row[2] for row in active],
                                                       on_tokens=[streams[row[0]].on_token for row in active]
                                                       if live else None,
                                                       **({"admit": admit,  # RSI3-LIVE-P2: stop on cancel
                                                           "cancelled": [cancelled_for(row[0]) for row in active]}
                                                          if live else {}),
                                                       **({"boundaries": turns}
                                                          if any(cut is not None for cut in turns) else {}))
                           if active else [])
            leftovers.extend(waiting)  # the run ended before a slot freed for them; they run right after
            if not isinstance(results, list) or len(results) != len(active) + len(joined):
                raise RuntimeError("invalid grouped result count")
            result_by_id = {row[0]: result for row, result in zip(active + joined, results)}
            pre_cancelled = skipped
            for request_id, prompt_ids, limit in report + joined:
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
                with lock:
                    stopped = result.finish_reason == "cancelled" and cancellations[request_id].is_set()
                if (not isinstance(tokens, (list, tuple)) or type(result.token_count) is not int
                        or result.token_count != len(tokens)
                        or not (0 if stopped else 1) <= result.token_count <= limit
                        or (result.finish_reason not in {"stop", "length"} and not stopped)
                        or not isinstance(result.text, str) or not isinstance(result.metadata, dict)):
                    raise RuntimeError("invalid grouped engine result")
                emitted_tokens = 0
                if request_id in streams:
                    emitted_tokens = streams[request_id].finish(result)
                    tokens = ()  # already sent as the engine read them back
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
                            "engine": {**result.metadata,
                                       "delivery": "streamed" if request_id in streams else "buffered_completion",
                                       "computed_tokens": result.token_count}}
                if command.get("trace_prompt_identity") is True:
                    terminal["prompt_ids_sha256"] = hashlib.sha256(json.dumps(prompt_ids, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                if selection is not None:
                    terminal["selection"] = selection
                _emit(terminal)
            report = leftovers + (drain_joins() if live else [])
            leftovers = []
        _emit({"type": "batch_done", "batch_id": batch_id, "request_ids": all_ids, "memory": _memory_counters()})
    except ValueError:
        _note_failure("batch")
        _safe_batch_error("invalid_request", batch_id)
    except Exception:
        _note_failure("batch")
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
        _note_failure("batch")
        _safe_batch_error("invalid_request", batch_id)
    except Exception:
        _note_failure("batch")
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
    parser.add_argument("--no-thinking", action="store_true",
                        help="render the chat template with thinking off (CPU5)")
    parser.add_argument("--chat-reuse", action="store_true",
                        help="current_engine: reuse KV state across chat turns (ChatTurnsPlan, SERVE7)")
    parser.add_argument("--engine-configuration", default="current_profile",
                        choices=("current_profile", "baseline_interactive", "core_interactive",
                                 "baseline_throughput", "core_throughput", "rust2", "dense_stacked"))
    parser.add_argument("--prefix-cache-max-entries", type=int, default=4)
    parser.add_argument("--prefix-cache-max-bytes", type=int, default=1024**3)
    parser.add_argument("--selection-evidence", default="[]")
    parser.add_argument("--compute-dtype", choices=("float32", "native", "dequantize"), default=None)
    parser.add_argument("--engine-knobs", default=None)  # KP1: class-A knobs when no stored profile applies
    args = parser.parse_args(argv)
    global CHAT_REUSE, THINKING
    CHAT_REUSE = bool(args.chat_reuse) and args.execution_variant == "current_engine"
    THINKING = not args.no_thinking
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
        rust2 = args.execution_variant == "current_engine" and args.engine_configuration == "rust2"
        model, tokenizer, stream_generate, device = _load(spec, lazy=rust2)
        import mlx.core as mx
        cuda_cache_limit = _apply_cuda_cache_limit(mx)
        if args.compute_dtype is not None:
            if args.execution_variant != "reference":
                raise ValueError("compute_dtype is available on the reference worker only")
            import mlx.core as mx
            if args.compute_dtype == "float32":
                model.set_dtype(mx.float32)  # floating parameters only; opt-in, changes output
            elif args.compute_dtype == "dequantize":  # CPU3: dense float32 weights; opt-in, changes output
                from ironmule.numeric_plans import dequantize_model
                dequantize_model(model)
            else:
                from ironmule.cuda_native import install
                install(model, mx.device_info() if mx.cuda.is_available() else None)
        if identity is not None:
            assert_model_unchanged(spec, identity)
        if args.execution_variant == "prefix_reuse":
            from ironmule_product.prefix_session import WorkerPrefixSession
            prefix_session = WorkerPrefixSession(model, tokenizer,
                binding_sha256=identity["identity_sha256"], scope_nonce=secrets.token_hex(16),
                max_entries=args.prefix_cache_max_entries, max_bytes=args.prefix_cache_max_bytes)
        elif rust2:
            from ironmule_product.rust2_bridge import Rust2Bridge
            engine_bridge = Rust2Bridge(os.environ.get("IRONMULE_RUST2_BINARY", ""), spec["snapshot_path"],
                                        tokenizer.eos_token_ids, os.environ.get("IRONMULE_RUST2_DRAFT") or None)
        elif args.execution_variant == "current_engine":
            from ironmule_product.engine_bridge import CurrentEngineBridge
            from ironmule_product.types import ModelSpec
            engine_bridge = CurrentEngineBridge.from_loaded(
                model, tokenizer, ModelSpec.from_dict(spec), configuration=args.engine_configuration,
                class_a_knobs=json.loads(args.engine_knobs) if args.engine_knobs is not None else None)
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
        # OPS1: the device MLX actually computes on (CPU4 serves on the CPU device), not a constant.
        "device": "gpu" if device == mx.gpu else "cpu", "stop_handling": "parent", "context_limit": CONTEXT_LIMIT,
        "execution_variant": args.execution_variant,
        **({"automatic_batch_qualified": _automatic_batch_qualified(selection_evidence, identity)}
           if automatic_runtime is not None else {}),
        **({"engine": engine_bridge.metadata()} if engine_bridge is not None else {}),
        **telemetry,
        "sources_sha256": _sources_digest(),  # RSI3-LIVE (f): every loaded source file of this worker
        "cuda_cache_limit_bytes": cuda_cache_limit,  # MEM2: None = MLX's default
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
                        or args.engine_configuration not in BATCH_CONFIGURATIONS):
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
