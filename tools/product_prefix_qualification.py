"""PROD11 native correctness qualification for identical full-prompt reuse.

Inert unless ``--execute`` is supplied.  The controller never imports MLX; one
owned child loads one frozen model and emits bounded metadata-only JSON frames.
This is a correctness study, not a performance benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ironmule.prod11_native_correctness.v1"
CHILD_SCHEMA = "ironmule.prod11_native_correctness.child.v1"
FRAME_LIMIT = 2 * 1024 * 1024
MAX_TOKENS = 8
EXPECTED_PROMPT_TOKENS = 1077
MODELS = {
    "mlx-community/gemma-3-1b-it-4bit": "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
    "mlx-community/gemma-3-4b-it-4bit": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
    "mlx-community/gemma-3-12b-it-4bit": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
}
CHECKS = {"wrong_binding", "wrong_scope", "wrong_key", "stale_clear", "stale_close",
          "clone_isolation", "cancel_recovery", "entry_eviction", "byte_oversize_skip",
          "canonical_unchanged", "byte_eviction"}
USED_MODULES = ("ironmule_product.prefix_reuse", "ironmule_product.model_policy",
                "ironmule_product.calibration", "ironmule_product.readiness",
                "ironmule_product.state", "friday_evidence.canonical",
                "friday_evidence.open_observation", "friday_evidence.process_memory",
                "friday_evidence.events", "friday_evidence.identity")


class QualificationFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class CheckpointReached(RuntimeError):
    pass


def orchard_messages() -> list[dict[str, str]]:
    unit = "Public orchard note: apples grow on trees and need sunlight. "
    return [{"role": "user", "content": unit * 88 + "END-OF-PUBLIC-ORCHARD-NOTE."}]


def digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metadata_only(value: Any) -> None:
    if isinstance(value, dict):
        if {"text", "tokens", "prompt", "prompt_ids", "snapshot", "scope_nonce",
            "messages", "content", "token_ids", "answers"}.intersection(value):
            raise QualificationFailure("raw_data_exposed")
        for child in value.values():
            _metadata_only(child)
    elif isinstance(value, list):
        for child in value:
            _metadata_only(child)


def validate_child_report(value: Any, model_id: str, revision: str) -> dict[str, Any]:
    _metadata_only(value)
    if not isinstance(value, dict) or value.get("schema") != CHILD_SCHEMA:
        raise QualificationFailure("child_schema_invalid")
    if value.get("model_id") != model_id or value.get("revision") != revision:
        raise QualificationFailure("child_identity_invalid")
    if value.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS:
        raise QualificationFailure("prompt_count_invalid")
    if value.get("status") != "passed" or value.get("correctness_gate") is not True:
        raise QualificationFailure("native_correctness_failed")
    if ("gpu" not in str(value.get("device", "")).lower()
            or value.get("performance_claim") is not False
            or value.get("activation_allowed") is not False
            or any(type(value.get(key)) is not int or value[key] <= 0
                   for key in ("mlx_active_bytes", "mlx_peak_bytes"))):
        raise QualificationFailure("child_device_or_claim_invalid")
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != CHECKS or any(v is not True for v in checks.values()):
        raise QualificationFailure("child_checks_incomplete")
    stock, candidate, traces = (value.get(key) for key in ("stock", "candidate", "traces"))
    if any(not isinstance(rows, list) or len(rows) != 4 for rows in (stock, candidate, traces)):
        raise QualificationFailure("child_schedule_invalid")
    for row in stock + candidate + [value.get("recovery_cold"), value.get("recovery_hit")]:
        if (not isinstance(row, dict) or row != stock[0]
                or row.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS
                or type(row.get("completion_tokens")) is not int
                or not 1 <= row["completion_tokens"] <= MAX_TOKENS
                or row.get("finish_reason") not in {"stop", "length"}
                or any(not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
                       for key in ("output_sha256", "token_sha256", "text_sha256", "logprobs_sha256"))):
            raise QualificationFailure("child_outputs_invalid")
    if (any(not isinstance(row, dict) or row.get("status") != "completed" for row in traces)
            or traces[0].get("cache_hit") is not False or traces[0].get("cache_stored") is not True
            or traces[0].get("reused_tokens") != 0
            or any(row.get("cache_hit") is not True or row.get("cache_stored") is not False
                   or row.get("reused_tokens") != EXPECTED_PROMPT_TOKENS - 1
                   for row in traces[1:])):
        raise QualificationFailure("child_cache_trace_invalid")
    checkpoint = value.get("checkpoint")
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("layers"), list)
            or not checkpoint["layers"] or checkpoint.get("sha256") != digest_json(checkpoint["layers"])
            or checkpoint != value.get("canonical_after_hits")
            or type(checkpoint.get("nbytes")) is not int or checkpoint["nbytes"] <= 0):
        raise QualificationFailure("child_checkpoint_invalid")
    return value


def _emit(value: dict[str, Any]) -> None:
    _metadata_only(value)
    frame = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(frame.encode()) >= FRAME_LIMIT:
        raise QualificationFailure("child_frame_oversize")
    print(frame, flush=True)


def _finish_child(result: dict[str, Any], returncode: int) -> int:
    _emit({"type": "finished", "result": result})
    # Parent records the live final state, then explicitly closes this owned child.
    command = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT + 1))
    if command != {"type": "shutdown"}:
        raise QualificationFailure("shutdown_command_invalid")
    return returncode


def _array_record(array: Any, mx: Any, np: Any) -> dict[str, Any]:
    # Diagnostic materialisation only; raw bytes never leave this function.
    raw = np.asarray(array.view(mx.uint8)).tobytes(order="C")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _cache_record(caches: list[Any], mx: Any, np: Any) -> dict[str, Any]:
    layers = []
    for cache in caches:
        state = cache.state
        arrays = state if isinstance(state, (list, tuple)) else (state,)
        layers.append({"class": type(cache).__name__,
                       "meta_state": list(cache.meta_state) if isinstance(cache.meta_state, tuple) else cache.meta_state,
                       "state": [_array_record(value, mx, np) for value in arrays],
                       "nbytes": cache.nbytes})
    return {"layers": layers, "nbytes": sum(layer["nbytes"] for layer in layers),
            "sha256": digest_json(layers)}


def _response_record(responses: list[Any], mx: Any, np: Any) -> dict[str, Any]:
    if not responses or responses[-1].finish_reason not in {"stop", "length"}:
        raise QualificationFailure("generation_incomplete")
    tokens = [response.token for response in responses]
    text = "".join(response.text for response in responses)
    logprob_hashes = [_array_record(response.logprobs, mx, np) for response in responses]
    return {"output_sha256": digest_json({"tokens": tokens, "text": text,
                                           "finish_reason": responses[-1].finish_reason}),
            "token_sha256": digest_json(tokens), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "logprobs_sha256": digest_json(logprob_hashes), "completion_tokens": len(tokens),
            "prompt_tokens": responses[-1].prompt_tokens, "finish_reason": responses[-1].finish_reason}


def _checkpoint(model: Any, tokenizer: Any, prompt_ids: list[int], stream_generate: Any,
                make_prompt_cache: Any, mx: Any, np: Any) -> tuple[dict[str, Any], list[Any]]:
    caches = make_prompt_cache(model)
    def stop(processed: int, total: int) -> None:
        if processed > 0 and processed == total - 1:
            raise CheckpointReached()
    generated = stream_generate(model, tokenizer, prompt_ids, max_tokens=MAX_TOKENS,
                                prompt_cache=caches, prefill_step_size=2048,
                                logits_processors=None, draft_model=None, kv_bits=None,
                                prompt_progress_callback=stop)
    try:
        try:
            next(generated)
        except CheckpointReached:
            pass
        else:
            raise QualificationFailure("checkpoint_not_reached")
    finally:
        generated.close()
    return _cache_record(caches, mx, np), caches


def _child() -> int:
    result: dict[str, Any] = {"schema": CHILD_SCHEMA, "status": "started",
        "correctness_gate": False, "performance_claim": False, "activation_allowed": False,
        "stock": [], "candidate": [], "traces": []}
    try:
        request = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT + 1))
        model_id, revision, snapshot_path = request["model_id"], request["revision"], request["snapshot_path"]
        binding = request.get("binding_sha256")
        if MODELS.get(model_id) != revision:
            raise QualificationFailure("snapshot_not_frozen")
        result.update(model_id=model_id, revision=revision)
        if not isinstance(binding, str) or len(binding) != 64:
            raise QualificationFailure("binding_invalid")
        from ironmule_product.model_policy import validate_model_config
        config = validate_model_config(snapshot_path, revision)
        with redirect_stdout(sys.stderr):
            import mlx.core as mx
            import numpy as np
            from mlx_lm import load, stream_generate
            from mlx_lm.models.cache import make_prompt_cache
            from ironmule_product.prefix_reuse import IdenticalPromptReuse, PrivatePrefixCache
            if not mx.metal.is_available():
                raise QualificationFailure("metal_unavailable")
            mx.set_default_device(mx.gpu)
            model, tokenizer = load(snapshot_path, tokenizer_config={"trust_remote_code": False},
                                    model_config={**config, "model_file": None}, revision=revision)
            prompt_ids = list(tokenizer.apply_chat_template(orchard_messages(), tokenize=True,
                                                             add_generation_prompt=True))
        if len(prompt_ids) != EXPECTED_PROMPT_TOKENS:
            raise QualificationFailure("prompt_count_invalid")
        result.update(device=str(mx.default_device()), prompt_tokens=len(prompt_ids),
                      prompt_ids_sha256=digest_json(prompt_ids))
        _emit({"type": "ready", "pid": os.getpid(), "device": str(mx.default_device()),
               "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory()})
        stock = result["stock"]
        with redirect_stdout(sys.stderr):
            for repeat in range(4):
                stock.append(_response_record(list(stream_generate(model, tokenizer, prompt_ids,
                                      max_tokens=MAX_TOKENS)), mx, np))
                # Emit after leaving redirected stdout below, through the saved pipe.
                with redirect_stdout(sys.__stdout__):
                    _emit({"type": "progress", "phase": "stock", "index": repeat, "sample": stock[-1]})
            checkpoint, _ = _checkpoint(model, tokenizer, prompt_ids, stream_generate,
                                        make_prompt_cache, mx, np)
            short_checkpoint, short_caches = _checkpoint(
                model, tokenizer, prompt_ids[:513], stream_generate, make_prompt_cache, mx, np
            )
        result.update(checkpoint=checkpoint, short_checkpoint=short_checkpoint)
        _emit({"type": "progress", "phase": "checkpoint", "checkpoint": checkpoint,
               "short_checkpoint": short_checkpoint})
        reference = stock[-1]
        if any(row != reference for row in stock):
            raise QualificationFailure("stock_not_repeatable")

        scope = uuid.uuid4().hex
        private = PrivatePrefixCache(binding_sha256=binding, scope_nonce=scope,
                                     max_entries=2, max_bytes=max(checkpoint["nbytes"] * 3, 1))
        adapter = IdenticalPromptReuse(model, tokenizer, private,
                                       expected_binding_sha256=binding, scope_nonce=scope)
        candidate, traces = result["candidate"], result["traces"]
        with redirect_stdout(sys.stderr):
            for _ in range(4):
                candidate.append(_response_record(list(adapter.stream(prompt_ids,
                                                                       max_tokens=MAX_TOKENS)), mx, np))
                traces.append(adapter.last_trace.__dict__)
                with redirect_stdout(sys.__stdout__):
                    _emit({"type": "progress", "phase": "candidate", "index": len(candidate) - 1,
                           "sample": candidate[-1], "trace": traces[-1]})
                if candidate[-1] != reference:
                    raise QualificationFailure("candidate_output_mismatch")
        key = tuple(prompt_ids)
        canonical = private._entries[key].snapshot
        canonical_after_cold = _cache_record(canonical, mx, np)
        result["canonical_after_hits"] = canonical_after_cold
        if canonical_after_cold != checkpoint:
            raise QualificationFailure("checkpoint_cache_mismatch")
        if not traces[0]["cache_stored"] or traces[0]["cache_hit"]:
            raise QualificationFailure("cold_trace_invalid")
        if any(not row["cache_hit"] or row["reused_tokens"] != len(prompt_ids) - 1
               or row["status"] != "completed" for row in traces[1:]):
            raise QualificationFailure("hit_trace_invalid")

        # Real-cache metadata/error paths. No concurrent model execution occurs.
        wrong_binding, reason_binding = private.restore(prompt_ids,
            binding_sha256="0" * 64, scope_nonce=scope)
        wrong_scope, reason_scope = private.restore(prompt_ids,
            binding_sha256=binding, scope_nonce="different-private-scope")
        different_valid_token = next(token for token in prompt_ids if token != prompt_ids[-1])
        wrong_key, reason_key = private.restore(prompt_ids[:-1] + [different_valid_token],
            binding_sha256=binding, scope_nonce=scope)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(private.restore, prompt_ids,
                                   binding_sha256=binding, scope_nonce=scope) for _ in range(2)]
            (clone_a, _), (clone_b, _) = [future.result() for future in futures]
        if clone_a is None or clone_b is None or any(a is b for a, b in zip(clone_a, clone_b)):
            raise QualificationFailure("clone_isolation_invalid")
        with redirect_stdout(sys.stderr):
            clone_outputs = [_response_record(list(stream_generate(model, tokenizer, [prompt_ids[-1]],
                                 max_tokens=MAX_TOKENS, prompt_cache=clone, prefill_step_size=2048,
                                 logits_processors=None, draft_model=None, kv_bits=None)), mx, np)
                             for clone in (clone_a, clone_b)]
        if any(row["output_sha256"] != reference["output_sha256"] or
               row["logprobs_sha256"] != reference["logprobs_sha256"] for row in clone_outputs):
            raise QualificationFailure("clone_generation_mismatch")
        canonical_after_hits = _cache_record(canonical, mx, np)
        if canonical_after_hits != checkpoint:
            raise QualificationFailure("canonical_mutated")
        result.update(canonical_after_hits=canonical_after_hits, clone_outputs=clone_outputs)
        _emit({"type": "progress", "phase": "clone_isolation", "canonical_sha256": canonical_after_hits["sha256"],
               "clone_outputs": clone_outputs})

        eviction_cache = PrivatePrefixCache(binding_sha256=binding, scope_nonce=scope,
                                            max_entries=1, max_bytes=max(checkpoint["nbytes"] * 2, 1))
        eviction_epoch, _ = eviction_cache.begin_cold(binding_sha256=binding, scope_nonce=scope)
        first_store = eviction_cache.commit(prompt_ids, canonical, epoch=eviction_epoch,
                                            binding_sha256=binding, scope_nonce=scope)
        second_store = eviction_cache.commit(prompt_ids[:513], short_caches, epoch=eviction_epoch,
                                             binding_sha256=binding, scope_nonce=scope)
        entry_evicted = (first_store == second_store == "stored" and
                         eviction_cache.stats().entries == 1 and
                         eviction_cache.stats().evictions == 1)
        byte_cache = PrivatePrefixCache(binding_sha256=binding, scope_nonce=scope,
                                        max_entries=1, max_bytes=max(checkpoint["nbytes"] - 1, 1))
        byte_epoch, _ = byte_cache.begin_cold(binding_sha256=binding, scope_nonce=scope)
        byte_skipped = byte_cache.commit(prompt_ids, canonical, epoch=byte_epoch,
                                         binding_sha256=binding, scope_nonce=scope) == "oversize"
        byte_eviction_cache = PrivatePrefixCache(binding_sha256=binding, scope_nonce=scope,
            max_entries=2, max_bytes=checkpoint["nbytes"] + short_checkpoint["nbytes"] - 1)
        byte_eviction_epoch, _ = byte_eviction_cache.begin_cold(binding_sha256=binding, scope_nonce=scope)
        large_stored = byte_eviction_cache.commit(prompt_ids, canonical, epoch=byte_eviction_epoch,
                                                  binding_sha256=binding, scope_nonce=scope)
        small_stored = byte_eviction_cache.commit(prompt_ids[:513], short_caches, epoch=byte_eviction_epoch,
                                                  binding_sha256=binding, scope_nonce=scope)
        byte_evicted = (large_stored == small_stored == "stored"
                       and byte_eviction_cache.stats().entries == 1
                       and byte_eviction_cache.stats().evictions == 1
                       and byte_eviction_cache.stats().bytes == short_checkpoint["nbytes"])

        epoch, _ = private.begin_cold(binding_sha256=binding, scope_nonce=scope)
        if epoch is None:
            raise QualificationFailure("cold_epoch_missing")
        private.clear()
        stale_clear = private.commit(prompt_ids, canonical, epoch=epoch,
                                     binding_sha256=binding, scope_nonce=scope)
        # Repopulate, then verify a real cancelled cold request cannot commit.
        cancel_cache = PrivatePrefixCache(binding_sha256=binding, scope_nonce=scope,
                                          max_entries=1, max_bytes=max(checkpoint["nbytes"] * 2, 1))
        cancel_adapter = IdenticalPromptReuse(model, tokenizer, cancel_cache,
                                              expected_binding_sha256=binding, scope_nonce=scope)
        with redirect_stdout(sys.stderr):
            cancelled = cancel_adapter.stream(prompt_ids, max_tokens=MAX_TOKENS)
            next(cancelled); cancelled.close()
        if cancel_cache.stats().entries or cancel_adapter.last_trace.status != "cancelled":
            raise QualificationFailure("cancel_commit_invalid")
        with redirect_stdout(sys.stderr):
            recovery_cold = _response_record(list(cancel_adapter.stream(prompt_ids, max_tokens=MAX_TOKENS)), mx, np)
            recovery_hit = _response_record(list(cancel_adapter.stream(prompt_ids, max_tokens=MAX_TOKENS)), mx, np)
        if recovery_cold != reference or recovery_hit != reference:
            raise QualificationFailure("cancel_recovery_invalid")
        result.update(recovery_cold=recovery_cold, recovery_hit=recovery_hit,
                      recovery_hit_trace=cancel_adapter.last_trace.__dict__)
        if not cancel_adapter.last_trace.cache_hit:
            raise QualificationFailure("cancel_recovery_not_hit")
        close_epoch, _ = cancel_cache.begin_cold(binding_sha256=binding, scope_nonce=scope)
        cancel_cache.close()
        stale_close = cancel_cache.commit(prompt_ids, canonical, epoch=close_epoch,
                                           binding_sha256=binding, scope_nonce=scope)

        checks = {"wrong_binding": wrong_binding is None and reason_binding == "wrong_binding",
                  "wrong_scope": wrong_scope is None and reason_scope == "wrong_scope",
                  "wrong_key": wrong_key is None and reason_key == "cache_miss",
                  "stale_clear": stale_clear == "stale_epoch", "stale_close": stale_close == "cache_closed",
                  "clone_isolation": True, "cancel_recovery": True,
                  "entry_eviction": entry_evicted, "byte_oversize_skip": byte_skipped,
                  "byte_eviction": byte_evicted,
                  "canonical_unchanged": canonical_after_hits == checkpoint}
        if not all(checks.values()):
            raise QualificationFailure("error_path_gate_failed")
        result.update({"schema": CHILD_SCHEMA, "status": "passed", "correctness_gate": True,
                  "model_id": model_id, "revision": revision, "device": str(mx.default_device()),
                  "prompt_tokens": len(prompt_ids), "prompt_ids_sha256": digest_json(prompt_ids),
                  "stock": stock, "candidate": candidate, "traces": traces,
                  "checkpoint": checkpoint, "canonical_after_hits": canonical_after_hits,
                  "short_checkpoint": short_checkpoint,
                  "checks": checks, "cache_stats": cancel_cache.stats().__dict__,
                  "performance_claim": False, "activation_allowed": False,
                  "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory()})
    except BaseException as exc:
        code = getattr(exc, "code", type(exc).__name__)
        result.update(status="failed", correctness_gate=False, error_code=code)
        return _finish_child(result, 1)
    return _finish_child(result, 0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.identity import runtime_identity
    from friday_evidence.open_observation import OpenObservation
    from ironmule_product.calibration import model_lease
    from ironmule_product.readiness import hardware_identity, probe
    from ironmule_product.state import ProductStore
    from product_load_screen import _exclusive_write, _installed_package_proof
    from product_long_context_reference import _provider_distribution_proof, _snapshot_metadata_proof
    from product_open_validation import ReferenceProcess

    if args.output.exists() or args.output.is_symlink():
        raise QualificationFailure("output_exists")
    store = ProductStore(args.state_dir.expanduser().absolute())
    spec = store.model(args.model)
    if MODELS.get(spec.model_id) != spec.revision: raise QualificationFailure("snapshot_not_frozen")
    source_paths = [Path(__file__), ROOT / "docs/PROD11_NATIVE_CORRECTNESS_SPEC.md",
                    ROOT / "ironmule_product/prefix_reuse.py", ROOT / "tools/product_open_validation.py",
                    ROOT / "tools/product_load_screen.py", ROOT / "tools/product_long_context_reference.py"]
    manifest = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source_paths}
    report = {"schema": SCHEMA, "run_id": uuid.uuid4().hex, "status": "started",
              "model_id": spec.model_id, "revision": spec.revision, "source_before": manifest(),
              "performance_claim": False, "activation_allowed": False, "workers": [],
              "observations": [], "progress": [], "host_observations": []}

    class QualificationProcess(ReferenceProcess):
        def __init__(self, observer):
            self.observer = observer
            self.buffer = bytearray()
            self.process = subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--child"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT)
            observer.bind(self.process, "prefix_correctness")

    def evidence(suffix):
        import importlib.util
        host = {**probe(), **hardware_identity()}
        report["host_observations"].append(host)
        journal.append(report["run_id"], "validation", {"state": "host_observation", "observation": host})
        for name, call in (("source", manifest), ("installed", lambda: _installed_package_proof(USED_MODULES)),
                           ("provider", _provider_distribution_proof),
                           ("model", lambda: _snapshot_metadata_proof(spec.snapshot_path)),
                           ("identity", lambda: runtime_identity(spec.as_dict(), host))):
            try:
                report[name + "_" + suffix] = call()
            except BaseException as exc:
                report.setdefault("evidence_errors", {})[name + "_" + suffix] = type(exc).__name__
                if suffix == "before":
                    raise
        installed_spec = importlib.util.find_spec("ironmule_product.prefix_reuse")
        if installed_spec is not None and installed_spec.origin is not None:
            report["candidate_file_" + suffix] = hashlib.sha256(Path(installed_spec.origin).read_bytes()).hexdigest()
        if report.get("candidate_file_" + suffix) != report["source_" + suffix]["ironmule_product/prefix_reuse.py"]:
            raise QualificationFailure("installed_candidate_differs_from_source")

    try:
        with EventJournal(store.root / "prefix-correctness.sqlite3") as journal, model_lease(store):
            journal.append(report["run_id"], "run_started", {"model_id": spec.model_id,
                           "revision": spec.revision, "source_sha256": canonical_sha256(report["source_before"])})
            child = None
            observer = OpenObservation(on_sample=lambda row: journal.append(report["run_id"], "validation",
                                                                            {"state": "resource", "observation": row}))
            try:
                evidence("before")
                binding = digest_json({"model_id": spec.model_id, "revision": spec.revision,
                    "source": report["source_before"], "installed": report["installed_before"],
                    "model": report["model_before"], "identity": report["identity_before"],
                    "options": {"prefill_step_size": 2048, "greedy": True,
                                "logits_processors": None, "draft": None, "kv_bits": None}})
                child = QualificationProcess(observer)
                child.send({"model_id": spec.model_id, "revision": spec.revision,
                            "snapshot_path": spec.snapshot_path, "binding_sha256": binding})
                while True:
                    event = child.read()
                    _metadata_only(event)
                    if event.get("type") == "ready":
                        if ("worker_ready" in report or event.get("pid") != child.process.pid
                                or "gpu" not in str(event.get("device", "")).lower()
                                or any(type(event.get(k)) is not int or event[k] <= 0
                                       for k in ("mlx_active_bytes", "mlx_peak_bytes"))):
                            raise QualificationFailure("worker_not_gpu_ready")
                        report["worker_ready"] = event
                        journal.append(report["run_id"], "worker_started", event)
                    elif event.get("type") == "progress":
                        if "worker_ready" not in report or len(report["progress"]) >= 16:
                            raise QualificationFailure("worker_progress_invalid")
                        report["progress"].append(event)
                        journal.append(report["run_id"], "sample", event)
                        print(json.dumps({"run_id": report["run_id"], "phase": event.get("phase"),
                                          "index": event.get("index")}), flush=True)
                    elif event.get("type") == "finished":
                        report["child"] = event.get("result")
                        observer.sample(force=True)
                        journal.append(report["run_id"], "validation", {"state": "child_result", "result": report["child"]})
                        if "worker_ready" not in report:
                            raise QualificationFailure("worker_never_ready")
                        validate_child_report(report["child"], spec.model_id, spec.revision)
                        expected_progress = [("stock", i) for i in range(4)] + [("checkpoint", None)] + [
                            ("candidate", i) for i in range(4)] + [("clone_isolation", None)]
                        if [(e.get("phase"), e.get("index")) for e in report["progress"]] != expected_progress:
                            raise QualificationFailure("progress_schedule_invalid")
                        break
                    else:
                        raise QualificationFailure("worker_protocol_invalid")
                report["status"] = "passed"
            except KeyboardInterrupt:
                report.update(status="cancelled", error_code="user_cancelled")
            except BaseException as exc:
                report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
            finally:
                if child is not None:
                    process = child.process
                    report["partial_frame"] = {"bytes": len(child.buffer), "sha256": hashlib.sha256(child.buffer).hexdigest()}
                    try:
                        child.close()
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(getattr(exc, "code", type(exc).__name__))
                        if report["status"] == "passed":
                            report.update(status="failed", error_code="worker_cleanup_failed")
                    finally:
                        report["workers"].append({"pid": process.pid, "returncode": process.poll(),
                                                  "closed": process.poll() is not None})
                report["observations"] = observer.rows
                report["observation_summary"] = observer.summary()
                try:
                    evidence("after")
                except BaseException as exc:
                    report.setdefault("evidence_errors", {})["after_collection"] = getattr(exc, "code", type(exc).__name__)
                if report["status"] == "passed":
                    if report.get("evidence_errors") or any(report.get(name + "_before") != report.get(name + "_after")
                            for name in ("source", "installed", "provider", "model", "identity")):
                        report.update(status="failed", error_code="identity_changed")
                    elif observer.summary()["errors"] or not observer.summary()["memory_sample_count"]:
                        report.update(status="failed", error_code="telemetry_incomplete")
                journal.append(report["run_id"], "run_finished", {"status": report["status"],
                    "error_code": report.get("error_code"), "report_sha256": canonical_sha256(report)})
    except BaseException as exc:
        report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
    finally:
        _exclusive_write(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child: return _child()
    if not args.execute: parser.error("native execution requires --execute")
    if args.state_dir is None or args.model is None or args.output is None: parser.error("missing required execution arguments")
    result = run(args)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "error_code": result.get("error_code")}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
