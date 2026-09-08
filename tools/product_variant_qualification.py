"""PROD14 native correctness screen for one explicit product worker variant.

The controller imports neither MLX nor model code. Native execution is closed
behind ``--execute`` and records correctness/lifecycle evidence, never speedup.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_open_validation import (
    MODELS, ValidationFailure, cases, digest, http_sample,
    observed_call, output_metadata, product_sample,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ironmule.prod14_variant_qualification.v1"
VARIANTS = ("prefix_reuse", "current_engine")
ORACLES = {
    "mlx-community/gemma-3-1b-it-4bit": ROOT / "research/raw/PROD10_1B_open_20260907_attempt1.json",
    "mlx-community/gemma-3-4b-it-4bit": ROOT / "research/raw/PROD10_4B_open_20260907_attempt1.json",
    "mlx-community/gemma-3-12b-it-4bit": ROOT / "research/raw/PROD10_12B_open_20260907_attempt2.json",
}
POLICY = {
    "request_timeout_s": None, "startup_timeout_s": None,
    "work_budget_s": None, "continuous_budget_s": None,
    "duty_cycle_limit": None, "required_pause_s": 0,
    "rss_stop_bytes": None, "swap_stop_bytes": None,
    "host_readiness_gate": False, "automatic_retry": False,
}


class FreshReferenceRequired(ValidationFailure):
    def __init__(self, reason: str):
        super().__init__("fresh_reference_required")
        self.reason = reason


def long_case(max_tokens: int = 8) -> dict:
    case = dict(cases()[0])
    case["max_tokens"] = max_tokens
    return case


def exact_output(actual: dict, expected: dict) -> bool:
    keys = ("output_sha256", "token_sha256", "text_sha256", "prompt_tokens",
            "completion_tokens", "finish_reason")
    return all(actual.get(key) == expected.get(key) for key in keys)


def load_correctness_oracle(model_id: str, revision: str, current_identity: dict,
                            current_provider: dict) -> tuple[dict, dict]:
    """Load a frozen PROD10 oracle without importing its timing observations."""
    path = ORACLES[model_id]
    raw_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        value = json.loads(path.read_text())
        identity = value["identity_before"]
        rows = [row for row in value["samples"]
                if row.get("backend") == "stock" and row.get("case") == "long_8"]
        binding_keys = ("model_sha256", "environment_sha256", "hardware_sha256")
        output_keys = ("output_sha256", "token_sha256", "text_sha256", "prompt_tokens",
                       "completion_tokens", "finish_reason", "prompt_ids_sha256")
        stable = len(rows) == 4 and all(
            all(row.get(key) == rows[0].get(key) for key in output_keys) for row in rows)
        valid = (value.get("status") == "passed" and value.get("model_id") == model_id
                 and value.get("revision") == revision and identity.get("revision") == revision
                 and all(identity.get(key) == current_identity.get(key) for key in binding_keys)
                 and value.get("identity_after") == identity
                 and value.get("provider_before") == current_provider
                 and value.get("provider_after") == current_provider
                 and stable and all(isinstance(rows[0].get(key), str) and rows[0][key]
                                    for key in ("output_sha256", "token_sha256", "text_sha256",
                                                "prompt_ids_sha256")))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        valid, rows = False, []
    evidence = {"path": (path.relative_to(ROOT).as_posix()
                         if path.is_relative_to(ROOT) else str(path)), "sha256": raw_sha256,
                "reused_for_correctness_only": True, "timings_reused": False,
                "stock_reused_records": len(rows) if valid else 0}
    if not valid:
        raise FreshReferenceRequired("oracle_status_identity_provider_or_rows_invalid")
    golden = {key: rows[0][key] for key in output_keys}
    return golden, evidence


def prefix_expectation(metadata: dict | None, *, hit: bool) -> None:
    prefix = metadata.get("prefix_cache") if isinstance(metadata, dict) else None
    trace = prefix.get("trace") if isinstance(prefix, dict) else None
    stats = prefix.get("stats") if isinstance(prefix, dict) else None
    expected_reused = 1076 if hit else 0
    if (not isinstance(prefix, dict) or prefix.get("status") != "accepted"
            or prefix.get("commit_status") != ("no_pending_commit" if hit else "stored")
            or not isinstance(trace, dict) or trace.get("status") != "completed"
            or trace.get("cache_hit") is not hit
            or trace.get("cache_stored") is not (not hit)
            or trace.get("reused_tokens") != expected_reused
            or not isinstance(stats, dict) or stats.get("entries", 0) < 1
            or stats.get("bytes", 0) < 1):
        raise ValidationFailure("prefix_hit_sequence_invalid")


def engine_metadata(metadata: dict | None, *, completion: bool = True) -> dict:
    engine = metadata.get("engine") if isinstance(metadata, dict) else None
    required = {"current_identity", "profile_source", "selected_knobs"}
    if not isinstance(engine, dict) or not required.issubset(engine):
        raise ValidationFailure("engine_metadata_incomplete")
    if completion and (engine.get("fallback_used") is not False or engine.get("fallback_count") != 0):
        raise ValidationFailure("engine_fallback_observed")
    return engine


def cancelled_sample(client, spec, observer, prompt_ids_sha256: str) -> dict:
    """Cancel after the first actual worker token and drain the terminal event."""
    from ironmule_product.types import GenerationRequest
    case = long_case(128)
    request = GenerationRequest(spec.model_id,
        tuple((m["role"], m["content"]) for m in case["messages"]),
        max_tokens=128, temperature=0.0, top_p=1.0)
    cancel = threading.Event()
    actual_event = threading.Event()
    tokens, pieces, done = [], [], None

    def consume():
        nonlocal done
        stream = client.stream(request, cancel=cancel, timeout=None)
        try:
            for event in stream:
                if event["type"] == "token":
                    tokens.append(event["token_id"])
                    pieces.append(event["text"])
                    actual_event.set()
                    cancel.set()
                elif event["type"] == "done":
                    done = event
        finally:
            stream.close()

    observed_call(consume, observer)
    metadata = client.last_generation_metadata
    prefix = metadata.get("prefix_cache") if isinstance(metadata, dict) else None
    trace = prefix.get("trace") if isinstance(prefix, dict) else None
    stats = prefix.get("stats") if isinstance(prefix, dict) else None
    if (not actual_event.is_set() or done is None or done.get("finish_reason") != "cancelled"
            or not isinstance(trace, dict) or trace.get("cache_stored") is not False
            or not isinstance(stats, dict) or stats.get("entries") != 0
            or not isinstance(metadata, dict)
            or metadata.get("prompt_ids_sha256") != prompt_ids_sha256):
        raise ValidationFailure("prefix_cancellation_invalid")
    return {"backend": "candidate_cancel", "case": "long_128",
            "actual_token_observed": True, "partial_completion_tokens": len(tokens),
            "partial_token_sha256": digest(tokens),
            "partial_text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
            "finish_reason": "cancelled", "metadata": metadata}


def _source_manifest(model_id: str) -> dict[str, str]:
    paths = (
        Path(__file__), ROOT / "docs/PROD14_NATIVE_SCREEN_SPEC.md",
        ROOT / "tools/product_open_validation.py",
        ROOT / "tools/product_load_screen.py",
        ROOT / "tools/product_long_context_reference.py",
        ROOT / "friday_evidence/canonical.py", ROOT / "friday_evidence/events.py",
        ROOT / "friday_evidence/identity.py", ROOT / "friday_evidence/open_observation.py",
        ROOT / "friday_evidence/process_memory.py", ROOT / "friday_evidence/storage.py",
        ROOT / "ironmule_product/backend.py", ROOT / "ironmule_product/worker.py",
        ROOT / "ironmule_product/service.py", ROOT / "ironmule_product/http_server.py",
        ROOT / "ironmule_product/state.py", ROOT / "ironmule_product/types.py",
        ROOT / "ironmule_product/model_policy.py", ROOT / "ironmule_product/calibration.py",
        ROOT / ("ironmule_product/prefix_reuse.py"),
        ROOT / ("ironmule_product/prefix_session.py"),
        ROOT / ("ironmule_product/engine_bridge.py"), ORACLES[model_id],
    )
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def run(state_dir: Path, model_id: str, variant: str, output: Path) -> dict:
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.identity import runtime_identity
    from friday_evidence.open_observation import OpenObservation
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.calibration import model_lease
    from ironmule_product.http_server import create_server
    from ironmule_product.readiness import hardware_identity, probe
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from product_load_screen import _exclusive_write, _installed_package_proof
    from product_long_context_reference import _provider_distribution_proof, _snapshot_metadata_proof

    if output.exists() or output.is_symlink():
        raise ValidationFailure("output_exists")
    if variant not in VARIANTS:
        raise ValidationFailure("variant_invalid")
    store = ProductStore(state_dir)
    spec = store.model(model_id)
    if MODELS.get(model_id) != spec.revision:
        raise ValidationFailure("snapshot_not_frozen")
    run_id = uuid.uuid4().hex
    manifest = lambda: _source_manifest(model_id)
    report = {"schema": SCHEMA, "run_id": run_id, "model_id": model_id,
              "revision": spec.revision, "variant": variant, "status": "started",
              "policy": POLICY, "performance_claim": False, "activation_allowed": False,
              "source_before": manifest(), "samples": [], "workers": [],
              "host_observations": [], "output_accounting": {
                  "stock_reused_records": 4, "stock_new_runs": 0,
                  "planned_candidate_new_full": 8 if variant == "prefix_reuse" else 7,
                  "planned_candidate_partial_cancelled": 1 if variant == "prefix_reuse" else 0}}
    reference = client = server = service = observer = process = None
    server_thread = None
    journal = EventJournal(state_dir / "variant-qualification.sqlite3")
    host = None
    used = ("friday_evidence.canonical", "friday_evidence.events", "friday_evidence.identity",
            "friday_evidence.open_observation", "friday_evidence.process_memory",
            "friday_evidence.storage", "ironmule_product.backend", "ironmule_product.worker",
            "ironmule_product.service", "ironmule_product.http_server", "ironmule_product.state",
            "ironmule_product.types", "ironmule_product.model_policy",
            "ironmule_product.calibration", "ironmule_product.prefix_reuse",
            "ironmule_product.prefix_session",
            "ironmule_product.engine_bridge")
    def cleanup_while_journal_open():
        nonlocal reference, client, server, service
        for action in ((lambda: server.shutdown()) if server else None,
                       (lambda: server.server_close()) if server else None,
                       (lambda: service.close()) if service else None,
                       (lambda: client.close()) if client else None):
            if action:
                try: action()
                except BaseException as exc: report.setdefault("cleanup_errors", []).append(type(exc).__name__)
        server = service = client = None
        if reference is not None:
            try:
                row = reference.close(); report["workers"].append({"backend": "stock", **row})
            except BaseException as exc: report.setdefault("cleanup_errors", []).append(type(exc).__name__)
            reference = None
    def finalize_while_journal_open(exc_type, exc, _traceback):
        if exc is not None:
            report.update(status="failed", error_code=getattr(exc, "code", exc_type.__name__),
                          error_type=exc_type.__name__)
            if isinstance(getattr(exc, "reason", None), str):
                report["error_reason"] = exc.reason
        if process is not None and not any(row.get("pid") == process.pid for row in report["workers"]):
            report["workers"].append({"backend": variant, "pid": process.pid,
                "returncode": process.poll(), "closed": process.poll() is not None})
        if observer is not None:
            report["observation_summary"] = observer.summary()
        try:
            if host is not None:
                fresh_host = {**probe(), **hardware_identity()}
                report["host_observations"].append(fresh_host)
                report["identity_after"] = runtime_identity(spec.as_dict(), fresh_host)
            report["provider_after"] = _provider_distribution_proof()
            report["metadata_after"] = _snapshot_metadata_proof(spec.snapshot_path)
            report["installed_after"] = _installed_package_proof(used)
            report["source_after"] = manifest()
            for key in ("identity", "provider", "metadata", "installed", "source"):
                before, after = f"{key}_before", f"{key}_after"
                if before in report and report[before] != report[after]:
                    raise ValidationFailure(f"{key}_changed")
        except BaseException as after_exc:
            report.update(status="failed", error_code=getattr(after_exc, "code", type(after_exc).__name__),
                          error_type=type(after_exc).__name__)
        if report.get("cleanup_errors"):
            report.update(status="failed", error_code=report.get("error_code") or "cleanup_failed")
        if report["status"] == "started":
            report["status"] = "passed"
        report["output_accounting"]["actual_new_full"] = sum(
            row.get("backend") in {"candidate_direct", "candidate_http", "candidate_recovery"}
            for row in report["samples"])
        report["output_accounting"]["actual_partial_cancelled"] = sum(
            row.get("backend") == "candidate_cancel" for row in report["samples"])
        report["output_accounting"]["stock_reused_records"] = sum(
            row.get("backend") == "stock_oracle_reused" for row in report["samples"])
        report["terminal_payload_sha256"] = digest({"status": report["status"],
            "error_code": report.get("error_code"), "samples": report["samples"],
            "workers": report["workers"], "source_after": report.get("source_after")})
        journal.append(run_id, "run_finished", {"status": report["status"],
            "error_code": report.get("error_code"), "samples": len(report["samples"]),
            "terminal_payload_sha256": report["terminal_payload_sha256"],
            "report_sha256": digest(report), "workers": report["workers"]})
        return False
    try:
        with ExitStack() as stack:
            stack.enter_context(journal)
            stack.enter_context(model_lease(store))
            stack.push(finalize_while_journal_open)
            stack.callback(cleanup_while_journal_open)
            journal.append(run_id, "run_started", {"model_id": model_id, "revision": spec.revision,
                "variant": variant, "source_sha256": canonical_sha256(report["source_before"])})
            report["installed_before"] = _installed_package_proof(used)
            report["provider_before"] = _provider_distribution_proof()
            report["metadata_before"] = _snapshot_metadata_proof(spec.snapshot_path)
            host = {**probe(), **hardware_identity()}
            report["host_observations"].append(host)
            report["identity_before"] = runtime_identity(spec.as_dict(), host)
            observer = OpenObservation(on_sample=lambda row: journal.append(run_id, "validation",
                {"state": "resource_observation", "observation": row}))

            def save(row):
                report["samples"].append(row)
                journal.append(run_id, "validation", {"state": "request_sample", "sample": row})

            baseline, report["stock_oracle"] = load_correctness_oracle(
                model_id, spec.revision, report["identity_before"], report["provider_before"])
            for repeat in range(4):
                save({"backend": "stock_oracle_reused", "case": "long_8", "repeat": repeat,
                      "reused_record": True, "timing_reused": False, **baseline})

            client = MLXWorkerClient(spec, startup_timeout=None, execution_variant=variant,
                                     trace_prompt_identity=True)
            def startup(_pid):
                nonlocal process
                process = client._process
                observer.bind(process, variant)
                observer.sample()
            ready = client.start(startup_guard=startup)
            report["candidate_ready"] = ready
            if variant == "current_engine":
                report["engine_ready_metadata"] = engine_metadata(
                    {"engine": ready.get("engine")}, completion=False)

            direct_count = 4
            for repeat in range(direct_count):
                sample = observed_call(lambda: product_sample(client, spec, long_case(), {}), observer)
                metadata = client.last_generation_metadata
                if metadata.get("prompt_ids_sha256") != baseline["prompt_ids_sha256"]:
                    raise ValidationFailure("candidate_prompt_identity_mismatch")
                if variant == "prefix_reuse":
                    prefix_expectation(metadata, hit=repeat > 0)
                else:
                    engine_metadata(metadata)
                    # The engine bridge is completion-only: its buffered token
                    # replay cannot establish time to first native token.
                    sample.pop("first_token_seconds", None)
                sample.update(backend="candidate_direct", case="long_8", repeat=repeat,
                              warmup=repeat == 0, metadata=metadata,
                              matches_reference=exact_output(sample, baseline))
                save(sample)
                if not sample["matches_reference"]:
                    raise ValidationFailure("candidate_reference_mismatch")

            if variant == "prefix_reuse":
                cleared = client.clear_prefix_cache()
                report["clear_before_http"] = cleared
                if cleared["stats"]["entries"] or cleared["stats"]["bytes"]:
                    raise ValidationFailure("prefix_clear_invalid")

                report["clear_before_cancel"] = client.clear_prefix_cache()
                cancellation = cancelled_sample(
                    client, spec, observer, baseline["prompt_ids_sha256"])
                cancellation["worker_pid"] = process.pid
                save(cancellation)
                recovery = observed_call(lambda: product_sample(client, spec, long_case(), {}), observer)
                if client.last_generation_metadata.get("prompt_ids_sha256") != baseline["prompt_ids_sha256"]:
                    raise ValidationFailure("candidate_prompt_identity_mismatch")
                prefix_expectation(client.last_generation_metadata, hit=False)
                recovery.update(backend="candidate_recovery", case="long_8",
                    metadata=client.last_generation_metadata,
                    matches_reference=exact_output(recovery, baseline))
                save(recovery)
                if not recovery["matches_reference"]:
                    raise ValidationFailure("recovery_reference_mismatch")
                report["cancellation_recovery"] = {"same_worker_pid": process.pid,
                    "worker_ready": client.ready, "exact_recovery": True}

                # The recovery populated the cache; HTTP must start cold.
                cleared = client.clear_prefix_cache()
                report["clear_before_http"] = cleared
                if cleared["stats"]["entries"] or cleared["stats"]["bytes"]:
                    raise ValidationFailure("prefix_clear_invalid")

            with tempfile.TemporaryDirectory(prefix="ironmule-prod14-service-") as tmp:
                transport = ProductStore(Path(tmp)); transport.setup("server")
                transport.set_request_timeout(None); transport.register_model(spec)
                service = ProductService(transport, backend=client, spec=spec)
                server = create_server(service, port=0)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start(); port = server.server_address[1]
                for repeat in range(3):
                    sample = observed_call(lambda: http_sample(port, model_id, long_case()), observer)
                    metadata = client.last_generation_metadata
                    if metadata.get("prompt_ids_sha256") != baseline["prompt_ids_sha256"]:
                        raise ValidationFailure("candidate_prompt_identity_mismatch")
                    if variant == "prefix_reuse":
                        prefix_expectation(metadata, hit=repeat > 0)
                    else:
                        engine_metadata(metadata)
                    sample.update(backend="candidate_http", case="long_8", repeat=repeat,
                                  metadata=metadata, matches_reference=(sample["text_sha256"] == baseline["text_sha256"]
                                  and sample["prompt_tokens"] == baseline["prompt_tokens"]
                                  and sample["completion_tokens"] == baseline["completion_tokens"]
                                  and sample["finish_reason"] == baseline["finish_reason"]))
                    save(sample)
                    if not sample["matches_reference"]:
                        raise ValidationFailure("http_reference_mismatch")
                server.shutdown(); server.server_close(); server_thread.join(timeout=5); server = None
                service.close(); service = None

            candidate_process = process
            client.close(); client = None
            report["workers"].append({"backend": variant, "pid": candidate_process.pid,
                "returncode": candidate_process.poll(), "closed": candidate_process.poll() is not None})
            if candidate_process.poll() != 0:
                raise ValidationFailure("candidate_shutdown_failed")
            observer.bind(None, "complete")
            if observer.summary()["errors"]:
                raise ValidationFailure("resource_observation_incomplete")
    except BaseException as exc:
        if report["status"] == "started":
            report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__),
                          error_type=type(exc).__name__, source_after=manifest())
            if isinstance(getattr(exc, "reason", None), str):
                report["error_reason"] = exc.reason
            report["terminal_payload_sha256"] = digest({"status": report["status"],
                "error_code": report["error_code"], "samples": report["samples"],
                "workers": report["workers"], "source_after": report["source_after"]})
    finally:
        _exclusive_write(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute or None in (args.state_dir, args.model, args.variant, args.output):
        parser.error("--execute, --state-dir, --model, --variant and --output are required")
    result = run(args.state_dir, args.model, args.variant, args.output)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "error_code": result.get("error_code"), "samples": len(result["samples"])}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
