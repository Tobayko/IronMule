"""Explicit DATA1 commands: planning and status never start an accelerator."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from ..events import EventJournal
from ..canonical import canonical_sha256
from .contracts import (BACKENDS, PARTITIONS, ContractError, code_digest, load_json,
                        make_spec, spec_digest, validate_spec, write_json_new)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ironmule data", allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, default=Path.cwd() / ".friday-data" / "portable")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "dataset", "train"):
        sub.add_parser(name, allow_abbrev=False)
    quota = sub.add_parser("quota", allow_abbrev=False)
    quota.add_argument("--resource", choices=("gpu", "tpu"), default="gpu")
    quota.add_argument("--output", type=Path)
    plan = sub.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--capture", required=True, type=Path)
    plan.add_argument("--backend", required=True, choices=BACKENDS)
    plan.add_argument("--mode", choices=("smoke", "measure"), default="smoke")
    plan.add_argument("--case-limit", type=int, default=1)
    plan.add_argument("--pairs", type=int, default=12)
    plan.add_argument("--output", type=Path)
    capture = sub.add_parser("capture", allow_abbrev=False)
    capture.add_argument("--partition", choices=PARTITIONS, required=True)
    capture.add_argument("--limit", type=int, default=1)
    capture.add_argument("--case-index", type=int, default=0)
    capture.add_argument("--execute", action="store_true")
    capture.add_argument("--product-state-dir", type=Path)
    capture.add_argument("--cache-root", type=Path)
    run = sub.add_parser("run", allow_abbrev=False)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--product-state-dir", type=Path)
    run.add_argument("--account-preflight", type=Path)
    run.add_argument("--owner")
    run.add_argument("--accelerator")
    imp = sub.add_parser("import", allow_abbrev=False)
    imp.add_argument("--report", type=Path, required=True)
    imp.add_argument("--spec", type=Path, required=True)
    dashboard = sub.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=8789)
    probe = sub.add_parser("provider-smoke", allow_abbrev=False)
    probe.add_argument("--backend", choices=("cuda", "tpu"), required=True)
    probe.add_argument("--owner", required=True)
    probe.add_argument("--accelerator", required=True)
    probe.add_argument("--account-preflight", type=Path, required=True)
    probe.add_argument("--kaggle-executable", default=None)
    probe.add_argument("--use-codex-kaggle-credentials", action="store_true")
    probe.add_argument("--execute", action="store_true")
    for command in (quota, run):
        command.add_argument("--kaggle-executable", default=None)
        command.add_argument("--use-codex-kaggle-credentials", action="store_true")
    return parser


def _kaggle_executable(explicit: str | None) -> str:
    if explicit:
        return explicit
    installed = shutil.which("kaggle")
    isolated = Path.cwd() / ".friday-data/tooling/kaggle/bin/kaggle"
    if installed:
        return installed
    if isolated.is_file():
        return str(isolated)
    raise ContractError("kaggle_cli_missing")


def _record(state_dir: Path, run_id: str, kind: str, payload: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with EventJournal(state_dir / "control.sqlite3") as journal:
        journal.append(run_id, kind, payload)


def _capture_origin(state_dir: Path, spec: dict) -> str:
    """Require locally supervised real captures, not a source-kind assertion."""
    known: dict[str, dict] = {}
    proofs = []
    for manifest_path in sorted((state_dir / "captures").glob("*/data/capture.json")):
        parent = manifest_path.parent.parent
        supervisor_path = parent / "supervisor.json"
        if not supervisor_path.is_file():
            continue
        supervision = load_json(supervisor_path)
        if (supervision.get("status") != "finished" or supervision.get("code_unchanged") is not True
                or supervision.get("module") != "friday_evidence.portable.capture"):
            continue
        with EventJournal(state_dir / "supervisor-events.sqlite3", read_only=True) as journal:
            terminal = journal.latest(run_id=supervision["run_id"])
        if not terminal or terminal["kind"] != "run_finished" or terminal["payload"].get("status") != "finished":
            continue
        manifest = load_json(manifest_path)
        if manifest.get("status") != "captured":
            continue
        for case in manifest.get("cases", []):
            known[case["case_id"]] = case
        proofs.append(canonical_sha256({"manifest": manifest, "supervisor": supervision}))
    for case in spec["cases"]:
        if canonical_sha256(case) != canonical_sha256(known.get(case["case_id"])):
            raise ContractError("case_has_no_verified_local_capture")
    return canonical_sha256(sorted(proofs))


def _capture(args) -> dict:
    from .capture import MODEL_ID, REVISION
    if not 1 <= args.limit <= 4 or args.case_index < 0 or args.case_index + args.limit > 4:
        raise ContractError("capture_limit_must_be_one_to_four")
    if not args.execute:
        return {"status": "planned", "model_id": MODEL_ID, "partition": args.partition,
                "case_limit": args.limit, "case_index": args.case_index, "hardware_started": False}
    from ironmule_inventory import discover_models
    from .supervisor import run_local_worker
    if args.cache_root:
        roots = [args.cache_root]
    else:
        local = Path.cwd() / ".friday-data/models/hub"
        roots = [local] if local.is_dir() else None
    matches = [row for row in discover_models(cache_roots=roots, loader="mlx_lm")
               if row["model_id"] == MODEL_ID and row["revision"] == REVISION and row["status"] == "available"]
    if len(matches) != 1:
        raise ContractError("frozen_model_missing_or_ambiguous")
    model_spec = {key: matches[0][key] for key in ("model_id", "revision", "snapshot_path", "weight_bytes")}
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    captures = args.state_dir / "captures"
    captures.mkdir(mode=0o700, exist_ok=True)
    destination = captures / uuid.uuid4().hex
    destination.mkdir(mode=0o700)
    data = destination / "data"
    data.mkdir(mode=0o700, exist_ok=False)
    model_file = destination / "model-spec.json"
    write_json_new(model_file, model_spec)
    outcome = run_local_worker("friday_evidence.portable.capture", ["--model-spec", str(model_file),
        "--partition", args.partition, "--output-dir", str(data), "--limit", str(args.limit),
        "--case-index", str(args.case_index)],
        destination, timeout_seconds=min(900, 90*args.limit + 60*(args.limit-1) + 60), state_dir=args.state_dir,
        product_state_dir=args.product_state_dir)
    return {"status": outcome["status"], "capture_file": str(data / "capture.json"),
            "supervisor_file": str(destination / "supervisor.json"), "performance_claim": False}


def _run_local(args, spec: dict) -> dict:
    from .supervisor import run_local_worker
    from .corpus import import_report
    destination = args.state_dir / "runs" / spec["run_id"]
    result = run_local_worker("friday_evidence.portable.runner", ["--spec", str(args.spec.resolve()),
        "--data-dir", str(args.data_dir.resolve()), "--output-dir", str(destination.resolve())],
        destination, timeout_seconds=spec["server_seconds"], state_dir=args.state_dir,
        product_state_dir=args.product_state_dir)
    report_path = destination / f"{spec['run_id']}.report.json"
    imported = None
    outcome_status = result["status"]
    if report_path.is_file():
        report = load_json(report_path)
        if result["status"] != "finished" or result.get("code_unchanged") is not True:
            # An otherwise complete child report cannot override parent failure.
            _record(args.state_dir, spec["run_id"], "validation", {"accepted": False, "reason": "supervisor_rejected"})
            imported = import_report(args.state_dir, report, spec)
            if result.get("error_code") == "worker_nonzero_exit" and report.get("status") == "censored":
                outcome_status = "censored"
        else:
            if report.get("status") in ("complete", "smoke"):
                _record(args.state_dir, spec["run_id"], "validation", {
                    "verified_execution": True, "backend": spec["backend"],
                    "spec_sha256": spec_digest(spec), "report_sha256": canonical_sha256(report),
                    "terminal_state": "complete", "supervisor_sha256": canonical_sha256(result)})
            imported = import_report(args.state_dir, report, spec)
    return {"status": outcome_status, "supervisor_file": str(destination / "supervisor.json"),
            "report_file": str(report_path), "import": imported, "performance_claim": False}


def _run_cloud(args, spec: dict) -> dict:
    from .bundle import prepare_bundle
    from .corpus import import_report
    from .kaggle import KaggleCli, KaggleCliError
    from .quota import AccountPreflight, QuotaController
    if not args.account_preflight or not args.owner or not args.accelerator:
        raise ContractError("cloud_requires_current_account_session_preflight_owner_and_free_sku")
    if spec["backend"] == "cuda" and args.accelerator != "NvidiaTeslaT4":
        raise ContractError("cuda_pilot_requires_verified_t4")
    if spec["backend"] == "tpu" and not args.accelerator.startswith("Tpu"):
        raise ContractError("tpu_pilot_requires_verified_tpu_sku")
    preflight = load_json(args.account_preflight)
    allowed = {field.name for field in fields(AccountPreflight)}
    values = {key: value for key, value in preflight.items() if key in allowed}
    if isinstance(values.get("supported_free_skus"), list):
        values["supported_free_skus"] = tuple(values["supported_free_skus"])
    account = AccountPreflight(**values)
    account.validate(time.time())
    client = KaggleCli(_kaggle_executable(args.kaggle_executable))
    resource = "gpu" if spec["backend"] == "cuda" else "tpu"
    quota = client.quota(resource=resource)
    controller = QuotaController(args.state_dir / "quota.sqlite3")
    controller.preflight(quota, account)
    destination = args.state_dir / "cloud" / spec["run_id"]
    started = time.monotonic()
    reservation = controller.reserve(quota, account, run_slug=f"data1-{spec['run_id']}",
        timeout_seconds=spec["server_seconds"], smoke=spec["mode"] == "smoke")
    spec = {**spec, "quota_reservation_sha256": canonical_sha256(reservation.to_dict())}
    try:
        bundle = prepare_bundle(spec, args.data_dir, destination, owner=args.owner)
        # Staging does not start an accelerator; the reservation still covers
        # all elapsed orchestration time conservatively.
        dataset_id = client.create_dataset(bundle["dataset_dir"])
        dataset_deadline = time.monotonic() + 120
        while True:
            dataset_state = client.dataset_state(dataset_id)
            if dataset_state == "ready":
                break
            if dataset_state == "failed" or time.monotonic() >= dataset_deadline:
                raise KaggleCliError("private_dataset_did_not_become_ready")
            time.sleep(5)
    except BaseException:
        controller.mark_submission_unknown(reservation)
        raise
    write_json_new(destination / "launch-spec.json", spec)
    write_json_new(destination / "reservation.json", reservation.to_dict())
    kernel_id = bundle["kernel_id"]
    metadata_path = Path(bundle["notebook_dir"]) / "kernel-metadata.json"
    try:
        client.push(bundle["notebook_dir"], accelerator=args.accelerator,
                    timeout_seconds=spec["server_seconds"], account=account)
    except BaseException:
        controller.mark_submission_unknown(reservation)
        raise
    terminal = None
    try:
        while time.monotonic() - started < spec["server_seconds"] + 120:
            state = client.job_state(kernel_id, metadata_path=metadata_path)
            if state in ("complete", "failed", "cancelled"):
                terminal = state
                break
            time.sleep(10)
        if terminal is None:
            controller.mark_terminal_unknown(reservation)
            raise KaggleCliError("session_end_unconfirmed_requires_account_review")
        after = client.quota(resource=resource)
        reconciliation = controller.reconcile_terminal(reservation, after,
            actual_elapsed_seconds=time.monotonic()-started, terminal=True)
    except BaseException:
        # Preserve the reservation on ambiguous polling, interrupt or quota errors.
        controller.mark_terminal_unknown(reservation)
        raise
    results = destination / "results"
    results.mkdir(mode=0o700)
    client.output(kernel_id, results, metadata_path=metadata_path)
    reports = list(results.rglob(f"{spec['run_id']}.report.json"))
    imported = None
    collection_success = False
    if len(reports) == 1:
        report = load_json(reports[0])
        if terminal == "complete" and report.get("status") in ("complete", "smoke"):
            collection_success = True
            _record(args.state_dir, spec["run_id"], "validation", {
                "verified_execution": True, "backend": spec["backend"],
                "spec_sha256": spec_digest(spec), "report_sha256": canonical_sha256(report),
                "terminal_state": "complete", "reservation_sha256": spec["quota_reservation_sha256"],
                "reconciliation_sha256": canonical_sha256(reconciliation)})
        imported = import_report(args.state_dir, report, spec)
    if not collection_success:
        # No retry/campaign continuation after a failed accelerator job.
        controller.mark_terminal_unknown(reservation)
    outcome = {"status": "complete" if collection_success else "failed", "provider_terminal_state": terminal,
               "run_id": spec["run_id"], "kernel_id": kernel_id,
               "quota": reconciliation, "import": imported, "performance_claim": False}
    write_json_new(destination / "outcome.json", outcome)
    return outcome


def _dispatch(args) -> dict | None:
    command = args.command
    if command == "capture":
        return _capture(args)
    if command == "plan":
        capture = load_json(args.capture)
        if capture.get("schema") != "ironmule.capture.v1" or capture.get("status") != "captured":
            raise ContractError("complete_real_capture_required")
        if not 1 <= args.case_limit <= 12:
            raise ContractError("invalid_case_limit")
        spec = make_spec(capture["cases"][:args.case_limit], args.backend, mode=args.mode, pairs=args.pairs)
        if args.output:
            write_json_new(args.output, spec)
        return {"status": "planned", "spec_sha256": spec_digest(spec), "spec": spec, "hardware_started": False}
    if command == "quota":
        from .kaggle import KaggleCli
        observed = KaggleCli(_kaggle_executable(args.kaggle_executable)).quota(resource=args.resource)
        result = {"schema": "ironmule.quota-observation.v1", **observed.to_dict(), "hardware_started": False}
        if args.output:
            write_json_new(args.output, result)
        return result
    if command == "run":
        spec = validate_spec(load_json(args.spec))
        if spec.get("code_sha256") != code_digest():
            raise ContractError("code_changed_since_plan")
        if not args.execute:
            return {"status": "planned", "run_id": spec["run_id"], "backend": spec["backend"],
                    "server_seconds": spec["server_seconds"], "hardware_started": False}
        origin_sha = _capture_origin(args.state_dir, spec)
        _record(args.state_dir, spec["run_id"], "run_started", {"backend": spec["backend"], "spec_sha256": spec_digest(spec)})
        _record(args.state_dir, spec["run_id"], "validation", {"capture_origin_sha256": origin_sha})
        try:
            outcome = _run_local(args, spec) if spec["backend"] == "mlx" else _run_cloud(args, spec)
        except Exception as exc:
            _record(args.state_dir, spec["run_id"], "run_finished", {"status": "failed", "error_type": type(exc).__name__})
            raise
        _record(args.state_dir, spec["run_id"], "run_finished", {"status": outcome["status"]})
        return outcome
    if command in ("dataset", "import"):
        from .corpus import build_dataset, import_report
        return build_dataset(args.state_dir) if command == "dataset" else import_report(args.state_dir, load_json(args.report), load_json(args.spec))
    if command == "train":
        from .learning import train_and_evaluate
        return train_and_evaluate(args.state_dir)
    if command == "status":
        from .dashboard import snapshot
        return snapshot(args.state_dir)
    if command == "dashboard":
        from .dashboard import serve
        serve(args.state_dir, port=args.port)
        return None
    if command == "provider-smoke":
        if not args.execute:
            return {"status":"planned", "backend":args.backend,
                    "accelerator":args.accelerator, "hardware_started":False,
                    "performance_claim":False}
        from .kaggle_probe import load_account_preflight, run_provider_smoke
        return run_provider_smoke(state_dir=args.state_dir, backend=args.backend,
            owner=args.owner, accelerator=args.accelerator,
            account=load_account_preflight(args.account_preflight),
            executable=_kaggle_executable(args.kaggle_executable))
    raise ContractError("unknown_data_command")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.state_dir = args.state_dir.expanduser().absolute()
    context = nullcontext()
    if getattr(args, "use_codex_kaggle_credentials", False):
        from .credentials import configured_kaggle_credentials
        context = configured_kaggle_credentials()
    try:
        with context:
            result = _dispatch(args)
        if result is not None:
            print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("status") in {
            "blocked", "failed", "deferred", "censored", "rejected", "cancelled"} else 0
    except (ValueError, RuntimeError, OSError) as exc:
        # Do not leak private paths, provider output or tokens via exception text.
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__,
                          "reason": str(exc) if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", str(exc)) else type(exc).__name__,
                          "performance_claim": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
