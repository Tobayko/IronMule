"""Synthetic control-path tests for the portable free-tier quota boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess

import pytest

from friday_evidence.portable.kaggle import (
    KaggleCli,
    KaggleCliError,
    parse_dataset_state,
    parse_job_state,
    parse_terminal_status,
    validate_private_dataset_metadata,
    validate_private_metadata,
)
from friday_evidence.portable.quota import (
    AccountPreflight,
    QuotaController,
    QuotaError,
    QuotaSnapshot,
)


NOW = 1_000.0


def _snapshot(*, remaining: float = 99_000.0, observed: float = NOW) -> QuotaSnapshot:
    return QuotaSnapshot(
        resource="GPU",
        total_seconds=100_000.0,
        used_seconds=100_000.0 - remaining,
        remaining_seconds=remaining,
        refresh_at_unix_s=2_000.0,
        observed_at_unix_s=observed,
    )


def _account(
    *, session: bool = True, checked: float = NOW, additional_usage: bool = False,
    skus: tuple[str, ...] = ("NvidiaTeslaT4",),
) -> AccountPreflight:
    return AccountPreflight(True, True, session, checked, checked, additional_usage, skus, True)


def _controller(tmp_path) -> QuotaController:
    return QuotaController(tmp_path / "quota-events.sqlite", clock=lambda: NOW)


def test_snapshot_parses_explicit_current_gpu_csv_and_serializes() -> None:
    snapshot = QuotaSnapshot.from_csv(
        "currentGPU,total,used,remain,refreshAt\ntrue,100000,1000,99000,2000\n",
        observed_at_unix_s=NOW,
    )
    assert snapshot.resource == "gpu"
    assert snapshot.is_fresh(NOW + 60)
    assert snapshot.to_dict()["remaining_seconds"] == 99_000.0


def test_snapshot_parses_actual_kaggle_224_quota_csv_with_utc_naive_refresh() -> None:
    actual_cli_csv = (
        "resource,used,remaining,total,refreshAt\n"
        "GPU,0.00h,30.00h,30.00h,2026-09-12T00:00:00\n"
        "TPU,0.00h,20.00h,20.00h,2026-09-12T00:00:00\n"
    )
    observed = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()
    snapshot = QuotaSnapshot.from_csv(actual_cli_csv, observed_at_unix_s=observed, resource="gpu")
    assert snapshot.total_seconds == 108_000
    assert snapshot.remaining_seconds == 108_000
    assert snapshot.display_resolution_seconds == 36
    assert snapshot.refresh_at_unix_s == datetime(2026, 9, 12, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize(
    "csv_payload",
    [
        "resource,total,used,remaining,refreshAt\nGPU,100,NaN,100,2000\n",
        "resource,total,used,remaining,refreshAt\nGPU,100,5,80,2000\n",
        "resource,total,used,remaining,refreshAt\nGPU,100,5,95,999\n",
    ],
)
def test_snapshot_rejects_invalid_or_inconsistent_provider_data(csv_payload: str) -> None:
    with pytest.raises(QuotaError):
        QuotaSnapshot.from_csv(csv_payload, observed_at_unix_s=NOW)


def test_preflight_requires_fresh_quota_and_independent_account_session_evidence(tmp_path) -> None:
    controller = _controller(tmp_path)
    with pytest.raises(QuotaError, match="active account session"):
        controller.preflight(_snapshot(), _account(session=False))
    with pytest.raises(QuotaError, match="stale"):
        controller.preflight(_snapshot(observed=NOW - 61), _account())
    with pytest.raises(QuotaError, match="additional active"):
        controller.preflight(_snapshot(), _account(additional_usage=True))
    unverified = AccountPreflight(True, True, True, NOW, NOW, False, ("NvidiaTeslaT4",), False)
    with pytest.raises(QuotaError, match="prerequisites"):
        controller.preflight(_snapshot(), unverified)


def test_reservation_is_bounded_global_and_never_retries_a_slug(tmp_path) -> None:
    controller = _controller(tmp_path)
    first = controller.reserve(
        _snapshot(), _account(), run_slug="data1-smoke-a", timeout_seconds=180, smoke=True
    )
    assert first.reserved_seconds == 300
    with pytest.raises(QuotaError, match="global project job"):
        controller.reserve(
            _snapshot(), _account(), run_slug="data1-next-b", timeout_seconds=180, smoke=True
        )
    controller.mark_submission_unknown(first)
    state = controller.status()
    assert state["frozen"] is True
    assert state["freeze_reason"] == "submission_unknown"
    with pytest.raises(QuotaError, match="frozen"):
        controller.reserve(
            _snapshot(), _account(), run_slug="data1-smoke-a", timeout_seconds=180, smoke=True
        )


def test_reviewed_failed_provider_run_keeps_charge_and_unfreezes(tmp_path) -> None:
    controller = _controller(tmp_path)
    reservation = controller.reserve(
        _snapshot(), _account(), run_slug="data1-reviewed-a", timeout_seconds=180, smoke=True)
    controller.mark_terminal_unknown(reservation)
    assert controller.status()["frozen"] is True
    reviewed = controller.review_frozen_terminal(
        reservation, _snapshot(), terminal_state="failed", evidence_sha256="a" * 64)
    assert reviewed["charged_seconds"] == 300
    status = controller.status()
    assert status["frozen"] is False
    assert status["active_run_slugs"] == []
    with pytest.raises(QuotaError, match="uniquely reviewable"):
        controller.review_frozen_terminal(
            reservation, _snapshot(), terminal_state="failed", evidence_sha256="a" * 64)


def test_reservation_requires_two_hour_remainder_and_approved_duration(tmp_path) -> None:
    controller = _controller(tmp_path)
    with pytest.raises(QuotaError, match="required reserve"):
        controller.reserve(
            _snapshot(remaining=7_400), _account(), run_slug="data1-low-a", timeout_seconds=180, smoke=True
        )
    with pytest.raises(QuotaError, match="approved limit"):
        controller.reserve(
            _snapshot(), _account(), run_slug="data1-long-a", timeout_seconds=901
        )


def test_reduced_quota_window_cap_persistently_freezes(tmp_path) -> None:
    controller = _controller(tmp_path)
    reservation = controller.reserve(
        _snapshot(), _account(), run_slug="data1-shrink-a", timeout_seconds=180, smoke=True
    )
    controller.reconcile_terminal(reservation, _snapshot(), actual_elapsed_seconds=180, terminal=True)
    shrunk = QuotaSnapshot("gpu", 2_000, 1_000, 1_000, 2_000, NOW)
    with pytest.raises(QuotaError, match="total shrank"):
        controller.reserve(
            shrunk, _account(), run_slug="data1-shrink-b", timeout_seconds=180, smoke=True
        )
    assert controller.status()["frozen"] is True


def test_reconciliation_charges_reservation_and_overshoot_freezes(tmp_path) -> None:
    controller = _controller(tmp_path)
    reservation = controller.reserve(
        _snapshot(), _account(), run_slug="data1-run-a", timeout_seconds=180, smoke=True
    )
    result = controller.reconcile_terminal(
        reservation, _snapshot(), actual_elapsed_seconds=240, terminal=True
    )
    assert result["charged_seconds"] == 300
    assert controller.status()["active_run_slugs"] == []

    controller2 = QuotaController(tmp_path / "overshoot.sqlite", clock=lambda: NOW)
    overflow = controller2.reserve(
        _snapshot(), _account(), run_slug="data1-run-b", timeout_seconds=180, smoke=True
    )
    result2 = controller2.reconcile_terminal(
        overflow, _snapshot(), actual_elapsed_seconds=301, terminal=True
    )
    assert result2["charged_seconds"] == 301
    assert controller2.status()["frozen"] is False


def test_reconciliation_requires_same_quota_window_and_consistent_delta(tmp_path) -> None:
    controller = _controller(tmp_path)
    reservation = controller.reserve(
        _snapshot(), _account(), run_slug="data1-delta-a", timeout_seconds=180, smoke=True
    )
    shifted = QuotaSnapshot("gpu", 100_000, 1_400, 98_600, 2_001, NOW)
    with pytest.raises(QuotaError, match="reset crossing"):
        controller.reconcile_terminal(reservation, shifted, actual_elapsed_seconds=180, terminal=True)
    assert controller.status()["frozen"] is True

    other = QuotaController(tmp_path / "delta.sqlite", clock=lambda: NOW)
    rounded_before = QuotaSnapshot("gpu", 100_000, 1_000, 98_999, 2_000, NOW)
    second = other.reserve(
        rounded_before, _account(), run_slug="data1-delta-b", timeout_seconds=180, smoke=True
    )
    # Provider display consumption must agree in both used and remaining fields.
    inconsistent = QuotaSnapshot("gpu", 100_000, 1_302, 98_699, 2_000, NOW)
    with pytest.raises(QuotaError, match="delta is inconsistent"):
        other.reconcile_terminal(second, inconsistent, actual_elapsed_seconds=180, terminal=True)
    assert other.status()["frozen"] is True


def test_observed_quota_cost_above_reservation_is_persistently_frozen(tmp_path) -> None:
    controller = _controller(tmp_path)
    reservation = controller.reserve(
        _snapshot(), _account(), run_slug="data1-cost-a", timeout_seconds=180, smoke=True
    )
    consumed = QuotaSnapshot("gpu", 100_000, 1_302, 98_698, 2_000, NOW)
    with pytest.raises(QuotaError, match="observed quota cost"):
        controller.reconcile_terminal(reservation, consumed, actual_elapsed_seconds=180, terminal=True)
    assert controller.status()["frozen"] is True


def test_readonly_status_does_not_create_a_journal_or_lock(tmp_path) -> None:
    controller = _controller(tmp_path)
    with pytest.raises(QuotaError, match="does not exist"):
        controller.status()
    assert not controller.journal_path.exists()
    assert not controller.lock_path.exists()


def test_private_metadata_and_adapter_use_safe_argv_only(tmp_path) -> None:
    metadata = tmp_path / "kernel-metadata.json"
    (tmp_path / "private-run.py").write_text("print('control-only')\n", encoding="utf-8")
    metadata.write_text(
        '{"id":"owner/private-run","title":"private run","is_private":true,"enable_internet":false,'
        '"code_file":"private-run.py","machine_shape":"NvidiaTeslaT4"}',
        encoding="utf-8",
    )
    seen: list[tuple[list[str], dict]] = []

    def runner(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="secret-value")

    client = KaggleCli(runner=runner, clock=lambda: NOW)
    assert validate_private_metadata(tmp_path)["is_private"] is True
    assert client.push(
        tmp_path, accelerator="NvidiaTeslaT4", timeout_seconds=180, account=_account()
    ) == "owner/private-run"
    assert seen[0][0] == [
        "kaggle", "kernels", "push", "-p", str(tmp_path), "--accelerator", "NvidiaTeslaT4", "--timeout", "180",
    ]
    assert seen[0][1]["shell"] is False
    assert seen[0][1]["capture_output"] is True


def test_adapter_redacts_provider_errors_and_parses_official_text_status(tmp_path) -> None:
    def failed(_argv, **_kwargs):
        return subprocess.CompletedProcess(["kaggle"], 1, stdout="api-token=secret", stderr="api-token=secret")

    with pytest.raises(KaggleCliError) as captured:
        KaggleCli(runner=failed).quota()
    assert "secret" not in str(captured.value)
    with pytest.raises(KaggleCliError):
        parse_terminal_status('owner/private-run has status "KernelWorkerStatus.RUNNING"')
    assert parse_job_state('owner/private-run has status "KernelWorkerStatus.RUNNING"') == "running"
    assert parse_terminal_status('owner/private-run has status "KernelWorkerStatus.COMPLETE"') == "complete"
    (tmp_path / "kernel-metadata.json").write_text(
        '{"id":"x","is_private":false,"enable_internet":false,"code_file":"x.py"}', encoding="utf-8"
    )
    (tmp_path / "x.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(KaggleCliError):
        validate_private_metadata(tmp_path)


def test_adapter_status_uses_documented_text_command_without_json(tmp_path) -> None:
    (tmp_path / "run.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text(
        '{"id":"owner/run","title":"run","is_private":true,"enable_internet":false,"code_file":"run.py"}',
        encoding="utf-8",
    )
    seen: list[list[str]] = []

    def runner(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout='owner/run has status "KernelWorkerStatus.COMPLETE"\n', stderr=""
        )

    assert KaggleCli(runner=runner).status("owner/run", metadata_path=tmp_path) == "complete"
    assert seen == [["kaggle", "kernels", "status", "owner/run"]]
    queued = KaggleCli(
        runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 0, stdout='owner/run has status "KernelWorkerStatus.QUEUED"\n', stderr=""
        )
    )
    assert queued.job_state("owner/run", metadata_path=tmp_path) == "queued"
    with pytest.raises(KaggleCliError):
        parse_job_state('owner/run has status "KernelWorkerStatus.MYSTERY"')


def test_dataset_status_uses_documented_json_projection() -> None:
    seen = []

    def runner(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"status":"ready"}', stderr="")

    client = KaggleCli(runner=runner)
    assert client.dataset_state("owner/private-data") == "ready"
    assert seen == [["kaggle", "datasets", "status", "owner/private-data", "--format", "json"]]
    assert parse_dataset_state('{"status":"creating"}') == "pending"
    assert parse_dataset_state('{"status":"error"}') == "failed"
    with pytest.raises(KaggleCliError):
        parse_dataset_state('{"status":"unknown"}')


def test_adapter_rejects_unverified_tpu_and_unsafe_metadata(tmp_path) -> None:
    (tmp_path / "run.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text(
        '{"id":"owner/run","title":"run","is_private":true,"enable_internet":false,"code_file":"run.py"}',
        encoding="utf-8",
    )
    with pytest.raises(KaggleCliError, match="verified free SKU"):
        KaggleCli(runner=lambda *_args, **_kwargs: None, clock=lambda: NOW).push(
            tmp_path, accelerator="TpuV38", timeout_seconds=180, account=_account()
        )
    (tmp_path / "kernel-metadata.json").write_text(
        '{"id":"owner/run","title":"run","is_private":true,"enable_internet":true,"code_file":"run.py"}',
        encoding="utf-8",
    )
    with pytest.raises(KaggleCliError, match="disable internet"):
        validate_private_metadata(tmp_path)


def test_push_rejects_title_slug_mismatch_before_provider_call(tmp_path) -> None:
    (tmp_path / "run.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text(
        '{"id":"owner/declared-slug","title":"different title","is_private":true,'
        '"enable_internet":false,"code_file":"run.py"}', encoding="utf-8")
    with pytest.raises(KaggleCliError, match="resolve exactly"):
        KaggleCli(runner=lambda *_args, **_kwargs: None, clock=lambda: NOW).push(
            tmp_path, accelerator="NvidiaTeslaT4", timeout_seconds=180, account=_account())


def test_dataset_creation_is_private_and_upload_surface_is_bounded(tmp_path) -> None:
    (tmp_path / "case.npy").write_bytes(b"not-a-real-array-control-only")
    (tmp_path / "source.zip").write_bytes(b"PK\x03\x04control-only")
    (tmp_path / "dataset-metadata.json").write_text(
        '{"id":"owner/data1-captures","title":"DATA1 captures",'
        '"licenses":[{"name":"other"}]}',
        encoding="utf-8",
    )
    seen: list[list[str]] = []

    def runner(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="provider-output-may-be-secret", stderr="")

    assert validate_private_dataset_metadata(tmp_path)["id"] == "owner/data1-captures"
    assert KaggleCli(runner=runner).create_dataset(tmp_path) == "owner/data1-captures"
    assert seen == [["kaggle", "datasets", "create", "-p", str(tmp_path)]]
    assert "--public" not in seen[0]


def test_dataset_metadata_rejects_public_or_unapproved_uploads(tmp_path) -> None:
    (tmp_path / "case.npy").write_bytes(b"control")
    (tmp_path / "notes.txt").write_text("not an approved artifact", encoding="utf-8")
    (tmp_path / "dataset-metadata.json").write_text(
        '{"id":"owner/data1-captures","title":"DATA1 captures",'
        '"licenses":[{"name":"CC0-1.0"}]}',
        encoding="utf-8",
    )
    with pytest.raises(KaggleCliError, match="other license"):
        validate_private_dataset_metadata(tmp_path)
