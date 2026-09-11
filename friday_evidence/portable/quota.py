"""Fail-closed accounting for bounded, free-tier remote jobs.

This module deliberately does not submit work.  It turns a fresh quota report
and independently observed account/session evidence into an append-only
reservation record.  A caller must still use the provider adapter to submit
and to reconcile a confirmed terminal result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import fcntl
import io
import math
from pathlib import Path
import re
import time
import uuid
from typing import Any, Iterator

from friday_evidence.events import EventJournal, EventJournalError


class QuotaError(RuntimeError):
    """Quota evidence, reservation, or reconciliation is unsafe."""


_FRESH_SECONDS = 60.0
_POST_JOB_RESERVE_SECONDS = 7_200.0
_SUBMISSION_OVERHEAD_SECONDS = 120.0
_SMOKE_LIMIT_SECONDS = 180.0
_REGULAR_LIMIT_SECONDS = 900.0
_RUN_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_RESOURCE_NAMES = {"gpu", "tpu"}
# Kaggle CLI 2.2.4 renders ``quota_refresh_time.isoformat()``.  Its actual
# authenticated ``quota --csv`` response is a timezone-less UTC ISO string;
# retain that narrowly pinned interpretation instead of treating it as local
# time.  Direct structured/typed timestamps still require an explicit zone.
_KAGGLE_CLI_NAIVE_UTC_VERSION = "2.2.4"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QuotaError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QuotaError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise QuotaError(f"{name} must be a finite nonnegative number")
    return result


def _timestamp(value: Any, name: str, *, allow_naive_utc: bool = False) -> float:
    if isinstance(value, str) and any(marker in value for marker in ("-", "T", "Z")):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise QuotaError(f"{name} must be a positive timestamp") from exc
        if parsed.tzinfo is None:
            if not allow_naive_utc:
                raise QuotaError(f"{name} must include a timezone")
            parsed = parsed.replace(tzinfo=timezone.utc)
        value = parsed.astimezone(timezone.utc).timestamp()
    result = _finite_number(value, name)
    if result <= 0:
        raise QuotaError(f"{name} must be a positive timestamp")
    return result


def _key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _pick(row: Mapping[str, str], *names: str) -> str | None:
    normalized = {_key(key): value.strip() for key, value in row.items() if key is not None}
    for name in names:
        value = normalized.get(_key(name))
        if value is not None and value != "":
            return value
    return None


def _seconds(value: Any, name: str) -> float:
    """Accept a numeric seconds value or a conservative HH:MM:SS string."""
    if isinstance(value, str) and ":" in value:
        parts = value.strip().split(":")
        if len(parts) not in {2, 3} or any(not item.isdigit() for item in parts):
            raise QuotaError(f"{name} must be expressed in seconds")
        values = [int(item) for item in parts]
        if len(values) == 2:
            values.insert(0, 0)
        hours, minutes, seconds = values
        if minutes >= 60 or seconds >= 60:
            raise QuotaError(f"{name} must be expressed in seconds")
        return float(hours * 3600 + minutes * 60 + seconds)
    return _finite_number(value, name)


def _quota_seconds(value: Any, name: str) -> tuple[float, float]:
    """Parse the CLI's unit-bearing quota fields and their resolution."""
    if isinstance(value, str) and value.strip().lower().endswith("h"):
        hours = _finite_number(value.strip()[:-1], name)
        # Kaggle CLI 2.2.4 emits f"{hours:.2f}h".  One display quantum is
        # 0.01 h = 36 seconds; retain it as a bound, not fabricated precision.
        return hours * 3600.0, 36.0
    return _seconds(value, name), 0.0


@dataclass(frozen=True)
class QuotaSnapshot:
    """One resource's provider-reported quota, tied to an observation time."""

    resource: str
    total_seconds: float
    used_seconds: float
    remaining_seconds: float
    refresh_at_unix_s: float
    observed_at_unix_s: float
    display_resolution_seconds: float = 0.0

    def __post_init__(self) -> None:
        resource = self.resource.lower().strip()
        if resource not in _RESOURCE_NAMES:
            raise QuotaError("quota resource must be GPU or TPU")
        object.__setattr__(self, "resource", resource)
        total = _finite_number(self.total_seconds, "quota total")
        used = _finite_number(self.used_seconds, "quota used")
        remaining = _finite_number(self.remaining_seconds, "quota remaining")
        refresh = _timestamp(self.refresh_at_unix_s, "quota refreshAt")
        observed = _timestamp(self.observed_at_unix_s, "quota observedAt")
        resolution = _finite_number(self.display_resolution_seconds, "quota display resolution")
        if refresh <= observed:
            raise QuotaError("quota refreshAt must be after quota observation")
        # Providers may round a display by a second.  Anything larger is
        # inconsistent evidence rather than spare quota.
        if abs((used + remaining) - total) > max(1.0, 1.5 * resolution):
            raise QuotaError("quota total, used, and remaining are inconsistent")
        object.__setattr__(self, "total_seconds", total)
        object.__setattr__(self, "used_seconds", used)
        object.__setattr__(self, "remaining_seconds", remaining)
        object.__setattr__(self, "refresh_at_unix_s", refresh)
        object.__setattr__(self, "observed_at_unix_s", observed)
        object.__setattr__(self, "display_resolution_seconds", resolution)

    @classmethod
    def from_csv(
        cls,
        payload: str,
        *,
        observed_at_unix_s: float,
        resource: str | None = None,
    ) -> "QuotaSnapshot":
        """Parse a provider CSV without guessing missing or ambiguous fields.

        Accepted rows have a resource column (``resource``/``accelerator``)
        or a boolean-ish current GPU/TPU marker.  The accounting columns may
        be named ``total``, ``used``, ``remaining``/``remain`` and
        ``refreshAt``/``resetAt``.  Multiple matching rows require the caller
        to select a resource explicitly.
        """
        if not isinstance(payload, str) or not payload.strip():
            raise QuotaError("quota CSV is empty")
        try:
            rows = list(csv.DictReader(io.StringIO(payload)))
        except csv.Error as exc:
            raise QuotaError("quota CSV is invalid") from exc
        if not rows or not rows[0] or None in rows[0]:
            raise QuotaError("quota CSV has no valid header")
        wanted = None if resource is None else resource.lower().strip()
        if wanted is not None and wanted not in _RESOURCE_NAMES:
            raise QuotaError("quota resource must be GPU or TPU")
        candidates: list[tuple[str, Mapping[str, str]]] = []
        for row in rows:
            declared = _pick(row, "resource", "accelerator", "type", "current")
            found: str | None = None
            if declared is not None and declared.lower().strip() in _RESOURCE_NAMES:
                found = declared.lower().strip()
            else:
                for candidate in _RESOURCE_NAMES:
                    marker = _pick(row, f"current{candidate}", f"is{candidate}")
                    if marker is not None and marker.lower() not in {"", "0", "false", "no", "none", "null"}:
                        found = candidate
                        break
            if found is not None and (wanted is None or found == wanted):
                candidates.append((found, row))
        if len(candidates) != 1:
            raise QuotaError("quota CSV must identify exactly one requested resource")
        selected_resource, row = candidates[0]
        total = _pick(row, "totalSeconds", "total", "quotaTotal")
        used = _pick(row, "usedSeconds", "used", "quotaUsed")
        remaining = _pick(row, "remainingSeconds", "remaining", "remain", "quotaRemaining")
        refresh = _pick(row, "refreshAt", "resetAt", "refreshAtUnixS", "reset")
        if None in {total, used, remaining, refresh}:
            raise QuotaError("quota CSV is missing required accounting columns")
        total_seconds, total_resolution = _quota_seconds(total, "quota total")
        used_seconds, used_resolution = _quota_seconds(used, "quota used")
        remaining_seconds, remaining_resolution = _quota_seconds(remaining, "quota remaining")
        if len({total_resolution, used_resolution, remaining_resolution}) != 1:
            raise QuotaError("quota CSV mixes unit-bearing and raw accounting values")
        return cls(
            resource=selected_resource,
            total_seconds=total_seconds,
            used_seconds=used_seconds,
            remaining_seconds=remaining_seconds,
            refresh_at_unix_s=_timestamp(refresh, "quota refreshAt", allow_naive_utc=True),
            observed_at_unix_s=_timestamp(observed_at_unix_s, "quota observedAt"),
            display_resolution_seconds=total_resolution,
        )

    def is_fresh(self, now_unix_s: float, max_age_s: float = _FRESH_SECONDS) -> bool:
        now = _timestamp(now_unix_s, "current time")
        age = _finite_number(max_age_s, "maximum quota age")
        return 0 <= now - self.observed_at_unix_s <= age and now < self.refresh_at_unix_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "total_seconds": self.total_seconds,
            "used_seconds": self.used_seconds,
            "remaining_seconds": self.remaining_seconds,
            "refresh_at_unix_s": self.refresh_at_unix_s,
            "observed_at_unix_s": self.observed_at_unix_s,
            "display_resolution_seconds": self.display_resolution_seconds,
        }


@dataclass(frozen=True)
class AccountPreflight:
    """Explicit evidence required before a quota can authorize a free job."""

    free_account_verified: bool
    no_paid_linkage_verified: bool
    active_session_verified: bool
    account_checked_at_unix_s: float
    session_checked_at_unix_s: float
    # This is affirmative evidence that no other project/provider job is
    # active, not merely that a session listing endpoint responded.
    additional_usage: bool = False
    supported_free_skus: tuple[str, ...] = ()
    # Browser/account evidence that every prerequisite for the listed SKUs is
    # satisfied (for example Kaggle phone verification for GPU access).
    accelerator_access_verified: bool = False

    def validate(self, now_unix_s: float, max_age_s: float = _FRESH_SECONDS) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.free_account_verified,
                self.no_paid_linkage_verified,
                self.active_session_verified,
                self.additional_usage,
                self.accelerator_access_verified,
            )
        ):
            raise QuotaError("account preflight flags must be booleans")
        if not self.free_account_verified:
            raise QuotaError("free account status is not verified")
        if not self.no_paid_linkage_verified:
            raise QuotaError("absence of paid linkage is not verified")
        if not self.active_session_verified:
            raise QuotaError("active account session with zero other jobs is not verified")
        if self.additional_usage:
            raise QuotaError("additional active account usage is present")
        if not self.accelerator_access_verified:
            raise QuotaError("accelerator account prerequisites are not verified")
        if not isinstance(self.supported_free_skus, tuple) or any(
            not isinstance(item, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,63}", item)
            for item in self.supported_free_skus
        ):
            raise QuotaError("supported free accelerator evidence is invalid")
        now = _timestamp(now_unix_s, "current time")
        max_age = _finite_number(max_age_s, "maximum preflight age")
        for label, checked in (
            ("account preflight", self.account_checked_at_unix_s),
            ("session preflight", self.session_checked_at_unix_s),
        ):
            checked_at = _timestamp(checked, label)
            if not 0 <= now - checked_at <= max_age:
                raise QuotaError(f"{label} evidence is stale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_account_verified": self.free_account_verified,
            "no_paid_linkage_verified": self.no_paid_linkage_verified,
            "active_session_verified": self.active_session_verified,
            "account_checked_at_unix_s": self.account_checked_at_unix_s,
            "session_checked_at_unix_s": self.session_checked_at_unix_s,
            "additional_usage": self.additional_usage,
            "supported_free_skus": list(self.supported_free_skus),
            "accelerator_access_verified": self.accelerator_access_verified,
        }


@dataclass(frozen=True)
class Reservation:
    run_slug: str
    run_id: str
    resource: str
    timeout_seconds: float
    reserved_seconds: float
    quota_refresh_at_unix_s: float
    created_at_unix_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_slug": self.run_slug,
            "run_id": self.run_id,
            "resource": self.resource,
            "timeout_seconds": self.timeout_seconds,
            "reserved_seconds": self.reserved_seconds,
            "quota_refresh_at_unix_s": self.quota_refresh_at_unix_s,
            "created_at_unix_s": self.created_at_unix_s,
        }


def _event_pages(journal: EventJournal) -> Iterator[dict[str, Any]]:
    after = 0
    while True:
        page = journal.events(limit=1000, after_seq=after)
        if not page:
            return
        yield from page
        after = int(page[-1]["seq"])
        if len(page) < 1000:
            return


class QuotaController:
    """One-project, one-active-job controller backed by ``EventJournal``.

    The journal's immutable records are the budget ledger.  The controller
    uses an advisory ``flock`` only to serialize decisions; read-only status
    intentionally opens neither a missing lock file nor a missing journal.
    """

    def __init__(
        self,
        journal_path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        lock_path: str | Path | None = None,
    ) -> None:
        self.journal_path = Path(journal_path)
        self.lock_path = Path(lock_path) if lock_path is not None else self.journal_path.with_suffix(self.journal_path.suffix + ".lock")
        self._clock = clock

    @staticmethod
    def new_run_slug(prefix: str = "data1") -> str:
        clean = re.sub(r"[^a-z0-9-]", "-", prefix.lower()).strip("-") or "data1"
        return f"{clean[:45]}-{uuid.uuid4().hex[:24]}"

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.lock_path.open("a", encoding="utf-8")
        except OSError as exc:
            raise QuotaError("cannot open quota controller lock") from exc
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise QuotaError("cannot acquire quota controller lock") from exc
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _state(self, journal: EventJournal) -> dict[str, Any]:
        reservations: dict[str, dict[str, Any]] = {}
        frozen = False
        freeze_reason: str | None = None
        for event in _event_pages(journal):
            payload = event["payload"]
            action = payload.get("quota_action") if isinstance(payload, dict) else None
            if action == "reserve":
                reservations[str(payload["run_slug"])] = dict(payload)
            elif action in {
                "submission_unknown", "terminal_unknown", "quota_unknown",
                "quota_shrunk", "overshoot", "provider_failed",
            }:
                frozen = True
                freeze_reason = str(action)
            elif action in {"reconciled", "reviewed_resolution"}:
                stored = reservations.get(str(payload.get("run_slug")))
                if stored is not None:
                    stored["reconciled"] = True
                    stored["charged_seconds"] = payload["charged_seconds"]
                if action == "reviewed_resolution":
                    frozen = False
                    freeze_reason = None
        return {"reservations": reservations, "frozen": frozen, "freeze_reason": freeze_reason}

    @staticmethod
    def _require_slug(run_slug: str) -> str:
        if not isinstance(run_slug, str) or not _RUN_SLUG_RE.fullmatch(run_slug):
            raise QuotaError("run slug must be a lowercase, unique safe identifier")
        return run_slug

    @staticmethod
    def _timeout(timeout_seconds: float, smoke: bool) -> float:
        timeout = _finite_number(timeout_seconds, "job timeout")
        if timeout <= 0:
            raise QuotaError("job timeout must be positive")
        limit = _SMOKE_LIMIT_SECONDS if smoke else _REGULAR_LIMIT_SECONDS
        if timeout > limit:
            raise QuotaError("job timeout exceeds the approved limit")
        return timeout

    def preflight(self, snapshot: QuotaSnapshot, account: AccountPreflight) -> None:
        now = _timestamp(self._clock(), "current time")
        if not snapshot.is_fresh(now):
            raise QuotaError("quota snapshot is stale or already reset")
        account.validate(now)

    def reserve(
        self,
        snapshot: QuotaSnapshot,
        account: AccountPreflight,
        *,
        run_slug: str,
        timeout_seconds: float,
        smoke: bool = False,
    ) -> Reservation:
        """Record a non-releasable pre-submit reservation after all checks."""
        run_slug = self._require_slug(run_slug)
        timeout = self._timeout(timeout_seconds, smoke)
        self.preflight(snapshot, account)
        reserve = timeout + _SUBMISSION_OVERHEAD_SECONDS
        # A CLI field printed to 0.01 h can overstate true total/remaining by
        # half a display quantum.  Spend only against their lower bound.
        display_half = snapshot.display_resolution_seconds / 2.0
        cap = min(
            max(0.0, snapshot.total_seconds - display_half) * 0.10,
            _POST_JOB_RESERVE_SECONDS,
        )
        now = _timestamp(self._clock(), "current time")
        run_id = uuid.uuid4().hex
        with self._exclusive_lock(), EventJournal(self.journal_path) as journal:
            # A contended writer lock may have consumed the allowed evidence
            # age; re-check at the point the irreversible ledger write occurs.
            self.preflight(snapshot, account)
            state = self._state(journal)
            if state["frozen"]:
                raise QuotaError("quota controller is frozen pending manual review")
            if run_slug in state["reservations"]:
                raise QuotaError("run slug already has a reservation; retries are forbidden")
            active = [item for item in state["reservations"].values() if not item.get("reconciled")]
            if active:
                raise QuotaError("a global project job is already reserved")
            already_reserved = sum(
                float(item["reserved_seconds"])
                for item in state["reservations"].values()
                if item.get("resource") == snapshot.resource
                and float(item.get("quota_refresh_at_unix_s", -1)) == snapshot.refresh_at_unix_s
            )
            if already_reserved > cap:
                # A provider may revise a window total down after we have
                # reserved from it.  There is no safe way to forgive that
                # difference automatically, even after process restart.
                try:
                    journal.append(run_id, "pause", {
                        "quota_action": "quota_shrunk",
                        "resource": snapshot.resource,
                        "quota_refresh_at_unix_s": snapshot.refresh_at_unix_s,
                        "reserved_seconds": already_reserved,
                        "current_cap_seconds": cap,
                        "recorded_at_unix_s": now,
                    })
                except EventJournalError as exc:
                    raise QuotaError("cannot persist reduced quota state") from exc
                raise QuotaError("quota-window total shrank below prior reservations; controller is frozen")
            if reserve > cap:
                raise QuotaError("job reservation exceeds the current quota-window cap")
            if snapshot.remaining_seconds - display_half - reserve < _POST_JOB_RESERVE_SECONDS:
                raise QuotaError("quota remaining after reservation is below the required reserve")
            if already_reserved + reserve > cap:
                raise QuotaError("quota-window reservation cap would be exceeded")
            payload = {
                "quota_action": "reserve",
                **Reservation(
                    run_slug, run_id, snapshot.resource, timeout, reserve,
                    snapshot.refresh_at_unix_s, now,
                ).to_dict(),
                "snapshot": snapshot.to_dict(),
                "account_preflight": account.to_dict(),
            }
            try:
                journal.append(run_id, "worker_started", payload)
            except EventJournalError as exc:
                raise QuotaError("cannot persist quota reservation") from exc
        return Reservation(run_slug, run_id, snapshot.resource, timeout, reserve, snapshot.refresh_at_unix_s, now)

    def mark_submission_unknown(self, reservation: Reservation) -> None:
        self._mark_frozen(reservation, "submission_unknown")

    def mark_terminal_unknown(self, reservation: Reservation) -> None:
        self._mark_frozen(reservation, "terminal_unknown")

    def mark_quota_unknown(self, reservation: Reservation) -> None:
        self._mark_frozen(reservation, "quota_unknown")

    def mark_provider_failed(self, reservation: Reservation) -> None:
        self._mark_frozen(reservation, "provider_failed")

    def review_frozen_terminal(
        self, reservation: Reservation, snapshot: QuotaSnapshot, *,
        terminal_state: str, evidence_sha256: str,
    ) -> dict[str, Any]:
        """Resolve one frozen reservation after explicit, external review.

        This does not refund the reservation. It requires a fresh quota from
        the same window and a content hash binding the independently observed
        provider status/report. Only terminal failure/cancellation is admitted;
        successful evidence follows the normal reconciliation/import path.
        """
        if terminal_state not in {"failed", "cancelled"}:
            raise QuotaError("reviewed terminal state must be failed or cancelled")
        if not isinstance(evidence_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
            raise QuotaError("review evidence SHA-256 is invalid")
        now = _timestamp(self._clock(), "current time")
        if (
            not snapshot.is_fresh(now)
            or snapshot.resource != reservation.resource
            or snapshot.refresh_at_unix_s != reservation.quota_refresh_at_unix_s
            or snapshot.observed_at_unix_s < reservation.created_at_unix_s
        ):
            raise QuotaError("review requires fresh matching quota evidence")
        with self._exclusive_lock(), EventJournal(self.journal_path) as journal:
            state = self._state(journal)
            stored = state["reservations"].get(reservation.run_slug)
            active = [item for item in state["reservations"].values() if not item.get("reconciled")]
            if (
                not state["frozen"] or stored is None
                or stored.get("run_id") != reservation.run_id
                or stored.get("reconciled") or len(active) != 1
            ):
                raise QuotaError("frozen reservation is not uniquely reviewable")
            payload = {
                "quota_action": "reviewed_resolution", "run_slug": reservation.run_slug,
                "terminal_state": terminal_state, "charged_seconds": reservation.reserved_seconds,
                "evidence_sha256": evidence_sha256, "snapshot": snapshot.to_dict(),
                "recorded_at_unix_s": now,
            }
            try:
                journal.append(reservation.run_id, "run_finished", payload)
            except EventJournalError as exc:
                raise QuotaError("cannot persist reviewed quota resolution") from exc
        return {"run_slug": reservation.run_slug, "charged_seconds": reservation.reserved_seconds,
                "terminal_state": terminal_state, "frozen": False}

    def _mark_frozen(self, reservation: Reservation, action: str) -> None:
        with self._exclusive_lock(), EventJournal(self.journal_path) as journal:
            state = self._state(journal)
            stored = state["reservations"].get(reservation.run_slug)
            if stored is None or stored.get("run_id") != reservation.run_id:
                raise QuotaError("reservation is not present in this ledger")
            try:
                journal.append(reservation.run_id, "pause", {
                    "quota_action": action,
                    "run_slug": reservation.run_slug,
                    "reserved_seconds": reservation.reserved_seconds,
                    "recorded_at_unix_s": _timestamp(self._clock(), "current time"),
                })
            except EventJournalError as exc:
                raise QuotaError("cannot persist frozen quota state") from exc

    def reconcile_terminal(
        self,
        reservation: Reservation,
        snapshot: QuotaSnapshot,
        *,
        actual_elapsed_seconds: float,
        terminal: bool,
    ) -> dict[str, Any]:
        """Charge a confirmed terminal job using fresh provider evidence only.

        The reservation is never released: the charge is the larger of the
        pre-submit reserve and actual elapsed cost.  A larger actual cost is
        recorded and freezes subsequent automation for manual review.
        """
        if not terminal:
            self.mark_terminal_unknown(reservation)
            raise QuotaError("job is not confirmed terminal; reservation remains frozen")
        now = _timestamp(self._clock(), "current time")
        if not snapshot.is_fresh(now) or snapshot.resource != reservation.resource:
            self.mark_quota_unknown(reservation)
            raise QuotaError("fresh matching quota is required for reconciliation")
        if snapshot.refresh_at_unix_s != reservation.quota_refresh_at_unix_s:
            self.mark_quota_unknown(reservation)
            raise QuotaError("quota reset crossing cannot reconcile a reservation")
        if snapshot.observed_at_unix_s < reservation.created_at_unix_s:
            self.mark_quota_unknown(reservation)
            raise QuotaError("quota observation predates the reservation")
        actual = _finite_number(actual_elapsed_seconds, "actual elapsed time")
        if actual <= 0:
            self.mark_terminal_unknown(reservation)
            raise QuotaError("actual elapsed time must be positive")
        charge = max(reservation.reserved_seconds, actual)
        overshoot = actual > reservation.reserved_seconds
        with self._exclusive_lock(), EventJournal(self.journal_path) as journal:
            state = self._state(journal)
            stored = state["reservations"].get(reservation.run_slug)
            if state["frozen"] or stored is None or stored.get("run_id") != reservation.run_id:
                raise QuotaError("quota controller cannot reconcile this reservation")
            if stored.get("reconciled"):
                raise QuotaError("reservation was already reconciled")
            before = stored.get("snapshot")
            if not isinstance(before, dict):
                self._append_quota_unknown(journal, reservation, now)
                raise QuotaError("reservation lacks pre-submit quota evidence")
            try:
                before_used = _finite_number(before["used_seconds"], "pre-submit quota used")
                before_remaining = _finite_number(before["remaining_seconds"], "pre-submit quota remaining")
                before_observed = _timestamp(before["observed_at_unix_s"], "pre-submit quota observation")
                before_refresh = _timestamp(before["refresh_at_unix_s"], "pre-submit quota reset")
            except (KeyError, QuotaError):
                self._append_quota_unknown(journal, reservation, now)
                raise QuotaError("reservation lacks valid pre-submit quota evidence")
            if (
                before_refresh != snapshot.refresh_at_unix_s
                or snapshot.observed_at_unix_s < max(before_observed, reservation.created_at_unix_s)
            ):
                self._append_quota_unknown(journal, reservation, now)
                raise QuotaError("quota evidence sequence is unresolved")
            used_delta = snapshot.used_seconds - before_used
            remaining_delta = before_remaining - snapshot.remaining_seconds
            # Each CLI observation is rounded to a 36-second quantum.  A
            # before/after delta in each field may vary by one full quantum;
            # comparing used and remaining deltas has a 72-second uncertainty
            # budget.  Structured snapshots retain their historical 1-second
            # tolerance.
            delta_rounding = (
                before.get("display_resolution_seconds", 0.0)
                + snapshot.display_resolution_seconds
            )
            try:
                delta_rounding = _finite_number(delta_rounding, "quota delta rounding")
            except QuotaError:
                self._append_quota_unknown(journal, reservation, now)
                raise QuotaError("reservation lacks valid quota rounding evidence")
            rounding_allowance = max(1.0, delta_rounding)
            if (
                used_delta < -rounding_allowance
                or remaining_delta < -rounding_allowance
                or abs(used_delta - remaining_delta) > rounding_allowance
            ):
                self._append_quota_unknown(journal, reservation, now)
                raise QuotaError("quota delta is inconsistent")
            observed_quota_delta = max(0.0, used_delta, remaining_delta)
            charge = max(reservation.reserved_seconds, actual, observed_quota_delta)
            overshoot = observed_quota_delta > reservation.reserved_seconds + rounding_allowance
            payload = {
                "quota_action": "reconciled",
                "run_slug": reservation.run_slug,
                "actual_elapsed_seconds": actual,
                "charged_seconds": charge,
                "observed_quota_delta_seconds": observed_quota_delta,
                "snapshot": snapshot.to_dict(),
                "recorded_at_unix_s": now,
            }
            try:
                journal.append(reservation.run_id, "run_finished", payload)
                if overshoot:
                    journal.append(reservation.run_id, "pause", {
                        "quota_action": "overshoot",
                        "run_slug": reservation.run_slug,
                        "reserved_seconds": reservation.reserved_seconds,
                        "actual_elapsed_seconds": actual,
                        "recorded_at_unix_s": now,
                    })
            except EventJournalError as exc:
                raise QuotaError("cannot persist quota reconciliation") from exc
        if overshoot:
            raise QuotaError("observed quota cost exceeded its reservation; controller is frozen")
        return {"run_slug": reservation.run_slug, "charged_seconds": charge, "frozen": False}

    @staticmethod
    def _append_quota_unknown(journal: EventJournal, reservation: Reservation, now: float) -> None:
        try:
            journal.append(reservation.run_id, "pause", {
                "quota_action": "quota_unknown",
                "run_slug": reservation.run_slug,
                "reserved_seconds": reservation.reserved_seconds,
                "recorded_at_unix_s": now,
            })
        except EventJournalError as exc:
            raise QuotaError("cannot persist frozen quota state") from exc

    def status(self, *, read_only: bool = True) -> dict[str, Any]:
        """Read ledger state without creating a journal or lock in read-only mode."""
        if not read_only:
            with self._exclusive_lock(), EventJournal(self.journal_path) as journal:
                return self._status_from_journal(journal)
        if not self.journal_path.exists():
            raise QuotaError("quota journal does not exist")
        # Do not create the lock merely to report status.  A held existing lock
        # is an uncertainty, so status fails rather than racing a writer.
        if self.lock_path.exists():
            try:
                handle = self.lock_path.open("r", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
            except OSError as exc:
                raise QuotaError("quota controller status is locked") from exc
        try:
            with EventJournal(self.journal_path, read_only=True) as journal:
                return self._status_from_journal(journal)
        except EventJournalError as exc:
            raise QuotaError("cannot read quota journal") from exc

    def _status_from_journal(self, journal: EventJournal) -> dict[str, Any]:
        state = self._state(journal)
        reservations = list(state["reservations"].values())
        return {
            "frozen": state["frozen"],
            "freeze_reason": state["freeze_reason"],
            "active_run_slugs": sorted(item["run_slug"] for item in reservations if not item.get("reconciled")),
            "reservations": reservations,
        }


__all__ = [
    "AccountPreflight",
    "QuotaController",
    "QuotaError",
    "QuotaSnapshot",
    "Reservation",
]
