"""Fail-closed readiness checks and the process-wide hardware lease.

This module intentionally only uses the Python standard library.  It does not
try to infer private GPU counters: a machine is considered ready only when the
publicly observable safety facts are both readable and stable.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class ReadinessError(RuntimeError):
    """Base error for readiness and lease failures."""


class LeaseBusy(ReadinessError):
    """Another process currently owns the lease."""


class LeaseError(ReadinessError):
    """The lease path or ownership record is unsafe or invalid."""


class OutputTruncated(ReadinessError):
    """A system command exceeded the probe's bounded output budget."""


@dataclass(frozen=True)
class ProbeSnapshot:
    """One complete public-API observation of the host.

    ``None`` is deliberate: unknown is never converted to a safe value.  The
    extra fields make the snapshot useful for audit/history without exposing a
    claim about a private Apple GPU utilization API.
    """

    timestamp: float = 0.0
    ac_connected: bool | None = None
    low_power: bool | None = None
    swap_used_bytes: int | None = None
    swap_total_bytes: int | None = None
    memory_available_bytes: int | None = None
    memory_total_bytes: int | None = None
    load_1m: float | None = None
    cpu_percent: float | None = None
    workload_active: bool | None = None
    process_tree_readable: bool = False
    process_evidence: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)) or not _finite(self.timestamp):
            raise ValueError("timestamp must be finite")
        for name in ("ac_connected", "low_power", "workload_active", "process_tree_readable"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(name + " must be bool or None")
        for name in ("swap_used_bytes", "swap_total_bytes", "memory_available_bytes", "memory_total_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(name + " must be a non-negative integer or None")
        for name in ("load_1m", "cpu_percent"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not _finite(value) or value < 0):
                raise ValueError(name + " must be a finite non-negative number or None")
        if self.swap_total_bytes is not None and self.swap_used_bytes is not None and self.swap_used_bytes > self.swap_total_bytes:
            raise ValueError("swap used cannot exceed total")
        if self.memory_total_bytes is not None and self.memory_available_bytes is not None and self.memory_available_bytes > self.memory_total_bytes:
            raise ValueError("available memory cannot exceed total")
        object.__setattr__(self, "process_evidence", tuple(self.process_evidence))
        object.__setattr__(self, "errors", tuple(self.errors))
        if any(not isinstance(item, str) or len(item) > 512 for item in self.process_evidence + self.errors):
            raise ValueError("probe text is invalid or unbounded")
        if self.fingerprint is not None and (not isinstance(self.fingerprint, str) or len(self.fingerprint) > 512):
            raise ValueError("fingerprint is invalid")

    @property
    def ac(self) -> bool | None:
        return self.ac_connected

    @property
    def swap_bytes(self) -> int | None:
        return self.swap_used_bytes

    @property
    def foreign_workload(self) -> bool | None:
        return self.workload_active

    @property
    def known(self) -> bool:
        return not self.errors and all(
            value is not None
            for value in (
                self.ac_connected,
                self.low_power,
                self.swap_used_bytes,
                self.memory_available_bytes,
                self.load_1m,
                self.cpu_percent,
                self.workload_active,
            )
        ) and self.process_tree_readable


@dataclass(frozen=True)
class ReadinessPolicy:
    """Conservative sampling policy.  Values are intentionally bounded."""

    min_samples: int = 3
    sample_interval_seconds: float = 0.25
    max_load_1m: float = 0.75
    max_cpu_percent: float = 35.0
    max_swap_growth_bytes: int = 0
    memory_stability_fraction: float = 0.05
    load_stability_delta: float = 0.25
    deadline_seconds: float = 30.0
    min_memory_available_bytes: int = 1
    min_memory_available_fraction: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) or self.min_samples < 2 or self.min_samples > 100:
            raise ValueError("min_samples must be at least two")
        numbers = (self.sample_interval_seconds, self.max_load_1m, self.max_cpu_percent, self.memory_stability_fraction, self.load_stability_delta, self.deadline_seconds, self.min_memory_available_fraction)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not _finite(value) for value in numbers):
            raise TypeError("readiness policy values must be finite numbers")
        if self.sample_interval_seconds < 0 or self.deadline_seconds <= 0:
            raise ValueError("sampling times must be positive")
        if self.max_load_1m < 0 or self.max_cpu_percent < 0:
            raise ValueError("load limits must be non-negative")
        if isinstance(self.max_swap_growth_bytes, bool) or not isinstance(self.max_swap_growth_bytes, int) or self.max_swap_growth_bytes < 0 or isinstance(self.min_memory_available_bytes, bool) or not isinstance(self.min_memory_available_bytes, int) or self.min_memory_available_bytes < 0 or not 0 <= self.memory_stability_fraction <= 1 or self.load_stability_delta < 0 or not 0 <= self.min_memory_available_fraction <= 1:
            raise ValueError("invalid readiness limits")


@dataclass(frozen=True)
class ReadinessDecision:
    ready: bool
    reasons: tuple[str, ...]
    samples: tuple[ProbeSnapshot, ...] = field(default_factory=tuple)
    checked_at: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool) or isinstance(self.checked_at, bool) or not isinstance(self.checked_at, (int, float)) or not _finite(self.checked_at):
            raise ValueError("invalid readiness decision")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "samples", tuple(self.samples))
        if any(not isinstance(reason, str) or len(reason) > 256 for reason in self.reasons):
            raise ValueError("invalid readiness reason")
        if any(not isinstance(sample, ProbeSnapshot) for sample in self.samples):
            raise TypeError("decision samples must be ProbeSnapshot values")

    @property
    def no_recommendation(self) -> bool:
        return not self.ready

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


class Probe(Protocol):
    def sample(self) -> ProbeSnapshot: ...


def _clock_now(clock: Any) -> float:
    if clock is None:
        return time.monotonic()
    if callable(clock):
        return float(clock())
    for name in ("monotonic", "now"):
        method = getattr(clock, name, None)
        if method is not None:
            return float(method())
    raise TypeError("clock must be callable or expose monotonic()/now()")


def _finite(value: Any) -> bool:
    try:
        return bool(value == value and abs(float(value)) != float("inf"))
    except (TypeError, ValueError, OverflowError):
        return False


def _sleep(sleeper: Any, seconds: float) -> None:
    if seconds <= 0:
        return
    if sleeper is None:
        time.sleep(seconds)
    elif callable(sleeper):
        sleeper(seconds)
    else:
        sleeper.sleep(seconds)


def _sample_probe(probe: Probe | Callable[[], ProbeSnapshot]) -> ProbeSnapshot:
    value = probe() if callable(probe) else probe.sample()
    if not isinstance(value, ProbeSnapshot):
        raise TypeError("probe must return ProbeSnapshot")
    return value


def _stable(samples: Sequence[ProbeSnapshot], policy: ReadinessPolicy) -> bool:
    if len(samples) < policy.min_samples:
        return False
    first = samples[0]
    if any(sample.ac_connected != first.ac_connected for sample in samples):
        return False
    if any(sample.low_power != first.low_power for sample in samples):
        return False
    if any(sample.workload_active != first.workload_active for sample in samples):
        return False
    loads = [sample.load_1m for sample in samples]
    if any(value is None for value in loads):
        return False
    if max(loads) - min(loads) > policy.load_stability_delta:
        return False
    memory = [sample.memory_available_bytes for sample in samples]
    if any(value is None for value in memory):
        return False
    base = max(memory[0], 1)
    if max(memory) - min(memory) > base * policy.memory_stability_fraction:
        return False
    swaps = [sample.swap_used_bytes for sample in samples]
    if any(value is None for value in swaps):
        return False
    if max(swaps) - min(swaps) > policy.max_swap_growth_bytes:
        return False
    return True


def check_readiness(
    probe: Probe | Callable[[], ProbeSnapshot],
    policy: ReadinessPolicy | None = None,
    *,
    sleeper: Any = None,
    clock: Any = None,
    deadline: float | None = None,
) -> ReadinessDecision:
    """Take separated samples and return a conservative decision.

    The deadline is checked before every sample and sleep.  A test that reaches
    its deadline at the boundary is inconclusive, never ready.
    """

    policy = policy or ReadinessPolicy()
    started = _clock_now(clock)
    end = deadline if deadline is not None else started + policy.deadline_seconds
    samples: list[ProbeSnapshot] = []
    reasons: list[str] = []
    for index in range(policy.min_samples):
        if _clock_now(clock) >= end:
            reasons.append("deadline_exceeded")
            break
        try:
            sample = _sample_probe(probe)
        except Exception as exc:  # fail closed; preserve only bounded type text
            reasons.append("probe_failed:" + type(exc).__name__)
            break
        samples.append(sample)
        if index + 1 < policy.min_samples:
            if _clock_now(clock) >= end:
                reasons.append("deadline_exceeded")
                break
            wait = min(policy.sample_interval_seconds, max(0.0, end - _clock_now(clock)))
            _sleep(sleeper, wait)

    if len(samples) < policy.min_samples:
        if not reasons:
            reasons.append("insufficient_samples")
        return ReadinessDecision(False, tuple(dict.fromkeys(reasons)), tuple(samples), _clock_now(clock))

    for sample in samples:
        if sample.errors:
            reasons.append("probe_error")
        if not sample.process_tree_readable:
            reasons.append("process_tree_unreadable")
        if sample.ac_connected is not True:
            reasons.append("ac_not_confirmed")
        if sample.low_power is not False:
            reasons.append("low_power_not_confirmed_off")
        if sample.workload_active is not False:
            reasons.append("foreign_workload_or_unknown")
        if sample.load_1m is None:
            reasons.append("load_unknown")
        elif sample.load_1m > policy.max_load_1m:
            reasons.append("load_too_high")
        if sample.cpu_percent is None:
            reasons.append("cpu_unknown")
        elif sample.cpu_percent > policy.max_cpu_percent:
            reasons.append("cpu_too_high")
        if sample.swap_used_bytes is None or sample.memory_available_bytes is None:
            reasons.append("memory_or_swap_unknown")
        elif policy.min_memory_available_fraction > 0 and sample.memory_total_bytes is None:
            reasons.append("memory_reserve_unknown")
        elif sample.memory_available_bytes < max(policy.min_memory_available_bytes, int((sample.memory_total_bytes or 0) * policy.min_memory_available_fraction)):
            reasons.append("memory_reserve_too_low")
    if not _stable(samples, policy):
        reasons.append("samples_unstable")
    unique = tuple(dict.fromkeys(reasons))
    return ReadinessDecision(not unique, unique, tuple(samples), _clock_now(clock))


_SWAP_UNITS = {
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}


def _locale_decimal(text: str) -> float:
    """Parse a number that macOS may print with a comma decimal mark.

    ``sysctl vm.swapusage`` and ``ps -o %cpu`` follow the user's locale, so on
    a German system they emit ``1675,38M`` and ``0,5``. The previous parser
    accepted a dot only, so on such a machine the readiness gate could never
    certify memory or the process tree and therefore never permitted a single
    gated measurement - failing closed, but failing always.

    Ambiguity is refused rather than guessed: a value carrying both separators,
    or more than one of either, raises instead of being interpreted as a
    thousands group.
    """

    candidate = text.strip()
    if not candidate:
        raise ReadinessError("number_unreadable")
    dots, commas = candidate.count("."), candidate.count(",")
    if (dots and commas) or dots > 1 or commas > 1:
        raise ReadinessError("number_ambiguous")
    normalised = candidate.replace(",", ".")
    try:
        value = float(normalised)
    except ValueError as exc:
        raise ReadinessError("number_unreadable") from exc
    if not _finite(value):
        raise ReadinessError("number_unreadable")
    return value


def _parse_swap_value(text: str, name: str) -> int:
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z]+)", text, re.I)
    if not match:
        raise ReadinessError("swap_value_unknown")
    unit = match.group(2).upper()
    if unit not in _SWAP_UNITS:
        raise ReadinessError("swap_unit_unknown")
    number = _locale_decimal(match.group(1))
    if not _finite(number):
        raise ReadinessError("swap_value_unknown")
    return int(number * _SWAP_UNITS[unit])


class MacSystemProbe:
    """Read Apple host state through bounded, absolute, shell-free commands."""

    COMMANDS = {
        "pmset": "/usr/bin/pmset",
        "vm_stat": "/usr/bin/vm_stat",
        "sysctl": "/usr/sbin/sysctl",
        "ps": "/bin/ps",
    }

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        clock: Any = None,
        timeout_seconds: float = 1.0,
        max_output_bytes: int = 128 * 1024,
    ) -> None:
        if timeout_seconds <= 0 or max_output_bytes < 1024:
            raise ValueError("invalid probe bounds")
        self._runner = runner or subprocess.run
        self._clock = clock
        self._timeout = timeout_seconds
        self._max_output = max_output_bytes

    def _run(self, argv: Sequence[str]) -> str:
        try:
            result = self._runner(
                list(argv),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            # stderr is diagnostics only and is never parsed as measurement.
            out = result.stdout if isinstance(result.stdout, str) else ""
            if len(out.encode("utf-8", "replace")) > self._max_output:
                raise OutputTruncated("output_truncated")
            text = out
            if getattr(result, "returncode", 0) != 0:
                raise ReadinessError("command_failed:" + argv[0])
            return text
        except OutputTruncated:
            raise
        except (OSError, subprocess.SubprocessError, ReadinessError) as exc:
            raise ReadinessError(type(exc).__name__) from exc

    def sample(self) -> ProbeSnapshot:
        timestamp = _clock_now(self._clock)
        errors: list[str] = []
        ac: bool | None = None
        low: bool | None = None
        swap: int | None = None
        swap_total: int | None = None
        mem_available: int | None = None
        mem_total: int | None = None
        process_readable = False
        workload: bool | None = None
        evidence: list[str] = []
        try:
            power = self._run((self.COMMANDS["pmset"], "-g", "batt"))
            source_matches = re.findall(r"^Now drawing from ['\"]([^'\"]+)['\"]\s*$", power, re.M)
            if len(source_matches) != 1:
                raise ReadinessError("power_source_ambiguous")
            source = source_matches[0]
            if source == "AC Power":
                ac = True
            elif source == "Battery Power":
                ac = False
            else:
                raise ReadinessError("power_source_unknown")
            custom = self._run((self.COMMANDS["pmset"], "-g", "custom"))
            section: str | None = None
            low_power_key_seen = False
            active_low_power_values: list[str | None] = []
            for line in custom.splitlines():
                section_match = re.match(r"^\s*(AC Power|Battery Power):\s*$", line)
                if section_match:
                    section = section_match.group(1)
                    continue
                match = re.match(r"^\s*lowpowermode(?:\s+(.*?))?\s*$", line, re.I)
                if match:
                    low_power_key_seen = True
                    if section == source:
                        active_low_power_values.append(match.group(1))
            if low_power_key_seen:
                if len(active_low_power_values) != 1:
                    raise ReadinessError("low_power_missing" if not active_low_power_values else "low_power_ambiguous")
                value = active_low_power_values[0]
                if value not in {"0", "1"}:
                    raise ReadinessError("low_power_unknown")
                low = value == "1"
            else:
                # Older macOS/hardware may omit the setting entirely.  There
                # is no low-power setting to enable in that case, so represent
                # the capability as explicitly unsupported/off.  If a key is
                # present anywhere, however, the active profile must resolve
                # exactly and cannot silently fall back to False.
                low = False
        except ReadinessError as exc:
            errors.append(str(exc) if str(exc) in {"output_truncated", "low_power_ambiguous", "low_power_missing", "low_power_unknown", "power_source_ambiguous"} else "power_unreadable")
        try:
            vm = self._run((self.COMMANDS["vm_stat"],))
            page_match = re.search(r"page size of (\d+) bytes", vm)
            if not page_match:
                raise ReadinessError("page_size_unknown")
            page = int(page_match.group(1))
            values: dict[str, int] = {}
            for name, value in re.findall(r"^([^:]+):\s+(\d+)\.", vm, re.M):
                values[name.strip()] = int(value)
            if "Pages free" not in values or "Pages speculative" not in values:
                raise ReadinessError("available_pages_unknown")
            # Inactive/purgeable accounting can overlap; count only disjoint
            # free and speculative pools for a conservative lower bound.
            free_pages = values["Pages free"] + values["Pages speculative"]
            mem_available = free_pages * page
        except (ReadinessError, ValueError) as exc:
            errors.append(str(exc) if str(exc) in {"page_size_unknown", "available_pages_unknown", "output_truncated"} else "memory_unreadable")
        try:
            memory = self._run((self.COMMANDS["sysctl"], "-n", "hw.memsize"))
            if not re.fullmatch(r"\s*\d+\s*", memory):
                raise ReadinessError("memory_total_unknown")
            mem_total = int(memory.strip())
            if mem_available is not None and mem_available > mem_total:
                raise ReadinessError("available_memory_exceeds_total")
            swap_text = self._run((self.COMMANDS["sysctl"], "vm.swapusage"))
            swap = _parse_swap_value(swap_text, "used")
            swap_total = _parse_swap_value(swap_text, "total")
            if swap > swap_total:
                raise ReadinessError("swap_exceeds_total")
        except (ReadinessError, ValueError) as exc:
            errors.append(str(exc) if str(exc) in {"swap_unit_unknown", "swap_exceeds_total", "memory_total_unknown", "available_memory_exceeds_total", "output_truncated"} else "swap_unreadable")
        load: float | None
        try:
            load = float(os.getloadavg()[0])
        except (OSError, IndexError):
            load = None
            errors.append("load_unreadable")
        cpu: float | None = None
        try:
            # ``comm`` is sufficient for known runtimes and avoids importing
            # unbounded argv text (which may contain prompts or tokens).
            ps = self._run((self.COMMANDS["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="))
            process_readable = True
            active_matches = 0
            cpu_values: list[float] = []
            records: dict[int, tuple[int, int, str, float | None, str]] = {}
            malformed = False
            for line in ps.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split(None, 5)
                if len(fields) < 6:
                    malformed = True
                    continue
                try:
                    uid, pid, ppid = int(fields[0]), int(fields[1]), int(fields[2])
                except ValueError:
                    malformed = True
                    continue
                state, cpu_text, command = fields[3], fields[4], fields[5]
                parsed_cpu: float | None = None
                try:
                    parsed_cpu = _locale_decimal(cpu_text)
                    if parsed_cpu < 0:
                        raise ValueError
                    cpu_values.append(parsed_cpu)
                except (ReadinessError, ValueError, TypeError):
                    malformed = True
                records[pid] = (uid, ppid, state, parsed_cpu, command)
            try:
                current_uid = os.getuid()
                if isinstance(current_uid, bool) or not isinstance(current_uid, int) or current_uid < 0:
                    raise ValueError
            except (AttributeError, OSError, TypeError, ValueError):
                raise ReadinessError("current_uid_unknown")
            own_pid = os.getpid()
            ancestors: set[int] = set()
            cursor = own_pid
            seen: set[int] = set()
            while cursor in records and cursor not in seen:
                seen.add(cursor)
                parent = records[cursor][1]
                if parent <= 0 or parent == cursor:
                    break
                ancestors.add(parent)
                cursor = parent
            ignored = ancestors | {own_pid}
            for pid, (uid, ppid, state, parsed_cpu, command) in records.items():
                if pid in ignored:
                    continue
                lowered = command.lower()
                relevant = any(token in lowered for token in ("claude", "mlx", "mlx_lm", "mlx-lm", "python", "node", "model", "gemma"))
                activity_hint = any(token in lowered for token in ("generate", "inference", "serve", "server", "gemma", "model", "mlx"))
                active = bool(re.match(r"^[rud]", state.lower())) or (parsed_cpu is not None and parsed_cpu > 1.0) or activity_hint
                # Known model/runtime names are blocked regardless of UID.
                # Unknown active processes are foreign only for this user;
                # system daemons are not blanket-blocked.
                if active and (relevant or uid == current_uid):
                    active_matches += 1
                    evidence.append(command[:240])
            if malformed:
                process_readable = False
            if own_pid not in records:
                process_readable = False
                errors.append("self_pid_unknown")
            cpu = sum(cpu_values) if cpu_values else None
            workload = active_matches > 0
        except ReadinessError as exc:
            errors.append(str(exc) if str(exc) == "output_truncated" else "process_tree_unreadable")
        return ProbeSnapshot(
            timestamp=timestamp,
            ac_connected=ac,
            low_power=low,
            swap_used_bytes=swap,
            swap_total_bytes=swap_total,
            memory_available_bytes=mem_available,
            memory_total_bytes=mem_total,
            load_1m=load,
            cpu_percent=cpu,
            workload_active=workload,
            process_tree_readable=process_readable,
            process_evidence=tuple(evidence[:16]),
            errors=tuple(errors),
        )


def _safe_lease_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise LeaseError("lease path must be a regular non-symlink file")
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise LeaseError("lease parent is unsafe")
    current = parent
    while True:
        if current.is_symlink() and str(current) not in {"/var", "/tmp"}:
            raise LeaseError("lease ancestor is a symlink")
        if current.parent == current:
            break
        current = current.parent
    return target


class ReadinessGate:
    """Reusable injected gate facade used at each session checkpoint."""

    def __init__(self, probe: Probe | Callable[[], ProbeSnapshot], policy: ReadinessPolicy | None = None, *, sleeper: Any = None, clock: Any = None) -> None:
        self.probe, self.policy, self.sleeper, self.clock = probe, policy or ReadinessPolicy(), sleeper, clock

    def check(self, *, deadline: float | None = None) -> ReadinessDecision:
        return check_readiness(self.probe, self.policy, sleeper=self.sleeper, clock=self.clock, deadline=deadline)


MacProbe = MacSystemProbe


class HardwareLease:
    """Exclusive, non-blocking ownership mark for all hardware work."""

    def __init__(self, path: str | os.PathLike[str], *, fingerprint: str, clock: Any = None) -> None:
        if not fingerprint or not isinstance(fingerprint, str):
            raise LeaseError("fingerprint is required")
        self.path = _safe_lease_path(path)
        self.fingerprint = fingerprint
        self._clock = clock
        self._fd: int | None = None
        self._token: str | None = None
        self.owner: Mapping[str, Any] | None = None
        self._held = False

    def acquire(self) -> "HardwareLease":
        if self._fd is not None:
            raise LeaseError("lease already acquired")
        # Keep the inode permanently.  flock, rather than an owner timestamp,
        # is the authority; a crashed owner is therefore recoverable naturally.
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                raise LeaseError("lease symlink refused") from exc
            raise LeaseError("cannot open lease") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise LeaseError("lease must be regular and 0600")
            path_info = self.path.lstat()
            if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
                raise LeaseError("lease path changed")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise LeaseBusy("lease is held") from exc
                raise
            token = uuid.uuid4().hex
            started_at = _clock_now(self._clock)
            owner = {
                "token": token,
                "pid": os.getpid(),
                "started": started_at,
                "start": started_at,
                "fingerprint": self.fingerprint,
            }
            payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
            os.ftruncate(fd, 0)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise LeaseError("lease owner write failed")
                view = view[written:]
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.read(fd, len(payload) + 1) != payload:
                raise LeaseError("lease owner readback failed")
            self._fd, self._token, self.owner, self._held = fd, token, owner, True
            return self
        except Exception:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise

    def _read_owner(self) -> dict[str, Any]:
        if self._fd is None or self._token is None or not self._held:
            raise LeaseError("lease is not held")
        os.lseek(self._fd, 0, os.SEEK_SET)
        raw = os.read(self._fd, 8192)
        try:
            value = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseError("lease owner record invalid") from exc
        if not isinstance(value, dict) or value.get("token") != self._token or value.get("fingerprint") != self.fingerprint:
            raise LeaseError("lease ownership mismatch")
        return value

    def validate(self) -> bool:
        try:
            owner = self._read_owner()
        except LeaseError:
            return False
        try:
            info = self.path.lstat()
            fd_info = os.fstat(self._fd) if self._fd is not None else None
            if fd_info is None or (info.st_dev, info.st_ino) != (fd_info.st_dev, fd_info.st_ino):
                return False
        except OSError:
            return False
        return owner.get("pid") == os.getpid()

    def heartbeat(self) -> bool:
        if not self.validate() or self._fd is None:
            return False
        owner = dict(self._read_owner())
        owner["heartbeat"] = _clock_now(self._clock)
        payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        view = memoryview(payload)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                return False
            view = view[written:]
        os.fsync(self._fd)
        os.lseek(self._fd, 0, os.SEEK_SET)
        if os.read(self._fd, len(payload) + 1) != payload:
            return False
        self.owner = owner
        return True

    def release(self) -> None:
        if self._fd is None or not self._held:
            return
        fd = self._fd
        try:
            if self.validate():
                released = {"released": True, "token": self._token}
                payload = json.dumps(released, sort_keys=True, separators=(",", ":")).encode()
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        break
                    view = view[written:]
                os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                if os.read(fd, len(payload) + 1) != payload:
                    raise LeaseError("lease release readback failed")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            self._fd = None
            self._token = None
            self.owner = None
            self._held = False

    def __enter__(self) -> "HardwareLease":
        return self.acquire()

    def __exit__(self, *_: Any) -> None:
        self.release()


__all__ = [
    "HardwareLease", "LeaseBusy", "LeaseError", "MacProbe", "MacSystemProbe", "ProbeSnapshot",
    "ReadinessGate",
    "ReadinessDecision", "ReadinessError", "ReadinessPolicy", "check_readiness",
]
