"""Private, crash-safe campaign accounting for the RunPod operator on Linux."""
import contextlib
import fcntl
import json
import math
import os
import tempfile
import time

INITIAL_BUDGET = 15.0
MAX_BUDGET = 20.0
CLEANUP_SECONDS = 600
# Container disk costs $0.10/GB/month. Use a 28-day month as an upper estimate.
DISK_USD_PER_GB_HOUR = 0.10 / (28 * 24)


def money(value):
    if isinstance(value, bool):
        raise ValueError("expected a monetary amount")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("expected a finite, nonnegative monetary amount")
    return amount


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def save(path, state):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as temp:
        json.dump(state, temp, indent=1, allow_nan=False)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp.name, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def limit(state):
    amount = money(state.get("budget", INITIAL_BUDGET))
    if amount > MAX_BUDGET:
        raise ValueError("campaign budget exceeds authorization")
    return amount


def held(state):
    pod = state.get("pod")
    external = sum(money(value) for value in state.get("external", {}).values())
    if pod:
        if "reserved" not in pod:
            raise ValueError("legacy pod needs cleanup before booking")
        accrued = max(0, time.time() - pod["started"]) / 3600 * money(pod["cost"])
        return external + max(money(pod["reserved"]), accrued + CLEANUP_SECONDS / 3600 * pod["cost"])
    return external


def reserve(state, amount, balance=None):
    total = money(state["spent"]) + held(state) + money(amount)
    if total > limit(state) or (balance is not None and amount > money(balance)):
        raise ValueError("campaign limit or available provider credit is insufficient")


def event(state, kind, **data):
    state.setdefault("events", []).append({"time": time.time(), "kind": kind, **data})
