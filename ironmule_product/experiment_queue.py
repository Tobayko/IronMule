"""Private mailbox for offline development. Production only writes coarse counters while idle."""
import fcntl
import hashlib
import re
import threading
import uuid
from collections import deque
from pathlib import Path

from .state import _atomic_write

WINDOW = 32


def host_id():
    # A Linux boot ID is shared across the processes using a kernel. Keep only its digest.
    return hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest()


def validate(ticket):
    if set(ticket) != {"version", "id", "producer", "model_sha256", "short", "document"}:
        raise ValueError("unexpected demand ticket fields")
    if (ticket["version"] != 1 or not re.fullmatch("[0-9a-f]{32}", ticket["id"])
            or any(not re.fullmatch("[0-9a-f]{64}", ticket[key]) for key in ("producer", "model_sha256"))
            or any(type(ticket[key]) is not int or ticket[key] < 0 for key in ("short", "document"))
            or ticket["short"] + ticket["document"] != WINDOW):
        raise ValueError("invalid demand ticket")
    return ticket


class ExperimentQueue:
    def __init__(self, directory, completed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.host = host_id()
        self.serving = (self.directory / ("serving-" + self.host)).open("a")
        fcntl.flock(self.serving, fcntl.LOCK_SH)
        self.last = completed
        self.completed = completed
        self.recent = deque(maxlen=WINDOW)
        self.lock = threading.Lock()

    def close(self):
        self.serving.close()

    def observe(self, jobs):
        """Single-profile engines lack width-agent traffic memory. Retain a bounded length class only."""
        with self.lock:
            for job in jobs:
                if not job.hardware_probe and job.reason in ("length", "eos"):
                    self.recent.append(len(job.ids) < 96)
                    self.completed += 1

    def idle(self, agent, model_sha256):
        observed = agent.requests if agent is not None else self.completed
        if observed < self.last + WINDOW:
            return
        if agent is None:
            with self.lock:
                short = sum(self.recent)
                observed = self.completed
        else:
            # Read existing completion memory before doing filesystem work.
            with agent._lock:
                observed = agent.requests
                if len(agent.memory) < WINDOW:
                    return
                short = sum(row[1] < 96 for row in list(agent.memory)[-WINDOW:])
        pending = self.directory / "pending.json"
        with (self.directory / "writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if pending.exists():
                return
            ticket = {"version": 1, "id": uuid.uuid4().hex, "producer": self.host,
                      "model_sha256": model_sha256, "short": short, "document": WINDOW - short}
            _atomic_write(pending, validate(ticket))
            self.last = observed
