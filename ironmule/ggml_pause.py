"""PF3 pause diagnostics for the ggml engine's native calls (2026-11-24); off unless IRONMULE_GGML_PAUSE names a file.

Some packed prefill calls on the A100 took 1.2-2.0 s longer than their rows explain (PF2, first waves after a server
start). This probe tells apart what the calling thread did meanwhile, without profiler permissions:

* per call, the change of the calling thread's CPU time (user, system), its minor and major page faults, its run delay
  (time runnable but not running, /proc schedstat), the process's major faults and the cgroup's CPU throttling;
* during a call, every `every_ms` a sample of the thread's scheduler state (R running, S sleeping, D uninterruptible),
  the system call it is blocked in with its file (an ioctl on /dev/nvidia* is a driver call, a futex a wait on another
  thread) and its kernel wait channel where the kernel shows it;
* once a call has run `spy_ms`, one `py-spy dump --native` of the process where py-spy is installed and allowed (it
  stops the process while it reads the stacks, so it lengthens that call).

Samples are kept in memory and written only for calls of `min_ms` or more, after the call returned; the per-call deltas
go to the engine trace line. Reading /proc needs no extra permission for the process's own threads; what the kernel
refuses (a wait channel shown as 0, a ptrace-denied py-spy) is recorded as refused, not guessed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

SYSCALLS = {"0": "read", "1": "write", "7": "poll", "16": "ioctl", "23": "select", "24": "sched_yield",
            "35": "nanosleep", "202": "futex", "228": "clock_gettime", "230": "clock_nanosleep", "232": "epoll_wait",
            "281": "epoll_pwait", "271": "ppoll", "9": "mmap", "11": "munmap", "10": "mprotect", "28": "madvise"}


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _stat_fields(text: str) -> list[str]:
    # fields after the command name, which may contain spaces: index 0 is field 3 (state)
    return text[text.rfind(")") + 2:].split() if text else []


def _cgroup_throttle() -> tuple[int, int]:
    """(throttled periods, throttled microseconds) of this cgroup; (-1, -1) where neither cgroup version shows them."""
    for path, scale in (("/sys/fs/cgroup/cpu.stat", 1), ("/sys/fs/cgroup/cpu/cpu.stat", 1000),
                        ("/sys/fs/cgroup/cpu,cpuacct/cpu.stat", 1000)):
        text = _read(path)
        if not text:
            continue
        kv = dict(line.split() for line in text.splitlines() if len(line.split()) == 2)
        usec = kv.get("throttled_usec")
        if usec is None and "throttled_time" in kv:  # v1: nanoseconds
            usec = int(kv["throttled_time"]) // scale
        if "nr_throttled" in kv:
            return int(kv["nr_throttled"]), int(usec or 0)
    return -1, -1


class PauseProbe:
    def __init__(self, path: str, min_ms: float = 1000.0, every_ms: float = 5.0, spy_ms: float = 0.0) -> None:
        self.path = path
        self.min_s = min_ms / 1e3
        self.spy_s = (spy_ms or min_ms) / 1e3
        self.every_s = every_ms / 1e3
        self.tick = os.sysconf("SC_CLK_TCK")
        self.pyspy = shutil.which("py-spy")
        self._wake = threading.Event()
        self._busy = False
        self._samples: list[str] = []
        self._tid = 0
        self._t0 = 0.0
        self._dumped = ""
        threading.Thread(target=self._sample_loop, name="ironmule-pause-probe", daemon=True).start()

    def _snapshot(self, tid: int) -> tuple:
        t = _stat_fields(_read(f"/proc/self/task/{tid}/stat"))
        p = _stat_fields(_read("/proc/self/stat"))
        sched = _read(f"/proc/self/task/{tid}/schedstat").split()
        thr = _cgroup_throttle()
        if len(t) < 13 or len(p) < 10:
            return ()
        # field numbers (1-based) 10 minflt, 12 majflt, 14 utime, 15 stime -> index - 3
        return (int(t[11]) / self.tick, int(t[12]) / self.tick, int(t[7]), int(t[9]), int(p[9]),
                int(sched[1]) / 1e9 if len(sched) > 1 else -1.0, thr[0], thr[1])

    def begin(self) -> None:
        self._tid = threading.get_native_id()
        self._before = self._snapshot(self._tid)
        self._samples = []
        self._dumped = ""
        self._t0 = time.monotonic()
        self._busy = True
        self._wake.set()

    def end(self, label: str) -> str:
        """Stop sampling; the per-call deltas as trace fields, and the samples written when the call was long."""
        self._busy = False
        self._wake.clear()
        took = time.monotonic() - self._t0
        after = self._snapshot(self._tid)
        fields = ""
        if self._before and after:
            d = [a - b for a, b in zip(after, self._before)]
            fields = (f" cpu_u={1e3 * d[0]:.0f} cpu_s={1e3 * d[1]:.0f} minflt={d[2]} majflt={d[3]} pmajflt={d[4]} "
                      f"rundelay={1e3 * d[5]:.1f} thr={d[6] if after[6] >= 0 else 'na'} "
                      f"thr_ms={d[7] / 1e3 if after[7] >= 0 else 'na'}")
        if took >= self.min_s:
            with open(self.path, "a") as fh:
                fh.write(f"# call {label} {self._t0:.6f} {took:.4f}{fields} pyspy={self._dumped or 'none'}\n")
                fh.write("\n".join(self._samples) + "\n")
        return fields

    def _sample_loop(self) -> None:
        while True:
            self._wake.wait()
            tid = self._tid
            while self._busy:
                now = time.monotonic()
                stat = _stat_fields(_read(f"/proc/self/task/{tid}/stat"))
                state = stat[0] if stat else "?"
                ticks = f"{stat[11]}/{stat[12]}" if len(stat) > 12 else "?"
                sc = _read(f"/proc/self/task/{tid}/syscall").split()
                call = "?"
                if sc:
                    if sc[0] == "running":
                        call = "running"
                    elif sc[0].lstrip("-").isdigit():
                        call = SYSCALLS.get(sc[0], f"sys{sc[0]}")
                        if call in ("ioctl", "read", "write", "poll") and len(sc) > 1:
                            try:
                                call += ":" + os.readlink(f"/proc/self/fd/{int(sc[1], 16)}").rsplit("/", 1)[-1]
                            except (OSError, ValueError):
                                pass
                wchan = _read(f"/proc/self/task/{tid}/wchan").strip() or "refused"
                self._samples.append(f"{now - self._t0:.4f} {state} {call} {wchan} {ticks}")
                if not self._dumped and self.pyspy and now - self._t0 >= self.spy_s:
                    out = f"{self.path}.pyspy.{int(now * 1e3)}"
                    try:
                        with open(out, "w") as fh:
                            subprocess.Popen([self.pyspy, "dump", "--native", "--pid", str(os.getpid())], stdout=fh,
                                             stderr=subprocess.STDOUT)
                        self._dumped = os.path.basename(out)
                    except OSError as exc:
                        self._dumped = f"failed:{exc.__class__.__name__}"
                time.sleep(self.every_s)
