"""Run on the owned pod. Hand off its original deadline before committing one bounded extension."""
import argparse
import base64
import json
import os
import signal
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARKER = ROOT / "extended"
READY = ROOT / "ready.json"
ENFORCE = ROOT / "enforce.py"
ORIGINAL = Path("/rp_watch.py")


def atomic(path, value):
    with tempfile.NamedTemporaryFile("w", dir=ROOT, delete=False) as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def identity(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def monitor(old, new):
    if not os.environ.get("RUNPOD_API_KEY") or not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("missing pod-scoped enforcement credentials")
    atomic(READY, {"pid": os.getpid(), "start": identity(os.getpid()), "old": old, "new": new})
    while True:
        deadline = old
        try:
            if json.loads(MARKER.read_text()) == new:
                deadline = new
        except (OSError, ValueError):
            pass
        if time.time() >= deadline:
            os.environ["RP_DEADLINE"] = str(deadline)
            os.execv(sys.executable, [sys.executable, str(ENFORCE)])
        time.sleep(min(2, max(0, deadline - time.time())))


def live(old, new):
    try:
        data = json.loads(READY.read_text())
        pid = data["pid"]
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (identity(pid) == data["start"] and str(Path(__file__).resolve()).encode() in argv
                and b"monitor" in argv and (data["old"], data["new"]) == (old, new))
    except (KeyError, ValueError, OSError):
        return False


def handoff(old, new, action):
    if os.geteuid() or int(os.environ["RP_DEADLINE"]) != old or not old < new <= old + 90 * 60:
        raise ValueError("invalid original lease or renewal")
    ROOT.chmod(0o700)
    deploy_source = base64.b64decode(os.environ["WATCH_B64"], validate=True)
    if ORIGINAL.read_bytes() != deploy_source:
        raise ValueError("original watcher differs from deployed source")
    if action == "revert":
        MARKER.unlink(missing_ok=True)
        print(json.dumps({"effective": old, "live": live(old, new)}))
        return
    if time.time() >= old - 45:
        raise ValueError("too close to the original deadline")
    if not live(old, new):
        if MARKER.exists():
            raise ValueError("unexpected committed marker without a live watcher")
        original = [p for p in Path("/proc").iterdir() if p.name.isdecimal() and old_watcher(p, old)]
        if len(original) != 1:
            raise ValueError("unable to identify exactly one original pod watchdog")
        ENFORCE.write_bytes(deploy_source)
        ENFORCE.chmod(0o500)
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "monitor", str(old), str(new)],
                         start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True)
        for _ in range(50):
            if live(old, new):
                break
            time.sleep(.1)
    if not live(old, new):
        raise ValueError("replacement did not acknowledge the old deadline")
    # The replacement now enforces the old deadline even if the controller or this
    # helper vanishes. Use pidfds to avoid signalling an unrelated recycled PID.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal() or not old_watcher(proc, old):
            continue
        fd = os.pidfd_open(int(proc.name))
        try:
            if old_watcher(proc, old) and live(old, new):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                done = select.poll()
                done.register(fd, select.POLLIN)
                if not done.poll(1800):
                    raise ValueError("original watchdog remains active")
        finally:
            os.close(fd)
    if not live(old, new) or time.time() >= old - 30:
        raise ValueError("pod handoff incomplete")
    atomic(MARKER, new)
    print(json.dumps({"effective": new, "live": True}))


def old_watcher(proc, old):
    try:
        argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if len(argv) != 2 or argv[1] != os.fsencode(ORIGINAL):
            return False
        return f"RP_DEADLINE={old}".encode() in (proc / "environ").read_bytes().split(b"\0")
    except OSError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("commit", "monitor", "revert"))
    parser.add_argument("old", type=int)
    parser.add_argument("new", type=int)
    a = parser.parse_args()
    if a.action == "monitor":
        monitor(a.old, a.new)
    else:
        handoff(a.old, a.new, a.action)
