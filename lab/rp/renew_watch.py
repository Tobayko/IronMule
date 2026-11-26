"""Re-arm the owned pod watchdog at its current cutoff before each bounded extension."""
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


def predecessor_process(directory, deadline):
    ready = json.loads((directory / "ready.json").read_text())
    if ready["new"] != deadline or json.loads((directory / "extended").read_text()) != deadline:
        raise ValueError("prior monitor does not own the requested deadline")
    proc = Path(f"/proc/{int(ready['pid'])}")
    try:
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        expected = [os.fsencode(directory / "handoff.py"), b"monitor", str(ready["old"]).encode(), str(deadline).encode()]
        if identity(ready["pid"]) == ready["start"] and argv[1:5] == expected:
            return proc
    except OSError:
        pass
    return None


def handoff(old, new, action, predecessor=None):
    if os.geteuid() or not old < new <= old + 90 * 60:
        raise ValueError("invalid original lease or renewal")
    if predecessor is None and int(os.environ["RP_DEADLINE"]) != old:
        raise ValueError("original cutoff differs without an identified previous monitor")
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

    def predecessors():
        if predecessor is not None:
            if (predecessor / "enforce.py").read_bytes() != deploy_source:
                raise ValueError("previous monitor enforcement source differs")
            proc = predecessor_process(predecessor, old)
            return [proc] if proc is not None else []
        return [p for p in Path("/proc").iterdir() if p.name.isdecimal() and old_watcher(p, old)]

    if not live(old, new):
        if MARKER.exists() or len(predecessors()) != 1:
            raise ValueError("unable to identify exactly one effective prior watchdog")
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
    # The replacement enforces old independently. Retire only the exact acknowledged process.
    for proc in predecessors():
        fd = os.pidfd_open(int(proc.name))
        try:
            if proc in predecessors() and live(old, new):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                done = select.poll()
                done.register(fd, select.POLLIN)
                if not done.poll(1800):
                    raise ValueError("previous watchdog remains active")
        finally:
            os.close(fd)
    if predecessors() or not live(old, new) or time.time() >= old - 30:
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
    parser.add_argument("--previous-tag", help="private path identity of the previously acknowledged monitor")
    a = parser.parse_args()
    prior_root = None
    if a.previous_tag is not None:
        if len(a.previous_tag) != 24 or any(c not in "0123456789abcdef" for c in a.previous_tag):
            parser.error("invalid previous monitor identity")
        prior_root = Path("/run/ironmule-renew-" + a.previous_tag)
        if prior_root == ROOT or prior_root.is_symlink():
            parser.error("unexpected previous monitor directory")
    if a.action == "monitor":
        monitor(a.old, a.new)
    else:
        handoff(a.old, a.new, a.action, prior_root)
