import inspect
import json
import os
import subprocess
import sys

import pytest

from ironmule import ab
from ironmule.runtime import Knobs


def _child_record(*, order=("baseline", "candidate"), pid=1):
    arm = {
        "total_ns": [1.0], "prefill_ns": [0.5], "decode_ns": [0.5],
        "logical_tokens": [7], "logical_tokens_per_repeat": [[7]],
        "physical_tokens_per_repeat": [[7]],
        "token_counts": [{"logical": 1, "physical": 1}],
        "stop_reasons": ["length"], "capacities": [64],
        "deterministic": True, "decode_steps": 0,
        "prompt_tokens": 1, "mlx_peak_bytes": 10,
    }
    return {"pid": pid, "arms": {"baseline": dict(arm), "candidate": dict(arm)},
            "order": list(order), "mlx_peak_bytes": 10,
            "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}}


def _pipe(text, *, close=True):
    """A real pipe holding `text`; with close=False its writer stays open, so it never ends."""
    read, write = os.pipe()
    os.write(write, text.encode())
    if close:
        os.close(write)
    return os.fdopen(read, "rb")


class _FakeProcess:
    pid = 123
    args = ["child"]

    def __init__(self, *, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = _pipe(stdout)
        self.stderr = _pipe(stderr)
        self.communicate_timeout = None
        self.actions = []
        self.wait_calls = 0

    def communicate(self, timeout=None):
        self.communicate_timeout = timeout
        self.stdout.close()
        self.stderr.close()
        return b"", b""

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self.returncode

    def poll(self):
        return self.returncode


def _run_setup(monkeypatch, payload=None):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        spec = json.loads(args[3])
        child = payload or _child_record(order=spec["order"], pid=100 + len(calls))
        return _FakeProcess(stdout="@@" + json.dumps(child) + "\n")

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    return calls


def _arms():
    return {"baseline": Knobs(), "candidate": Knobs(readback_every=2)}


def test_run_callbacks_receive_each_index_and_defensive_json_copy(monkeypatch):
    calls = _run_setup(monkeypatch)
    before = []
    completed = []

    def before_child(index, order):
        before.append((index, order))

    def on_child(index, record):
        completed.append((index, record))
        record["arms"]["baseline"]["logical_tokens"][0] = 999

    result = ab.run(_arms(), processes=2, repeats=7, warmup=2,
                    before_child=before_child, on_child=on_child)

    assert before == [(0, ["baseline", "candidate"]),
                      (1, ["candidate", "baseline"])]
    assert [index for index, _ in completed] == [0, 1]
    assert completed[0][1] is not result["raw"][0]
    assert result["raw"][0]["arms"]["baseline"]["logical_tokens"] == [7]
    assert len(calls) == 2
    assert "start_new_session" not in calls[0][1]
    assert result["token_count_identity"] is True
    assert result["stop_reason_identity"] is True


def test_run_uses_only_real_popen_compatible_pipe_kwargs(monkeypatch):
    """Keep the A/B launch contract valid for subprocess.Popen itself."""
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    real_signature = inspect.signature(subprocess.Popen)
    observed = []

    def strict_popen(args, *, stdout, stderr, cwd, env):
        # bind() is the regression guard: an unsupported Popen kwarg raises here.
        real_signature.bind(args, stdout=stdout, stderr=stderr, cwd=cwd, env=env)
        observed.append({"stdout": stdout, "stderr": stderr, "cwd": cwd, "env": env})
        spec = json.loads(args[3])
        child = _child_record(order=spec["order"], pid=100 + len(observed))
        return _FakeProcess(stdout="@@" + json.dumps(child) + "\n")

    monkeypatch.setattr(ab.subprocess, "Popen", strict_popen)
    result = ab.run(_arms(), processes=1, repeats=7, warmup=2)

    assert len(observed) == 1
    assert observed[0]["stdout"] is subprocess.PIPE
    assert observed[0]["stderr"] is subprocess.PIPE
    assert "start_new_session" not in observed[0]
    assert result["token_identity"] is True


def test_run_timeout_is_passed_to_subprocess_and_is_loud(monkeypatch):
    import importlib
    import time

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    process = _FakeProcess()
    process.args = ["secret", "prompt"]
    process.stdout.close()
    read, write = os.pipe()  # the writer stays open: the child neither ends nor exits
    os.write(write, b"partial secret output")
    process.stdout = os.fdopen(read, "rb")
    reaped = []
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(ab, "_terminate_child", lambda child: (reaped.append(child), child.communicate()))
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="child 0 timed out") as error:
            ab.run(_arms(), processes=1, child_timeout_seconds=0.5)
    finally:
        os.close(write)
    assert 0.5 <= time.monotonic() - started < 5
    assert reaped == [process]
    assert "secret" not in str(error.value)
    assert error.value.__cause__ is None

    with pytest.raises(ValueError, match="finite and positive"):
        ab.run(_arms(), child_timeout_seconds=0)
    with pytest.raises(ValueError, match="finite and positive"):
        ab.run(_arms(), child_timeout_seconds=10**1000)


@pytest.mark.parametrize("returncode,stdout,needle", [
    (7, "@@{}\n", "exited with status 7"),
    (0, "no marker\n", "no result marker"),
    (0, "@@{\"value\":NaN}\n", "invalid JSON"),
])
def test_run_rejects_nonzero_or_missing_child_marker(monkeypatch, returncode, stdout, needle):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(ab.subprocess, "Popen",
                        lambda *_args, **_kwargs: _FakeProcess(
                            returncode=returncode, stdout=stdout, stderr="secret"))
    with pytest.raises(RuntimeError, match=needle) as error:
        ab.run(_arms(), processes=1)
    assert "secret" not in str(error.value)



def test_run_names_only_the_exception_class_of_a_failed_child(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    stderr = 'Traceback (most recent call last):\n  File "child", line 1\nRuntimeError: secret prompt\n'
    monkeypatch.setattr(ab.subprocess, "Popen",
                        lambda *_args, **_kwargs: _FakeProcess(returncode=1, stdout="", stderr=stderr))
    with pytest.raises(ab.ABRunError, match=r"exited with status 1 \(RuntimeError\)") as error:
        ab.run(_arms(), processes=1)
    assert "secret" not in str(error.value)
    assert error.value.partial_evidence == {"exception_type": "RuntimeError"}


def test_exception_class_is_read_from_the_traceback_not_its_notes():
    """BACKLOG4: the guard's note followed the exception, so the last line named nothing."""
    noise = "Error in sitecustomize; set PYTHONVERBOSE for traceback:\nModuleNotFoundError: No module named 'x'\n"
    stderr = (noise + 'Traceback (most recent call last):\n  File "<string>", line 7, in <module>\n'
              "    load()\n    ^^^^^^\nRuntimeError: cudaMallocAsync(&data, size, stream) failed: out of memory\n"
              '@GUARD_FAILURE{"events": []}\n')
    assert ab._child_exception_type(stderr) == "RuntimeError"
    assert ab._child_exception_type(noise) is None
    assert ab._child_exception_type("") is None


def test_run_communication_error_reaps_child_before_raising(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    process = _FakeProcess()

    class BrokenSelector:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def register(self, *_args):
            raise OSError("pipe secret")

    def terminate_child(child):
        child.actions.append("terminate")
        child.communicate()

    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(ab, "_terminate_child", terminate_child)
    monkeypatch.setattr(ab.selectors, "DefaultSelector", BrokenSelector)
    with pytest.raises(ab.ABRunError, match="communication failed") as error:
        ab.run(_arms(), processes=1)
    assert process.actions == ["terminate"]
    assert "pipe secret" not in str(error.value)


def test_direct_child_cleanup_escalates_only_when_terminate_does_not_reap(monkeypatch):
    process = _FakeProcess()
    process.returncode = None
    wait_results = iter([subprocess.TimeoutExpired(["child"], 2), 9])

    def wait(timeout=None):
        process.wait_calls += 1
        result = next(wait_results)
        if isinstance(result, BaseException):
            raise result
        process.returncode = result
        return result

    process.wait = wait
    ab._terminate_child(process)
    assert process.actions == ["terminate", "kill"]
    assert process.wait_calls == 2


def test_direct_child_cleanup_kills_after_wait_oserror_while_child_is_alive():
    process = _FakeProcess()
    process.returncode = None
    wait_calls = 0

    def wait(timeout=None):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise OSError("wait unavailable")
        process.returncode = 9
        return 9

    def kill():
        process.actions.append("kill")

    process.wait = wait
    process.kill = kill
    with pytest.raises(RuntimeError, match="wait: OSError"):
        ab._terminate_child(process)
    assert process.actions == ["terminate", "kill"]
    assert process.poll() == 9, "wait OSError must not leave a live child behind"


def test_run_rejects_incomplete_child_before_aggregation(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs:
                        _FakeProcess(stdout="@@{}\n"))
    with pytest.raises(ab.ABRunError, match="incomplete result") as error:
        ab.run(_arms(), processes=1)
    assert error.value.child_index == 0
    assert error.value.partial_children == []


def test_guard_failure_marker_is_retained_in_ab_run_error(monkeypatch):
    import importlib
    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    failure = {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}
    process = _FakeProcess(returncode=2, stdout="@GUARD_FAILURE" + json.dumps(failure) + "\n")
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs: process)
    with pytest.raises(ab.ABRunError) as error:
        ab.run(_arms(), processes=1)
    assert error.value.partial_evidence == {"guard_failure": failure}


def _real_child(monkeypatch, script):
    """Launch a real Python child with ab.run's own pipes; `script` gets the spec as SPEC."""
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    started = []
    real_popen = subprocess.Popen

    def popen(args, **kwargs):
        process = real_popen([sys.executable, "-c", f"import json, sys\nSPEC = json.loads({args[3]!r})\n"
                              + script], **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(ab.subprocess, "Popen", popen)
    return started


def test_run_keeps_the_marker_of_a_child_that_fills_the_other_pipe(monkeypatch):
    """Q3a P2: 400 KiB on stderr exceeds any pipe buffer; reading stdout first would deadlock."""
    record = json.dumps(_child_record())
    script = ("sys.stderr.write('x' * 400 * 1024); sys.stderr.flush()\n"
              f"rec = json.loads({record!r}); rec['order'] = SPEC['order']\n"
              "print('@@' + json.dumps(rec), flush=True)\n")
    started = _real_child(monkeypatch, script)
    result = ab.run(_arms(), processes=2, child_timeout_seconds=30)
    assert [child["order"] for child in result["raw"]] == [["baseline", "candidate"],
                                                          ["candidate", "baseline"]]
    assert all(process.poll() == 0 for process in started)
    assert all(process.stdout.closed and process.stderr.closed for process in started)


def test_run_stops_a_child_that_overflows_its_output_and_reaps_it(monkeypatch):
    """Q3a P2: the child writes without end; the cap stops it, not a timeout or memory."""
    started = _real_child(monkeypatch, "while True:\n    sys.stdout.write('y' * 65536)\n")
    with pytest.raises(ab.ABRunError, match="child 0 output limit exceeded") as error:
        ab.run(_arms(), processes=1, child_timeout_seconds=30)
    assert error.value.partial_children == []
    assert len(started) == 1 and started[0].poll() is not None
