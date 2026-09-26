"""NEXT1-I: a two-card stage counts only when every rank left a complete, matching record.

`mlx.launch` exits 0 even when a rank crashed (PERF1 run 11), so an exit code or rank 0's
file alone proves nothing. `perf1.py` writes rank 0 to OUT.json and rank r to
OUT-rankR.json, each atomically; this checks that all of them exist, parse, name the same
arm, mode, model, code and input hashes and attempt, and carry ranks 0..size-1 of one
pipeline. A notebook that sets PERF1_ATTEMPT per stage passes it as `attempt`, so a file
left by an earlier attempt of the same stage does not count. After a stage, `gpu_idle()` shows
that no rank outlived it.

    from ranks import all_ranks, gpu_idle
    ok, why = all_ranks("/kaggle/working/e2e-qwen36-35b-a3b-kernel+p16.json", attempt="s3-a1")
    idle, who = gpu_idle()

Run as a script for its self-check.
"""
import json
import os
import subprocess
import tempfile


def all_ranks(path: str, size: int = 2, attempt: str | None = None) -> tuple[bool, str]:
    """(True, "complete") when every rank's record is there and agrees, else (False, why)."""
    files = [path] + [path.replace(".json", f"-rank{rank}.json") for rank in range(1, size)]
    records = []
    for name in files:
        try:
            with open(name) as stream:
                records.append(json.load(stream))
        except (OSError, ValueError) as exc:
            return False, f"{os.path.basename(name)}: {type(exc).__name__}"
    for key in ("arm", "mode", "model_path", "code_sha256", "input_sha256", "attempt"):
        values = {json.dumps(record.get(key), sort_keys=True) for record in records}
        if len(values) != 1:
            return False, f"ranks disagree on {key}"
    if not records[0].get("code_sha256"):
        return False, "no code hash"
    if attempt is not None and records[0].get("attempt") != attempt:
        return False, f"attempt {records[0].get('attempt')!r}, not {attempt!r}"
    pipelines = [record.get("pipeline") or {} for record in records]
    if sorted(p.get("rank", -1) for p in pipelines) != list(range(size)) or any(
            p.get("size") != size for p in pipelines):
        return False, "rank set incomplete: " + str(sorted(p.get("rank") for p in pipelines))
    return True, "complete"


def gpu_idle(smi_output: str | None = None) -> tuple[bool, str]:
    """(True, "idle") when no compute process holds a GPU, else (False, the pids).

    Run after a stage, from the notebook that owns the GPUs: a rank that outlived its stage
    shows up here, and a later stage would otherwise share its card. It only reports; it never
    terminates anything, since a process it did not start is not its to end.
    """
    if smi_output is None:
        try:
            smi_output = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"nvidia-smi: {type(exc).__name__}"
    pids = [line.strip() for line in smi_output.splitlines() if line.strip()]
    return (not pids), ("idle" if not pids else "still on a GPU: " + ", ".join(pids))


def _self_check() -> None:
    assert gpu_idle("") == (True, "idle")
    assert gpu_idle("4242\n 77 \n") == (False, "still on a GPU: 4242, 77")
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "e2e.json")
        base = {"arm": "kernel", "mode": "e2e", "model_path": "/m", "code_sha256": "c", "attempt": "a1"}
        assert all_ranks(out) == (False, "e2e.json: FileNotFoundError")
        with open(out, "w") as stream:
            json.dump({**base, "pipeline": {"rank": 0, "size": 2}}, stream)
        assert all_ranks(out) == (False, "e2e-rank1.json: FileNotFoundError")
        with open(out.replace(".json", "-rank1.json"), "w") as stream:
            json.dump({**base, "arm": "stock", "pipeline": {"rank": 1, "size": 2}}, stream)
        assert all_ranks(out) == (False, "ranks disagree on arm")
        with open(out.replace(".json", "-rank1.json"), "w") as stream:
            json.dump({**base, "pipeline": {"rank": 1, "size": 2}}, stream)
        assert all_ranks(out) == (True, "complete")
        assert all_ranks(out, attempt="a1") == (True, "complete")
        assert all_ranks(out, attempt="a2") == (False, "attempt 'a1', not 'a2'")
        with open(out.replace(".json", "-rank1.json"), "w") as stream:
            json.dump({**base, "code_sha256": "d", "pipeline": {"rank": 1, "size": 2}}, stream)
        assert all_ranks(out) == (False, "ranks disagree on code_sha256")
        for name in (out, out.replace(".json", "-rank1.json")):
            with open(name) as stream:
                record = json.load(stream)
            with open(name, "w") as stream:
                json.dump({**record, "code_sha256": None}, stream)
        assert all_ranks(out) == (False, "no code hash")
    print("ranks self-check ok")


if __name__ == "__main__":
    _self_check()
