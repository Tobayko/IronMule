"""NEXT1-I: a two-card stage counts only when every rank left a complete, matching record.

`mlx.launch` exits 0 even when a rank crashed (PERF1 run 11), so an exit code or rank 0's
file alone proves nothing. `perf1.py` writes rank 0 to OUT.json and rank r to
OUT-rankR.json, each atomically; this checks that all of them exist, parse, name the same
arm, mode and model, and carry ranks 0..size-1 of one pipeline.

    from ranks import all_ranks
    ok, why = all_ranks("/kaggle/working/e2e-qwen36-35b-a3b-kernel+p16.json")

Run as a script for its self-check.
"""
import json
import os
import tempfile


def all_ranks(path: str, size: int = 2) -> tuple[bool, str]:
    """(True, "complete") when every rank's record is there and agrees, else (False, why)."""
    files = [path] + [path.replace(".json", f"-rank{rank}.json") for rank in range(1, size)]
    records = []
    for name in files:
        try:
            with open(name) as stream:
                records.append(json.load(stream))
        except (OSError, ValueError) as exc:
            return False, f"{os.path.basename(name)}: {type(exc).__name__}"
    for key in ("arm", "mode", "model_path"):
        values = {json.dumps(record.get(key), sort_keys=True) for record in records}
        if len(values) != 1:
            return False, f"ranks disagree on {key}"
    pipelines = [record.get("pipeline") or {} for record in records]
    if sorted(p.get("rank", -1) for p in pipelines) != list(range(size)) or any(
            p.get("size") != size for p in pipelines):
        return False, "rank set incomplete: " + str(sorted(p.get("rank") for p in pipelines))
    return True, "complete"


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "e2e.json")
        base = {"arm": "kernel", "mode": "e2e", "model_path": "/m"}
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
    print("ranks self-check ok")


if __name__ == "__main__":
    _self_check()
