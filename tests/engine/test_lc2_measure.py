"""LC2's harness (`tools/lc2_measure.py`): commands, kernel and its fixed reading on synthetic reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lc2_measure", ROOT / "tools" / "lc2_measure.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _cell(kind, rate, first, n):
    rows = [{"first_s": first, "end_s": first + 63 / rate, "tokens": 64, "text": "x", "error": None} for _ in range(n)]
    return {"kind": kind, "at_once": kind == "load", "complete": True, "requests": rows}


def _arm(arm, run, load, quiet, first=0.05):
    return {"arm": arm, "run": run, "started": True,
            "cells": [_cell("load", load, first, 8) for _ in range(5)] + [_cell("quiet", quiet, first, 16)]}


def _report(i=(300.0, 60.0), others=None):
    others = others or {"L": (300.0, 60.0), "Lf": (330.0, 62.0), "V": (600.0, 70.0)}
    arms = [_arm("I", 1, *i)] + [_arm(a, 1, *v) for a, v in others.items()] + [_arm("I", 2, *i)]
    return {"protocol": "lc2", "arms": arms}


def test_the_reading_names_the_fastest_engine_and_whether_mlx_holds_its_own():
    out = tool.analyse(_report())
    assert out["valid"] and out["fastest"] == "V" and out["next"].startswith("the agent on V")
    assert out["against"]["L"]["holds_its_own"] and not out["against"]["V"]["holds_its_own"]
    assert out["goal_1_2_against_l"] is False
    alone = tool.analyse(_report(i=(700.0, 90.0)))
    assert alone["fastest"] == "I" and alone["next"] == "MLX stays the CUDA base"
    assert alone["goal_1_2_against_l"] is True


def test_an_arm_that_failed_is_named_and_the_rest_is_read():
    report = _report()
    report["arms"][3] = {"arm": "V", "run": 1, "started": False, "error": "the server did not become ready",
                         "cells": []}
    out = tool.analyse(report)
    assert out["valid"] and out["failed"] == {"V1": "the server did not become ready"} and "V" not in out["against"]
    report["arms"].pop()
    assert tool.analyse(report)["valid"] is False


def test_each_arm_gets_its_fixed_server_command():
    paths = {"ironmule": "ironmule", "ironmule_model": tool.MLX_MODEL, "state": "/s", "llama_server": "ls",
             "gguf": "m.gguf", "vllm_python": "py"}
    command, model, usage = tool.commands("I", 9, paths)
    assert command[:5] == ["ironmule", "serve", "--model", tool.MLX_MODEL, "--agent"] and not usage
    plain, _, _ = tool.commands("L", 9, paths)
    flash, _, _ = tool.commands("Lf", 9, paths)
    assert "-fa" not in plain and flash == plain + ["-fa"] and "-np" in plain
    vllm, model, usage = tool.commands("V", 9, paths)
    assert model == tool.HF_MODEL and usage and vllm[vllm.index("--dtype") + 1] == "half"
    assert vllm[vllm.index("--max-num-seqs") + 1] == "8" and tool.ORDER == ("I", "L", "Lf", "V", "I")


def test_the_kernel_is_private_on_a_t4_and_pinned_to_a_full_commit(tmp_path):
    with pytest.raises(ValueError):
        tool.write_kernel(tmp_path, "owner", "abc")
    sha = "0123456789abcdef0123456789abcdef01234567"
    tool.write_kernel(tmp_path, "owner", sha)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and metadata["enable_gpu"] and metadata["machine_shape"] == "NvidiaTeslaT4"
    run = (tmp_path / "run.py").read_text()
    assert run.count(sha) == 3 and "lc2_measure.py" in run
    compile(run, "run.py", "exec")
