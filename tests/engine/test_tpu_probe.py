"""TP0's probe (tools/tpu_probe.py): its kernel pins a commit and asks for a TPU, and its analysis reads a report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tpu_probe", ROOT / "tools" / "tpu_probe.py")
tpu_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpu_probe)


def test_the_kernel_runs_one_pinned_commit_on_a_tpu(tmp_path):
    with pytest.raises(ValueError):
        tpu_probe.write_kernel(tmp_path / "bad", "tobayko", "main")
    folder = tpu_probe.write_kernel(tmp_path / "k", "tobayko", "a" * 40)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["enable_tpu"] and not metadata["enable_gpu"] and metadata["is_private"]
    assert metadata["machine_shape"] == "TpuV5E8" and metadata["id"] == "tobayko/ironmule-tpu-probe"
    run = (folder / "run.py").read_text()
    assert "a" * 40 in run and "/kaggle/working/tpu_probe_report.json" in run
    assert '\nif __name__ == "__main__":\n' not in run  # the entry runs main once, with the kernel's arguments


def test_the_analysis_turns_product_times_into_a_ceiling_not_a_speed():
    shapes = [(4, 8), (8, 4)]
    rows = [{"k": k, "n": n, "rows": m, "seconds": 0.001 * m} for k, n in shapes for m in (1, 8)]
    result = tpu_probe.analyse({"layers": 2, "jax": rows, "mlx_cpu": {"error": "ModuleNotFoundError: mlx"}})
    one = result["runtimes"]["jax"]["1"]
    assert one["matrix_step_s"] == pytest.approx(2 * 2 * 0.001)
    assert one["ceiling_tok_s"] == pytest.approx(1 / 0.004)
    assert one["weight_gb_s"] == pytest.approx(2 * 2 * (2 * 32) / 0.004 / 1e9)
    assert result["runtimes"]["jax"]["8"]["ceiling_tok_s"] == pytest.approx(8 / 0.032)
    assert result["runtimes"]["mlx_cpu"] == {"error": "ModuleNotFoundError: mlx"}  # a missing runtime stays visible


def test_a_smoke_run_records_what_is_missing_instead_of_failing():
    report = tpu_probe.probe(smoke=True)
    assert report["smoke"] and report["schema"] == tpu_probe.SCHEMA
    for name in ("jax", "mlx_cpu"):
        assert isinstance(report[name], list) or "error" in report[name]
    assert set(tpu_probe.analyse(report)["runtimes"]) == {"jax", "mlx_cpu"}
