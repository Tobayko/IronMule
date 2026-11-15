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


# ---- TP0b: the whole-step probe (tools/tpu_step_probe.py) ----

step_spec = importlib.util.spec_from_file_location("tpu_step_probe", ROOT / "tools" / "tpu_step_probe.py")
step_probe = importlib.util.module_from_spec(step_spec)
step_spec.loader.exec_module(step_probe)


def test_the_step_kernel_runs_one_pinned_commit_on_a_tpu(tmp_path):
    with pytest.raises(ValueError):
        step_probe.write_kernel(tmp_path / "bad", "tobayko", "main")
    folder = step_probe.write_kernel(tmp_path / "k", "tobayko", "b" * 40)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["enable_tpu"] and not metadata["enable_gpu"] and metadata["is_private"]
    assert metadata["id"] == "tobayko/ironmule-tpu-step-probe" and metadata["machine_shape"] == "TpuV5E8"
    run = (folder / "run.py").read_text()
    assert "b" * 40 in run and "/kaggle/working/tpu_step_probe_report.json" in run
    assert '\nif __name__ == "__main__":\n' not in run


def test_the_step_analysis_decides_by_the_chained_step_against_both_t4_references():
    def report(one_s, eight_s):
        return {"smoke": False, "t4_reference": step_probe.T4_REFERENCE, "results": [
            {"rows": 1, "capacity": 2048, "position": 1024, "per_call_s": 0.05, "chained_step_s": one_s},
            {"rows": 8, "capacity": 2048, "position": 1024, "per_call_s": 0.05, "chained_step_s": eight_s},
            {"rows": 64, "capacity": 512, "position": 256, "error": "RESOURCE_EXHAUSTED"}]}
    fast = step_probe.analyse(report(0.010, 0.012))
    assert fast["cases"]["1 rows, 2048 cache"]["chained_tok_s"] == pytest.approx(100.0)
    assert fast["decision"].startswith("TP1 reopens")
    assert fast["cases"]["64 rows, 512 cache"] == {"error": "RESOURCE_EXHAUSTED"}
    assert step_probe.analyse(report(0.010, 0.040))["decision"].startswith("TP1 stays closed"), "8 rows: 200 tok/s"
    assert step_probe.analyse(report(0.020, 0.012))["decision"].startswith("TP1 stays closed"), "1 row: 50 tok/s"
    assert step_probe.analyse({"smoke": False, "results": []})["decision"].startswith("no decision")


def test_the_whole_step_matches_a_plain_numpy_step_and_its_chain_matches_single_steps():
    """Run only where JAX is installed (the Kaggle TPU session, or a local venv); it checks the probe's arithmetic,
    so its time is a time of the model's real decode step and not of something cheaper."""
    jax = pytest.importorskip("jax")
    import numpy as np
    import jax.numpy as jnp

    config = step_probe.SMOKE
    params = step_probe.weights(config, seed=3)
    rows, capacity, position = 2, 16, 5
    keys, values = step_probe.cache(config, rows, capacity)
    rng = np.random.default_rng(0)  # earlier positions already hold keys and values, as in a running decode
    keys = keys.at[:, :, :, :position].set(jnp.asarray(rng.normal(size=keys[:, :, :, :position].shape), jnp.bfloat16))
    values = values.at[:, :, :, :position].set(
        jnp.asarray(rng.normal(size=values[:, :, :, :position].shape), jnp.bfloat16))
    tokens = jnp.asarray([7, 200], jnp.int32)
    got, _, _ = jax.jit(step_probe.make_step(config, logits=True))(params, tokens, keys, values, jnp.int32(position))

    f = {k: np.asarray(v, np.float32) for k, v in params.items()}
    K, V = np.asarray(keys, np.float32), np.asarray(values, np.float32)
    d, heads, kv = config["head_dim"], config["heads"], config["kv_heads"]
    inv = 1.0 / (config["theta"] ** (np.arange(0, d, 2) / d))
    cos, sin = np.cos(position * inv), np.sin(position * inv)

    def norm(x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config["eps"]) * w

    def rope(x):
        a, b = x[..., : d // 2], x[..., d // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    x = f["embed"][np.asarray(tokens)]
    for layer in range(config["layers"]):
        h = norm(x, f["n_attn"][layer])
        q = rope(norm((h @ f["wq"][layer]).reshape(rows, heads, d), f["n_q"][layer]))
        k = rope(norm((h @ f["wk"][layer]).reshape(rows, kv, d), f["n_k"][layer]))
        v = (h @ f["wv"][layer]).reshape(rows, kv, d)
        K[layer, :, :, position], V[layer, :, :, position] = k, v
        out = np.zeros((rows, heads, d))
        for r in range(rows):
            for head in range(heads):
                g = head // (heads // kv)
                s = K[layer, r, g, : position + 1] @ q[r, head] * d ** -0.5
                p = np.exp(s - s.max())
                out[r, head] = (p / p.sum()) @ V[layer, r, g, : position + 1]
        x = x + out.reshape(rows, heads * d) @ f["wo"][layer]
        h = norm(x, f["n_mlp"][layer])
        gate = h @ f["wg"][layer]
        x = x + (gate / (1 + np.exp(-gate)) * (h @ f["wu"][layer])) @ f["wd"][layer]
    want = norm(x, f["n_out"]) @ f["embed"].T
    got = np.asarray(got)
    assert np.max(np.abs(got - want)) <= 0.05 * np.max(np.abs(want)), "bf16 against float32, within bf16 noise"

    chain = jax.jit(step_probe.make_chain(step_probe.make_step(config), 4))
    single = jax.jit(step_probe.make_step(config))
    t1, k1, v1 = chain(params, tokens, keys, values, jnp.int32(position))
    t2, k2, v2 = tokens, keys, values
    for i in range(4):
        t2, k2, v2 = single(params, t2, k2, v2, jnp.int32(position + i))
    assert np.array_equal(np.asarray(t1), np.asarray(t2)) and np.array_equal(np.asarray(k1), np.asarray(k2))
    report = step_probe.probe(smoke=True)
    assert report["smoke"] and len(report["results"]) == 2 and all("error" not in r for r in report["results"])
    assert step_probe.analyse(report)["decision"].startswith("smoke")
