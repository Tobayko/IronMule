"""Control-plane checks for the opt-in finite greedy candidate.

These tests intentionally do not import or execute MLX model code.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from ironmule_product import greedy
from ironmule_product.greedy import GreedyCompatibilityError, MAX_FINITE_TOKENS, bounded_generate_step


ROOT = Path(__file__).resolve().parents[2]


def test_import_does_not_initialize_mlx() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import ironmule_product.greedy; print('mlx' in sys.modules, 'mlx_lm' in sys.modules)"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"


def test_worker_absolute_script_bootstraps_package_from_other_cwd(tmp_path: Path) -> None:
    script = (
        "import runpy; runpy.run_path(%r, run_name='worker_bootstrap_test'); "
        "from ironmule_product.greedy import STOCK_MLX_LM_VERSION; print(STOCK_MLX_LM_VERSION)"
    ) % str(ROOT / "ironmule_product" / "worker.py")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "0.31.3"


def test_version_mismatch_is_rejected_before_mlx_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(greedy.metadata, "version", lambda name: "0.31.4")
    candidate = bounded_generate_step([1], object())
    with pytest.raises(GreedyCompatibilityError, match="0.31.3"):
        next(candidate)


@pytest.mark.parametrize("max_tokens", [0, -1, True, MAX_FINITE_TOKENS + 1, 1.0])
def test_finite_limit_is_rejected_before_model_work(max_tokens: object) -> None:
    candidate = bounded_generate_step(object(), object(), max_tokens=max_tokens)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_tokens"):
        next(candidate)


@pytest.mark.parametrize(
    "option",
    [
        "sampler",
        "logits_processors",
        "prompt_cache",
        "kv_bits",
        "kv_group_size",
        "quantized_kv_start",
        "prompt_progress_callback",
        "input_embeddings",
        "draft_model",
        "num_draft_tokens",
    ],
)
def test_unsupported_generation_options_reject_before_model_work(option: str) -> None:
    candidate = bounded_generate_step(object(), object(), **{option: None})
    with pytest.raises(ValueError, match=option):
        next(candidate)


def test_prefill_step_size_is_the_only_optional_control() -> None:
    candidate = bounded_generate_step(object(), object(), prefill_step_size=0)
    with pytest.raises(ValueError, match="prefill_step_size"):
        next(candidate)


def test_empty_prompt_rejects_before_model_work() -> None:
    candidate = bounded_generate_step([], object())
    with pytest.raises(ValueError, match="prompt"):
        next(candidate)


def test_candidate_is_explicitly_pinned_to_finite_stock_compatibility() -> None:
    source = (ROOT / "ironmule_product" / "greedy.py").read_text(encoding="utf-8")
    assert "mlx-lm 0.31.3" in source
    assert "MIT licensed" in source
