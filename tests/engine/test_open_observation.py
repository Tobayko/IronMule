"""Control-plane contracts for observation-only worker telemetry."""

from __future__ import annotations

import inspect

import pytest

from friday_evidence.open_observation import Observer


def test_public_api_is_neutral_and_has_expected_parameters() -> None:
    assert list(inspect.signature(Observer).parameters) == ["on_sample", "interval_seconds"]
    assert list(inspect.signature(Observer.bind).parameters) == ["self", "process", "label"]
    assert list(inspect.signature(Observer.sample).parameters) == ["self", "force"]
    assert list(inspect.signature(Observer.summary).parameters) == ["self"]


@pytest.mark.parametrize("interval", [True, -0.1, float("inf"), "1"])
def test_invalid_interval_is_a_programmer_error(interval: object) -> None:
    with pytest.raises(ValueError):
        Observer(interval_seconds=interval)  # type: ignore[arg-type]


def test_invalid_bind_arguments_are_programmer_errors() -> None:
    observer = Observer()
    with pytest.raises(TypeError, match="process"):
        observer.bind(object(), "worker")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="label"):
        observer.bind(None, 1)  # type: ignore[arg-type]


def test_empty_summary_is_an_observation_report_not_a_gate() -> None:
    assert Observer().summary() == {
        "sample_count": 0,
        "skipped_interval_count": 0,
        "memory_sample_count": 0,
        "swap_sample_count": 0,
        "max_memory_bytes": {
            "rss_bytes": None,
            "physical_footprint_bytes": None,
            "peak_footprint_bytes": None,
            "wired_bytes": None,
        },
        "latest_system_swap_delta_bytes": None,
        "errors": {},
    }
