"""A vector may only say what was measured, and must survive a round trip saying it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ironmule import characterization as ch

ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / "research" / "raw" / "B71_vector_m1max_20260910_v3.json"
DATASET = ROOT / "research" / "raw" / "B72_cost_dataset_20260910.json"


def _vector():
    return ch.load(json.loads(VECTOR.read_text())["vector"])


def test_the_measured_vector_round_trips():
    vector = _vector()
    assert ch.load(vector.to_json()).as_dict() == vector.as_dict()


def test_an_empty_vector_is_all_missing_and_still_valid():
    empty = ch.HardwareCharacterizationVector(
        ch.SCHEMA, ch.StaticFacts(), ch.MeasuredResponses(), ch.Conditions())
    assert set(empty.missing()) == set(ch.MEASURED_FIELDS)
    assert ch.load(empty.to_json()).missing() == empty.missing()


def test_nothing_is_imputed():
    """The geometry probe withheld its field on a loaded machine. It must stay withheld."""
    vector = _vector()
    assert "geometry_response" in vector.missing()
    assert vector.measured.geometry_response == ()
    assert "geometry_4_8_ratio" not in vector.relations


def test_a_relation_never_appears_without_its_inputs():
    measured = ch.MeasuredResponses(
        width_response=(ch.Measurement(1.0, "ns_per_row", "e", context={"width": 1}),))
    relations = ch.relations_for(measured)
    assert "m16_cost_per_row_vs_m1" not in relations
    assert "m8_cost_per_row_vs_m4" not in relations


def test_every_relation_names_where_it_came_from():
    for name, row in _vector().relations.items():
        assert row["from"] and all(isinstance(item, str) for item in row["from"]), name
        assert row["note"], name


def test_every_measurement_carries_its_evidence():
    measured = _vector().measured
    for name in ch.MEASURED_FIELDS:
        value = getattr(measured, name)
        entries = value if isinstance(value, tuple) else ([value] if value else [])
        for entry in entries:
            assert entry.evidence_id, name
            assert entry.unit, name


def test_a_missing_list_that_disagrees_is_refused():
    raw = json.loads(_vector().to_json())
    raw["missing"] = []
    with pytest.raises(ch.CharacterizationError):
        ch.load(raw)


def test_a_measurement_with_fields_out_of_place_is_refused():
    raw = json.loads(_vector().to_json())
    raw["measured"]["cache_residency_ratio"]["note"] = "extra"
    with pytest.raises(ch.CharacterizationError):
        ch.load(raw)


def test_a_wrong_schema_is_refused():
    raw = json.loads(_vector().to_json())
    raw["schema"] = "ironmule.hardware_characterization_vector.v2"
    with pytest.raises(ch.CharacterizationError):
        ch.load(raw)


def test_no_single_score_is_produced():
    """A machine fast at one thing and slow at another is what one number destroys."""
    relations = _vector().relations
    assert all(isinstance(row, dict) and "value" in row for row in relations.values())
    assert not any(name in relations for name in ("score", "rating", "index", "overall"))


# -- the dataset for B72 -----------------------------------------------------


def test_every_dataset_row_has_the_exact_fields():
    record = json.loads(DATASET.read_text())
    assert record["schema"] == "ironmule.cost_dataset.v1"
    for row in record["rows"]:
        assert set(row) == set(record["row_fields"])
        assert row["evidence_id"] and row["validity"]
        assert row["measured_cost"]["value"] is not None


def test_no_blocked_or_invalid_evidence_reached_the_dataset():
    record = json.loads(DATASET.read_text())
    for row in record["rows"]:
        assert row["validity"] not in ("BLOCKED", "EXPLORATORY_INVALID", "NOT_STARTED")
    assert record["excluded_sources"], "the exclusions must be listed, not silent"


def test_the_dataset_says_it_has_one_machine():
    record = json.loads(DATASET.read_text())
    fingerprints = {row["hardware_features"]["hardware_fingerprint"] for row in record["rows"]}
    assert len(fingerprints) == 1
    assert "one machine" in record["one_machine"]
