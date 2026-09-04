from __future__ import annotations

import copy
import hashlib
import math
import unittest

from friday_h01.analysis import analyze_trace
from friday_h01.canonical import canonical_sha256
from friday_h01.constants import (
    COOLDOWN_NS,
    MAX_GAP_OVERSHOOT_NS,
    SCHEMA_VERSION,
    SESSION_INVALID_STATUS,
    SESSION_ORDER,
    TOTAL_SAMPLES,
)
from friday_h01.protocol import (
    ProtocolError,
    build_manifest,
    build_trace,
    validate_manifest,
    validate_result,
    validate_trace,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def make_manifest(session_id: str = "C0", *, environment_label: str = "environment") -> dict:
    fixture_body = {
        "a_sha256": _digest("fixture-a"),
        "b_sha256": _digest("fixture-b"),
        "metadata_sha256": _digest("fixture-metadata"),
    }
    fixture = {**fixture_body, "fixture_sha256": canonical_sha256(fixture_body)}
    source = {
        "parent_phase": "H0",
        "parent_run_id": "h0-parent-offline-fixture",
        "parent_manifest_sha256": _digest("parent-manifest"),
        "parent_result_sha256": _digest("parent-result"),
        "parent_bundle_sha256": _digest("parent-bundle"),
    }
    return build_manifest(
        session_id,
        fixture=fixture,
        study_spec_sha256=_digest("h01-study-spec-v2"),
        code_sha256=_digest("h01-code-v2"),
        environment_sha256=_digest(environment_label),
        source=source,
    )


def make_trace(
    session_id: str = "C0",
    durations: list[int] | None = None,
    *,
    gap_overshoots: list[int] | None = None,
    environment_label: str = "environment",
) -> tuple[dict, dict]:
    manifest = make_manifest(session_id, environment_label=environment_label)
    trace = build_trace(
        manifest,
        durations or [1_000_000] * TOTAL_SAMPLES,
        gap_overshoots_ns=gap_overshoots,
    )
    return manifest, trace


def integer_leaf_paths(value: object, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(value) is int:
        return [path]
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in integer_leaf_paths(child, path + (key,))
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in integer_leaf_paths(child, path + (index,))
        ]
    return []


def replace_leaf(value: dict, path: tuple[object, ...], replacement: object) -> dict:
    changed = copy.deepcopy(value)
    cursor: object = changed
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    return changed


def path_text(root: str, path: tuple[object, ...]) -> str:
    text = root
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text


def rebind_manifest_hashes(manifest: dict) -> None:
    schedule_body = {
        key: value for key, value in manifest["schedule"].items() if key != "sha256"
    }
    manifest["schedule"]["sha256"] = canonical_sha256(schedule_body)
    identity = {key: value for key, value in manifest.items() if key != "run_id"}
    session_id = manifest["session"]["id"]
    manifest["run_id"] = f"h01-{session_id.lower()}-{canonical_sha256(identity)}"


class H01ProtocolTests(unittest.TestCase):
    def test_manifest_schema_is_closed_and_six_run_ids_are_distinct(self) -> None:
        run_ids = set()
        for session_id in SESSION_ORDER:
            with self.subTest(session_id=session_id):
                manifest = make_manifest(session_id)
                self.assertEqual(validate_manifest(manifest), manifest)
                self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
                run_ids.add(manifest["run_id"])
        self.assertEqual(len(run_ids), 6)
        extra = make_manifest()
        extra["unknown"] = None
        with self.assertRaises(ProtocolError):
            validate_manifest(extra)

    def test_real_gap_contract_and_cooldown_continuity(self) -> None:
        overshoots = [index % (MAX_GAP_OVERSHOOT_NS + 1) for index in range(TOTAL_SAMPLES)]
        manifest, trace = make_trace(gap_overshoots=overshoots)
        self.assertEqual(validate_trace(manifest, trace), trace)
        self.assertNotIn("actual_gap_ns", trace["samples"][0])
        for index, sample in enumerate(trace["samples"]):
            actual = sample["gap_end_ns"] - sample["gap_start_ns"]
            self.assertEqual(actual, sample["requested_gap_ns"] + overshoots[index])
            self.assertEqual(sample["gap_end_ns"], sample["start_ns"])
        transition = trace["samples"][32]
        previous = trace["samples"][31]
        self.assertEqual(
            transition["gap_start_ns"],
            previous["start_ns"] + previous["duration_ns"] + COOLDOWN_NS,
        )

    def test_trace_rejects_delete_duplicate_reorder_timestamp_and_digest_mutations(self) -> None:
        manifest, trace = make_trace()
        mutations: dict[str, dict] = {}
        deleted = copy.deepcopy(trace)
        del deleted["samples"][7]
        mutations["delete"] = deleted
        duplicated = copy.deepcopy(trace)
        duplicated["samples"][7] = copy.deepcopy(duplicated["samples"][6])
        mutations["duplicate"] = duplicated
        reordered = copy.deepcopy(trace)
        reordered["samples"][6], reordered["samples"][7] = (
            reordered["samples"][7],
            reordered["samples"][6],
        )
        mutations["reorder"] = reordered
        timestamp = copy.deepcopy(trace)
        timestamp["samples"][1]["start_ns"] += 1
        mutations["timestamp"] = timestamp
        digest = copy.deepcopy(trace)
        digest["schedule_sha256"] = "0" * 64
        mutations["digest"] = digest
        for name, mutation in mutations.items():
            with self.subTest(name=name), self.assertRaises(ProtocolError):
                validate_trace(manifest, mutation)

    def test_extra_pause_and_rebound_are_invalid_and_cannot_reuse_result_hash(self) -> None:
        manifest, trace = make_trace()
        valid_result = analyze_trace(manifest, trace)
        extra_pause = copy.deepcopy(trace)
        sample = extra_pause["samples"][40]
        sample["gap_end_ns"] += MAX_GAP_OVERSHOOT_NS + 1
        sample["start_ns"] = sample["gap_end_ns"]
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, extra_pause)
        invalid = analyze_trace(manifest, extra_pause)
        self.assertEqual(invalid["status"], SESSION_INVALID_STATUS)
        self.assertEqual(validate_result(invalid, manifest, extra_pause), invalid)
        with self.assertRaises(ProtocolError):
            validate_result(valid_result, manifest, extra_pause)

        rebound = copy.deepcopy(trace)
        rebound_sample = rebound["samples"][12]
        rebound_sample["gap_end_ns"] = rebound_sample["gap_start_ns"] - 1
        rebound_sample["start_ns"] = rebound_sample["gap_end_ns"]
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, rebound)

    def test_fixture_environment_source_and_telemetry_are_bound(self) -> None:
        manifest, trace = make_trace()
        mutations = []
        fixture = copy.deepcopy(trace)
        fixture["fixture"]["a_sha256"] = "0" * 64
        mutations.append(fixture)
        environment = copy.deepcopy(trace)
        environment["environment_sha256"] = "0" * 64
        mutations.append(environment)
        source = copy.deepcopy(trace)
        source["source"]["parent_result_sha256"] = "0" * 64
        mutations.append(source)
        telemetry = copy.deepcopy(trace)
        telemetry["telemetry"]["thermal_state"] = {"value": None, "missing_reason": None}
        mutations.append(telemetry)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ProtocolError):
                validate_trace(manifest, mutation)

    def test_int64_bool_nonfinite_and_schema_contracts_fail_closed(self) -> None:
        manifest, trace = make_trace()
        bad_index = copy.deepcopy(manifest)
        bad_index["session"]["index"] = False
        with self.assertRaises(ProtocolError):
            validate_manifest(bad_index)
        with self.assertRaises(ProtocolError):
            build_trace(manifest, [True] + [1_000_000] * (TOTAL_SAMPLES - 1))
        too_large = copy.deepcopy(trace)
        too_large["samples"][0]["duration_ns"] = 1 << 63
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, too_large)
        nonfinite = copy.deepcopy(trace)
        nonfinite["samples"][0]["duration_ns"] = math.inf
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, nonfinite)
        extra = copy.deepcopy(trace)
        extra["samples"][0]["unexpected"] = 1
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, extra)

    def test_every_manifest_integer_leaf_rejects_bool_before_identity_reconstruction(self) -> None:
        manifest = make_manifest()
        paths = integer_leaf_paths(manifest)
        self.assertEqual(len(paths), 575)
        for path in paths:
            mutation = replace_leaf(manifest, path, True)
            rebind_manifest_hashes(mutation)
            expected_path = path_text("manifest", path)
            with self.subTest(path=expected_path), self.assertRaises(ProtocolError) as raised:
                validate_manifest(mutation)
            self.assertIn(expected_path, str(raised.exception))
        false_manifest = replace_leaf(manifest, ("session", "index"), False)
        rebind_manifest_hashes(false_manifest)
        with self.assertRaises(ProtocolError):
            validate_manifest(false_manifest)
        float_manifest = replace_leaf(manifest, ("schema_version",), float(SCHEMA_VERSION))
        rebind_manifest_hashes(float_manifest)
        with self.assertRaises(ProtocolError):
            validate_manifest(float_manifest)

    def test_every_trace_integer_leaf_rejects_bool_before_binding_comparison(self) -> None:
        manifest, trace = make_trace()
        paths = integer_leaf_paths(trace)
        self.assertEqual(len(paths), 1011)
        for path in paths:
            mutation = replace_leaf(trace, path, True)
            expected_path = path_text("trace", path)
            with self.subTest(path=expected_path), self.assertRaises(ProtocolError) as raised:
                validate_trace(manifest, mutation)
            self.assertIn(expected_path, str(raised.exception))
        with self.assertRaises(ProtocolError):
            validate_trace(manifest, replace_leaf(trace, ("samples", 0, "sample_index"), False))
        with self.assertRaises(ProtocolError):
            validate_trace(
                manifest,
                replace_leaf(trace, ("cooldown", "requested_ns"), float(COOLDOWN_NS)),
            )


if __name__ == "__main__":
    unittest.main()
