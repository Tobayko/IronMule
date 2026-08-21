from __future__ import annotations

import copy
import unittest

from friday_h01.canonical import canonical_sha256
from friday_h01.constants import (
    BURN_IN_BLOCKS,
    LONG_LABEL,
    MAIN_BLOCKS,
    SCHEMA_VERSION,
    SESSION_ORDER,
    SESSION_SPECS,
    SHORT_LABEL,
    TOTAL_SAMPLES,
)
from friday_h01.schedule import ScheduleError, materialize_schedule, validate_schedule


def _integer_leaf_paths(value: object, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(value) is int:
        return [path]
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _integer_leaf_paths(child, path + (key,))
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _integer_leaf_paths(child, path + (index,))
        ]
    return []


def _replace(value: dict, path: tuple[object, ...], replacement: object) -> dict:
    changed = copy.deepcopy(value)
    cursor: object = changed
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    body = {key: child for key, child in changed.items() if key != "sha256"}
    changed["sha256"] = canonical_sha256(body)
    return changed


def _path_text(root: str, path: tuple[object, ...]) -> str:
    text = root
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text


class H01ScheduleTests(unittest.TestCase):
    def test_six_sessions_have_deterministic_disjoint_materializations(self) -> None:
        hashes = set()
        self.assertEqual(tuple(SESSION_SPECS), SESSION_ORDER)
        self.assertEqual(len({spec[2] for spec in SESSION_SPECS.values()}), 6)
        for session_id in SESSION_ORDER:
            with self.subTest(session_id=session_id):
                first = materialize_schedule(session_id)
                second = materialize_schedule(session_id)
                body = {key: value for key, value in first.items() if key != "sha256"}
                self.assertEqual(first, second)
                self.assertEqual(first["schema_version"], SCHEMA_VERSION)
                self.assertEqual(first["sha256"], canonical_sha256(body))
                self.assertEqual(first["seed"], SESSION_SPECS[session_id][2])
                self.assertEqual(validate_schedule(first), first)
                hashes.add(first["sha256"])
        self.assertEqual(len(hashes), 6)

    def test_each_phase_block_is_balanced_and_requested_gaps_are_exact(self) -> None:
        entries = materialize_schedule("C0")["entries"]
        self.assertEqual(len(entries), TOTAL_SAMPLES)
        cursor = 0
        for phase, blocks in (("burn_in", BURN_IN_BLOCKS), ("main", MAIN_BLOCKS)):
            phase_index = 0
            for block_index in range(blocks):
                block = entries[cursor : cursor + 4]
                self.assertEqual([row["sample_index"] for row in block], list(range(cursor, cursor + 4)))
                self.assertEqual([row["phase_index"] for row in block], list(range(phase_index, phase_index + 4)))
                self.assertEqual({row["phase"] for row in block}, {phase})
                self.assertEqual({row["block_index"] for row in block}, {block_index})
                self.assertEqual([row["position"] for row in block], [0, 1, 2, 3])
                self.assertEqual([row["gap_label"] for row in block].count(SHORT_LABEL), 2)
                self.assertEqual([row["gap_label"] for row in block].count(LONG_LABEL), 2)
                self.assertEqual(
                    sorted(row["requested_gap_ns"] for row in block),
                    [50_000_000, 50_000_000, 750_000_000, 750_000_000],
                )
                cursor += 4
                phase_index += 4

    def test_any_materialized_schedule_mutation_is_rejected(self) -> None:
        original = materialize_schedule("V2")
        mutations = []
        changed_order = copy.deepcopy(original)
        changed_order["entries"][0], changed_order["entries"][1] = (
            changed_order["entries"][1],
            changed_order["entries"][0],
        )
        mutations.append(changed_order)
        changed_digest = copy.deepcopy(original)
        changed_digest["sha256"] = "0" * 64
        mutations.append(changed_digest)
        unknown_key = copy.deepcopy(original)
        unknown_key["unexpected"] = 1
        mutations.append(unknown_key)
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ScheduleError):
                validate_schedule(mutation)

    def test_every_integer_leaf_rejects_bool_before_materialization_comparison(self) -> None:
        schedule = materialize_schedule("C0")
        paths = _integer_leaf_paths(schedule)
        self.assertEqual(len(paths), 562)
        for path in paths:
            mutation = _replace(schedule, path, True)
            expected_path = _path_text("schedule", path)
            with self.subTest(path=expected_path), self.assertRaises(ScheduleError) as raised:
                validate_schedule(mutation)
            self.assertIn(expected_path, str(raised.exception))
        with self.assertRaises(ScheduleError):
            validate_schedule(_replace(schedule, ("entries", 1, "position"), False))
        with self.assertRaises(ScheduleError):
            validate_schedule(_replace(schedule, ("schema_version",), float(SCHEMA_VERSION)))


if __name__ == "__main__":
    unittest.main()
