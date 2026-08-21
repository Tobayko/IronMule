from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from friday_h0.protocol import close_manifest
from friday_h0.storage import Storage as H0Storage
from friday_h01.canonical import canonical_sha256
from friday_h01 import import_h0 as import_module
from friday_h01.import_h0 import (
    ADAPTER_REGISTRY_SCHEMA_VERSION,
    CLAIMED_KNOWN_MALFORMED,
    COMPLETED_ADAPTER,
    INVALID_ADAPTER,
    MATCHED,
    OUTCOME_LEGACY_OBSERVATION,
    PARSER_COMPLETED_V1,
    RECOGNIZED_NO_WARMUP,
    RUNTIME_UNAVAILABLE_ADAPTER,
    STATIC_ADAPTER_REGISTRY,
    UNSUPPORTED_GENERATION,
    W1V3_COMPLETED_ADAPTER,
    AdapterDescriptor,
    AdapterRegistry,
    LegacyImportError,
    audit_h0_legacy_warmups,
    build_legacy_entity_binding,
    declared_schema_tags,
    inventory_h0_generations,
    structural_fingerprint,
)
from friday_h01.storage import Storage, StorageConflict
from tests.test_manifest import valid_manifest


FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "h01"
ARCHIVED_FIXTURES = {
    RUNTIME_UNAVAILABLE_ADAPTER: (
        "h0_runtime_unavailable_246e_v1.json",
        "7f2734234984994e4625f8246da8f7c80f05280ced895167637a4aca14d74813",
    ),
    COMPLETED_ADAPTER: (
        "h0_completed_legacy_5f62_v1.json",
        "22876adef92c0ccd93e50be82e50db349e9e261e1111b573a49bc1c4d1802f0c",
    ),
    INVALID_ADAPTER: (
        "h0_warmup_unstable_aae3_v1.json",
        "f5058482ded55742eab50ef5b29ab9e82d8b9bac256e36eb4f70abfc74967f7d",
    ),
    W1V3_COMPLETED_ADAPTER: (
        "h0_completed_w1v3_101c_v1.json",
        "2f60fb80eb8610c2fec658d6495cfb3eba2ab9407caae951ee96a4dc0a1ee82a",
    ),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archived_fixture(adapter_id: str) -> dict[str, object]:
    filename, expected_file_sha256 = ARCHIVED_FIXTURES[adapter_id]
    path = FIXTURE_ROOT / filename
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise AssertionError(f"archived fixture file hash changed: {filename}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("archived fixture must be an object")
    body = {key: child for key, child in value.items() if key != "fixture_body_sha256"}
    if value.get("fixture_body_sha256") != canonical_sha256(body):
        raise AssertionError(f"archived fixture body hash changed: {filename}")
    if value.get("fixture_schema") != "friday_h01.archived_h0_common_result_fixture.v1":
        raise AssertionError("archived fixture schema is not registered")
    return value


def _persist_archived_fixture(
    path: Path, fixture: dict[str, object], *, created_at: int | None = None
) -> None:
    closed = close_manifest(copy.deepcopy(fixture["manifest"]))
    payload = fixture["common_result_payload"]
    if not isinstance(payload, dict) or not isinstance(payload.get("bundle"), dict):
        raise AssertionError("archived Common Result fixture is malformed")
    bundle = payload["bundle"]
    with H0Storage.open(path) as storage:
        storage.persist_common_result(
            closed,
            copy.deepcopy(bundle["result"]),
            created_at_unix_ns=(
                fixture["source_created_at_unix_ns"] if created_at is None else created_at
            ),
            raw_samples=copy.deepcopy(bundle["raw_samples"]),
            scalar_metrics=copy.deepcopy(bundle["scalar_metrics"]),
            correctness_metrics=copy.deepcopy(bundle["correctness_metrics"]),
            artifacts=copy.deepcopy(bundle["artifacts"]),
        )


def _warmup_list(fixture: dict[str, object], adapter_id: str) -> list[int]:
    result = fixture["common_result_payload"]["bundle"]["result"]
    benchmark = result["evidence"]["benchmark_evidence"]
    if adapter_id == INVALID_ADAPTER:
        return list(benchmark["failure_diagnostic"]["details"]["warmups_ns"])
    return list(benchmark["arms"]["baseline"]["warmup"]["durations_ns"])


def _adapter_contract() -> dict[str, object]:
    return {
        "common_result_ready": False,
        "reason": "single-process measurements require aggregation before any global decision",
        "mapping": {
            "runtime_unavailable": "invalid/baseline_fallback",
            "invalid*": "invalid/baseline_fallback",
            "measurement_complete": "aggregation_required",
            "baseline_reference": "not_run",
        },
    }


def _closed(run_id: str, *, provenance_digit: str = "a"):
    manifest = valid_manifest("eager_baseline")
    manifest["run_id"] = run_id
    if provenance_digit != "a":
        manifest["provenance"]["code_sha256"] = provenance_digit * 64
    return close_manifest(manifest)


def _result(closed: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": closed.run_id,
        "mode": "eager_baseline",
        "manifest_sha256": closed.sha256,
        "status": "completed",
        "classification": "measurement_complete",
        "action": "baseline_fallback",
        "error": None,
        "evidence": {
            "rss_peak_bytes": 2_000,
            "rss_missing_reason": None,
            "benchmark_classification": "baseline_reference",
            "benchmark_action": "not_run",
            "aggregation_required": False,
            "adapter_contract": _adapter_contract(),
            "benchmark_evidence": {},
        },
    }


def _persist(path: Path, closed: object, result: dict[str, object], created: int) -> None:
    with H0Storage.open(path) as storage:
        storage.persist_common_result(closed, result, created_at_unix_ns=created)


def _selector_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "structural_fingerprint_algorithm": "sha256_recursive_json_structure_v1",
        "schema_tag_algorithm": "recursive_schema_version_paths_v1",
        "source_status": "completed",
        "parent_code_sha256": "1" * 64,
        "parent_spec_sha256": "2" * 64,
        "parent_environment_sha256": "3" * 64,
        "result_structure_sha256": "4" * 64,
        "result_schema_tags_sha256": "5" * 64,
        "diagnostic_present": False,
        "diagnostic_structure_sha256": None,
        "diagnostic_schema_tags_sha256": None,
    }


class H01LegacyInventoryTests(unittest.TestCase):
    def test_structural_fingerprint_is_value_independent_but_shape_exact(self) -> None:
        base = {
            "schema_version": 1,
            "nested": [{"flag": True, "label": "alpha", "value": 7}, None, 3.5],
        }
        changed_values = {
            "schema_version": 9,
            "nested": [{"flag": False, "label": "omega", "value": -8}, None, -4.25],
        }
        first = structural_fingerprint(base)
        self.assertEqual(first, structural_fingerprint(copy.deepcopy(base)))
        self.assertEqual(first, structural_fingerprint(changed_values))
        self.assertNotEqual(first, structural_fingerprint({**base, "extra": None}))
        self.assertNotEqual(first, structural_fingerprint({"schema_version": 1, "nested": []}))

        type_changed = copy.deepcopy(base)
        type_changed["nested"][0]["value"] = 7.0
        self.assertNotEqual(first, structural_fingerprint(type_changed))
        bool_changed_to_int = copy.deepcopy(base)
        bool_changed_to_int["nested"][0]["flag"] = 1
        self.assertNotEqual(first, structural_fingerprint(bool_changed_to_int))

        first_tags = declared_schema_tags(base)
        second_tags = declared_schema_tags(changed_values)
        self.assertEqual(first_tags, [{"path": "/schema_version", "type_class": "int", "value": 1}])
        self.assertEqual(second_tags[0]["value"], 9)
        self.assertNotEqual(canonical_sha256(first_tags), canonical_sha256(second_tags))

        for invalid in ((1, 2), {"x": float("nan")}, {"x": 1 << 63}):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(LegacyImportError):
                    structural_fingerprint(invalid)

    def test_archived_generations_are_verified_selected_and_parsed_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "h0.sqlite3"
            target = root / "h01.sqlite3"
            ordered_adapters = (
                RUNTIME_UNAVAILABLE_ADAPTER,
                COMPLETED_ADAPTER,
                INVALID_ADAPTER,
                W1V3_COMPLETED_ADAPTER,
            )
            fixtures = [_archived_fixture(adapter) for adapter in ordered_adapters]
            for fixture in fixtures:
                _persist_archived_fixture(source, fixture)
            source_before = _file_sha256(source)

            outcome = audit_h0_legacy_warmups(source)

            self.assertEqual(source_before, _file_sha256(source))
            self.assertFalse(target.exists())
            self.assertEqual(
                outcome.report["counts"], {"eligible": 4, "importable": 3, "excluded": 1}
            )
            self.assertEqual(
                [row["adapter"] for row in outcome.report["candidates"]],
                list(ordered_adapters),
            )
            self.assertEqual(
                [row["source_run_id"] for row in outcome.report["candidates"]],
                [fixture["source_run_id"] for fixture in fixtures],
            )
            excluded = outcome.report["candidates"][0]
            self.assertEqual(excluded["disposition"], "excluded")
            self.assertEqual(excluded["exclusion_reason"], RECOGNIZED_NO_WARMUP)
            self.assertIsNone(excluded["entity_id"])
            self.assertEqual(len(outcome.bundles), 3)
            bundles = {bundle["manifest"]["adapter"]: bundle for bundle in outcome.bundles}
            for adapter in ordered_adapters[1:]:
                with self.subTest(adapter=adapter):
                    fixture = _archived_fixture(adapter)
                    expected_warmup = _warmup_list(fixture, adapter)
                    bundle = bundles[adapter]
                    observation = bundle["trace"]["observation"]
                    descriptor = next(
                        item
                        for item in STATIC_ADAPTER_REGISTRY.descriptors
                        if item.adapter_id == adapter
                    )
                    self.assertEqual(observation["warmup_ns"], expected_warmup)
                    self.assertEqual(
                        observation["raw_warmup_sha256"], canonical_sha256(expected_warmup)
                    )
                    self.assertEqual(observation["descriptor_sha256"], descriptor.descriptor_sha256)
                    self.assertEqual(observation["parser_id"], descriptor.parser_id)
                    self.assertEqual(bundle["result"]["action"], "no_h0_conclusion")
                    self.assertIs(bundle["result"]["stationarity_supported"], False)
                    self.assertIs(bundle["result"]["h0_reclassification"], False)
            self.assertEqual(
                bundles[INVALID_ADAPTER]["trace"]["observation"]["source_diagnostic"],
                fixtures[2]["common_result_payload"]["bundle"]["result"]["evidence"]
                ["benchmark_evidence"]["failure_diagnostic"],
            )
            report_body = {
                key: value for key, value in outcome.report.items() if key != "report_sha256"
            }
            self.assertEqual(outcome.report["report_sha256"], canonical_sha256(report_body))

            with Storage.open(target) as storage:
                for bundle in outcome.bundles:
                    arguments = {
                        key: bundle[key]
                        for key in (
                            "entity_id",
                            "entity_kind",
                            "status",
                            "created_at_unix_ns",
                            "manifest",
                            "trace",
                            "result",
                            "lineage",
                        )
                    }
                    self.assertEqual(storage.persist_bundle(**arguments).state, "inserted")
                    self.assertEqual(storage.persist_bundle(**arguments).state, "idempotent")
                collision = dict(arguments)
                collision["created_at_unix_ns"] += 1
                with self.assertRaises(StorageConflict):
                    storage.persist_bundle(**collision)

    def test_execute_is_one_atomic_verified_and_idempotent_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "h0.sqlite3"
            target = root / "h01.sqlite3"
            for adapter in (
                RUNTIME_UNAVAILABLE_ADAPTER,
                COMPLETED_ADAPTER,
                INVALID_ADAPTER,
                W1V3_COMPLETED_ADAPTER,
            ):
                _persist_archived_fixture(source, _archived_fixture(adapter))
            source_before = _file_sha256(source)
            dry_run = audit_h0_legacy_warmups(source)

            first = audit_h0_legacy_warmups(
                source, execute=True, target=target
            )

            self.assertEqual(source_before, _file_sha256(source))
            self.assertTrue(target.is_file())
            self.assertEqual(first.report["execution_mode"], "adapter_execute")
            self.assertNotEqual(
                first.report["report_sha256"], dry_run.report["report_sha256"]
            )
            self.assertEqual(first.bundles, dry_run.bundles)
            self.assertEqual(
                [item.state for item in first.persistence], ["inserted"] * 3
            )
            self.assertEqual(
                [item.entity_id for item in first.persistence],
                [bundle["entity_id"] for bundle in first.bundles],
            )
            with Storage.open(target, read_only=True) as storage:
                with storage.read_transaction():
                    rows = storage.verified_rows()
            self.assertEqual(
                [row["bundle"] for row in rows], list(first.bundles)
            )

            target_after_first = _file_sha256(target)
            second = audit_h0_legacy_warmups(
                source, execute=True, target=target
            )
            self.assertEqual(
                [item.state for item in second.persistence], ["idempotent"] * 3
            )
            self.assertEqual(first.report, second.report)
            self.assertEqual(target_after_first, _file_sha256(target))
            self.assertEqual(source_before, _file_sha256(source))

            rollback_target = root / "rollback.sqlite3"
            planned = first.bundles[0]
            arguments = {
                key: copy.deepcopy(planned[key])
                for key in (
                    "entity_id",
                    "entity_kind",
                    "status",
                    "created_at_unix_ns",
                    "manifest",
                    "trace",
                    "result",
                    "lineage",
                )
            }
            arguments["created_at_unix_ns"] += 1
            with Storage.open(rollback_target) as storage:
                storage.persist_bundle(**arguments)
            with self.assertRaises(LegacyImportError):
                audit_h0_legacy_warmups(
                    source, execute=True, target=rollback_target
                )
            with Storage.open(rollback_target, read_only=True) as storage:
                with storage.read_transaction():
                    rollback_rows = storage.verified_rows()
            self.assertEqual(len(rollback_rows), 1)
            self.assertEqual(
                rollback_rows[0]["bundle"]["created_at_unix_ns"],
                planned["created_at_unix_ns"] + 1,
            )
            self.assertEqual(source_before, _file_sha256(source))

    def test_old_and_w1v3_parsers_do_not_use_the_current_h0_normalizer(self) -> None:
        self.assertNotIn("normalize_mlx_common_result", inspect.getsource(import_module))
        for adapter in (COMPLETED_ADAPTER, W1V3_COMPLETED_ADAPTER):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "h0.sqlite3"
                fixture = _archived_fixture(adapter)
                _persist_archived_fixture(source, fixture)
                outcome = audit_h0_legacy_warmups(source)
                self.assertEqual(outcome.report["counts"]["importable"], 1)
                self.assertEqual(outcome.bundles[0]["manifest"]["adapter"], adapter)

    def test_value_change_keeps_selector_but_changes_raw_hash_and_entity(self) -> None:
        fixture = _archived_fixture(COMPLETED_ADAPTER)
        changed = copy.deepcopy(fixture)
        warmup = changed["common_result_payload"]["bundle"]["result"]["evidence"] \
            ["benchmark_evidence"]["arms"]["baseline"]["warmup"]
        warmup["durations_ns"][0] += 1
        warmup["samples"][0]["value"] += 1
        results = []
        selectors = []
        for source_fixture in (fixture, changed):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "h0.sqlite3"
                _persist_archived_fixture(source, source_fixture)
                inventory = inventory_h0_generations(source)
                selectors.append(inventory["candidates"][0]["selector"])
                results.append(audit_h0_legacy_warmups(source))
        self.assertEqual(selectors[0], selectors[1])
        first_record = results[0].report["candidates"][0]
        changed_record = results[1].report["candidates"][0]
        self.assertNotEqual(first_record["raw_warmup_sha256"], changed_record["raw_warmup_sha256"])
        self.assertNotEqual(first_record["entity_id"], changed_record["entity_id"])
        self.assertNotEqual(first_record["h01_bundle_sha256"], changed_record["h01_bundle_sha256"])

    def test_registry_selectors_are_generation_not_run_specific(self) -> None:
        forbidden = {"run_id", "created_at_unix_ns", "warmup_ns", "durations_ns"}
        for descriptor in STATIC_ADAPTER_REGISTRY.descriptors:
            with self.subTest(adapter=descriptor.adapter_id):
                self.assertTrue(forbidden.isdisjoint(descriptor.selector))
                encoded = json.dumps(descriptor.selector, sort_keys=True)
                self.assertNotIn("h0-eager_baseline", encoded)
                self.assertEqual(descriptor.registry_schema_version, ADAPTER_REGISTRY_SCHEMA_VERSION)

    def test_inventory_replays_all_candidates_orders_and_excludes_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "h0.sqlite3"
            target = root / "h01.sqlite3"
            rows = (
                (_closed("run-z"), 20),
                (_closed("run-first"), 10),
                (_closed("run-a", provenance_digit="b"), 20),
            )
            for closed, created in rows:
                _persist(source, closed, _result(closed), created)
            before = _file_sha256(source)

            first = inventory_h0_generations(source)
            second = inventory_h0_generations(source)
            self.assertEqual(first, second)
            self.assertEqual(before, _file_sha256(source))
            self.assertFalse(target.exists())
            self.assertEqual(
                first["counts"],
                {
                    "eligible": 3,
                    MATCHED: 0,
                    UNSUPPORTED_GENERATION: 3,
                    CLAIMED_KNOWN_MALFORMED: 0,
                },
            )
            self.assertEqual(
                [row["source_run_id"] for row in first["candidates"]],
                ["run-first", "run-a", "run-z"],
            )
            self.assertEqual(len(first["adapter_descriptors"]), 4)
            body = {key: value for key, value in first.items() if key != "inventory_sha256"}
            self.assertEqual(first["inventory_sha256"], canonical_sha256(body))
            for index, candidate in enumerate(first["candidates"]):
                self.assertEqual(candidate["selection_index"], index)
                self.assertEqual(candidate["full_bundle_verification"], "verified")
                self.assertEqual(candidate["registry_match"]["state"], UNSUPPORTED_GENERATION)
                self.assertNotIn("warmup", json.dumps(candidate["selector"], sort_keys=True))
                self.assertEqual(
                    candidate["result_declared_schema_tags"][0],
                    {"path": "/schema_version", "type_class": "int", "value": 1},
                )
                for digest in candidate["stored_hashes"].values():
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")

            compatibility = audit_h0_legacy_warmups(source)
            self.assertEqual(compatibility.report["counts"], {"eligible": 3, "importable": 0, "excluded": 3})
            self.assertEqual(compatibility.bundles, ())
            self.assertEqual(compatibility.persistence, ())
            self.assertTrue(
                all(
                    row["exclusion_reason"] == UNSUPPORTED_GENERATION
                    for row in compatibility.report["candidates"]
                )
            )
            execute_outcome = audit_h0_legacy_warmups(
                source, execute=True, target=target
            )
            self.assertEqual(execute_outcome.report["execution_mode"], "adapter_execute")
            self.assertEqual(execute_outcome.persistence, ())
            self.assertFalse(target.exists())
            with self.assertRaises(LegacyImportError):
                audit_h0_legacy_warmups(source, execute=True)
            with self.assertRaises(LegacyImportError):
                audit_h0_legacy_warmups(source, target=target)
            with self.assertRaises(LegacyImportError):
                audit_h0_legacy_warmups(source, execute=1, target=target)
            with self.assertRaises(LegacyImportError):
                audit_h0_legacy_warmups(source, execute=True, target=source)
            self.assertEqual(before, _file_sha256(source))

    def test_registry_states_and_descriptor_entity_binding(self) -> None:
        selector = _selector_fixture()
        descriptor = AdapterDescriptor.from_selector(
            adapter_id="historical_completed_generation_v1",
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id="closed_completed_warmup_v1",
            selector=selector,
        )
        replay = AdapterDescriptor.from_selector(
            adapter_id="historical_completed_generation_v1",
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id="closed_completed_warmup_v1",
            selector=copy.deepcopy(selector),
        )
        self.assertEqual(descriptor, replay)
        self.assertEqual(descriptor.descriptor_sha256, replay.descriptor_sha256)
        registry = AdapterRegistry((descriptor,))
        matched = registry.match(selector)
        self.assertEqual(matched.state, MATCHED)
        self.assertEqual(matched.adapter_id, descriptor.adapter_id)
        self.assertEqual(matched.adapter_descriptor_sha256, descriptor.descriptor_sha256)

        malformed = copy.deepcopy(selector)
        malformed["result_structure_sha256"] = "6" * 64
        self.assertEqual(registry.match(malformed).state, CLAIMED_KNOWN_MALFORMED)
        unknown = copy.deepcopy(selector)
        unknown["parent_code_sha256"] = "7" * 64
        self.assertEqual(registry.match(unknown).state, UNSUPPORTED_GENERATION)
        self.assertEqual(len(STATIC_ADAPTER_REGISTRY.descriptors), 4)

        changed_descriptor = AdapterDescriptor.from_selector(
            adapter_id="historical_completed_generation_v1",
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id="closed_completed_warmup_v2",
            selector=selector,
        )
        hashes = {
            "parent_manifest_sha256": "8" * 64,
            "parent_result_sha256": "9" * 64,
            "parent_evidence_sha256": "a" * 64,
            "parent_bundle_sha256": "b" * 64,
            "raw_warmup_sha256": "c" * 64,
        }
        first = build_legacy_entity_binding(
            descriptor,
            adapter_registry_sha256=registry.registry_sha256,
            source_run_id="run-1",
            source_created_at_unix_ns=1,
            **hashes,
        )
        replayed = build_legacy_entity_binding(
            descriptor,
            adapter_registry_sha256=registry.registry_sha256,
            source_run_id="run-1",
            source_created_at_unix_ns=1,
            **hashes,
        )
        changed = build_legacy_entity_binding(
            changed_descriptor,
            adapter_registry_sha256=registry.registry_sha256,
            source_run_id="run-1",
            source_created_at_unix_ns=1,
            **hashes,
        )
        self.assertEqual(first, replayed)
        self.assertNotEqual(descriptor.descriptor_sha256, changed_descriptor.descriptor_sha256)
        self.assertNotEqual(first["entity_id"], changed["entity_id"])
        self.assertEqual(
            first["identity"]["adapter_descriptor_sha256"], descriptor.descriptor_sha256
        )
        self.assertNotEqual(
            canonical_sha256(
                {
                    "adapter": descriptor.adapter_id,
                    "parser_id": descriptor.parser_id,
                    "descriptor_sha256": descriptor.descriptor_sha256,
                }
            ),
            canonical_sha256(
                {
                    "adapter": changed_descriptor.adapter_id,
                    "parser_id": changed_descriptor.parser_id,
                    "descriptor_sha256": changed_descriptor.descriptor_sha256,
                }
            ),
        )

        with self.assertRaises(LegacyImportError):
            AdapterRegistry((descriptor, replay))
        invalid_selector = copy.deepcopy(selector)
        invalid_selector["schema_version"] = True
        with self.assertRaises(LegacyImportError):
            AdapterDescriptor.from_selector(
                adapter_id="bad",
                outcome=OUTCOME_LEGACY_OBSERVATION,
                parser_id="bad",
                selector=invalid_selector,
            )

    def test_claimed_shape_mismatch_aborts_before_parser(self) -> None:
        fixture = _archived_fixture(COMPLETED_ADAPTER)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "h0.sqlite3"
            _persist_archived_fixture(source, fixture)
            real_fingerprint = structural_fingerprint

            def shape_mutation(value: object) -> dict[str, object]:
                fingerprint = real_fingerprint(value)
                if isinstance(value, dict) and {"status", "classification", "evidence"} <= set(value):
                    fingerprint = dict(fingerprint)
                    fingerprint["sha256"] = "0" * 64
                return fingerprint

            with (
                patch("friday_h01.import_h0.structural_fingerprint", side_effect=shape_mutation),
                patch(
                    "friday_h01.import_h0._parse_descriptor",
                    side_effect=AssertionError("parser must not run for claimed malformed shape"),
                ),
                self.assertRaisesRegex(LegacyImportError, "claimed known generation"),
            ):
                audit_h0_legacy_warmups(source)

    def test_wrapper_and_child_tamper_stop_before_adapter_parser(self) -> None:
        fixture = _archived_fixture(COMPLETED_ADAPTER)
        for mutation in ("wrapper", "child"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "h0.sqlite3"
                _persist_archived_fixture(source, fixture)
                connection = sqlite3.connect(source)
                try:
                    if mutation == "wrapper":
                        row = connection.execute(
                            "SELECT sql FROM sqlite_master WHERE type='trigger' "
                            "AND name='status_events_append_only_update'"
                        ).fetchone()
                        connection.execute("DROP TRIGGER status_events_append_only_update")
                        connection.execute(
                            "UPDATE status_events SET payload_hash=? WHERE event_kind='common_result'",
                            ("0" * 64,),
                        )
                        connection.execute(row[0])
                    else:
                        trigger = connection.execute(
                            "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
                            "AND tbl_name='raw_samples' AND sql LIKE '%UPDATE%' LIMIT 1"
                        ).fetchone()
                        if trigger is not None:
                            connection.execute(f'DROP TRIGGER "{trigger[0]}"')
                        connection.execute(
                            "UPDATE raw_samples SET value=value+1 WHERE rowid=(SELECT MIN(rowid) FROM raw_samples)"
                        )
                        if trigger is not None:
                            connection.execute(trigger[1])
                    connection.commit()
                finally:
                    connection.close()
                with (
                    patch(
                        "friday_h01.import_h0._parse_descriptor",
                        side_effect=AssertionError("parser must not run before full H0 replay"),
                    ),
                    self.assertRaises(LegacyImportError),
                ):
                    audit_h0_legacy_warmups(source)

    def test_tamper_or_schema_drift_stops_before_any_fingerprint(self) -> None:
        for mutation in ("bundle", "schema"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "h0.sqlite3"
                closed = _closed(f"run-{mutation}")
                _persist(source, closed, _result(closed), 1)
                connection = sqlite3.connect(source)
                try:
                    if mutation == "bundle":
                        trigger = connection.execute(
                            "SELECT sql FROM sqlite_master WHERE type='trigger' "
                            "AND name='status_events_append_only_update'"
                        ).fetchone()[0]
                        connection.execute("DROP TRIGGER status_events_append_only_update")
                        connection.execute(
                            "UPDATE status_events SET payload_hash=? WHERE event_kind='common_result'",
                            ("0" * 64,),
                        )
                        connection.execute(trigger)
                    else:
                        connection.execute("CREATE VIEW unexpected_inventory_view AS SELECT 1 AS value")
                    connection.commit()
                finally:
                    connection.close()
                before = _file_sha256(source)
                with patch(
                    "friday_h01.import_h0.structural_fingerprint",
                    side_effect=AssertionError("fingerprint must not run before replay"),
                ):
                    with self.assertRaises(LegacyImportError):
                        inventory_h0_generations(source)
                self.assertEqual(before, _file_sha256(source))


if __name__ == "__main__":
    unittest.main()
