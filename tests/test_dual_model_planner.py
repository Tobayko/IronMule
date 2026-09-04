"""Offline contract tests for the cycle-15 two-model planner study.

These tests never load a model.  Mocks are used only to exercise rejected
preconditions, child-process failures, and partial-result handling; no mocked
value is treated as hardware evidence.
"""

from __future__ import annotations

import hashlib
import http.client
import inspect
import importlib.util
import io
import json
import base64
import os
import stat
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT_ROOT / "experiments" / "dual_model_planner"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


measure = load("dual_model_planner_measure", EXPERIMENT / "measure_dual_model_planner.py")
worker = load("dual_model_planner_worker", EXPERIMENT / "worker.py")


EXPECTED_RESPONSE = '{"candidate_id":"persistent_service_qualification"}'
CODEBLOCK_RESPONSE = f"```json\n{EXPECTED_RESPONSE}\n```"
PROMPT_HASH = "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
PREREGISTRATION_HASH = "246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe"
EXPECTED_MODEL_SPECS = {
    "1b": {
        "model_id": "mlx-community/gemma-3-1b-it-4bit",
        "revision": "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
    },
    "4b": {
        "model_id": "mlx-community/gemma-3-4b-it-4bit",
        "revision": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
    },
}
RENDERED_PROMPT_BYTES = worker.PLANNER_PROMPT.encode("utf-8")
RENDERED_PROMPT_B64 = base64.b64encode(RENDERED_PROMPT_BYTES).decode("ascii")
RENDERED_PROMPT_HASH = hashlib.sha256(RENDERED_PROMPT_BYTES).hexdigest()
STAT_MANIFEST = {
    "config.json": {"device": 1, "inode": 10, "size": 2, "mtime_ns": 100},
    "tokenizer_config.json": {"device": 1, "inode": 11, "size": 2, "mtime_ns": 101},
    "model.safetensors": {"device": 1, "inode": 12, "size": 7, "mtime_ns": 102},
}


def snapshot_identity(model_key: str) -> dict[str, object]:
    spec = measure.MODEL_SPECS[model_key]
    return {
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "model_snapshot_weight_files": ["model.safetensors"],
        "model_snapshot_weight_bytes": 123,
        "model_source": "validated_project_local_snapshot",
        "snapshot_path": f"/project-local/{model_key}/snapshot",
        "snapshot_files_sha256": {"model.safetensors": "a" * 64},
        "snapshot_sha256": "b" * 64,
        "execution_stat_manifest": json.loads(json.dumps(STAT_MANIFEST)),
        "weight_sha256": {"model.safetensors": "a" * 64},
    }


def raw_event(
    model_key: str,
    pid: int,
    *,
    text: str = EXPECTED_RESPONSE,
    token: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    if token is None:
        token = 7 if model_key == "1b" else 8
    prompt_ids = [101, 102, 103]
    tokens = [token, 1]
    spec = measure.MODEL_SPECS[model_key]
    return {
        "device": "Device(gpu, 0)",
        "event": "complete",
        "finish_reason": finish_reason,
        "load_count": 1,
        "max_output_tokens": measure.MAX_OUTPUT_TOKENS,
        "model_key": model_key,
        "model_id": spec["model_id"],
        "model_load_ns": 100,
        "model_work_ns": 2_000_000,
        "mlx_peak_bytes": 2_000_000,
        "output_tokens": len(tokens),
        "pid": pid,
        "prefill_step_size": worker.PREFILL_STEP_SIZE,
        "prompt_sha256": worker.PROMPT_SHA256,
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": len(prompt_ids),
        "rendered_prompt_b64": RENDERED_PROMPT_B64,
        "rendered_prompt_sha256": RENDERED_PROMPT_HASH,
        "rss_peak_bytes": 3_000_000,
        "sampler_temperature": 0.0,
        "snapshot_integrity": {
            "bound_snapshot_sha256": snapshot_identity(model_key)["snapshot_sha256"],
            "bound_weight_sha256": snapshot_identity(model_key)["weight_sha256"],
            "before_load_stat_manifest": json.loads(json.dumps(STAT_MANIFEST)),
            "after_load_stat_manifest": json.loads(json.dumps(STAT_MANIFEST)),
        },
        "snapshot_path": snapshot_identity(model_key)["snapshot_path"],
        "snapshot_sha256": snapshot_identity(model_key)["snapshot_sha256"],
        "snapshot_revision": spec["revision"],
        "text": text,
        "token_rate": 1_000.0,
        "tokens": tokens,
        "ttft_ns": 1_000_000,
        "weight_sha256": snapshot_identity(model_key)["weight_sha256"],
        "worker_watchdog_seconds": measure.WORKER_WATCHDOG_SECONDS,
    }


def validated_run(
    model_key: str,
    pid: int,
    *,
    pair_id: int = 1,
    position: int = 1,
    text: str = EXPECTED_RESPONSE,
    token: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    value = measure._validate_event(
        raw_event(
            model_key,
            pid,
            text=text,
            token=token,
            finish_reason=finish_reason,
        ),
        pid,
        model_key,
        snapshot_identity(model_key),
    )
    value.update(
        {
            "pair_id": pair_id,
            "power_source": "ac_power",
            "process_wall_ns": 5_000_000,
            "schedule_position": position,
            "swap_before_bytes": 0,
            "swap_after_bytes": 0,
            "swap_delta_bytes": 0,
        }
    )
    return value


def six_runs(model_key: str, *, text: str = EXPECTED_RESPONSE) -> list[dict[str, object]]:
    return [
        validated_run(
            model_key,
            1000 + index,
            pair_id=index + 1,
            position=index + 1,
            text=text,
        )
        for index in range(6)
    ]


def twelve_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    position = 0
    for pair_id, order in enumerate(measure.PAIR_SCHEDULE, start=1):
        for model_key in order:
            position += 1
            runs.append(
                validated_run(
                    model_key,
                    2000 + position,
                    pair_id=pair_id,
                    position=position,
                )
            )
    return runs


class FrozenProtocolTests(unittest.TestCase):
    def test_preregistration_hash_and_fixed_ids_are_closed(self) -> None:
        digest = hashlib.sha256((EXPERIMENT / "PREREGISTRATION.md").read_bytes()).hexdigest()
        self.assertEqual(digest, PREREGISTRATION_HASH)
        self.assertEqual(digest, measure.FROZEN_PREREGISTRATION_SHA256)
        self.assertEqual(measure.STUDY_ID, "dual-model-evidence-planner-20260824-01")
        self.assertEqual(measure.RUN_ID, "dual-model-evidence-planner-validation-20260824-01")
        self.assertEqual(measure.MODEL_SPECS, EXPECTED_MODEL_SPECS)
        self.assertEqual(measure.MODEL_SPECS, worker.MODEL_SPECS)
        self.assertEqual(worker.PROMPT_SHA256, PROMPT_HASH)
        self.assertEqual(
            worker.ALLOWED_CANDIDATES,
            (
                "persistent_service_qualification",
                "batched_readback",
                "host_readback_upper_bound",
                "kv_cache_preallocation_ab",
            ),
        )
        self.assertEqual(worker.MAX_OUTPUT_TOKENS, 32)
        self.assertEqual(worker.PREFILL_STEP_SIZE, 256)
        self.assertEqual(measure.WORKER_TIMEOUT_SECONDS, 90.0)
        self.assertEqual(measure.MAX_EVENT_BYTES, 1_000_000)
        self.assertEqual(measure.MAX_MEMORY_BYTES, 5 * 1024**3)
        self.assertEqual(measure.PACING_TARGET, 0.10)
        self.assertEqual(measure.BOOTSTRAP_SEED, 20260824)
        self.assertEqual(measure.BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(measure.PAIR_COUNT, 6)
        self.assertEqual(measure.RUN_COUNT, 12)
        self.assertEqual(measure.POLICY.gpu_work_limit_s, 120.0)
        self.assertEqual(measure.POLICY.continuous_gpu_limit_s, 6.0)
        self.assertEqual(measure.POLICY.required_break_s, 4.0)
        self.assertEqual(measure.POLICY.duty_window_s, 60.0)
        self.assertEqual(measure.POLICY.duty_cycle_limit, 0.15)
        self.assertEqual(measure.POLICY.wall_limit_s, 1_200.0)
        self.assertEqual(measure.POLICY.candidate_cooldown_s, 60.0)

    def test_preregistration_has_no_trailing_whitespace(self) -> None:
        text = (EXPERIMENT / "PREREGISTRATION.md").read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            self.assertEqual(
                line,
                line.rstrip(" \t"),
                f"PREREGISTRATION.md:{line_number} has trailing whitespace",
            )

    def test_preregistration_separates_parent_content_hashes_from_timed_child_work(self) -> None:
        text = (EXPERIMENT / "PREREGISTRATION.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        self.assertIn("parent hashes the two execution manifests once again", normalized)
        self.assertIn("outside the measured child process wall interval", normalized)
        self.assertIn("resolved-file stat manifest", normalized)
        self.assertIn("device, inode, size and `mtime_ns`", normalized)
        self.assertIn("stat checks are recorded but are not charged as GPU work", normalized)
        self.assertNotIn("child content-manifest hashing is performed", normalized)

    def test_balanced_schedule_has_three_orders_and_twelve_positions(self) -> None:
        self.assertEqual(len(measure.PAIR_SCHEDULE), 6)
        self.assertEqual(measure.PAIR_SCHEDULE.count(("1b", "4b")), 3)
        self.assertEqual(measure.PAIR_SCHEDULE.count(("4b", "1b")), 3)
        flat = [model for pair in measure.PAIR_SCHEDULE for model in pair]
        self.assertEqual(flat.count("1b"), 6)
        self.assertEqual(flat.count("4b"), 6)
        runs = twelve_runs()
        self.assertEqual(len(runs), 12)
        self.assertEqual([run["schedule_position"] for run in runs], list(range(1, 13)))
        self.assertEqual(len({run["pid"] for run in runs}), 12)
        self.assertEqual({run["load_count"] for run in runs}, {1})
        for pair_id in range(1, 7):
            self.assertEqual(
                [run["model_key"] for run in runs if run["pair_id"] == pair_id],
                list(measure.PAIR_SCHEDULE[pair_id - 1]),
            )

    def test_normal_invocation_never_reaches_hardware(self) -> None:
        with mock.patch.object(measure, "execute") as execute:
            self.assertEqual(measure.main([]), 78)
            execute.assert_not_called()
        self.assertEqual(worker.main(["--self-check"]), 0)

    def test_main_handles_a_minimal_partial_report_defensively(self) -> None:
        minimal = {
            "decision": "resource_or_budget_failed",
            "formal_claim": False,
            "run_id": measure.RUN_ID,
            "runs": [{"validated_event": True}],
            "study_id": measure.STUDY_ID,
        }
        with (
            mock.patch.object(measure, "execute", return_value=minimal) as execute,
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = measure.main(["--execute", "--run-id", measure.RUN_ID])
        execute.assert_called_once_with(measure.RUN_ID)
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["decision"], "resource_or_budget_failed")
        self.assertEqual(report["run_id"], measure.RUN_ID)
        self.assertEqual(report["runs_completed"], 1)
        self.assertIs(report["formal_claim"], False)


class WorkerContractTests(unittest.TestCase):
    def test_only_the_canonical_answer_is_accepted(self) -> None:
        self.assertEqual(worker.parse_choice(EXPECTED_RESPONSE), worker.EXPECTED_CANDIDATE)
        invalid = (
            "",
            " "+EXPECTED_RESPONSE,
            EXPECTED_RESPONSE+" ",
            "\n"+EXPECTED_RESPONSE,
            EXPECTED_RESPONSE+"\n",
            "\ufeff"+EXPECTED_RESPONSE,
            '{"candidate_id": "persistent_service_qualification"}',
            '{\n"candidate_id":"persistent_service_qualification"\n}',
            CODEBLOCK_RESPONSE,
            "preface "+EXPECTED_RESPONSE,
            EXPECTED_RESPONSE+" trailing",
            '{"candidate_id":"persistent_service_qualification","extra":1}',
            '{"candidate_id":"persistent_service_qualification","candidate_id":"batched_readback"}',
            '{"candidate_id":"unknown"}',
            "NaN",
            "null",
            '[]',
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(worker.WorkerError):
                    worker.parse_choice(value)

    def test_structural_alternatives_are_rejected_by_the_exact_byte_contract(self) -> None:
        for candidate in worker.ALLOWED_CANDIDATES:
            value = json.dumps({"candidate_id": candidate}, separators=(",", ":"))
            if candidate == worker.EXPECTED_CANDIDATE:
                self.assertEqual(worker.parse_choice(value), candidate)
            else:
                with self.assertRaises(worker.WorkerError):
                    worker.parse_choice(value)

    def test_prompt_ids_tokenize_the_exact_rendered_prompt_bytes(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls: list[tuple[object, bool, bool]] = []
                self.encoded: tuple[str, bool] | None = None

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.calls.append((messages, tokenize, add_generation_prompt))
                if tokenize:
                    raise AssertionError("prompt IDs must come from the rendered bytes")
                return "rendered prompt"

            def encode(self, text, *, add_special_tokens):
                self.encoded = (text, add_special_tokens)
                return [11, 12, 13]

        tokenizer = Tokenizer()
        prompt_ids, rendered = worker._prompt_ids(tokenizer)
        self.assertEqual(prompt_ids, [11, 12, 13])
        self.assertEqual(rendered, b"rendered prompt")
        self.assertEqual(len(tokenizer.calls), 1)
        messages, tokenize, add_generation_prompt = tokenizer.calls[0]
        self.assertFalse(tokenize)
        self.assertTrue(add_generation_prompt)
        self.assertEqual(messages, [{"role": "user", "content": worker.PLANNER_PROMPT}])
        self.assertEqual(tokenizer.encoded, ("rendered prompt", False))

    def test_prompt_tokenizer_failures_are_rejected_without_model_work(self) -> None:
        class BadTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return [True]

        with self.assertRaises(worker.WorkerError):
            worker._prompt_ids(BadTokenizer())

    def test_worker_protocol_requires_registered_parent_and_model_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(worker.WorkerError):
                worker._run_worker("1b")

    def test_worker_binds_resolver_path_and_parent_hashes_without_child_content_hashing(self) -> None:
        source = inspect.getsource(worker)
        run_source = inspect.getsource(worker._run_worker)
        self.assertIn("resolve_local_model_snapshot", run_source)
        self.assertIn("snapshot.revision", run_source)
        self.assertIn("_snapshot_stat_manifest(snapshot)", run_source)
        self.assertNotIn("_snapshot_identity(snapshot)", run_source)
        self.assertNotIn("_sha256_file(", run_source)
        self.assertIn("load(str(snapshot.path))", run_source)
        self.assertIn("snapshot_sha256", source)
        self.assertIn("weight_sha256", source)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
            self.assertIn(field, source)

    def test_child_snapshot_stat_manifest_is_cheap_and_covers_the_loaded_tree(self) -> None:
        class Snapshot:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.weight_files = ("model.safetensors",)

        with tempfile.TemporaryDirectory() as directory:
            revision = "a" * 40
            repository = Path(directory).resolve() / "models--fixture--model"
            root = repository / "snapshots" / revision
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            snapshot = Snapshot(root)
            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("child must not read model bytes for identity"),
            ):
                before_path, before = worker._snapshot_stat_manifest(snapshot)
            self.assertEqual(before_path, str(root))
            self.assertEqual(set(before), {"config.json", "tokenizer_config.json", "model.safetensors"})
            for entry in before.values():
                self.assertEqual(set(entry), {"dev", "inode", "path", "size", "mtime_ns"})
                self.assertTrue(
                    all(type(entry[field]) is int for field in ("dev", "inode", "size", "mtime_ns"))
                )
                self.assertIsInstance(entry["path"], str)
            stat_path = root / "model.safetensors"
            stat_path.write_bytes(b"changed")
            stat = stat_path.stat()
            os.utime(stat_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            after_path, after = worker._snapshot_stat_manifest(snapshot)
            self.assertEqual(after_path, before_path)
            self.assertNotEqual(before, after)
            external = Path(directory).resolve() / "outside.safetensors"
            external.write_bytes(b"outside")
            stat_path.unlink()
            stat_path.symlink_to(external)
            with self.assertRaises(worker.WorkerError):
                worker._snapshot_stat_manifest(snapshot)
        with mock.patch.dict(
            os.environ,
            {
                "FRIDAY_DUAL_PARENT_PID": str(os.getpid()),
                "FRIDAY_DUAL_RUN_ID": worker.RUN_ID,
                "FRIDAY_DUAL_MODEL_KEY": "4b",
            },
        ):
            with self.assertRaises(worker.WorkerError):
                worker._run_worker("1b")


class GateAndDecisionTests(unittest.TestCase):
    def test_identity_and_determinism_require_six_fresh_single_loads(self) -> None:
        runs = six_runs("1b")
        self.assertTrue(measure._identity_gate("1b", runs))
        self.assertTrue(measure._determinism_gate(runs))
        for field, value in (
            ("tokens", [99, 1]),
            ("text", CODEBLOCK_RESPONSE),
            ("finish_reason", "length"),
            ("prompt_token_ids", [9, 9]),
            ("pid", runs[0]["pid"]),
            ("load_count", 2),
            ("device", "Device(cpu, 0)"),
            ("model_id", measure.MODEL_SPECS["4b"]["model_id"]),
        ):
            changed = [dict(run) for run in runs]
            changed[1][field] = value
            if field in {"tokens", "text", "finish_reason", "prompt_token_ids"}:
                self.assertFalse(measure._determinism_gate(changed), field)
            else:
                self.assertTrue(measure._determinism_gate(changed), field)
            if field in {"pid", "load_count", "device", "model_id"}:
                self.assertFalse(measure._identity_gate("1b", changed), field)
            else:
                self.assertTrue(measure._identity_gate("1b", changed), field)

    def test_different_models_may_have_different_token_ids(self) -> None:
        runs = twelve_runs()
        one_b = [run for run in runs if run["model_key"] == "1b"]
        four_b = [run for run in runs if run["model_key"] == "4b"]
        self.assertNotEqual(one_b[0]["tokens"], four_b[0]["tokens"])
        self.assertTrue(measure._determinism_gate(one_b))
        self.assertTrue(measure._determinism_gate(four_b))
        self.assertTrue(measure._prompt_identity_gate(runs))
        changed = [dict(run) for run in runs]
        changed[0]["rendered_prompt_b64"] = base64.b64encode(b"different prompt").decode("ascii")
        self.assertFalse(measure._prompt_identity_gate(changed))

    def test_cross_model_decoded_bytes_are_reported_without_quality_scoring(self) -> None:
        runs = twelve_runs()
        by_pair: dict[int, dict[str, dict[str, object]]] = {}
        for run in runs:
            by_pair.setdefault(run["pair_id"], {})[run["model_key"]] = run
        equal = [
            by_pair[pair_id]["1b"]["text"].encode("utf-8")
            == by_pair[pair_id]["4b"]["text"].encode("utf-8")
            for pair_id in range(1, 7)
        ]
        self.assertEqual(equal, [True] * 6)
        self.assertEqual(sum(equal), 6)
        self.assertNotEqual(by_pair[1]["1b"]["token_sha256"], by_pair[1]["4b"]["token_sha256"])
        summary = measure._cross_model_text(runs)
        self.assertEqual(summary["exact_text_equal_count"], 6)
        self.assertEqual(summary["exact_text_equal_total"], "6/6")
        self.assertTrue(summary["informational_only"])
        self.assertEqual([row["exact_text_equal"] for row in summary["pairs"]], [True] * 6)
        for row in summary["pairs"]:
            self.assertIsInstance(row["1b_token_sha256"], str)
            self.assertIsInstance(row["4b_token_sha256"], str)
            self.assertEqual(
                row["1b_text_utf8_sha256"],
                hashlib.sha256(EXPECTED_RESPONSE.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                row["4b_text_utf8_sha256"],
                hashlib.sha256(EXPECTED_RESPONSE.encode("utf-8")).hexdigest(),
            )
        self.assertNotIn("quality", summary)

        by_pair[3]["4b"]["text"] = CODEBLOCK_RESPONSE
        changed = [
            by_pair[pair_id]["1b"]["text"].encode("utf-8")
            == by_pair[pair_id]["4b"]["text"].encode("utf-8")
            for pair_id in range(1, 7)
        ]
        self.assertEqual(sum(changed), 5)
        changed_summary = measure._cross_model_text(runs)
        self.assertEqual(changed_summary["exact_text_equal_count"], 5)
        self.assertEqual(changed_summary["exact_text_equal_total"], "5/6")
        self.assertNotIn("quality", changed_summary)

    def test_contract_and_priority_are_separate_and_noncritical(self) -> None:
        valid = six_runs("1b")
        self.assertTrue(measure._contract_gate(valid))
        self.assertTrue(measure._priority_gate(valid))
        codeblock = six_runs("1b", text=CODEBLOCK_RESPONSE)
        self.assertFalse(measure._contract_gate(codeblock))
        self.assertFalse(measure._priority_gate(codeblock))
        alternative = six_runs("1b")
        for run in alternative:
            run["candidate_id"] = "batched_readback"
            run["parser"] = {
                "contract_ok": False,
                "contract_candidate_id": None,
                "contract_error": "planner answer is structurally valid but not byte-exact",
                "structural_ok": True,
                "structural_candidate_id": "batched_readback",
                "structural_error": None,
            }
        self.assertFalse(measure._contract_gate(alternative))
        self.assertFalse(measure._priority_gate(alternative))

    def test_within_model_token_mismatch_is_a_correctness_failure(self) -> None:
        changed = six_runs("1b")
        changed[1]["tokens"] = [99, 1]
        summary = measure._model_summary("1b", changed)
        self.assertFalse(summary["deterministic"])
        self.assertFalse(summary["correctness_pass"])
        self.assertEqual(
            measure.decision_for(
                one_b_pass=False,
                four_b_pass=True,
                pairwise=None,
                one_b_peak_rss=None,
                four_b_peak_rss=None,
                correctness_failure=True,
            ),
            "correctness_failed",
        )

    def test_resource_gate_rejects_partial_memory_or_swap_evidence(self) -> None:
        runs = twelve_runs()
        self.assertTrue(measure._resource_gate(runs, 0))
        self.assertFalse(measure._resource_gate(runs[:-1], 0))
        swapped = [dict(run) for run in runs]
        swapped[0]["swap_delta_bytes"] = 1
        self.assertFalse(measure._resource_gate(swapped, 0))
        oversized = [dict(run) for run in runs]
        oversized[0]["rss_peak_bytes"] = measure.MAX_MEMORY_BYTES + 1
        self.assertFalse(measure._resource_gate(oversized, 0))
        self.assertFalse(measure._resource_gate(runs, None))

    def test_all_decision_outputs_and_correctness_precedence_are_frozen(self) -> None:
        pairwise = {
            "complete": True,
            "ratios_1b_div_4b": {"process_wall": {"median": 1.04}},
        }
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=False,
                pairwise=None,
                one_b_peak_rss=None,
                four_b_peak_rss=None,
            ),
            "planner_1b_qualified_exact_case",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=False,
                four_b_pass=True,
                pairwise=None,
                one_b_peak_rss=None,
                four_b_peak_rss=None,
            ),
            "planner_4b_qualified_exact_case",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=False,
                four_b_pass=False,
                pairwise=None,
                one_b_peak_rss=None,
                four_b_peak_rss=None,
            ),
            "no_planner_qualified",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise=pairwise,
                one_b_peak_rss=75,
                four_b_peak_rss=100,
            ),
            "both_qualified_1b_preferred",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise={
                    "complete": True,
                    "ratios_1b_div_4b": {"process_wall": {"median": 1.05}},
                },
                one_b_peak_rss=75,
                four_b_peak_rss=100,
            ),
            "both_qualified_1b_preferred",
        )
        for ratio, rss in ((1.050001, 75), (1.05, 76)):
            with self.subTest(ratio=ratio, rss=rss):
                self.assertEqual(
                    measure.decision_for(
                        one_b_pass=True,
                        four_b_pass=True,
                        pairwise={
                            "complete": True,
                            "ratios_1b_div_4b": {"process_wall": {"median": ratio}},
                        },
                        one_b_peak_rss=rss,
                        four_b_peak_rss=100,
                    ),
                    "both_qualified_no_automatic_preference",
                )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise=pairwise,
                one_b_peak_rss=75,
                four_b_peak_rss=100,
                correctness_failure=True,
            ),
            "correctness_failed",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise=pairwise,
                one_b_peak_rss=75,
                four_b_peak_rss=100,
                terminal_failure=True,
            ),
            "resource_or_budget_failed",
        )
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise=pairwise,
                one_b_peak_rss=75,
                four_b_peak_rss=100,
                correctness_failure=True,
                terminal_failure=True,
            ),
            "resource_or_budget_failed",
        )


class EvidenceAndGatePreconditionTests(unittest.TestCase):
    def test_snapshot_identity_hashes_all_files_and_weights(self) -> None:
        class Snapshot:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.weight_files = ("model.safetensors",)

            def report_identity(self):
                return {"model_id": "fixture", "model_revision": "a" * 40}

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve() / "models--fixture--model"
            root = repository / "snapshots" / ("a" * 40)
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            result = measure._snapshot_identity(Snapshot(root))
            self.assertEqual(result["weight_sha256"]["model.safetensors"], hashlib.sha256(b"weights").hexdigest())
            self.assertIn("config.json", result["snapshot_files_sha256"])
            self.assertEqual(len(result["snapshot_sha256"]), 64)
            external = Path(directory).resolve() / "outside.safetensors"
            external.write_bytes(b"outside")
            (root / "model.safetensors").unlink()
            (root / "model.safetensors").symlink_to(external)
            with self.assertRaises(measure.StudyError):
                measure._snapshot_identity(Snapshot(root))

    def test_event_revision_and_prompt_fingerprints_are_closed(self) -> None:
        for field, value in (
            ("snapshot_revision", "wrong-revision"),
            ("snapshot_path", "/other/snapshot"),
            ("snapshot_sha256", "0" * 64),
            ("weight_sha256", {"model.safetensors": "0" * 64}),
            (
                "snapshot_integrity",
                {
                    "before_load_stat_manifest": {
                        "model.safetensors": {
                            "device": 9,
                            "inode": 9,
                            "size": 9,
                            "mtime_ns": 9,
                        }
                    },
                    "after_load_stat_manifest": json.loads(json.dumps(STAT_MANIFEST)),
                },
            ),
            ("prompt_sha256", "0" * 64),
            ("rendered_prompt_sha256", "0" * 64),
            ("rendered_prompt_b64", base64.b64encode(b"different prompt").decode("ascii")),
            ("worker_watchdog_seconds", 5.0),
        ):
            event = raw_event("1b", 4242)
            event[field] = value
            with self.subTest(field=field):
                with self.assertRaises(measure.WorkerError):
                    measure._validate_event(event, 4242, "1b", snapshot_identity("1b"))

    def test_dirty_tree_is_rejected_before_execution(self) -> None:
        def git_result(*args: str) -> str:
            if args[:2] == ("rev-parse", "HEAD"):
                return "a" * 40
            return " M user-file.py"

        with mock.patch.object(measure, "_git", side_effect=git_result):
            with self.assertRaises(measure.StudyError):
                measure._require_clean_worktree()

    def test_registered_snapshot_revision_mismatch_is_rejected_preflight(self) -> None:
        class Snapshot:
            path = PROJECT_ROOT
            revision = "wrong-revision"

        expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve()
        with (
            mock.patch.object(measure.sys, "executable", str(expected_python)),
            mock.patch.object(measure, "_sha256", return_value=measure.FROZEN_PREREGISTRATION_SHA256),
            mock.patch.object(measure, "_require_clean_worktree", return_value=("a" * 40, "")),
            mock.patch.object(measure, "_require_target_environment"),
            mock.patch.object(measure, "require_ac_power", return_value="ac_power"),
            mock.patch.object(measure, "resolve_local_model_snapshot", return_value=Snapshot()),
        ):
            with self.assertRaises(measure.StudyError):
                measure._preflight(measure.RUN_ID)

    def test_target_hardware_gate_fails_closed_without_model_work(self) -> None:
        with mock.patch.object(measure.platform, "machine", return_value="x86_64"):
            with self.assertRaises(measure.StudyError):
                measure._require_target_environment()

    def test_ac_power_is_required_by_preflight(self) -> None:
        expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve()
        with (
            mock.patch.object(measure.sys, "executable", str(expected_python)),
            mock.patch.object(measure, "_sha256", return_value=measure.FROZEN_PREREGISTRATION_SHA256),
            mock.patch.object(measure, "_require_clean_worktree", return_value=("a" * 40, "")),
            mock.patch.object(measure, "_require_target_environment"),
            mock.patch.object(measure, "require_ac_power", side_effect=RuntimeError("battery")),
        ):
            with self.assertRaises(RuntimeError):
                measure._preflight(measure.RUN_ID)

    def test_offline_environment_removes_python_injection(self) -> None:
        poisoned = {name: "unsafe" for name in measure.UNSAFE_PYTHON_ENVIRONMENT}
        with mock.patch.dict(os.environ, {**poisoned, "HF_HUB_OFFLINE": "0"}):
            env = measure._worker_environment("1b", snapshot_identity("1b"))
        for name in poisoned:
            self.assertNotIn(name, env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["FRIDAY_DUAL_MODEL_KEY"], "1b")
        self.assertEqual(env["FRIDAY_DUAL_SNAPSHOT_PATH"], snapshot_identity("1b")["snapshot_path"])
        self.assertEqual(env["FRIDAY_DUAL_SNAPSHOT_REVISION"], snapshot_identity("1b")["model_revision"])
        self.assertEqual(env["FRIDAY_DUAL_SNAPSHOT_SHA256"], snapshot_identity("1b")["snapshot_sha256"])
        self.assertEqual(
            json.loads(env["FRIDAY_DUAL_WEIGHT_SHA256"]),
            snapshot_identity("1b")["weight_sha256"],
        )
        incomplete = snapshot_identity("1b")
        incomplete.pop("snapshot_sha256")
        with self.assertRaises(measure.StudyError):
            measure._worker_environment("1b", incomplete)

    def test_private_marker_directory_and_exclusive_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = (root / "private").resolve()
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            measure._require_private_directory(private)
            os.chmod(private, 0o755)
            with self.assertRaises(measure.StudyError):
                measure._require_private_directory(private)
            os.chmod(private, 0o700)
            link = root / "link"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaises(measure.StudyError):
                measure._require_private_directory(link)
            marker = private / "attempt.json"
            measure._exclusive_json(marker, {"study_id": measure.STUDY_ID}, 0o600)
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                measure._exclusive_json(marker, {"study_id": measure.STUDY_ID}, 0o600)

    def test_existing_marker_or_result_blocks_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            marker = root / "attempt.json"
            result.write_text("{}", encoding="utf-8")
            marker.write_text("{}", encoding="utf-8")
            with mock.patch.object(measure, "RESULT_PATH", result), mock.patch.object(
                measure, "ATTEMPT_PATH", marker
            ):
                with self.assertRaises(measure.StudyError):
                    measure._preflight("wrong-run-id")
                with self.assertRaises(measure.StudyError):
                    measure._preflight(measure.RUN_ID)


class WorkerFailureAndBudgetTests(unittest.TestCase):
    class FakeProcess:
        pid = 777
        returncode = 0

        def __init__(self, payload: bytes = b"{}\n", *, timeout: bool = False) -> None:
            self.payload = payload
            self.timeout = timeout
            self.communicate_calls = 0
            self.wait_calls = 0
            self.stdout = io.BytesIO(payload)

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.timeout and self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.payload, b""

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.timeout and self.wait_calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def run_worker_with(self, process: FakeProcess):
        with (
            mock.patch.object(measure.subprocess, "Popen", return_value=process),
            mock.patch.object(measure, "_swap_used_bytes", return_value=0),
            mock.patch.object(measure, "require_ac_power", return_value="ac_power"),
        ):
            return measure._run_worker(
                pair_id=1,
                schedule_position=1,
                model_key="1b",
                snapshot_identity=snapshot_identity("1b"),
            )

    def test_timeout_terminates_owned_worker_without_retry(self) -> None:
        process = self.FakeProcess(timeout=True)
        with mock.patch.object(measure, "_terminate_process") as terminate:
            with self.assertRaises(measure.WorkerError):
                self.run_worker_with(process)
        terminate.assert_called_once_with(process)
        self.assertEqual(process.wait_calls, 1)
        self.assertEqual(process.communicate_calls, 0)

    def test_output_limit_and_malformed_event_are_rejected(self) -> None:
        with self.assertRaises(measure.WorkerError):
            self.run_worker_with(self.FakeProcess(b"x" * (measure.MAX_EVENT_BYTES + 1)))
        with self.assertRaises(measure.WorkerError):
            self.run_worker_with(self.FakeProcess(b"not-json\n"))
        with self.assertRaises(measure.WorkerError):
            measure._decode_event(b"{}\n{}\n")
        with self.assertRaises(measure.WorkerError):
            measure._decode_event(b"NaN\n")
        with self.assertRaises(measure.WorkerError):
            measure._decode_event(b'{"event":"error","error_type":"x","message":"bad"}\n')

    def test_validated_event_is_returned_when_a_post_event_check_aborts(self) -> None:
        process = self.FakeProcess(
            (
                json.dumps(raw_event("1b", self.FakeProcess.pid), separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
        with (
            mock.patch.object(measure.subprocess, "Popen", return_value=process),
            mock.patch.object(measure, "_swap_used_bytes", side_effect=(0, None)),
            mock.patch.object(measure, "require_ac_power", return_value="ac_power"),
            mock.patch.object(
                measure.time,
                "perf_counter_ns",
                side_effect=(10_000_000, 20_000_000),
            ),
        ):
            value = measure._run_worker(
                pair_id=1,
                schedule_position=1,
                model_key="1b",
                snapshot_identity=snapshot_identity("1b"),
            )
        self.assertEqual(value["candidate_id"], measure.EXPECTED_CANDIDATE)
        self.assertTrue(value["parser"]["contract_ok"])
        self.assertEqual(value["abort_reason"], "swap_usage_unavailable_after_worker")
        self.assertIsNone(value["swap_after_bytes"])
        self.assertIsNone(value["swap_delta_bytes"])

    def test_stdout_limit_is_enforced_without_unbounded_communicate_capture(self) -> None:
        class StreamLimitedProcess:
            pid = 778
            returncode = 0

            def __init__(self) -> None:
                self.stdout = tempfile.TemporaryFile(mode="w+b")
                self.stdout.write(b"x" * (measure.MAX_EVENT_BYTES + 1))
                self.stdout.seek(0)

            def communicate(self, timeout=None):
                raise AssertionError("unbounded communicate capture is forbidden")

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                return None

            def kill(self):
                return None

        process = StreamLimitedProcess()
        with (
            mock.patch.object(measure.subprocess, "Popen", return_value=process),
            mock.patch.object(measure, "_swap_used_bytes", return_value=0),
            mock.patch.object(measure, "require_ac_power", return_value="ac_power"),
        ):
            try:
                with self.assertRaises(measure.WorkerError):
                    measure._run_worker(
                        pair_id=1,
                        schedule_position=1,
                        model_key="1b",
                        snapshot_identity=snapshot_identity("1b"),
                    )
            finally:
                process.stdout.close()

    def test_partial_result_is_preserved_after_second_worker_failure(self) -> None:
        first = validated_run("1b", 1001)
        captured: dict[str, object] = {}

        class FakeGuard:
            def __init__(self, _policy):
                self.calls: list[str] = []

            def before_candidate(self):
                self.calls.append("before")

            def record_gpu(self, _seconds):
                self.calls.append("record")

            def required_break(self):
                self.calls.append("break")

            def finish_candidate(self):
                self.calls.append("finish")

            def summary(self):
                return {
                    "duty_cycle_limit": 0.15,
                    "gpu_work_seconds": 1.0,
                    "max_continuous_gpu_seconds": 1.0,
                    "wall_seconds": 2.0,
                }

        class PostSnapshot:
            def __init__(self, model_key: str) -> None:
                self.model_key = model_key
                self.revision = EXPECTED_MODEL_SPECS[model_key]["revision"]

        with tempfile.TemporaryDirectory() as directory:
            attempt_dir = Path(directory) / "attempt-dir"
            attempt_dir.mkdir(mode=0o700)
            result_path = Path(directory) / "result.json"
            with (
                mock.patch.object(measure, "ATTEMPT_DIR", attempt_dir),
                mock.patch.object(measure, "ATTEMPT_PATH", attempt_dir / "attempt.json"),
                mock.patch.object(measure, "RESULT_PATH", result_path),
                mock.patch.object(
                    measure,
                    "_preflight",
                return_value=("a" * 40, "", "ac_power", {"1b": snapshot_identity("1b"), "4b": snapshot_identity("4b")}, 0),
                ),
                mock.patch.object(measure, "_provenance", return_value={
                    "code_sha256": "c" * 64,
                    "environment_sha256": "e" * 64,
                    "environment": {"packages": {}},
                    "git_dirty_state": False,
                    "git_revision": "a" * 40,
                    "hardware": {},
                    "prompt_sha256": PROMPT_HASH,
                }),
                mock.patch.object(measure, "_exclusive_json"),
                mock.patch.object(measure, "_require_private_directory"),
                mock.patch.object(
                    measure,
                    "resolve_local_model_snapshot",
                    side_effect=lambda model_id: PostSnapshot(
                        "1b" if model_id == EXPECTED_MODEL_SPECS["1b"]["model_id"] else "4b"
                    ),
                ),
                mock.patch.object(
                    measure,
                    "_snapshot_identity",
                    side_effect=lambda snapshot: snapshot_identity(snapshot.model_key),
                ),
                mock.patch.object(measure, "_atomic_result", side_effect=lambda value: captured.update(value)),
                mock.patch.object(measure, "_run_worker", side_effect=[first, measure.WorkerError("timeout")]),
                mock.patch.object(measure, "_swap_used_bytes", return_value=0),
                mock.patch.object(measure, "BudgetGuard", FakeGuard),
            ):
                report = measure.execute(measure.RUN_ID)
        self.assertEqual(report["runs"], [first])
        self.assertEqual(report["error"]["type"], "WorkerError")
        self.assertEqual(captured["runs"], [first])
        self.assertEqual(report["decision"], "resource_or_budget_failed")

    def test_post_run_finalization_failure_still_persists_partial_evidence(self) -> None:
        first = validated_run("1b", 1101)
        captured: dict[str, object] = {}

        class FakeGuard:
            def __init__(self, _policy):
                pass

            def before_candidate(self):
                pass

            def record_gpu(self, _seconds):
                pass

            def required_break(self):
                pass

            def finish_candidate(self):
                pass

            def summary(self):
                return {
                    "duty_cycle_limit": 0.15,
                    "gpu_work_seconds": 1.0,
                    "max_continuous_gpu_seconds": 1.0,
                    "wall_seconds": 2.0,
                }

        class PostSnapshot:
            def __init__(self, model_key: str) -> None:
                self.model_key = model_key
                self.revision = EXPECTED_MODEL_SPECS[model_key]["revision"]

        with tempfile.TemporaryDirectory() as directory:
            attempt_dir = Path(directory) / "attempt-dir"
            attempt_dir.mkdir(mode=0o700)
            result_path = Path(directory) / "result.json"
            with (
                mock.patch.object(measure, "ATTEMPT_DIR", attempt_dir),
                mock.patch.object(measure, "ATTEMPT_PATH", attempt_dir / "attempt.json"),
                mock.patch.object(measure, "RESULT_PATH", result_path),
                mock.patch.object(
                    measure,
                    "_preflight",
                    return_value=(
                        "a" * 40,
                        "",
                        "ac_power",
                        {"1b": snapshot_identity("1b"), "4b": snapshot_identity("4b")},
                        0,
                    ),
                ),
                mock.patch.object(
                    measure,
                    "_provenance",
                    return_value={
                        "code_sha256": "c" * 64,
                        "environment_sha256": "e" * 64,
                        "environment": {"packages": {}},
                        "git_dirty_state": False,
                        "git_revision": "a" * 40,
                        "hardware": {},
                        "prompt_sha256": PROMPT_HASH,
                    },
                ),
                mock.patch.object(measure, "_exclusive_json"),
                mock.patch.object(measure, "_require_private_directory"),
                mock.patch.object(
                    measure,
                    "resolve_local_model_snapshot",
                    side_effect=lambda model_id: PostSnapshot(
                        "1b" if model_id == EXPECTED_MODEL_SPECS["1b"]["model_id"] else "4b"
                    ),
                ),
                mock.patch.object(
                    measure,
                    "_snapshot_identity",
                    side_effect=lambda snapshot: snapshot_identity(snapshot.model_key),
                ),
                mock.patch.object(measure, "_atomic_result", side_effect=lambda value: captured.update(value)),
                mock.patch.object(measure, "_run_worker", side_effect=[first, measure.WorkerError("timeout")]),
                mock.patch.object(measure, "_swap_used_bytes", return_value=0),
                mock.patch.object(measure, "BudgetGuard", FakeGuard),
                mock.patch.object(measure, "_pairwise", side_effect=measure.StudyError("finalization failed")),
            ):
                report = measure.execute(measure.RUN_ID)
        self.assertEqual(report["runs"], [first])
        self.assertEqual(captured["runs"], [first])
        self.assertEqual(report["decision"], "resource_or_budget_failed")
        self.assertIsNotNone(report["error"])
        self.assertTrue(captured["partial_result"])
        self.assertTrue(
            any("pairwise aggregation failed" in item for item in captured["finalization_errors"])
        )

    def test_noncritical_contract_failure_keeps_all_twelve_scheduled_runs(self) -> None:
        runs: list[dict[str, object]] = []
        position = 0
        for pair_id, order in enumerate(measure.PAIR_SCHEDULE, start=1):
            for model_key in order:
                position += 1
                runs.append(
                    validated_run(
                        model_key,
                        3000 + position,
                        pair_id=pair_id,
                        position=position,
                        text=CODEBLOCK_RESPONSE if model_key == "1b" else EXPECTED_RESPONSE,
                    )
                )
        captured: dict[str, object] = {}

        class FakeGuard:
            def __init__(self, _policy):
                self.calls: list[str] = []

            def before_candidate(self):
                self.calls.append("before")

            def record_gpu(self, _seconds):
                self.calls.append("record")

            def required_break(self):
                self.calls.append("break")

            def finish_candidate(self):
                self.calls.append("finish")

            def summary(self):
                return {
                    "duty_cycle_limit": 0.15,
                    "gpu_work_seconds": 1.0,
                    "max_continuous_gpu_seconds": 1.0,
                    "wall_seconds": 2.0,
                }

        class PostSnapshot:
            def __init__(self, model_key: str) -> None:
                self.model_key = model_key
                self.revision = EXPECTED_MODEL_SPECS[model_key]["revision"]

        with tempfile.TemporaryDirectory() as directory:
            attempt_dir = Path(directory) / "attempt-dir"
            attempt_dir.mkdir(mode=0o700)
            result_path = Path(directory) / "result.json"
            with (
                mock.patch.object(measure, "ATTEMPT_DIR", attempt_dir),
                mock.patch.object(measure, "ATTEMPT_PATH", attempt_dir / "attempt.json"),
                mock.patch.object(measure, "RESULT_PATH", result_path),
                mock.patch.object(
                    measure,
                    "_preflight",
                    return_value=(
                        "a" * 40,
                        "",
                        "ac_power",
                        {"1b": snapshot_identity("1b"), "4b": snapshot_identity("4b")},
                        0,
                    ),
                ),
                mock.patch.object(
                    measure,
                    "_provenance",
                    return_value={
                        "code_sha256": "c" * 64,
                        "environment_sha256": "e" * 64,
                        "environment": {"packages": {}},
                        "git_dirty_state": False,
                        "git_revision": "a" * 40,
                        "hardware": {},
                        "prompt_sha256": PROMPT_HASH,
                    },
                ),
                mock.patch.object(measure, "_exclusive_json"),
                mock.patch.object(measure, "_require_private_directory"),
                mock.patch.object(
                    measure,
                    "resolve_local_model_snapshot",
                    side_effect=lambda model_id: PostSnapshot(
                        "1b" if model_id == EXPECTED_MODEL_SPECS["1b"]["model_id"] else "4b"
                    ),
                ),
                mock.patch.object(
                    measure,
                    "_snapshot_identity",
                    side_effect=lambda snapshot: snapshot_identity(snapshot.model_key),
                ),
                mock.patch.object(measure, "_atomic_result", side_effect=lambda value: captured.update(value)),
                mock.patch.object(measure, "_run_worker", side_effect=runs) as run_worker,
                mock.patch.object(measure, "_swap_used_bytes", return_value=0),
                mock.patch.object(measure, "BudgetGuard", FakeGuard),
            ):
                report = measure.execute(measure.RUN_ID)
        self.assertIsNone(report["error"])
        self.assertEqual(run_worker.call_count, 12)
        self.assertEqual(report["metrics"]["runs_completed"], 12)
        self.assertEqual(len(report["runs"]), 12)
        self.assertEqual(report["decision"], "planner_4b_qualified_exact_case")
        self.assertEqual(report["metrics"]["model_1b"]["contract_successes"], 0)
        self.assertEqual(report["metrics"]["model_4b"]["contract_successes"], 6)
        self.assertEqual(captured["metrics"]["runs_completed"], 12)

    def test_budget_charge_happens_before_required_pause(self) -> None:
        calls: list[str] = []

        class Guard:
            def record_gpu(self, seconds):
                calls.append(f"record:{seconds}")

            def required_break(self):
                calls.append("break")

        guard = Guard()
        charged_seconds = measure._record_gpu(guard, 100_000_000)
        measure._required_breaks(guard, charged_seconds)
        self.assertTrue(calls)
        self.assertTrue(calls[0].startswith("record:"))
        self.assertIn("break", calls[1:])

    def test_resource_failure_after_charge_skips_pause_and_persists_partial_result(self) -> None:
        guards: list[object] = []

        class Guard:
            def __init__(self, _policy):
                self.calls: list[str] = []
                guards.append(self)

            def before_candidate(self):
                self.calls.append("before")

            def record_gpu(self, _seconds):
                self.calls.append("record")

            def required_break(self):
                self.calls.append("break")

            def finish_candidate(self):
                self.calls.append("finish")

            def summary(self):
                return {
                    "duty_cycle_limit": 0.15,
                    "gpu_work_seconds": 1.0,
                    "max_continuous_gpu_seconds": 1.0,
                    "wall_seconds": 2.0,
                }

        class PostSnapshot:
            def __init__(self, model_key: str) -> None:
                self.model_key = model_key
                self.revision = EXPECTED_MODEL_SPECS[model_key]["revision"]

        for field, value in (
            ("abort_reason", "swap_usage_unavailable_after_worker"),
            ("swap_delta_bytes", 1),
            ("rss_peak_bytes", measure.MAX_MEMORY_BYTES + 1),
            ("mlx_peak_bytes", measure.MAX_MEMORY_BYTES + 1),
        ):
            with self.subTest(resource=field):
                first = validated_run("1b", 5001)
                first[field] = value
                captured: dict[str, object] = {}
                with tempfile.TemporaryDirectory() as directory:
                    attempt_dir = Path(directory) / "attempt-dir"
                    attempt_dir.mkdir(mode=0o700)
                    result_path = Path(directory) / "result.json"
                    with (
                        mock.patch.object(measure, "ATTEMPT_DIR", attempt_dir),
                        mock.patch.object(measure, "ATTEMPT_PATH", attempt_dir / "attempt.json"),
                        mock.patch.object(measure, "RESULT_PATH", result_path),
                        mock.patch.object(
                            measure,
                            "_preflight",
                            return_value=(
                                "a" * 40,
                                "",
                                "ac_power",
                                {"1b": snapshot_identity("1b"), "4b": snapshot_identity("4b")},
                                0,
                            ),
                        ),
                        mock.patch.object(
                            measure,
                            "_provenance",
                            return_value={
                                "code_sha256": "c" * 64,
                                "environment_sha256": "e" * 64,
                                "environment": {"packages": {}},
                                "git_dirty_state": False,
                                "git_revision": "a" * 40,
                                "hardware": {},
                                "prompt_sha256": PROMPT_HASH,
                            },
                        ),
                        mock.patch.object(measure, "_exclusive_json"),
                        mock.patch.object(measure, "_require_private_directory"),
                        mock.patch.object(
                            measure,
                            "resolve_local_model_snapshot",
                            side_effect=lambda model_id: PostSnapshot(
                                "1b"
                                if model_id == EXPECTED_MODEL_SPECS["1b"]["model_id"]
                                else "4b"
                            ),
                        ),
                        mock.patch.object(
                            measure,
                            "_snapshot_identity",
                            side_effect=lambda snapshot: snapshot_identity(snapshot.model_key),
                        ),
                        mock.patch.object(
                            measure,
                            "_atomic_result",
                            side_effect=lambda result: captured.update(result),
                        ),
                        mock.patch.object(measure, "_run_worker", return_value=first),
                        mock.patch.object(measure, "_swap_used_bytes", return_value=0),
                        mock.patch.object(measure, "BudgetGuard", Guard),
                    ):
                        report = measure.execute(measure.RUN_ID)
                guard = guards[-1]
                self.assertEqual(report["runs"], [first])
                self.assertEqual(report["decision"], "resource_or_budget_failed")
                self.assertTrue(report["partial_result"])
                self.assertEqual(captured["runs"], [first])
                self.assertTrue(captured["partial_result"])
                self.assertEqual(guard.calls, ["before", "record", "finish"])


class StatisticsTests(unittest.TestCase):
    def test_median_mad_and_pairwise_metrics_have_no_outlier_removal(self) -> None:
        values = [0.31, 0.33, 0.35, 0.34, 0.32, 0.36]
        self.assertEqual(measure._median(values), 0.335)
        self.assertAlmostEqual(measure._mad(values), 0.015)
        runs = twelve_runs()
        for run in runs:
            if run["model_key"] == "1b":
                run["ttft_ns"] = 2_000_000
                run["model_work_ns"] = 4_000_000
                run["process_wall_ns"] = 10_000_000
                run["token_rate"] = 20.0
            else:
                run["ttft_ns"] = 4_000_000
                run["model_work_ns"] = 8_000_000
                run["process_wall_ns"] = 20_000_000
                run["token_rate"] = 10.0
        summary = measure._model_summary("1b", [run for run in runs if run["model_key"] == "1b"])
        self.assertEqual(summary["metrics"]["ttft_seconds"]["median"], 0.002)
        self.assertEqual(summary["metrics"]["model_work_seconds"]["median"], 0.004)
        self.assertEqual(summary["metrics"]["process_wall_seconds"]["median"], 0.01)
        pairwise = measure._pairwise(runs)
        self.assertTrue(pairwise["complete"])
        self.assertEqual(pairwise["pair_ids"], list(range(1, 7)))
        self.assertEqual(pairwise["ratios_1b_div_4b"]["ttft"]["values"], [0.5] * 6)
        self.assertEqual(pairwise["ratios_1b_div_4b"]["model_work"]["values"], [0.5] * 6)
        self.assertEqual(pairwise["ratios_1b_div_4b"]["process_wall"]["values"], [0.5] * 6)
        self.assertEqual(pairwise["ratios_1b_div_4b"]["token_rate"]["values"], [2.0] * 6)

    def test_pairwise_requires_exactly_one_observation_per_model_per_pair(self) -> None:
        runs = twelve_runs()
        duplicate = dict(runs[-1])
        duplicate["model_key"] = "4b"
        runs.append(duplicate)
        result = measure._pairwise(runs)
        self.assertFalse(result["complete"])
        self.assertFalse(measure._cross_model_text(runs)["complete"])
        unexpected = twelve_runs()
        unexpected.append(dict(unexpected[0], pair_id=99))
        self.assertFalse(measure._pairwise(unexpected)["complete"])
        self.assertFalse(measure._cross_model_text(unexpected)["complete"])

    def test_incomplete_pairing_is_terminal_resource_failure(self) -> None:
        self.assertEqual(
            measure.decision_for(
                one_b_pass=True,
                four_b_pass=True,
                pairwise={"complete": False},
                one_b_peak_rss=70,
                four_b_peak_rss=100,
                terminal_failure=True,
            ),
            "resource_or_budget_failed",
        )

    def test_bootstrap_is_paired_seeded_and_deterministic(self) -> None:
        ratios = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
        first = measure._bootstrap_ci(ratios)
        second = measure._bootstrap_ci(ratios)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], measure.BOOTSTRAP_SEED)
        self.assertEqual(first["resamples"], measure.BOOTSTRAP_RESAMPLES)
        self.assertEqual(first["method"], "paired six-pair median-ratio bootstrap percentile")
        self.assertEqual(
            first["percentiles"],
            {"lower": 0.025, "upper": 0.975, "interpolation": "linear"},
        )
        self.assertAlmostEqual(first["lower"], 0.85)
        self.assertAlmostEqual(first["upper"], 1.25)
        self.assertLessEqual(first["lower"], first["upper"])
        self.assertIsNone(measure._bootstrap_ci(ratios[:-1]))
        self.assertIsNone(measure._bootstrap_ci([float("nan")] * 6))


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dashboard_path = EXPERIMENT / "dashboard.py"
        if not dashboard_path.is_file():
            raise AssertionError("dual-model dashboard.py is required before UI tests")
        cls.dashboard = load("dual_model_planner_dashboard", dashboard_path)

    def result(self) -> dict[str, object]:
        runs = twelve_runs()
        return {
            "decision": "both_qualified_no_automatic_preference",
            "formal_claim": False,
            "gates": {"all_runs_completed": True},
            "metrics": {"runs_completed": 12, "model_1b": {}, "model_4b": {}},
            "provenance": {"database_sha256": "d" * 64},
            "resources": {"swap_delta_bytes": 0},
            "run_id": measure.RUN_ID,
            "runs": runs,
            "study_id": measure.STUDY_ID,
        }

    def request(self, server, method: str, target: str, host: str = "127.0.0.1"):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        connection.request(method, target, headers={"Host": host})
        response = connection.getresponse()
        body = response.read()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def test_snapshot_is_read_only_hides_raw_text_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(self.result()), encoding="utf-8")
            before = hashlib.sha256(result_path.read_bytes()).hexdigest()
            value = self.dashboard.snapshot(result_path)
            after = hashlib.sha256(result_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(value["read_only"])
            self.assertIn("model_1b", value["metrics"])
            self.assertIn("model_4b", value["metrics"])
            self.assertIn("pairwise", value["metrics"])
            encoded = json.dumps(value, sort_keys=True)
            self.assertNotIn("secret raw answer", encoded)
            self.assertNotIn('"tokens"', encoded)
            self.assertNotIn('"text"', encoded)
            link = root / "link.json"
            link.symlink_to(result_path)
            with self.assertRaises(self.dashboard.DashboardError):
                self.dashboard.snapshot(link)

    def test_dashboard_requires_the_fixed_run_id_and_decision_allowlist(self) -> None:
        expected_decisions = {
            "planner_1b_qualified_exact_case",
            "planner_4b_qualified_exact_case",
            "both_qualified_1b_preferred",
            "both_qualified_no_automatic_preference",
            "no_planner_qualified",
            "resource_or_budget_failed",
            "correctness_failed",
        }
        self.assertEqual(self.dashboard.EXPECTED_RUN_ID, measure.RUN_ID)
        self.assertEqual(self.dashboard.ALLOWED_DECISIONS, expected_decisions)
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            for field, value in (
                ("run_id", "dual-model-evidence-planner-validation-wrong"),
                ("decision", "hardware_run_failed"),
                ("decision", "planner_1b_qualified_exact_case;run"),
            ):
                payload = self.result()
                payload[field] = value
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(field=field, value=value):
                    with self.assertRaises(self.dashboard.DashboardError):
                        self.dashboard.snapshot(result_path)
            for decision in expected_decisions:
                payload = self.result()
                payload["decision"] = decision
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(allowed_decision=decision):
                    self.assertEqual(
                        self.dashboard.snapshot(result_path)["decision"], decision
                    )

    def test_snapshot_uses_a_recursive_allowlist_for_untrusted_nested_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            payload = self.result()
            payload["gates"] = {
                "all_runs_completed": True,
                "text": "secret raw answer",
                "nested": {"tokens": [1, 2, 3], "evil": "drop me"},
            }
            payload["metrics"] = {
                "runs_completed": 12,
                "model_1b": {
                    "contract_successes": 6,
                    "text": "secret raw answer",
                    "tokens": [1, 2, 3],
                    "nested": {"text": "drop me"},
                },
                "model_4b": {"priority_successes": 6, "evil": {"tokens": [4]}},
                "pairwise": {"tokens": [5], "nested": {"text": "drop me"}},
                "cross_model_text": {"exact_text_equal_total": "6/6"},
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            value = self.dashboard.snapshot(result_path)
            encoded = json.dumps(value, sort_keys=True)
            self.assertNotIn("secret raw answer", encoded)
            self.assertNotIn('"text"', encoded)
            self.assertNotIn('"tokens"', encoded)
            self.assertNotIn("evil", encoded)

    def test_http_get_head_methods_host_rejection_and_hash_stability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(json.dumps(self.result()), encoding="utf-8")
            before = hashlib.sha256(result_path.read_bytes()).hexdigest()
            server = self.dashboard.ThreadingHTTPServer(
                ("127.0.0.1", 0), self.dashboard._handler(result_path)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                status, headers, body = self.request(server, "GET", "/")
                self.assertEqual(status, 200)
                self.assertGreater(len(body), 0)
                self.assertIn(b"textContent", body)
                head_status, head_headers, head_body = self.request(server, "HEAD", "/")
                self.assertEqual(head_status, 200)
                self.assertEqual(head_body, b"")
                self.assertEqual(head_headers.get("Content-Length"), headers.get("Content-Length"))
                api_status, _, api_body = self.request(server, "GET", "/api/snapshot")
                self.assertEqual(api_status, 200)
                self.assertNotIn(b'"tokens"', api_body)
                self.assertNotIn(b'"text"', api_body)
                api_head, _, api_head_body = self.request(server, "HEAD", "/api/snapshot")
                self.assertEqual(api_head, 200)
                self.assertEqual(api_head_body, b"")
                for target in ("/", "/api/snapshot"):
                    for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
                        write_status, write_headers, _ = self.request(server, method, target)
                        self.assertEqual(write_status, 405, (target, method))
                        self.assertEqual(write_headers.get("Allow"), "GET, HEAD")
                foreign_status, _, _ = self.request(server, "GET", "/", host="evil.example:8783")
                self.assertEqual(foreign_status, 421)
                foreign_head, _, _ = self.request(server, "HEAD", "/", host="evil.example:8783")
                self.assertEqual(foreign_head, 421)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            after = hashlib.sha256(result_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_html_uses_text_nodes_and_never_injects_raw_model_text(self) -> None:
        self.assertNotIn(b"innerHTML", self.dashboard.HTML)
        self.assertIn(b"textContent", self.dashboard.HTML)


if __name__ == "__main__":
    unittest.main()
