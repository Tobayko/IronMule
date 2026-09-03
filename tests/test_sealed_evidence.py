"""Every sealed evidence database must still verify today.

The hash chains are this project's deepest integrity mechanism, and nothing
ran them as a set. A chain verifies only when somebody asks it to.

.friday-data is gitignored, so each check skips when its database is absent
from a checkout. A skip is not a pass; it means this checkout carries no
evidence to verify.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".friday-data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: (database, module, class, expected record count as the documents claim)
RECORD_CHAINS = [
    ("head-skip-v1.sqlite3", "experiments.head_skip_formal.study", "Storage", 16),
    ("h1-v2.sqlite3", "friday_h1.storage", "Storage", 16),
    ("n10-v1.sqlite3", "friday_n10.storage", "Storage", 2),
    ("n10-v2.sqlite3", "friday_n10_v2.storage", "Storage", 16),
    ("phase1b-rmsnorm.sqlite3", "friday_phase1b.history", "History", 2),
    ("avo-router.sqlite3", "friday_avo_router.history", "History", 2),
    ("runtime.sqlite3", "friday_runtime.history", "History", 2),
    ("runtime-n10.sqlite3", "friday_runtime_n10.history", "History", 2),
    ("head-skip-runtime.sqlite3", "friday_head_skip_runtime.history", "History", 3),
]


@pytest.mark.parametrize(
    "database,module_name,class_name,expected",
    RECORD_CHAINS,
    ids=[entry[0].removesuffix(".sqlite3") for entry in RECORD_CHAINS],
)
def test_a_sealed_chain_still_verifies(database, module_name, class_name, expected):
    path = DATA / database
    if not path.is_file():
        pytest.skip(f"{database} is not present in this checkout")
    store_class = getattr(importlib.import_module(module_name), class_name)
    with store_class.open(path, read_only=True) as store:
        records = list(store.verified_records())
    assert len(records) == expected, (
        f"{database} holds {len(records)} records; the documents claim {expected}"
    )


def test_the_head_skip_history_has_the_shape_the_status_table_describes():
    path = DATA / "head-skip-v1.sqlite3"
    if not path.is_file():
        pytest.skip("head-skip-v1.sqlite3 is not present in this checkout")
    study = importlib.import_module("experiments.head_skip_formal.study")
    with study.Storage.open(path, read_only=True) as store:
        kinds: dict[str, int] = {}
        for record in store.verified_records():
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    # "versiegelte Präregistrierung, sechs bestandene A/A-Sessions, sechs
    # frische A/B-Sessions" — the table's wording, counted.
    assert kinds == {
        "preregistration": 1, "calibration_session": 6, "calibration_summary": 1,
        "confirmation_seal": 1, "confirmation_session": 6, "study_decision": 1,
    }


def test_the_optimizer_memory_chain_is_intact():
    path = DATA / "optimizer-v2.sqlite3"
    if not path.is_file():
        pytest.skip("optimizer-v2.sqlite3 is not present in this checkout")
    from friday_optimizer.memory import OptimizationMemoryV2

    with OptimizationMemoryV2.open_read_only(path) as view:
        report = view.integrity()
    assert report.chain_ok, report.error
    assert report.rows > 0


def test_the_shared_evidence_store_verifies():
    path = DATA / "research.sqlite3"
    if not path.is_file():
        pytest.skip("research.sqlite3 is not present in this checkout")
    from friday_evidence.storage import EvidenceStorage

    with EvidenceStorage.open(path) as storage:
        storage.verify_schema()
        rows = storage.verified_rows()
    assert rows


def test_every_h0_run_still_hashes_to_its_recorded_manifest():
    path = DATA / "h0.sqlite3"
    if not path.is_file():
        pytest.skip("h0.sqlite3 is not present in this checkout")
    from friday_h0.manifest import manifest_hash, validate_manifest
    from friday_h0.storage import Storage

    mismatches = []
    with Storage.open(path, read_only=True) as storage:
        storage._verify_identity()
        storage._verify_schema_integrity()
        rows = storage.connection.execute(
            "SELECT run_id, manifest_json, manifest_hash FROM runs"
        ).fetchall()
        for run_id, manifest_json, recorded in rows:
            if manifest_hash(validate_manifest(json.loads(manifest_json))) != recorded:
                mismatches.append(run_id)
    assert rows, "the H0 database records no run"
    assert not mismatches, f"manifest hash drifted for: {mismatches}"


def test_no_sealed_database_is_left_unchecked():
    """A new sealed database must be added here, not quietly ignored."""

    if not DATA.is_dir():
        pytest.skip("no evidence directory in this checkout")
    known = {entry[0] for entry in RECORD_CHAINS} | {
        "optimizer-v2.sqlite3", "research.sqlite3", "h0.sqlite3", "h01.sqlite3", "device-profile.sqlite3",
    }
    present = {path.name for path in DATA.glob("*.sqlite3")}
    assert not (present - known), f"unverified evidence databases: {sorted(present - known)}"
