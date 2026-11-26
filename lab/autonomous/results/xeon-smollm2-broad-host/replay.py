"""Replay the two fixed, independently seeded Xeon diagnostic runs from their raw wave records."""
import hashlib
import json
from math import isclose
from pathlib import Path

from training.code_trial import verify_deliveries


def main():
    root = Path(__file__).resolve().parent
    outcome = json.loads((root / "outcome.json").read_bytes())
    bindings = []
    for row in outcome["runs"]:
        raw = (root / f"{row['seed']}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        observation = json.loads(raw)
        record = observation["record"]
        verify_deliveries(record, row["seed"])
        assert row["source"] == record["commit"]
        assert row["correct"] == record["correct"] == record["completed"] == 16
        assert row["tokens"] == record["tokens"] == 48
        bindings.append(record["binding"])
        profiles = observation["measured_profiles"]
        assert len(profiles) == len(record["waves"]) == 4
        totals = []
        for wave, profile, saved_share in zip(record["waves"], profiles, row["wave_shares"], strict=True):
            total = sum(site[4] for site in profile["sites"])
            assert 0 <= total <= wave["wall_s"] <= profile["wall_s"]
            assert isclose(total / wave["wall_s"], saved_share)
            totals.append(total)
        share = sum(totals) / sum(wave["wall_s"] for wave in record["waves"])
        assert isclose(share, row["extra_share"])
        assert share < 1 - 1 / 1.02
        print(f"seed {row['seed']}   correct {record['correct']}/16   exclusive share {share:.4%}")
    assert len(bindings) == 2 and bindings[0] == bindings[1]
    assert outcome["decision"] == "defer" and outcome["author_order"] is None


if __name__ == "__main__":
    main()
