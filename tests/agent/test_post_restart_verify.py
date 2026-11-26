import copy
import json

import pytest

from lab.pk1 import post_restart_verify as verify

ARCHIVED = verify.FROZEN.parents[1] / "lab/autonomous/results/a100-restart"
DATA = ARCHIVED / "d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3"


def test_cpu_selection_replays_real_target_scores_and_refuses_changed_model_choice():
    contract = json.loads(verify.FROZEN.read_bytes())
    receipt = json.loads((DATA / "control.json").read_bytes())
    choice = json.loads((DATA / "pilot-trained.json").read_bytes())["selection"]
    verify.check_selection(choice, DATA / "startup-head.npz", receipt, contract)
    invalid = copy.deepcopy(choice)
    invalid["scores"][0][0] += .02
    with pytest.raises(ValueError, match="scores"):
        verify.check_selection(invalid, DATA / "startup-head.npz", receipt, contract)


def test_disjoint_times_bind_location_and_excluded_native_methods(tmp_path):
    source = tmp_path / "ggml_engine.py"
    source.write_text("class GgmlEngine:\n    def _decode(self):\n        self.L.llama_decode()\n")
    site = verify.post_restart_profile.sites(source)
    line, eligible, reason = site["GgmlEngine._decode"]
    assert eligible is False
    row = {"function": "GgmlEngine._decode", "path": verify.SOURCE, "line": line,
           "eligible": eligible, "reason": reason, "self_s": 0.1, "calls": 1}
    phases = [{"phase": "startup", "wall_s": .2, "unassigned_s": .1, "sites": [row]}]
    data = {"initialized_s": .2, "broker_elapsed_s": 1}
    verify.check_phases(phases, ["startup"], site, data)
    row["eligible"] = True
    with pytest.raises(ValueError, match="exclusion"):
        verify.check_phases(phases, ["startup"], site, data)
    row["eligible"] = False
    phases[0]["sites"].append(row)
    with pytest.raises(ValueError, match="duplicate"):
        verify.check_phases(phases, ["startup"], site, data)
    phases[0]["sites"].pop()
    phases[0]["unassigned_s"] = .2
    with pytest.raises(ValueError, match="sum"):
        verify.check_phases(phases, ["startup"], site, data)
