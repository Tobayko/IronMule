import numpy as np
import pytest

from ironmule.pilot import Pilot
from training.pilot_w4_profiles import NAMES, features, fit_head


def test_whole_profile_requires_consensus_support_and_local_receipt():
    zero = np.zeros
    head = [zero((1, 2)), zero(1), zero((1, 1)), zero(1), zero((3, 1)), np.array([.1, .8, .2])]
    different = [*head[:-1], np.array([.1, .2, .8])]

    def choose(heads, eligible=NAMES, support=True, error=.1):
        return Pilot.startup_profile(heads, features(16), NAMES, eligible, "q4_q6", in_support=support, margin_error=error)

    assert choose([head])["profile"] == "q4"
    assert choose([head], ["q4_q6"])["reason"] == "whole profile lacks local qualification"
    assert choose([head, different])["reason"] == "unresolved model margin"
    assert choose([head], error=1)["profile"] == "q4_q6"
    unknown = choose([head], support=False)
    assert unknown["profile"] == "q4_q6" and unknown["scores"] == []
    with pytest.raises(ValueError):
        choose([head], eligible=[])


def test_fit_is_reproducible_and_uses_only_training_groups():
    times = {1: [40, 30, 27], 2: [48, 33, 27], 32: [40, 30, 27]}
    first = fit_head(times, [1, 2], 7)
    other = fit_head({**times, 32: [1, 500, 900]}, [1, 2], 7)
    assert all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_independent_stream_fit_requires_raw_counterbalanced_contexts(tmp_path):
    from types import SimpleNamespace

    from training import pilot_w4_profiles as lane

    order = {"order_id": "observed"}
    files = {}
    groups = []
    for rnd in (0, 1):
        for profile in NAMES:
            cells = []
            for i, context in enumerate(lane.CONTEXTS):
                key = {"context": context, "width": 32}
                if rnd == 0 and profile == "stock":
                    groups.append(key)
                cells.append({**key, "prompt_lengths": [32] * 16 + [32 * (i + 1)] * 16, "warm_tokens": [[1] * 32] * 2,
                              "tokens": [[1] * 32] * 8, "seconds": (3 - NAMES.index(profile)) * (1 + .01 * rnd)})
            target = tmp_path / f"{rnd}-{profile}.json"
            lane.save(target, {"source_sha256": lane.trial.checksum(lane.Path(lane.__file__).with_name("pilot_w4_probe.py")),
                               "spec": {"order": "observed", "name": profile}, "cells": cells})
            files[target.name] = lane.trial.checksum(target)
    observations = tmp_path / "contexts.json"
    lane.save(observations, {"schema": "ironmule.w4-stream-contexts.v1", "order": "observed", "source_sha256": files,
                             "groups": groups})
    args = SimpleNamespace(contexts=observations, seed=12, weights=tmp_path / "head.npz", report=tmp_path / "fit.json")
    lane.fit_contexts(args, order)
    result = lane.read(args.report)
    assert {g[0] for g in result["train_groups"]} == {"32", "96"}
    assert result["added_groups"] == [["224", 32]] and result["uncertainty_groups"] == [["512", 32]]
    assert result["holdout"][0]["group"] == "serving_mix"
    heads, _, seed, features = lane.load_heads(args.weights, "observed")
    assert len(heads) == 3 and seed == 92741 and len(features) == 3
    (tmp_path / "0-q4.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        lane.fit_contexts(args, order)


def test_whole_profile_comparison_removes_only_qualified_mask_treatment():
    from training.pilot_w4_profiles import equal_conditions, whole_profile

    reference = {"mask_bits": [65535, 255], "padding": {"3": 8}, "pilot_costs": {"step_one": .002}}
    runtime = {"profile": reference, "threads": 4, "affinity": [0, 1, 2, 3]}
    q4 = whole_profile(reference, "q4")
    assert q4 == {**reference, "mask_bits": [65535, 0]}
    assert equal_conditions({"binding": {**runtime, "profile": q4}}, reference) == runtime
    assert equal_conditions({"binding": {**runtime, "threads": 2, "profile": q4}}, reference) != runtime
    assert equal_conditions({"binding": {**runtime, "profile": {**q4, "padding": {}}}}, reference) != runtime
    assert reference["mask_bits"] == [65535, 255]
