"""A KG5 decision is scoped to the observed tensor shape, machine and both complete measured stages."""
import hashlib
import json

import pytest

from lab.kg5 import bench
from lab.kg5.serve_qualified import read_verdict


def test_exact_shape_requires_linked_steps_and_http_with_factual_dispatch():
    identity = {"uuid": "gpu-a", "inputs_sha256": ["model-a", "build-a"]}
    order = [(0, "off"), (0, "on"), (1, "on"), (1, "off")]
    facts = {str(i): True for i in range(50)}
    evidence = {**identity, "shape": bench.SHAPE, "status": "passed",
                "pairs": [{"passed": True, "largest_sample_kl": 0.001, "step_saving": 0.11}] * 2}
    arms = [{"rep": rep, "arm": arm, "hits": int(arm == "on")} for rep, arm in order]
    steps = json.dumps({**evidence, "stage": "steps", "arms": arms}).encode()
    http = {**evidence, "stage": "http", "steps_sha256": hashlib.sha256(steps).hexdigest(),
            "arms": [{**a, "facts": facts, "fact_hits": [a["hits"]] * 2, "tok_s": 10 + a["hits"],
                      "ttft_median_s": 1, "ttft_p95_s": 2} for a in arms]}
    read_verdict(steps, json.dumps(http), identity)
    for key, value in [("steps_sha256", "unrelated"), ("status", "timeout"), ("shape", {}), ("uuid", "gpu-b")]:
        with pytest.raises(ValueError):
            read_verdict(steps, json.dumps({**http, key: value}), identity)
    http["arms"][2]["fact_hits"] = [0, 0]
    with pytest.raises(bench.Rejected):
        read_verdict(steps, json.dumps(http), identity)
