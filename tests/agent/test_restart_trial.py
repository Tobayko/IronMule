import copy

import pytest

from ironmule.factual_gate import cases
from lab.pk1 import native_trial as native
from lab.pk1 import restart_trial as trial
from training.serving_trial import digest


def row(seed, cold=50., reused=False):
    answers = [{"output_ids": [i], "prompt_digest": f"p{i}", "answer": str(i), "stop": "eos"} for i in range(32)]
    wave = {"completed": 32, "tokens": 32, "wall_s": 2., "ttfts_s": [1.] * 32,
            "output_sha256": digest(answers), "answers": answers, "geometry": {"steps": 2, "padded": 0}}
    facts = [{"answer": c.expected, "output_ids": [i + 10]} for i, c in enumerate(cases(seed))]
    return {"schema": "ironmule.qwen-restart.arm.v1", "seed": seed, "cold_s": cold, "initialized_s": cold-3,
            "broker_elapsed_s": cold+15, "cold": wave, "warm": wave.copy(), "facts": [True] * 18, "factual": facts,
            "factual_timing": {"ttfts_s": [1.] * 18, "wall_s": 2., "completed_s": cold+5}, "factual_first": [],
            "mask_bits": [0xff000000, 0], "mask": {"tokens_served": [32], "q6k_tokens_served": []}, "reused": reused,
            "binding": {"gpu_layers": 999, "seqs": 32, "seq_ctx": 2048, "speculate": True,
                        "flags": {"IRONMULE_" + k: v for k, v in native.FLAGS.items()},
                        "settings": {"flash_attn": True, "kv_unified": False, "kv_f32": False},
                        "native": {"sha256": {"backend/CUDA": "backend"}}, "fp": {"devices": ["test"]}},
            "source_sha256": "candidate" if reused else "base", "pilot_decisions": 23,
            "cuda_vendor": {tag: {"path": "/cuda/" + tag, "sha256": value} for tag, value in trial.A100_VENDOR.items()},
            "pilot_sha256": "pilot", "harness_sha256": native.checksum(trial.HARNESS),
            "peak_rss_kib": 300, "device_used_bytes": [40, 45, 45, 45]}


def test_cold_restart_requires_arm_equivalence_and_gain_above_aa_noise():
    aa = [row(trial.SEED), row(trial.SEED, 51)]
    pairs = [(row(trial.SEED+i), row(trial.SEED+i, 44, True)) for i in range(trial.PAIRS)]
    assert trial.judge(pairs, aa)["passed"]
    assert not trial.judge(pairs, [aa[0], row(trial.SEED, 60)])["passed"]
    perturbed = copy.deepcopy(pairs)
    perturbed[0][1]["warm"]["geometry"] = {"steps": 1}
    with pytest.raises(ValueError):
        trial.judge(perturbed, aa)
    wrong = copy.deepcopy(pairs[0][1])
    wrong["factual"][0]["answer"] = "doesn't know"
    with pytest.raises(ValueError):
        trial.fresh_pass(aa[0], wrong, trial.SEED)
    wrong_vendor = copy.deepcopy(pairs[0][1])
    wrong_vendor["cuda_vendor"]["libcublasLt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        trial.fresh_pass(aa[0], wrong_vendor, trial.SEED)


def test_stale_restart_must_return_to_the_prevailing_profile():
    reference = row(trial.SEED)
    assert trial.compare_pair(reference, row(trial.SEED), trial.SEED, require_hit=False)
    with pytest.raises(ValueError):
        trial.compare_pair(reference, row(trial.SEED, 41, True), trial.SEED, require_hit=False)


def test_factual_first_regrades_both_seeds_and_measures_first_token_from_process_start():
    seed = trial.SEED + 100
    ref, cand = row(seed), row(seed, 45., True)
    for arm in (ref, cand):
        arm["initialized_s"] -= 7
        for i, expected in enumerate((seed, seed ^ 419)):
            arm["factual_first"].append({"seed": expected, "completed_s": arm["initialized_s"] + (i + 1) * 2.2,
                                         "wall_s": 2., "ttfts_s": [.5] * 18, "correct": [True] * 18,
                                         "answers": [{"answer": c.expected, "output_ids": [1], "stop": "eos"} for c in cases(expected)]})
    result = trial.factual_pair(ref, cand, seed, ref["binding"])
    assert result["cold_factual_speedup"] > 1 and result["cold_factual_p95_ratio"] < 1
    assert result["warm_factual_p95_ratio"] == 1
    cand["factual_first"][1]["answers"][0]["answer"] = ""
    with pytest.raises(ValueError):
        trial.factual_pair(ref, cand, seed, ref["binding"])
