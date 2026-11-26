"""Prospective serving protocol for one measured improvement over the merged PR60 worker."""

from training.code_author import digest

SCHEMA = "ironmule.post-restart-source-order.v1"
BASE = "63db7745998383509602601d46ae4bbff0076648"
PARENT = "d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3"
CONTROL = "de74395813fd3d71edf6b0a860da01dc352f5a95d3c5f997dc7fc6799cb1d639"
HEAD = "a490979b3ebb77a2bcae6659c7ae44b70a813acb4fb05be4a2da7c19d7a46254"
AA_SEED = 96850
MIXED = tuple(range(96851, 96857))
FACTS = (96951, 96952)


def contract():
    return {
        "schema": "ironmule.followup-serving-contract.v1",
        "base_commit": BASE,
        "parent_order": PARENT,
        "parent_control_sha256": CONTROL,
        "weights_sha256": HEAD,
        "reference": "merged PR60 source with previously verified W4 receipt in both arms",
        "workload": "cold 32x32 mixed, warm 32x32 mixed, independent 18 facts. Two factual-first paired seeds each run two distinct 18-fact batches on a fresh process.",
        "cold_clock": "wall from before complete CPU startup selection through last rendered completion including decision time",
        "aa_seed": AA_SEED,
        "aa_processes": 2,
        "mixed_seeds": list(MIXED),
        "factual_first_seeds": list(FACTS),
        "pair_order": "alternate reference first then candidate first, within each family",
        "mixed_win": "all cold pairs faster, median >=1.02 and median relative gain > A/A spread",
        "factual_first_win": "both cold pairs faster and median >=1.02; median cold absolute TTFT p95 ratio <=1.02",
        "guardrails": "paired median warm wall and p95 ratios <=1.02 for mixed and factual (including first-factual holdout); each RSS and observed device footprint ratio <=1.05",
        "equivalence": "same source-specific target binding, W4 masks and receipt, full 32/18 completions, outputs and scheduling counters. Both regular arms use the same independently trained CPU head and complete cached profile.",
        "fallback": "changed/withdrawn/expired parent or unknown context cannot select new source. Numerical grades and retained old cold calibrator checks remain mandatory.",
        "fresh_target": "new mixed seed 97051 and factual-first seed 97052 with same full real-work checks before scoped activation; persisted withdrawal restores merged PR60",
        "repeat_limit": 1,
        "code_retries": 0,
        "activation_rollback": "real fresh work before/after persisted source withdrawal on same seed 97053",
    }


def contract_digest():
    return digest(contract())
