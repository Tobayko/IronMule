# Reproduce the mixed-prefill experiment

The CPU model uses timed calls from two independent A100 starts to rank six row budgets. Regenerate the pre-registered choice from the repository root with Python 3.12 and the project's development dependencies installed.

```sh
python -m lab.pk6.mixed_progress.plan > /tmp/mixed-plan.json
diff -u lab/autonomous/results/mixed-prefill-progress/plan.json /tmp/mixed-plan.json
python -m pytest -q tests/agent/test_mixed_progress.py
```

For the small-model CPU screen, prepare an isolated worktree at the frozen evaluator revision. Provide a locally installed CPU ggml build and matching SmolLM2-135M-Instruct Q4_K_M checkpoint. Use the same CPU affinity when comparing timings. The independent screen compares the reference and split scheduler on four new task groups, with eight requests per arm and group. A nonzero exit preserves any exact-output differences even when every answer has the correct factual value.

```sh
git worktree add --detach /tmp/ironmule-mixed 7fa0e4d8974c190192f056f6bc5b30dc72b82571
cd /tmp/ironmule-mixed
git apply --check lab/pk6/mixed_progress/engine.patch
git apply lab/pk6/mixed_progress/engine.patch
export IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_GGML_PACK=1 IRONMULE_LAB_CPU_PACK=1
python -m lab.pk6.mixed_progress.cpu "$GGUF" --seed 84351 --pairs 4 > /tmp/cpu-holdout.json
```

The original hardware series started at `7fa0e4d8974c190192f056f6bc5b30dc72b82571`. Its first native start exposed the stock-retry bug. The on-device [recovery](recover.py) pinned the correction from PR 68 and the unchanged external wave digest. The signed-off code and warm experiment state are both described by the archived target. For an offline replay of those original observations, use the compressed native traces included in the series.

```sh
data=lab/autonomous/results/l40s-mixed-prefill-progress
python -m lab.pk6.mixed_progress.grade "$data" --project
python -m lab.pk6.mixed_progress.learning "$data"
```

For a fresh collection use a revision containing both PR 67 and PR 68. Start from the root of that checkout. Supply an existing budgeted L40S lease with both RunPod watchdogs, sufficient remaining time and an empty private results directory. The collector builds for SM89, verifies the models and starts the five servers. A fresh series restores the initial kernel environment between starts.

```sh
revision=$(git rev-parse HEAD)
python -m lab.rp.mixed_progress --pod-id "$POD_ID" --state "$PRIVATE_RESULTS" --commit "$revision"
python -m lab.pk6.mixed_progress.grade "$PRIVATE_RESULTS" --project --expected-source "$revision"
```

On the same device, after a completed primary run, a separate virtual environment containing `vllm==0.34.0` can run the registered current-release supplement. Pass that environment as `--vllm`, the primary output directory as `--primary`, and the same verified model directory as `--models`. Collect sampled device telemetry into `current-device-load.csv` using the four columns recorded by the primary trial. Run `python -m lab.pk6.mixed_progress.release_score PRIMARY SUPPLEMENT` to replay it. This experiment uses the existing lease deadline.

Rebuild the searchable archive with `python -m training.experience_archive build --output /tmp/ironmule-experience.jsonl`. Forecasts and cross-backend speed measurements carry their provenance. The repository's ordinary serving command uses the qualified product scheduler.

For the current-version comparison, replay the recorded vLLM process with `python -m lab.pk6.mixed_progress.release_score lab/autonomous/results/l40s-mixed-prefill-progress lab/autonomous/results/l40s-vllm-034`. The measured package set, device identity, sampled memory and power, and both factual seed blocks are in the output.

For the shared-pool CPU regression, use the isolated worktree containing the scheduler patch, apply `pool-budget.patch` from this directory and run `python -m lab.pk6.mixed_progress.pool_cpu "$GGUF"` with GPU layers disabled. The stock and candidate cases share the same forced low CPU admission budget. The GPU run uses the actual trained Pilot. The archive distinguishes the initial split-stream experiment from this shared-pool follow-up.
