# Experience archive and trial deployment

From the repository root, build a private JSON Lines snapshot of the tracked notebook, runtime evidence and lab result files. The builder uses the repository's placement-trace validator and the per-step trainer's eligible examples. It also includes the current local CPU-thread trial checkpoint when available. The snapshot can be regenerated from the same source tree.

```sh
python -m training.experience_archive build --output /tmp/ironmule-experience.jsonl
python -m training.experience_archive search --archive /tmp/ironmule-experience.jsonl --query 'cross_pod_reference_changed'
python -m training.experience_archive search --archive /tmp/ironmule-experience.jsonl --query 'PS1a allocation' --limit 2
```

Each row has a source path, a SHA-256 digest of its source bytes, a locator within those bytes and a grouping key. Missing measurements are null. A placement-pair label concerns a worker route on the recorded host, with the listed background queues and complete worker settings. Its output hash covers the visible response and stop reason. The 200 historical placement rows include 80 with matched response digests and token counts. Repeats sharing a prompt-size family stay in one group across campaigns. The per-step thread trainer has separate context-family groups and a different action space.

The A100 import reads individual HTTP observations from the original pod 36 and 37 tarballs. The locator gives the tar member and line. PS1, ps1a and PK6 came from the maintainer's Claude sessions. The two different R configurations across pods and the vLLM AWQ comparison are excluded from learning labels. Cost rows refer to whole historical sessions. Notebook text is indexed as reported commentary. Forecasts from the CPU replay have simulated provenance, separate from actual observed traffic. Remaining raw files have searchable paths and content digests pending a protocol-specific importer. The per-trial dates, measurement scope, qualifications and known unknowns remain in the original sources. [Lab notebook](../docs/LAB.md) · [Placement protocol](hardware-placement.md)

## Switch and withdraw a measured revision

The release adapter operates on the single model, device binding and warmed public copy workload recorded by the evaluator. Its entry point runs four clients through the GGUF tokenizer, Pilot and engine. A qualification does not authorize changing a general HTTP service. Use the trial directory containing the six paired records as `TRIAL`.

```sh
python -m training.trial_release activate --state "$TRIAL" --order "$ORDER" --pr "$PR" --gguf "$GGUF"
python -m training.trial_release run --state "$TRIAL" --gguf "$GGUF" --seed 9400 --output /tmp/active-wave.json
python -m training.trial_release check --state "$TRIAL" --gguf "$GGUF"
python -m training.trial_release rollback --state "$TRIAL"
```

Activation verifies the merged GitHub PR on its order-specific qualification branch. The merge's first parent must equal the measured baseline. The reviewed tree, merged tree and all six measured candidate trees must match. The controller reads the runtime source hashes from the Git objects, verifies the evaluator, hardware binding, seed, work and output, then repeats the original gate. A fresh reference run on the target verifies placement before activation. The selected revision is stored in an atomically replaced relative symlink under `deployment/active`.

The check command measures six alternating pairs with fresh seeds, under the same resource, throughput, TTFT and correctness limits. A regression or evaluator failure writes a persistent withdrawal and restores the reference checkout. Subsequent runs select that reference, including after restart. Re-running a withdrawn order preserves its withdrawal. As a deployment exercise, the offline tests use real Git trees, synthetic evaluator results and a local substitute for the GitHub PR response.

```sh
python -m pytest -n 0 -q tests/agent/test_experience_archive.py tests/agent/test_trial_release.py
```
