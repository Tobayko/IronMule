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

The active workload runner also watches for regressions in its completed requests. It compares wave position, short/document phase, each lane's input and generated token counts, and the amount of reused prompt work. A reference envelope requires at least two distinct evaluation seeds with the same shape. The controller withdraws after three successive runs in which every matched wave exceeds the slowest reference span or first-token latency, with both phases represented. An unmatched run or recovery resets the counter. The order-specific counter survives restarts. This guard reuses completed timings without starting another benchmark.

### Scoped code releases on RunPod

Two additional adapters admit a single attributed change to the engine. The four-client EPYC copy experiment targets `GgmlEngine._prefill` at revision `79ee889`. The A100 Qwen3-32B repeat-startup experiment targets `GgmlEngine.__init__` at revision `33dc7bf`. Their release paths check the actual method boundary, complete qualification evidence, clean Git tree and CI on the exact proposal PR. Each order gets a `qualification/<order-id-prefix>` branch. The control host can publish using:

```sh
python -m lab.rp.cpu_code_release control --state "$CONTROL" --trial "$CPU_REPORTS" \
  --authorship "$AUTHOR" --reference "$AUTHOR/reference" --candidate "$AUTHOR/candidate" --order "$ORDER" --publish

python -m lab.rp.restart_campaign --state "$GPU_CAMPAIGN" --config "$GPU_CONFIG" \
  --order "$GPU_ORDER" --checkout "$QWEN_BASE" --publish --wait-seconds 3600
```

The control host transfers the merged receipt and its SHA-256 independently. All GitHub operations stay on that host. The target compares the observed model, hardware, runtime, code and workload against the receipt. A fresh candidate/reference measurement must pass before the atomic selection changes. The target checks exported Python packages, weights, tests and lab inputs against the hashes of the committed source. Use fresh worker-readable, immutable Git exports for both arms. Broker state belongs outside their read/write scope.

To recheck or operate the CPU four-client copy lane on the registered CPU affinity, set the same native library environment used for qualification and run the following commands with the same Linux `taskset` as the trial:

```sh
cpu=(--state "$TARGET_STATE" --reference "$REF" --candidate "$CAND" --order "$ORDER" --gguf "$GGUF"
     --head "$HEAD_NPZ" --worker-root "$WORKER_ROOT" --python "$WORKER_PYTHON" --receipt-sha256 "$TRUSTED_SHA")
python -m lab.rp.cpu_code_release activate "${cpu[@]}" --receipt "$CONTROL_RECEIPT"
python -m lab.rp.cpu_code_release run "${cpu[@]}" --seed 9352
python -m lab.rp.cpu_code_release exercise "${cpu[@]}" --seed 9353
python -m lab.rp.cpu_code_release rollback --state "$TARGET_STATE"
```

The CPU record covers six matched seeds including all actual delivered answers, the frozen scratch head and its qualification time, plus the fixed four-thread challenger. `run` uses the measured engine and seeded copy workload. It stores the actual selected source, scratch choices, serving time and full worker elapsed time. The restart campaign runs its corresponding commands on the current paid lease after successful native startup qualification. Its `run` records both 32-client completions, independently graded answers, cold time, warm time and full worker elapsed time. Both adapters exercise the persistent withdrawal in a copy of the target state by executing identical actual input on the selected candidate and the restored reference. Explicit rollback also survives a process restart. A product endpoint needs its own real product qualification.

The matched EPYC retry keeps the original code order and attributed source response. It captures a separate incumbent-calibrated profile and uses it across all arms on seeds 18351–18356. The fresh local probes still have to support each published thread decision. `cpu_code_release control --matched` checks these raw trials, derives a distinct qualification order and PR branch from the parent order and profile hash, and includes the verified calibration bytes in the control receipt. The target validates a fresh held-out pair at seed 19351. An optional `--profile "$TARGET_PROFILE"` checks the supplied file against the same digest before candidate use. A stale candidate profile triggers withdrawal. The reference can use the original calibration bytes in the verified receipt.
