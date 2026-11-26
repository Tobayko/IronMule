# Same-device local calibration source experiment

The second A100 RunPod campaign used the ordinary PR 57 server and the Qwen3-32B GGUF
at commit `8cf0a3106f09a3338eb53c28047af7e36fc7b340`. Two complete HTTP server
starts on that board each spent about 18.8 s measuring PK2. IronMule ranked local
startup methods with matching observed native geometry. Previously rejected methods
and the earlier PR 60 search family carry their earlier outcomes into the ranking.
The winning sensitivity projection was 4.63 s saved per qualifying fresh start.
This number describes a simulation. The two source inputs, rank, attributed Codex
reply and original constraints are preserved in the [order archive](../autonomous/results/a100-dynamic-calibration/).

The generated method measures all five thresholds on the local GPU. In the specified
geometry it may stop after the second repeat if the observed ranges agree on the
original choice and speed threshold. Ambiguity triggers the original third repeat.
The source keeps the fresh reference and candidate likelihood evaluations. The
opt-in applies to the bound experimental entry. The served language-model weights,
GPU kernels, numerical threshold and output limit are shared by the compared arms.

## CPU replay

Use a Linux source checkout with its CPU development dependencies installed.

```sh
series=lab/autonomous/results/a100-pr57-warm-repeat
trial=lab/autonomous/results/a100-dynamic-calibration
python -m lab.pk6.warm_result "$series"
python -m lab.pk6.startup_search commission "$series" "$trial" "$trial"
python -m lab.pk6.startup_search materialize "$trial" /tmp/ironmule-measured-pk2.py
python -m pytest -n 0 -q tests/agent/test_dynamic_calibration.py
```

The CPU replay rebuilds the qualified parent engine from the previous Codex response,
then applies the new response to its single allowed method. It works in a shallow
checkout. The regression test replays actual burst timings, opt-out behavior, noisy
measurements and a rejected numeric choice.

## Bounded target run

The HTTP study and ordered study have separate prospective protocols. The HTTP
A/A reference changed one warm text at request index 29 with the same seed, token
count and selected threshold. The grader stopped. A separately retained diagnostic
completed a full candidate HTTP start and answered all 18 factual questions.
The ordered study uses the existing native `GgmlEngine` and per-step Pilot. A trusted
client enqueues the entire batch in a deterministic order, checks returned token IDs,
stops and factual grades and records cold and warm time, RSS and GPU memory. The job
keeps every completed arm and stops at its first mandatory failure.

On the recorded physical RunPod target, after the warm comparison and before
the lease deadline, run the broker on the control host. It uses the existing
budget journal, a clean tracked baseline, a single-method candidate, root-owned
inputs and a separate unprivileged GPU worker. The ordered workload runs inside
the repository's Landlock and seccomp restrictions. The private control state
can be reused after interruption. A terminal source verdict is durable.

```sh
python -m lab.rp.rp account
OPENBLAS_NUM_THREADS=1 python -m lab.rp.ordered_campaign \
  --pod-id "$POD_ID" --state "$CAMPAIGN" --seconds 2500 --publish
python -m lab.pk6.ordered_grade "$CAMPAIGN/ordered"
```

Publication uses a source-only qualification PR. The broker checks the exact merged
tree and repeats both workload families on fresh seeds. The existing Pilot's profile
head fits all six mixed pairs from random weights and holds out the complete
factual-first family. The reference and always-qualified candidate are its two simple
rule comparators. The model can abstain. A live ablation uses its original random
weights on the same qualified choices. Each delivery gets a new process and work
identifier. A local withdrawal survives process restart. The exercise runs a real
candidate delivery followed by a real reference delivery and compares both work
digests. The evaluation charges the source, controller, process and full workload
including checks around the worker.

On a qualified target the following restores the local reference for subsequent
ordered requests. This command addresses the scoped deployment state.

```sh
python -m lab.pk6.ordered_release rollback --state /work/ordered-release/state
```

The physical device digest, CUDA and native binaries, GGUF and local W4 table bind
the evidence. A different machine needs its own observations and qualification.
Historical qualifying source, model and release receipts remain available in the
[repeat-start archive](../pk1/restart-trial.md). The [operator guide](../rp/README.md)
covers provider balance, reservations, watchdogs and teardown.
