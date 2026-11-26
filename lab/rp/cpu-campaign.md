# A bounded EPYC source experiment

The Linux control host owns the budget journal, hosted Codex response, clean Git checkouts and
release decisions. The target must already be an admitted RunPod lease with the restricted CPU
worker, CPU MLX, GGML, the pinned model and the candidate-independent broker installed. The EPYC
7742 adapter covers four pinned cores, four SmolLM2 135M copy clients, up to three generated tokens
and 512 tokens per sequence. Use a separate lane for a different source order and a separate local
state for each qualification protocol. Keep the private RunPod journal across restarts.

The private target JSON contains the seven fields below. Set the actual active pod ID from the
budget journal. Paths shown correspond to the recorded RunPod setup.

```json
{
  "pod_id": "YOUR-ACTIVE-POD-ID",
  "framework": "/opt/ironmule-worker/controller",
  "env": "/opt/ironmule-worker/env.sh",
  "python": "/opt/ironmule-worker/cpuvenv/bin/python",
  "lane": "/opt/ironmule-worker/cpu-eval",
  "worker_root": "/opt/ironmule-worker/tasks",
  "gguf": "/opt/ironmule-worker/models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
}
```

From the control checkout and its CPU Python environment, the following command plans from
observations, consumes one correlated subscription Codex request, stages complete Git exports
and submits one target experiment. `--matched` selects a single fresh reference calibration
followed by independent verification of that profile in every arm. The target uses new seeds
18351 through 18356 and six counterbalanced source, incumbent and four-thread rule comparisons.

```sh
python -m lab.rp.cpu_campaign --matched \
  --state /persistent/epyc-matched --target /private/target.json \
  --head evidence/runtime/cpu-pilot-epyc-candidate.npz \
  --prior evidence/runtime/cpu-pilot-epyc-discovery.json \
  --evidence lab/autonomous/results/runpod-epyc-pilot \
  --author-cli --wait-seconds 1200
```

Omit `--author-cli` for the hosted request/response transport. The saved outcome gives the request
path. Supply an existing correlated `--order order.json --authorship /persistent/authoring` to
replay the proposal with fresh checkouts. API billing requires independent prior budget admission.
The normal CPU pilot and offline replay make no external model calls.

`--prepare-only` verifies the lease, stages immutable files, and returns a waiting status before
dispatch. Reserve a quiet host window before resuming without this flag. Each new target job holds
a sibling file lock and runs under the lease deadline. The default qualification job starts with
`epyc-qualify-` and the first 16 hex characters of the author order. The matched protocol adds
`-matched-v2`. Use `--job-name` to adopt an existing operator-submitted job including its incident
suffix. A restarted control process inspects the target's files, job table and process arguments
before deciding to dispatch. The polling window is bounded by `--wait-seconds`. Source transfers and a protected GitHub CI check can outlast that window. With zero wait the command
takes one step and exits, and rerunning it resumes the campaign.

The output directory contains `order.json`, `target-binding.json`, `qualification/` and
`campaign-outcome.json`. Each completed raw arm includes actual answers, token IDs, thread
qualification, source and binary hashes, wall time, TTFT, memory and overhead. The control host
recomputes the complete verdict, checks fixed work and preserves refusals. The cost estimate
includes the whole lease to date. Use the budget journal for invoice reconciliation.

Add `--publish` to submit a qualified change through its scoped PR. Target activation requires the
verified merge and a fresh pair. The coordinator dispatches real work, exercises withdrawal with
identical input on an isolated state copy and then checks selection from a new process. Add
`--product-requests prompts.json` to qualify the warm `GgmlBridge` path independently and send
one real batch. The input shape is `{"prompts":["a prompt"],"max_tokens":3}`. Only eligible
four-client copy traffic uses the scoped candidate. The result records the actual selected arm,
answer bytes and persistent withdrawal state. The primary serving branch retains its prior default.

The first startup-per-arm protocol selected different thread profiles on its real target.
Its preserved first pair can be rechecked offline, with no active lease or author, using the
following command.

```sh
python - <<'PY'
from pathlib import Path
from lab.rp.cpu_campaign import profile_conflict
from training.code_trial import read
data = Path('lab/autonomous/results/runpod-epyc-pilot')
print(profile_conflict(data / 'prefill-code/qualification-v1', read(data / 'prefill-order.json')))
PY
```

For a qualified or rejected six-triplet result, replay the exact historical verdict from the
recorded source response and `qualification/` directory, even after the lease has been deleted.

```sh
python -m lab.rp.cpu_campaign --replay --matched --state /persistent/epyc-matched
```

If the original authoring directory was shared across protocols, pass the same
`--authorship /persistent/authoring` used during qualification.
