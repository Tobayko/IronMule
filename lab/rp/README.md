# Bounded RunPod measurements

Use one Linux control machine and the same private `RP_STATE` path throughout the campaign. The default is `~/.cache/ironmule/rp.json`. Keep this directory on persistent storage. It includes the job token. Use the control checkout for accounting and the evaluator, and give the candidate checkout only to the build and measurement job.

```sh
python lab/rp/rp.py account
python lab/rp/rp.py spend
python lab/rp/rp.py deploy --gpu 'NVIDIA A100-SXM4-80GB' --cloud SECURE --disk 80 --minutes 60 --max-hourly 2.50
python lab/rp/rp.py wait
# Submit the prepared experiment and fetch its results as each pair completes.
python lab/rp/rp.py stop
python lab/rp/rp.py account
python lab/rp/rp.py spend
```

The initial campaign limit is USD 15 across all attempts. For a concrete need, the operator may run `python lab/rp/rp.py extend 'reason'` to raise the limit once to USD 20. Before a separately metered API call, use `reserve-external NAME MAX_USD` with its maximum charge. After billing arrives, use `settle-external NAME ACTUAL_USD`. Unsettled reservations continue to count. After the limit, local training and replay can continue.

Deployment checks the account balance and quotes the specified cloud, GPU count and disk. The ledger reserves the maximum hourly rate for the entire lease plus ten minutes for cleanup and billing. For example, 60 minutes at a maximum USD 2.50 per hour requires USD 2.917 available. A local process starts before the create mutation and reconciles ambiguous creates by unique pod name. Duplicate creates are blocked while the reservation exists. A second process starts with the container and requests termination at the same UTC deadline using RunPod's pod-scoped credential. The control credential stays on the control machine. Neither process requires the model server. Both depend on the provider API being reachable. A failure of both control planes can delay cleanup.

After restarting the control host, run `python lab/rp/rp.py resume` to restart its deadline watcher or clean up an expired lease. Run `stop` for immediate cleanup. If the provider has yet to confirm removal, retry `stop`. A create whose outcome cannot be resolved holds its reservation for operator reconciliation. The history records the observed account delta and the wall-time estimate including a ten-minute tail separately. Admission uses the higher value. Account deltas may include unrelated usage. Historical executions appear in the existing ledger if the same state path was used.

Price basis for container disk is USD 0.10 per GB-month, conservatively spread over 28 days. Only ephemeral container disk is requested. [RunPod storage pricing](https://docs.runpod.io/pods/pricing) · [Pod-scoped credentials](https://docs.runpod.io/pods/references/environment-variables)

## Running an isolated native candidate

On a dedicated Linux pod, the root broker places the model, reference binary, bindings, evaluator and
candidate sources in root-owned trees with read-only access for a dedicated worker uid. Copy bindings
out of a private root home before starting the worker. The worker receives its own empty directory.
The following checks the boundary without compiling or loading candidate code.

```bash
install -d -m 0755 /opt/ironmule-worker
install -m 0444 lab/rp/restricted_worker.py /opt/ironmule-worker/restricted_worker.py
install -d -m 0700 -o 60000 -g 60000 /opt/ironmule-worker/preflight-out
python /opt/ironmule-worker/restricted_worker.py --preflight \
  --output /opt/ironmule-worker/preflight-out --readonly /opt/ironmule-worker/restricted_worker.py
```

The launcher needs libseccomp and Landlock ABI 3 or newer. It drops supplementary groups and the
entire capability bounding set, requires `no_new_privs`, denies sockets and applies a filesystem
allowlist before running the preflight in every worker. The parent sets a wall deadline and terminates
the process group. Per-process CPU time, descriptors, process count, core files and output file size
are limited. CUDA access uses the pod's device files. Check GPU execution separately on each host.
[Recorded A100 preflight](results/restricted-worker-preflight.json)

## Four-core CPU trial during a GPU-idle window

On a leased Linux host with an available five-minute interval, use a clean Git checkout and a working
GGML installation. Export its binding/library paths and choose four allowed physical cores. This
example uses logical CPUs 0–3 on the EPYC 7742 RunPod host with SMT off.

```bash
python lab/rp/cpu_pilot.py --cpus 0 1 2 3 --seconds 300 \
  --gguf /models/SmolLM2-135M-Instruct-Q4_K_M.gguf --state /work/cpu-trial-first
```

The runner trains on complete pairs from context bands 1 and 4 and replays band 6 on holdout. The
subsequent four-slot serving run starts fresh hardware qualification. The command uses CPU inference
with GPU visibility disabled and lowers its scheduling priority. `campaign.json` records the source
commit, affinity, phase timings and outcome. `trials.json` retains complete and interrupted series.
An empty holdout, failed correctness check or lack of qualified model decisions leaves a failed phase.
Send SIGTERM when GPU timing must resume. Preserve the state directory before terminating the pod.

An owned pod may renew its current lease once, by at most 90 minutes. The operator first
checks the provider's reported owner, balance and rate and reserves the additional duration
in the shared journal. The deployed pod's immutable original watcher transfers enforcement
to a replacement that initially uses the original deadline. The controller commits the new
deadline after a positive remote acknowledgement. A lost reply retains the reservation and
the earlier control deadline. Retry using identical minutes and reason. Start the handoff at
least three minutes before the initial deadline. Calling `stop` still terminates early.

```sh
python -m lab.rp.rp renew --minutes 90 --reason 'finish paired tests on the same calibrated GPU'
python -m lab.rp.rp spend
```

After verified teardown, export the redacted campaign journal. It contains the observed
account delta, per-lease posted USD if billing has caught up, conservative charges including
the retained cleanup tail, and external settlements. Pod IDs, account balances, job names,
tokens and free-text reasons stay in the private control journal. Refresh the export after
the provider posts delayed billing.

```sh
python -m lab.rp.rp report --output lab/rp/results/campaign-costs.json
```
