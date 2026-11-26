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
