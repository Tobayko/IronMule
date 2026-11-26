# Replay the first-token experiment

The GPU protocol and candidate are frozen at commit `b9051fcaf96fcdd53e95ebf0fa15db42b454877e`. The public scorer runs on a CPU. Use Python 3.11 or later in the repository environment.

```sh
python -m lab.pk6.first_token.phases --projections lab/autonomous/results/a100-warm-ttft
```

To extract the same projection from original traces, add `--traces "$RAW_TRACES"` and point `--projections` to a new output directory. Each series must contain the original streaming timelines and HTTP wave summaries. The extractor keeps only synthetic benchmark request IDs, timestamps, KV sequence IDs, row counts and stop reasons. It rejects conflicting clocks, partial work, missing or duplicate IDs and call shapes that disagree with the jobs.

For each wave the scorer also reports median time from send to the second text chunk and the p95 of each request's longest gap between chunks. A server can emit one early piece and then stall. These secondary readouts were specified after examining the historical six native waves and before the new GPU processes returned their first measurements. Only the gates in the [registered protocol](protocol.md) determine the paired decision.

To repeat the intervention on a separate authorised lease, use `lab.rp.rp` with an existing cumulative journal and watchdogs. On the target create distinct checkouts for the driver and measured runtime. The following commands assume `/work/IronMule` is a clean clone with the driver commit fetched.

```sh
cd /work/IronMule
git worktree add --detach /work/first-token-driver b9051fcaf96fcdd53e95ebf0fa15db42b454877e
git checkout --detach f3a69741f7a466ffc144a87b0395f4bfdb6e9b50
VLLM=0 JOBS=16 bash lab/rp/setup.sh 80
export IRONMULE_GGML_BINDINGS="$HOME/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16"
export LLAMA_CPP_LIB_PATH="$IRONMULE_GGML_BINDINGS/vendor/llama.cpp/build/bin"
PYTHONPATH="$PWD" python /work/first-token-driver/lab/pk6/first_token/trial.py \
  --checkout "$PWD" --models /models --state /work/state --output /work/first-token
```

Launch the target command as a job named `first-token-trial`. On the control host, `python -m lab.rp.first_token --pod-id "$POD_ID" --state "$RESULTS"` copies the files to persistent storage every eight seconds. Reuse the same private directory when resuming that lease. Stop the pod with `python -m lab.rp.rp stop` and verify account and journal status. Given the copied directory, run the following from the analysis checkout containing the scorer. Use a private copy when extracting projections alongside full traces.

```sh
python -m lab.pk6.first_token.phases --series "$RESULTS" --traces "$RESULTS" \
  --projections "$RESULTS" --labels R1 C1 C2 R2
python -m lab.pk6.first_token.grade "$RESULTS"
```
