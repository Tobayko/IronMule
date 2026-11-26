# Attributing a decode kernel and measuring a source variant

Run the profiler from the PR 57 source on a Linux CUDA worker with one visible GPU. Set
`IRONMULE_GGML_BINDINGS` and `LLAMA_CPP_LIB_PATH` to its PR 57 CUDA build. For a Git archive set
`IRONMULE_SOURCE_COMMIT` to the full revision used for the archive. The existing W4 tuner builds
and measures a table locally for the GPU, compiler and GGUF. The first A100 comparison uses
Qwen3-32B Q4_K_M, SHA-256 `efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689`.
Reserve the accelerator exclusively for calibration and timed work.

```sh
export PYTHONPATH="$PWD" IRONMULE_GGML_GPU_LAYERS=999 IRONMULE_W4=1
export IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_GGML_SEQ_CTX=2048
export IRONMULE_PK6=1 IRONMULE_PS1=1 IRONMULE_PS1_WARM=0 IRONMULE_DH2=0 IRONMULE_SV5=0
export IRONMULE_CUBLAS_MIN_ROWS=128 IRONMULE_CUDA_GRAPHS_ROWS=32
python lab/pk1/profile_engine.py --gguf "$GGUF" --width 32 --prompts 32 \
    --prompt-len 238 --decode 16 --order-out "$STATE/order.json"
```

The profiler warms up the decode shape, records two 16-step CUPTI device windows and looks up the
observed symbols in the checkout. The ranking estimates seconds saved over 31 steps if each
candidate shortened its own measured device duration by 10%. Source attribution identifies a
specialization. The dispatch table identifies the variants used for matrix shapes. Ops outside
IronMule's W4 source stay in the profile for diagnosis. The order records both windows and the
bound model, binary, source, table, hardware and runtime. This engine diagnostic uses fixed tokens.

To dispatch the measured site to the hosted Codex worker, spool the order with its exact
source, submit the worker's reply through the existing author transport, and rerun the author
command. The source SHA must still match the profiling receipt.

```sh
python -m lab.pk1.native_trial --mode author --order "$STATE/order.json" \
    --checkout "$BASE" --output "$STATE/authoring/candidate.cu"
```

The evaluator accepts a
separate CUDA source file containing the proposed kernel. Build it with the incumbent variant
indices and function ABI so both arms can consume the exact same dispatch table. `BASE` is the
unchanged reference checkout. `REFERENCE_SO` and `TABLE` are the local tuning artifacts bound to
the order. The controlling checkout supplies the evaluator, gate and sandbox.

```sh
python -m lab.pk1.native_trial --checkout "$BASE" --candidate-source "$PROPOSAL" \
    --reference-lib "$REFERENCE_SO" --table "$TABLE" --order "$STATE/order.json" \
    --gguf "$GGUF" --output "$STATE/trial/result.json"
```

The compiler and each measuring subprocess run in a separate Linux worker with a time limit.
The default worker uses bubblewrap. On an owned disposable pod that disallows nested
namespaces, supply `--worker-root /opt/ironmule-worker/tasks --worker-uid 60000` to use the
restricted UID worker. The operator stages root-owned read-only runtime, evaluator, model and
bindings in paths accessible to that UID. Its access and network preflight runs before each
job. The broker imports only one bounded regular output from the worker each time.
The screen checks the model's width-32 matrices with the existing float64 tolerance and compares
all returned elements against the unmodified binary. Six fresh process pairs alternate arm order
with a warm wave in each. Each arm runs a 32-client short/document mix through the tokenizer,
the CPU Pilot, native engine and output callbacks, then grades 18 factual answers. Reports bind
the actually mapped W4 artifact and table separately from the hardware and native GGML identity.
The fixed serving gate requires identical generated work, improvement in every pair, median
throughput gain of at least 2%, median p95 TTFT ratio at most one, and peak host RSS within 5%.
The per-arm timing includes ordinary controller decisions. The outer report separately records
build and experiment time for that invocation. A stopped run resumes from the saved order,
build receipt, screening and completed arms.

After a rejected order, replay its two original device windows against an unprofiled, factual-gated
reference arm on this binding. The next order needs a different source family and enough measured
device time to cover the 2% whole-wave gate using the original 10% kernel-duration assumption.
The planner estimates the cost of twelve arm processes using the measured reference duration and the
previous experiment's duration. A second rejection ends this search on the retained profile. Supply
the current lease deadline, GPU price and unused reservation from the cost journal.

```sh
python -m lab.pk1.native_order --prior "$STATE/order.json" --trial "$STATE/trial" \
    --reference "$STATE/trial/ref-0.json" --output "$STATE/next/order.json" \
    --seconds "$REMAINING_SECONDS" --hourly-usd "$HOURLY_USD" --usd "$REMAINING_USD"
```

A failing screen or pair leaves the reference binary selected. A positive result qualifies the
tested mixed engine workload on that binding. Attribution, kernel timings or device-name matches
alone supply no serving qualification. Any extension of PR 57's comparison with vLLM requires
the corresponding HTTP workloads on both deployments. [PR 57 benchmark protocol](../pk6/a100_pk6.sh)

## Control publication and scoped target selection

Copy the original order and adjacent profile, the full author directory, table, trial reports,
and both the build receipt and measured candidate binary to the controller. Run below from
the same trusted evaluator revision. The control command recomputes the numerical, factual,
resource and timing gates, reconstructs the candidate from the hosted worker reply and
reports the isolated Git tree. A rejected or incomplete measurement cannot create a PR.

```sh
python -m lab.pk1.native_release control --state "$CONTROL" --trial "$TRIAL" \
    --order "$ORDER" --candidate-source "$PROPOSAL" --table "$TABLE"
```

After a genuine qualification, repeat with `--publish` to submit the exact tree and wait for
both CI jobs. The command verifies the merge, parents and source bytes on the order's
qualification branch. Transfer `$CONTROL/deployment/control.json` to the target using
the trusted operator channel, along with the reported SHA-256. Keep the selected state,
receipts, libraries, and table outside worker write access.

```sh
target=(--state "$DEPLOY" --trial "$TRIAL" --checkout "$BASE" --candidate-source "$PROPOSAL"
        --reference-lib "$REFERENCE_SO" --gguf "$GGUF" --table "$TABLE")
python -m lab.pk1.native_release activate "${target[@]}" --receipt "$RECEIPT" --receipt-sha256 "$SHA256"
python -m lab.pk1.native_release run "${target[@]}" --seed 93742
python -m lab.pk1.native_release rollback --state "$DEPLOY"
```

On the restricted RunPod use the same `--worker-root` and `--worker-uid` as the original
trial. Activation runs both binaries on a fresh seeded 32-client workload and checks all 18
factual responses. `run` selects the recorded binary only for this engine workload and reports
fresh performance and correctness. A latency or elapsed-time excursion beyond the original
reference envelope triggers a matched reference check. Failed quality, hash, or matched
performance checks persist withdrawal before reference fallback. A different target needs
its own qualification. The release command has no effect on the default server configuration.
