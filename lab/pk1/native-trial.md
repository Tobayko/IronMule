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
all returned elements against the unmodified binary. A separate incumbent startup records the
padding table, measured prefill pass costs and Pilot step costs. Each arm restores those measurements
and sets both W4 width masks using the canonical reference profile before model initialization.
Each worker reads both mask registers before and after the workload. Six fresh process pairs alternate arm order
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

## Run the native campaign from the control host

A prepared RunPod lease can execute the handoffs above under one bounded control process. Save the existing pod ID and the following absolute **target** paths in a local JSON file. `framework` is the trusted evaluator checkout, `checkout` is the measured serving tree, `python` is the interpreter, and `env` is a root-owned shell file containing binding, runtime and CUDA paths. The remaining keys are `gguf`, `reference_lib`, `table`, `worker_root`, `state_root` and `pod_id`. Stage immutable assets for the restricted UID first. The control host has its own clean checkout of the order's base commit at `$BASE`.

```sh
python -m lab.rp.native_campaign --config "$TARGET_JSON" --state "$STATE" \
    --order "$STATE/order.json" --checkout "$BASE" --publish --wait-seconds 3600
```

With hosted Codex, the process writes the attributed development request to `$STATE/authoring/author/request.json`. A connected dispatcher supplies the reply. With a locally authenticated subscription Codex CLI, add `--author-cli` for automatic consumption. Run the same command after a restart. Each poll checks the existing lease and retains completed result files on the control host. After six qualifying pairs, the publication gate checks the GitHub merge, sends the content-bound activation receipt to the target and runs a new independent process with a fresh seed. A rejected order retains its candidate, intermediate artifacts, reason and cost window. The provider journal records the final removal and billing tail. The upper-bound amortization estimate charges the whole elapsed lease to this candidate, including setup and author wait, and uses its measured median saving per 32 completed requests.

## Qualify an HTTP deployment

The native target runner measures the 32-client engine directly. A published native winner can enter a separate product qualification through the campaign option below. The new path uses the ordinary ProductService, HTTP API, tokenizer, scheduler, bridge and existing CPU Pilot. Each GPU worker runs with the restricted worker UID, independent wall deadline and read-only inputs. Only the frontend opens the loopback HTTP listener or receives product requests. The control host's GitHub and pod credentials stay outside the worker.

```sh
python -m lab.rp.native_campaign --config "$CONFIG" --order "$ORDER" --checkout "$BASE" --state "$CAMPAIGN" \
  --publish --serve-qualified --http-reference-url http://127.0.0.1:8231 \
  --http-port 8230 --http-serve-seconds 120 --http-timeout 1800 --wait-seconds 1800
```

Choose an existing ordinary loopback product endpoint with locally readable health and model metadata that advertises the same loaded model revision and a passed GGUF quality gate. Requests outside this endpoint's single-user-message, greedy, 32-output-token and 1,536-byte prompt envelope receive HTTP 307 with the ordinary endpoint as the destination. Clients can follow the redirect with the original POST. For a dedicated traffic lane on a private pod, replace `--http-reference-url ...` with `--dedicated-http`. That lane accepts only its explicit profile. Other product servers keep their configured deployments.

The adapter first verifies the merged native control receipt and the bound GGUF, table, binaries, source, hardware and frozen Pilot profile. Each arm must independently pass the product's GGUF numeric/factual gate with the measured settings. Four fresh worker starts in R-C-C-R order cover six paired seed holdouts. Every seed issues streamed HTTP completions at 32, eight and one concurrent clients. The PR57 performance prompts request four sentences and allow 32 output tokens. A completion with the same text, stop reason and exact prompt/output counts on both arms certifies equivalent measured work. The report gives completion counts and output equivalence for this batch. At width 32 all six runs must improve throughput, their median paired speedup must reach 1.02×, and median paired p95 time to first token (TTFT) must hold or improve. At widths eight and one, median wall time and p95 TTFT may increase by at most 5%. Peak host RSS and observed GPU memory use may grow by at most 5% in each pair.

Each seed also issues a separate concurrent batch of 18 independently graded short answers using the QG1 exact-answer grader. Qualification records each response, its stop reason, token counts and client-observed TTFT, plus the batch wall time and total correct. All 18 answers must pass on both arms with equivalent output in every pair. Correct completions per second is the number of graded passes divided by the full batch wall time. Across the six seeds, the median candidate/reference ratio must be at least 1.00 for correct completions per second and at most 1.00 for batch p95 TTFT. These supplemental HTTP criteria were set before measuring a candidate on the product endpoint. Gate time, startup and HTTP experiment time consume the same reserved pod lease.

On success the campaign makes a new ordinary factual POST through a candidate server, withdraws its selection in a copy of the protected deployment state, and repeats the same POST after the reference worker takes over. It then starts the actual selected endpoint for the requested bounded lifetime, makes a fresh external factual request and saves the serving receipt. A persistent withdrawal survives restart. The live worker returns to the verified reference after failure or three consecutive measured timing misses among matching prompt size, output length and queue-delay buckets. An unmatched request has no timing vote. The control host continuously captures the gate reports and paired HTTP halves into `http-evidence/`.

The same target can later launch or withdraw the HTTP selection:

```sh
http_target=("${target[@]}" --worker-root "$WORKER_ROOT")
python -m lab.pk1.native_http qualify "${http_target[@]}" --timeout 1800
python -m lab.pk1.native_http exercise "${http_target[@]}" --port 8230 --timeout 600
python -m lab.pk1.native_http serve "${http_target[@]}" --port 8230 --reference-url http://127.0.0.1:8231 --lifetime 600
python -m lab.pk1.native_http rollback --state "$DEPLOY"
```

Here `target` contains the same state, GGUF, original checkout, reference library, candidate source, trial, dispatch table and dedicated worker root supplied to native activation. The measured CPU frontend smoke accepted two ordinary 32-token requests through its stock worker. CUDA product admission requires a positive native release and actual HTTP evidence on the target.
