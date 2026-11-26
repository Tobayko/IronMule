# Hardware coverage of the development loop

The Pilot scores actions on the CPU. The table describes the device that executes the GGUF work used to choose and evaluate code changes. A historical kernel or width comparison has its own model, binary, scheduler, quality gate and workload. The release adapter keys admission to the measured machine, model, runtime, settings and workload.

| Linux target | Detection and device computation | Measured model work | Status of the new development loop |
| --- | --- | --- | --- |
| Intel Xeon Platinum 8573C, four-core quota | The local GGML CPU build verified an F32 addition on 2026-11-26. No accelerator appeared in the host inventory. Explicit CUDA and ROCm doctor probes each returned “no registered GGML … device”. | SmolLM2-135M-Instruct Q4_K_M completed the four-client copy workload. A separate two-slot plain-decode study fitted a CPU thread head and replayed its choices. | The planner selected a tokenizer change. The evaluator rejected it after six pairs. The two-slot thread head also retained its reference in the quiet and contention audits. [Code trial](../lab/autonomous/results/cpu-tokenizer-01/) · [Pilot replay](../evidence/runtime/cpu-pilot-thread-replay.json) |
| RunPod AMD EPYC 7742, four pinned physical cores, CPU-only MLX 0.32.5 + GGML | CUDA visibility was empty and no model layer was offloaded. A separate Linux CPU wheel supplied MLX. | The scratch head selected four threads on the held-out context at one and four decode rows. Their combined measured decode time was 0.17545 s, tied with the all-four-thread rule, versus 0.46744 s for the shipped Pilot. | Four locally qualified scratch decisions were used across 16 correct requests in 6.16645 s of timed waves. The p95 first-token latency was 2.19551 s. Collection, training, qualification and serving totaled 231.97 s. Host-wide load counters limit attribution to the four pinned CPUs. [EPYC evidence](../lab/autonomous/results/runpod-epyc-pilot/) |
| RunPod NVIDIA A100 SXM4 80 GB, SmolLM2-135M-Instruct Q4_K_M | GGML registered CUDA, verified its device-local addition and loaded 31/31 layers onto the GPU. | The Pilot-off portable smoke passed both copy requests. All eight calibration and holdout cells in the independent width-1/4 comparison matched output and correctness. | The four-client Pilot-on evaluator failed after full offload in the native batched cuBLAS path. The one-client retry with padding disabled reached the same native function and recorded an unsupported-parameter error. These runs generated no qualified code trial. [A100 preflight evidence](../lab/autonomous/results/a100-smollm2-preflight/) |
| RunPod NVIDIA A100 SXM4 80 GB, Qwen3-32B Q4_K_M | PR 57 records native A100 builds and on-device qualification. | Historical PS1 + ps1a + PK6 gave 304.61 tok/s at 32 clients against 301.875 tok/s for PS1 + ps1a and 276.88 tok/s for vLLM 0.11.0 AWQ-Marlin. | A rebuilt PR 57 reference required PR 59 to restore W4 dispatch. The new W4 kernel variant was slower in the matched microbenchmark and failed qualification. A separate Python change reuses an exactly bound W4 measurement in a new worker. It passed six paired mixed-first runs, two factual-first pairs, and fresh target activation. Median cold delivery was 33.292 s with the original worker and 15.360 s with this change. The scratch head made the choice in fresh serving processes. [Portable release evidence](../lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3/) · [Native kernel trials](../lab/autonomous/results/a100-native/) · [PR 57 archive](../lab/pk1/results/pod37/) |
| Other NVIDIA CUDA families | GGML queries the registered device, visible memory, driver/runtime and device identity. Hardware indicators alone leave compute verification pending. | Existing width experiments cover particular A40, H100, L40S, RTX 4090 and RTX 6000 Ada configurations. Long Qwen3-32B on RTX 6000 Ada failed held-out token agreement. | CUDA evaluation is available for one visible device with full model offload. Each new family, device/runtime, model and workload needs its own reference, correctness and six paired measurements before a code change can be deployed. [Portable study and exceptions](../docs/LAB.md) |
| AMD ROCm | The builder discovers HIP targets. The inventory reports AMD PCI/driver data, and the GGML adapter probes the registered ROCm device. | The portable smoke and width calibration accept a HIP-enabled GGML build. This development campaign has no measured AMD device. | The isolated development evaluator currently accepts CPU and CUDA. ROCm needs an evaluator device-mount adapter, a bound full-offload smoke, measured reference and paired holdouts on the target. [Build targets](../tools/build_ggml.py) · [Portable calibration](../lab/portable_gpu/README.md) |

## Product worker kernel profiles

`ironmule serve --gguf` qualifies each worker in its own process. For the PR 57 Qwen3-32B Q4_K_M file (SHA-256 `efd97156…`), one GGML-registered NVIDIA A100-SXM4-80GB, all 65 layers requested and an identifiable CUDA runtime, it proposes PS1 + ps1a + PK6. PS1_WARM, PS1_CONCAT, DH2 and SV5 stay off. If that bundle fails the local quality gate, the worker retries with all six flags off. Other GPU/model combinations start qualification with all six off. A failed reference gate leaves the worker unavailable. CPU workers keep the legacy CPU flags. An explicit lab flag selects an experimental bundle and passes through the same gate.

The serving process verifies the effective flags, GGUF SHA-256, requested offload, device/driver identity, loaded native binaries and selected kernel-library contents before opening its first model context. Gate and GPU width caches include the profile. A passed quality gate establishes correctness on its own fixtures. Speed and deployment of a new code variant still require the independent paired workload on its target. Historical standalone scripts using `GgmlEngine` directly retain their original PR 57 defaults. [Profile selection](../ironmule/ggml_profiles.py) · [Verification](../ironmule_product/pinned_ggml.py) · [Standalone reference](../docs/HANDOVER_A100.md)

On the Xeon, a cold gate completed all three stages in 152.8 seconds with the stock settings. The worker served a two-token HTTP completion. A restart restored the same profile and repeated the completion. The measured CPU was shared. [CPU startup and restart evidence](../lab/autonomous/results/xeon-smollm2-product-profile/smoke.json)

## Verify a new binding

Load the environment file produced by the GGML build for the actual target. With the configured runtime, use the following probes before scheduling an experiment. Set `PROVIDER` to `cpu`, `cuda` or `rocm`. For GPU calibration select one visible card.

```sh
python -m ironmule_cli doctor --ggml "$PROVIDER" --json
python lab/portable_gpu/bench.py --gguf "$GGUF" --provider "$PROVIDER" --stage smoke --budget-s 60 --output smoke.json
python lab/portable_gpu/bench.py --gguf "$GGUF" --provider "$PROVIDER" --stage compare --transition --budget-s 240 --output width-calibration.json
```

The doctor reports passive host observations separately from a verified arithmetic result. The smoke uses the native loader's offload count, answers, generated IDs and stop reasons. Width calibration makes paired measurements on the selected backend with fresh holdout targets and includes context rebuilding during transitions. These files help select the next experiment. The code evaluator measures both versions again with its own fixed copy workload and matching Pilot, model, backend binary and device. [Implementation and calibration details](../lab/portable_gpu/README.md)

Read one trial's evidence and selected deployment without running a model. A missing release yields an empty checkout. A withdrawn release selects its reference. The status read validates the checkout and can complete a pending withdrawal. Its recorded device comes from the prior report. Use `check` to verify the active revision on the current target.

```sh
python -m training.trial_release status --state "$TRIAL"
python -m training.trial_release check --state "$TRIAL" --gguf "$GGUF"
```

### A100 restart support

The measured startup release runs one CUDA engine per isolated worker on the recorded A100, with the
same Qwen GGUF, W4 binary and dispatch table, GGML build, CUDA vendor libraries, mask tuple,
bounded workload and eligible CPU checkpoint. A missing or stale receipt starts local calibration.
The broker verifies the release and chooses a complete program before the worker starts.
The reference and candidate share PS1, ps1a and PK6. The improvement pays back through repeat
worker starts. Six warm waves averaged comparable service time in the two arms. A separately
measured automatic withdrawal remained active across process restart. Its first incumbent attempt
failed its local gate. The next worker served 32 cold, 32 warm and 18 correct factual answers.

```sh
order=d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
OPENBLAS_NUM_THREADS=1 python -m training.replay_qualified_restart "lab/autonomous/results/a100-restart/$order"
```

The merged source is present on the existing A100 development branch. The release receipt covers
the tested source tree and deployment scope. The ordinary streaming HTTP product has its own
numeric and factual calibration and separate evidence. [Qualified program](https://github.com/Tobayko/IronMule/pull/60) ·
[A100 integration](https://github.com/Tobayko/IronMule/pull/61) · [Startup and broker commands](../lab/pk1/restart-trial.md)

### Activated EPYC prefill code in the product bridge

After the EPYC code release has a verified publication and a fresh active target receipt, run the
separate product gate on that same target. Both root-owned checkouts must match the published receipt.
The broker starts two unprivileged workers, qualifies the same thread head, and alternates six
comparisons through `GgmlBridge.generate_many`. Each comparison includes four short/document waves
of four public copy requests. Each lane sends its first visible text through the worker pipe.
The gate times its arrival at the broker and checks the text against the completed answer.
The receipt also preserves each worker callback timestamp. Total batch latency includes the
request and completed reply over the broker pipe.

```sh
python -m lab.rp.cpu_product_release qualify --state "$STATE" --order "$ORDER" \
  --reference "$REF" --candidate "$CAND" --gguf "$GGUF" --head "$HEAD_FILE" \
  --python "$CPU_PYTHON" --worker-root "$WORKER_ROOT" --receipt-sha256 "$RECEIPT_SHA" --timeout 360
```

Use `serve` with the same arguments to keep both workers warm. After its JSON readiness line, send
one JSON object per line containing a `prompts` array and `max_tokens`. Replies include the selected
arm, complete answers, token IDs, stop reasons and timings. For a single batch, use `run --requests
PATH`. A four-lane batch in the qualified copy family can use the candidate. Other prompt content,
batch sizes within one to four, and output limits up to 64 use the reference. The worker context
limit is 512 tokens per request. A quality failure retries on the reference. Exceeding the measured
reference timing envelope withdraws the candidate for subsequent requests and processes.
This interface operates over private process pipes. HTTP measurement requires its own target run.

In the initial independent-start EPYC comparison, both arms gave 16 correct answers at seed 8351.
Their starting head profiles differed. The reference qualified one to four threads and the source
variant qualified two to four. The resulting 6.239 s and 2.402 s cannot qualify the code variant.

The matched protocol uses one new trusted reference calibration. Each isolated arm loads the
same measured step prices, original head and eligible offsets. Recorded visits schedule the
previously measured proposal for testing. Each process independently runs all four of the
original counterbalanced triplets, checks outputs and re-evaluates its own A/A noise and
controller cost. The serving head must choose the admitted action on real work. Supply this
immutable artifact through `--profile PATH` to the matched product adapter. Both its warm workers
verify the profile digest and requalify the same actions.

### Learned local calibration on the second A100

Two ordinary Qwen3-32B Q4_K_M server starts on the second RunPod A100 SXM4
80 GB led to a new source order for PK2. The ordered native entry qualified
and activated the generated source on that physical GPU with driver 595.112.07.
The three scratch CPU heads preserve the original deployment on a missing or
unqualified workload. The complete 16-chunk stock gate and factual checks
are in the target archive. The ordinary asynchronous HTTP source trial stopped
at its own reference A/A. Each new device needs fresh measurements for its
own binding. [Deployment, external requests and offline replay](../lab/pk6/local-calibration.md)
