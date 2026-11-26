# Hardware coverage of the development loop

The Pilot scores actions on the CPU. The table describes the device that executes the GGUF work used to choose and evaluate code changes. A historical kernel or width comparison has its own model, binary, scheduler, quality gate and workload. The release adapter keys admission to the measured machine, model, runtime, settings and workload.

| Linux target | Detection and device computation | Measured model work | Status of the new development loop |
| --- | --- | --- | --- |
| Intel Xeon Platinum 8573C, four-core quota | The local GGML CPU build verified an F32 addition on 2026-11-26. No accelerator appeared in the host inventory. Explicit CUDA and ROCm doctor probes each returned “no registered GGML … device”. | SmolLM2-135M-Instruct Q4_K_M completed the four-client copy workload. A separate two-slot plain-decode study fitted a CPU thread head and replayed its choices. | The planner selected a tokenizer change. The evaluator rejected it after six pairs. The two-slot thread head also retained its reference in the quiet and contention audits. [Code trial](../lab/autonomous/results/cpu-tokenizer-01/) · [Pilot replay](../evidence/runtime/cpu-pilot-thread-replay.json) |
| RunPod NVIDIA A100 SXM4 80 GB, SmolLM2-135M-Instruct Q4_K_M | GGML registered CUDA, verified its device-local addition and loaded 31/31 layers onto the GPU. | The Pilot-off portable smoke passed both copy requests. All eight calibration and holdout cells in the independent width-1/4 comparison matched output and correctness. | The four-client Pilot-on evaluator failed after full offload in the native batched cuBLAS path. The one-client retry with padding disabled reached the same native function and recorded an unsupported-parameter error. These runs generated no qualified code trial. [A100 preflight evidence](../lab/autonomous/results/a100-smollm2-preflight/) |
| RunPod NVIDIA A100 SXM4 80 GB, Qwen3-32B Q4_K_M | PR 57 records native A100 builds and on-device qualification. | Historical PS1 + ps1a + PK6 gave 304.61 tok/s at 32 clients against 301.875 tok/s for PS1 + ps1a and 276.88 tok/s for vLLM 0.11.0 AWQ-Marlin. | These settings and bundled binaries form the historical reference for this binding. A new CUDA change needs a fresh comparison against this reference. [Handover and scope](../docs/HANDOVER_A100.md) · [Original runs](../lab/pk1/results/pod37/) |
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
