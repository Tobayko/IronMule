# Qwen3-32B prompt dispatch on NVIDIA L40S

The study starts from PR 57 at `2e3c1bb69b8cedc2b889156e6dd0912bc1a6235d`. It tests real mixed prompt and decode calls on Ada. The production reference uses the existing CPU Pilot, native GGML and its local quality gate. The external process serves the official AWQ-Marlin artifact with vLLM 0.34.0. Both engines receive the same rendered token IDs, greedy decoding, 2,048-token contexts and a 32-token generation limit.

## Replay on CPU

From the repository root, run the following commands with the repository's Python dependencies. The HTTP replay checks the board, model and comparator digests, process windows, all native request IDs, seven streaming metrics, output counts, factual case identities and one-second device samples.

```bash
results=lab/autonomous/results/l40s-native-dispatch
python -m lab.pk6.native_dispatch.score "$results/http" > /tmp/ironmule-comparison.json
python -m lab.pk6.native_dispatch.tile_order "$results/http/primary" /tmp/ironmule-native-order
```

The order uses stock HTTP and independent CUPTI traces to rank two GGUF types and their compiled tile sizes. The predicted savings assume a 10% reduction in the sampled device interval. The target and the Codex response are bound by digest. The build replaces one Q4_K translation unit using the target's own compilation database, then relinks an isolated CUDA backend. The tile experiment retains native shared-memory and shape validation. The baseline and candidate can also run on the same alternative binary with the candidate disabled or enabled.

## Run on the registered GPU

Use one L40S, a native SM89 build from `lab/rp/setup.sh 89`, a verified Qwen3-32B-Q4_K_M GGUF, the official Qwen3-32B AWQ files and the vLLM 0.34.0 environment under `/opt/vllm034`. The RunPod operator supplies `RP_DEADLINE` from its reserved lease. Run the commands inside that reservation, with `IRONMULE_GGML_BINDINGS` and `LLAMA_CPP_LIB_PATH` pointing to the same native build throughout. The operator owns the watchdog, collection and cleanup. All GPU measurements on a host use `/work/native-dispatch.lock`.

```bash
model=/models/Qwen3-32B-Q4_K_M.gguf
python -m lab.pk6.native_dispatch.build fp8 /work/fp8 "$model"
python -m lab.pk6.native_dispatch.kernel_check --gguf "$model" \
  --lib /work/fp8/fp8.so --out /work/fp8-arithmetic
python -m lab.pk6.native_dispatch.operators --gguf "$model" \
  --lib /work/fp8/fp8.so --out /work/operator-measurements
python -m lab.pk6.native_dispatch.learn /work/operator-measurements /work/pilot-policy
```

The scratch-trained CPU head uses the existing Pilot selector. Training, uncertainty calibration and testing keep entire projection geometries together. The two independent activation seeds stay with each geometry. A route needs agreement across heads, a margin above calibration error, numerical agreement in the operator screen and a measured advantage over native GGML and cuBLAS. Only adjacent supported points can form a candidate interval. Interpolated widths require end-to-end qualification. An empty plan retains stock.

For a bounded serving run, keep the experiment sources from this PR in the disposable checkout used by the runner. Set up a detached checkout of the base revision before applying the instrumentation. The target's weight files, gate thresholds and reference executables stay fixed.

```bash
base=2e3c1bb69b8cedc2b889156e6dd0912bc1a6235d
git worktree add --detach /work/prompt-study "$base"
cp -R lab/pk6/native_dispatch /work/prompt-study/lab/pk6/
cd /work/prompt-study
export PYTHONPATH="$PWD"
python -m lab.pk6.native_dispatch.prepare .
bash lab/pk6/native_dispatch/serve.sh /work/http/primary \
  --lib /work/fp8/fp8.so --arms vllm R1 F1 F2 R2
```

Use fresh output directories. The two native blocks run with different seeds. A diagnostic D process samples actual CUDA symbols and never supplies performance labels. F uses a prompt-only FP8 KG1 plugin. I uses signed INT8 products and int32 accumulation. M extends the vendored Q4_K Marlin conversion to long prompts and has an explicit model lifetime and a bounded repack cache. U widens the shared physical batch and admission budget. V applies that same complete scheduling profile with F. N evaluates an independent non-flash split-KV stream configuration after a rejected flash probe. G adds the explicit PS1 and PK6 configuration to N. Each setting and generated library needs a separate model gate.

The threshold-only P measurements exposed the default-attention identity issue corrected in PR 70. They are excluded from selection. Preserve regression and fallback observations, including runs in which no candidate dispatch occurred. To restore the product reference, restart from the unmodified checkout and its original native binary. None of these research plugins is installed by default.

[Raw measurements, development order and comparison](../../autonomous/results/l40s-native-dispatch/)


## Pricing a candidate on this L40S

The operator screen measures five Q4_K and two Q6_K projections with two activation seeds. Its graph timings include dispatch and synchronization. The alternative-kernel arithmetic check also reports an error against sampled dequantized source channels. That check times repeated operations on one tensor, so its ranking is a reason to run the full server.

```bash
python -m lab.pk6.native_dispatch.replay_policy "$results/pilot" /tmp/pilot-holdout.json
python -m lab.pk6.native_dispatch.compact "$results/http/primary" /tmp/compact-pool.json
python -m lab.pk6.native_dispatch.build int8 /work/prompt-int8 "$model"
python -m lab.pk6.native_dispatch.build marlin /work/prompt-marlin "$model"
python -m lab.pk6.native_dispatch.build cached-int8 /work/prompt-cached-int8 "$model"
python -m lab.pk6.native_dispatch.grid_build /work/grid-calibration
LLAMA_CPP_LIB_PATH=/work/grid-calibration/bin python -m lab.pk6.native_dispatch.grid_screen \
  "$model" /work/grid-calibration /work/grid-measurements
```

The Q4_K grid screen varies the CUDA launch geometry and keeps the native quantization. Each alternate configuration must run all the timed graphs, agree numerically with stock and beat it by 10% on both activation seeds. Its output contains exact projection shapes and row counts. If the list is nonempty, build the proposed profile with `grid_build /work/grid-serving --plan /work/grid-measurements/plan.json` and qualify it with the full server.

The H process sizes its unified KV pool from the first stock workload. The individual context stays at 2,048 tokens and the native and scheduler capacities are recorded together. The J process retains packed INT8 feed-forward weights during one model's lifetime. L lets the frozen CPU head choose before server initialization. Its decision and CPU wall time are included in the target receipt.
