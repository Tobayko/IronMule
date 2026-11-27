# Qwen3-32B prompt dispatch on NVIDIA L40S

The study starts from PR 57 at `2e3c1bb69b8cedc2b889156e6dd0912bc1a6235d`. It tests real mixed prompt and decode calls on Ada. The production reference uses the existing CPU Pilot, native GGML and its local quality gate. The external process serves the official AWQ-Marlin artifact with vLLM 0.34.0. Both engines receive the same rendered token IDs, greedy decoding, 2,048-token contexts and a 32-token generation limit.

## Replay on CPU

From the repository root, run the following commands with the repository's Python dependencies. The HTTP replay checks the board, model and comparator digests, process windows, all native request IDs, seven streaming metrics, output counts, factual case identities and one-second device samples.

```bash
results="$(realpath lab/autonomous/results/l40s-native-dispatch)"
python -m lab.pk6.native_dispatch.score "$results/http" > /tmp/ironmule-comparison.json
python -m lab.pk6.native_dispatch.tile_order "$results/http/primary" /tmp/ironmule-native-order
```

The order uses stock HTTP and independent CUPTI traces to rank two GGUF types and their compiled tile sizes. The predicted savings assume a 10% reduction in the sampled device interval. The target and the Codex response are bound by digest. The build replaces one Q4_K translation unit using the target's own compilation database, then relinks an isolated CUDA backend. The tile experiment retains native shared-memory and shape validation. The baseline and candidate can also run on the same alternative binary with the candidate disabled or enabled.

### Archived vLLM targets

Each range covers six fixed-work waves on the recorded L40S. The ten-percent throughput column gives an engineering target at each concurrency. Replay `python -m lab.pk6.native_dispatch.reference` for the latency distributions and the original measurement times.

| Clients | vLLM output tokens/s, median (range) | 10% faster, tokens/s | Median time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 37.38 (32.17–37.53) | 41.12 | 0.047 s |
| 8 | 164.05 (163.64–165.33) | 180.45 | 0.716 s |
| 32 | 266.55 (265.08–269.20) | 293.20 | 1.709 s |

With the archive available, a native-only run reads the external target from its original receipts and checks the current IronMule model locally. The target manifest records which external waves supplied the comparison.

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
results="$(realpath lab/autonomous/results/l40s-native-dispatch)"
base=2e3c1bb69b8cedc2b889156e6dd0912bc1a6235d
git worktree add --detach /work/prompt-study "$base"
cp -R lab/pk6/native_dispatch /work/prompt-study/lab/pk6/
cd /work/prompt-study
export PYTHONPATH="$PWD"
python -m lab.pk6.native_dispatch.prepare .
bash lab/pk6/native_dispatch/serve.sh /work/http/primary \
  --reference-archive "$results" --lib /work/fp8/fp8.so --arms vllm R1 F1 F2 R2
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


The Y arm converts original Q4_K and Q6_K feed-forward matrices to FP16 once while a model owns the cache. Its GEMM accumulates into an FP32 destination. The device arithmetic check compares its output with independently dequantized rows. E runs automatic kernel generation from the unchanged PR 57 product checkout with the experimental CUDA flags cleared. Compare it with the explicitly selected stock R control to see what the ordinary launcher chooses. The fixed-work harness pins online admission learning during timing for every native arm.

To inspect startup, sampled energy and memory together with the streaming results, run `python -m lab.pk6.native_dispatch.verdict "$results" /tmp/native-verdict.json`. Where the late vLLM restart is available, the timing ratios must also hold against that process. The score retains the GGUF and AWQ artifact identities.

To restore the exact Python evaluator that produced a particular set of receipts, run the following command from a clone containing this PR and its base history. It verifies the content hashes before writing the detached checkout. The primary evaluator restores the engine SHA-256 recorded in its HTTP target manifest.

```bash
python -m lab.pk6.native_dispatch.materialize "$results/sources" primary /tmp/ironmule-primary
```

The optional draft experiment pairs the Qwen3-32B target with the official Qwen3-0.6B Q8_0 draft at revision `23749fefcc72300e3a2ad315e1317431b06b590a`. K permits the available draft model to propose first. O uses a repeated-context proposal of at least two tokens when available, then offers remaining eligible requests to the draft model. The large model verifies every proposed token. A trace records how many proposed rows it checked and accepted during measured requests.

To rank the remaining arithmetic candidates on CPU, run `python -m lab.pk6.native_dispatch.next_kernel "$results" /tmp/prompt-order`. For a bounded GPU run in the prepared checkout, supply a library from the matching arithmetic test, for example `--library marlin-check=/work/prompt_marlin.so --max-seconds 1800`. The selector checks the GPU, model evidence and native binary, then launches both seeds under the existing reservation. Pass multiple libraries to let the planner choose among them. Each run writes a new directory with its decision and receipts.

A separate vLLM control loads the identical 32B GGUF using the upstream out-of-tree GGUF plugin. The deployment comparison keeps its original AWQ-Marlin model. For the exact-weight control add `vllm-gguf-plugin==0.0.5` and `gguf==0.19.0` to an isolated Python path for vLLM 0.34.0. The x86-64 plugin wheel SHA-256 is `e795779830ad8907cb0e51bf661dd7784ebaf244b0b324af07a8d3925b6a9739`. When a decision needs a new exact-weight control, supply its reason with `--vllm-recheck-reason` to `python -m lab.pk6.native_dispatch.same_weights --out /work/same-weights --arms vllm`. The control uses the same tokenizer, target and reserved lease. [Upstream plugin](https://github.com/vllm-project/vllm-gguf-plugin)

#### Compact KV capacity and packed Q4_K

H1 reduced peak sampled memory during requests from 36,122 to 23,834 MiB. Both processes peaked at 45,160 MiB when startup and calibration were included. The first 32-client block measured 159.5 output tokens/s with H1, 155.2 with R1 and 266.5 with vLLM AWQ-Marlin. H1 answered all 32 factual requests correctly. Capacity trials use a separate archive because their 900-word documents change the timed workload.

The Marlin follow-up reserves 6 GiB beyond measured serving usage and limits the repack cache to the remaining whole GiB. A synchronized GGML screen compares disjoint weight buffers on two activation seeds. The generated header admits only measured widths whose complete graph used the plugin, had relative RMS error below 0.04 and cleared the 1.10 speed ratio twice. Each target model opens and closes its own packing session. W requires both its compiled program receipt and the compact-pool plan.

```bash
python -m lab.pk6.native_dispatch.compact "$results/http" /tmp/packed-memory.json --packed-memory
python -m lab.pk6.native_dispatch.packed_screen "$model" /work/prompt_marlin.so \
  "$results/marlin-check/result.json" /tmp/packed-memory.json /work/packed-screen
python -m lab.pk6.native_dispatch.build packed-marlin /work/packed-serving "$model" --policy /work/packed-screen
bash lab/pk6/native_dispatch/serve.sh /work/http/packed --arms W1 W2 \
  --lib /work/packed-serving/packed-marlin.so --routing-receipt /work/packed-serving/build.json \
  --pool-plan /tmp/compact-pool.json
```

#### Learning the size of a prefill chunk

In the first 32-client reference wave, all document requests reached the worker within 70 ms. The final document started prefill after 4.11 s. IronMule fitted its existing CPU step-cost model on native calls from R1. On the independent R2 restart, the mean absolute prediction error was 25.5 ms, versus 49.7 ms with the A100 parameters and 60.9 ms with a constant fitted to R1. Its simulation ranked six budgets and selected 768 rows. The simulation assumed simultaneous arrivals and cold prefixes.

Q allows already-admitted short answers to take one decode step alongside more prompt rows in the qualified unified KV pool. It preserves projected cell reservations for partially processed prompts. Each counted trial must show real chunk execution during performance and factual requests. The local SmolLM2 check uses 96 rows to exercise the same branch on a small model.

```bash
python -m lab.pk6.native_dispatch.chunk_plan "$results/http/primary" /tmp/chunk-plan.json
# Inside a fresh study checkout with its evaluator sources restored
python -m lab.pk6.native_dispatch.prepare . --shared-chunks
python -m lab.pk6.mixed_progress.pool_cpu /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf --chunk-rows 96
bash lab/pk6/native_dispatch/serve.sh /work/http/chunks --arms Q1 Q2 --policy /tmp/chunk-plan.json
```

#### Reuse the vLLM target

Run `python -m lab.pk6.native_dispatch.reference` on CPU before reserving a GPU. It reads the original streams and returns medians and observed ranges for one, eight and 32 clients. On this L40S the six 32-client waves ranged from 265.08 to 269.20 output tokens/s. The median was 266.55. A 10% lead over that median requires 293.20 tokens/s with the same work. Each latency metric has its own reference and spread.

The study runner checks the registered model, quantization, vLLM settings, workload, seeds and hardware before starting vLLM. A match writes `reused-vllm.json` with the original timestamps. A matching device class with a different physical fingerprint supplies a historical development target. Paired scoring requires the matching physical device and token digests. For an intentional recheck, record the concrete reason with `--vllm-recheck-reason`. A detached evaluator without the committed evidence can use `--reference-archive PATH`. The extra AWQ restart and optional same-GGUF control for this lease were both cancelled before either server started.

The initial grid selected seven shape/width combinations across two activation seeds. None of its selected widths appeared in the 289 outer calls during R1's timed performance requests. One appeared in the 15 calls during its factual wave. IronMule therefore skipped the two planned server starts and selected 21 widths from actual request windows. These cover 70.1% of the native time in R1's performance requests and 44.2% in its factual wave. The CUDA screen still measures full native graphs because their internal dispatch dimensions can differ from an outer call.

```bash
python -m lab.pk6.native_dispatch.workload_grid "$results/http/primary" \
  "$results/native-grid/plan.json" /tmp/workload-grid-order.json
LLAMA_CPP_LIB_PATH=/work/grid-calibration/bin python -m lab.pk6.native_dispatch.grid_screen \
  "$model" /work/grid-calibration /work/workload-measurements --workload-order /tmp/workload-grid-order.json
```

#### Memory-bounded W4 decoding

The ordinary generated-kernel startup ran out of GPU memory during calibration. IronMule
sized a four-geometry table from the stock serving footprint, retaining 128 measured
dispatch entries. With eight equal-length requests, the two CUDA worker pairs measured
143.8 vs 136.7 and 143.1 vs 135.2 output tokens/s. Each generated the same token IDs
as its paired reference and answered all 18 factual cases correctly. Single-client
throughput changed by less than 0.4% on both seeds. The candidate packed 2.44 GiB.

The S profile takes the selected table through the ordinary full-model quality gate and
HTTP workload. Its counter forwards to the same generated library and counts successful
native calls. CUDA graphs can replay previously captured calls. A successful serving
claim needs a dispatch witness during measured requests and the factual wave.

```bash
python -m lab.pk6.native_dispatch.w4_profile "$results/bounded-w4/followup" /tmp/w4-order.json
python -m lab.pk6.native_dispatch.w4_profile "$results/bounded-w4/followup" /work/bounded-profile \
  --library "$generated_kernel"
bash lab/pk6/native_dispatch/serve.sh /work/http/bounded --arms S1 S2 \
  --reference-archive "$results" --lib /work/bounded-profile/w4-count.so \
  --routing-receipt /work/bounded-profile/build.json
```

The build checks the original full calibration table beside `$generated_kernel`, the
selected rows, the generating library, the Qwen3-32B model digest and the physical board.

#### Let the draft model work at a higher load

With the existing two draft slots, K1 reached 55.87 output tokens/s at one client.
The target verified 122 proposed tokens over those three waves. It verified 20
proposed tokens across the eight-client waves, where throughput reached 112.84
tokens/s. The two-slot drafter stops when more than two requests are active.

IronMule used those observations to propose four slots and a limit of eight active
requests. The draft GGUF metadata puts the extra FP16 KV storage at 448 MiB. In the
C experiment, requests that already own a slot keep it. Among requests waiting for
a slot, the new scheduler favors those requiring the fewest new draft prompt tokens.
The target still verifies every proposed token. The trace records the active request
count at verification. A C run requires evidence that the higher limit actually served.

```bash
python -m lab.pk6.native_dispatch.draft_load "$results/http" "$results/draft-metadata.json" /tmp/draft-load.json
study="$(pwd)"
python -m lab.pk6.native_dispatch.materialize "$results/sources" draft-mix-load /work/replay-draft-load
cd /work/replay-draft-load
export PYTHONPATH="$PWD"
bash "$study/lab/pk6/native_dispatch/serve.sh" /work/http/draft-load --arms C1 C2 \
  --reference-archive "$results" --policy /tmp/draft-load.json
```

#### Replaying the long-document tests

The capacity workload sends 900-word inventories, with a mix of shared and distinct prefixes. Each
process serves the same workload at one, eight and 32 clients, then answers 32 short factual
questions. A separate quality workload asks for an exact six-digit seal embedded near the
beginning, middle or end of a document. The other two seals are distractors. It checks two new
sets of 32 documents and repeats the short factual test after the long ones.

```bash
python -m lab.pk6.native_dispatch.score "$results/controls/capacity" > /tmp/capacity.json
python -m lab.pk6.native_dispatch.document_quality assess "$results/controls/document-retrieval" /tmp/retrieval.json
python -m lab.pk6.native_dispatch.compact "$results/controls/capacity" /tmp/phases.json --explain-pressure
python -m lab.pk6.native_dispatch.host_load "$results" "$results/host-samples.jsonl.gz" /tmp/host-load.json
```

The host-load replay joins five-second CPU observations to the actual request windows. Its
sample count indicates which processes ran after that observer started. The vLLM archive covers
the earlier mixed workload. The long-document comparison uses IronMule's original physical KV pool.

#### Learning a pool profile from served traffic

`python -m lab.pk6.native_dispatch.pool_pilot fit "$results" /tmp/pool-pilot` trains three small
cost heads from random weights on CPU. It uses the same Pilot inference function as the native
operator study. All repetitions of seed 943819 train together. The six workload/concurrency
cells from seed 180341 evaluate the fitted choice. The report also scores the untrained heads,
a stock rule and a simple document-length rule. The independent reverse-order comparison and
the retrieval test determine which choices may serve. A width needs the existing ten-percent
criterion on every streaming metric in both long-document runs. Each correct reference
retrieval must also survive, with examples at all three document positions and all 32 short
factual answers correct. Unknown hardware or traffic outside the measured length range selects
stock. The decisions apply at process startup for a workload observed on that target.

On the qualified board, the bounded restart reads three waves of actual traffic before opening the
server. Run from the restored pool evaluator. Supply the hardware reservation in `RP_DEADLINE`, the
matching `LLAMA_CPP_LIB_PATH` and the absolute paths produced above. The original stock history for
the reverse-order long-document comparison is under the following path in the results archive.

```bash
history="$results/controls/capacity/capacity-confirm/R1/waves-R1.jsonl"
python -m lab.pk6.native_dispatch.pool_pilot serve \
  --policy /tmp/pool-pilot --pool-plan "$results/compact-pool.json" \
  --reference-archive "$results" --history "$history" --width 32 --seed-index 1 \
  --state /work/pool-deployment --workload stress --out /work/first-pool-restart
python -m lab.pk6.native_dispatch.pool_pilot withdraw /work/pool-deployment --reason 'rollback drill'
python -m lab.pk6.native_dispatch.pool_pilot serve \
  --policy /tmp/pool-pilot --pool-plan "$results/compact-pool.json" \
  --reference-archive "$results" --history "$history" --width 32 --seed-index 1 \
  --state /work/pool-deployment --workload stress --out /work/restored-pool-restart
```

The observed prompt lengths and concurrency are the model inputs. The selected server must pass its
own model gate and serve the fixed requests. Verification charges the full CPU fit, startup decision
and verification to the three 32-client waves. Model loading and server startup are reported
separately. A regression, incomplete trial or explicit withdrawal selects stock at the next start.

The CPU kernel planner reads the stored vLLM target when it prices the gap. At 32 clients the
six original IronMule waves need a median 3.13 seconds less wall time to reach 293.20 output
tokens/s. If everything outside native mixed prompt/decode calls stayed fixed, those calls
would need to run 2.23–3.11 times as fast, depending on the wave. The estimate uses each wave's
completed tokens and its request-bound native trace. [Per-wave calculation](../../autonomous/results/l40s-native-dispatch/mixed-time-target.json)
