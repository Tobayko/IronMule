# Handover: Qwen3-32B on one A100 against vLLM (+10 % at 32 clients)

State of 2026-11-25, branch `claude/blissful-pasteur-2uj5by`. This file is meant to let a developer or agent continue
without the chat that produced the work. Measurement details live in [LAB.md](LAB.md); this file routes to them.

## Subsequent CPU Pilot work on 2026-11-26

The original record below describes the warm HTTP comparison. A subsequent same-model
campaign measured repeated worker startup on a new A100 SXM4 80 GB. Its CPU Pilot selected
a complete, locally qualified W4 configuration. The source change reduced median first mixed
batch delivery from 33.292 to 15.360 seconds on six matched pairs. PR 60 qualified the source,
and PR 61 merged the same tree into this A100 branch. The release archive includes raw answers,
fresh target activation, withdrawal, a failed fallback calibration and successful reference
service after restart. [Hardware scope](../training/optimization-hardware.md#a100-restart-support) ·
[Training, source experiment and activation](../lab/pk1/restart-trial.md) ·
[Original HTTP result replay](../lab/pk6/replay-http.md)

The cache-replay search family for PR 60 came from the developers. IronMule issued the
order from two matching device observations, Codex implemented it, and the CPU Pilot
learned when to select the qualified source. In a subsequent search over observed startup
methods and ten sampling plans, IronMule selected a different experiment from the data.
Two blind full starts supported its forecast. Codex implemented the resulting order.
On the A100, the new code reduced native calls from 96 to 64 in the instrumented screen,
but won only four of six uninstrumented mixed pairs. The six batches together took
104.491 seconds, compared with 100.533 seconds on the PR 60 reference. All paired outputs
matched. The evaluator rejected the program and recorded the decision for the next run.
The scoped PR 60 source stayed selected. [Independent target accounting](../lab/autonomous/results/a100-calibration-sampling/target-accounting.json) ·
[Durable planning outcome](../lab/autonomous/results/a100-calibration-sampling/post-terminal-replan.json)

To refit the released startup head on the CPU, reproduce its held-out decisions and verify the two
source verdicts in a Linux source checkout with the repository dependencies installed, run

```sh
records=lab/autonomous/results
parent="$records/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3"
child="$records/a100-calibration-sampling/trial-4e87c90dad8d9f21db28a69501613f44a98b9399b1b5838316f276978de288e7"
OPENBLAS_NUM_THREADS=1 python -m training.replay_qualified_restart "$parent"
python -m training.sampling_evidence replay "$child"
```

The [bounded target commands](../lab/pk1/restart-trial.md) describe how to supply a new
reserved device, a pinned checkout, the Codex author and a persistent campaign directory.

The four-core EPYC serving experiment uses the Pilot's larger, 20,232-parameter thread
head. These commands refit it from the recorded CPU trials and compare the result with the archived weights.

```sh
out=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 python -m training.pilot_threads fit \
  --trials evidence/runtime/cpu-pilot-epyc-trials.json \
  --weights "$out/thread-head.npz" --report "$out/heldout.json"
cmp "$out/thread-head.npz" evidence/runtime/cpu-pilot-epyc-candidate.npz
```

## Result and validity domain

- **+10.02 % output throughput over vLLM at 32 concurrent clients** by the rule fixed before the run: median tok/s
  over all six 32-client waves of two fresh IronMule server starts (first waves included) against vLLM's median on
  the same pod. IronMule 304.6, vLLM 0.11.0 (AWQ-Marlin) 276.9 tok/s, ratio 1.1002
  ([pod 37](LAB.md#2026-11-25-a100-pod-37-pk6-kept----3046-against-2769-toks-at-32-clients-1002--the-10--reached-by-its-rule)).
- **The margin is nil.** Gap to the line -0.3 ms a median wave. Per server start 1.105 and 1.095. One run, not
  repeated (the session rule allowed one). Treat the +10 % as reached once, not as robust.
- Domain: one RunPod A100 SXM4 80 GB (secure cloud, US-WA-1, 32 vCPU), CUDA 12.8 image
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`; Qwen3-32B Q4_K_M GGUF (IronMule) against the
  official Qwen3-32B-AWQ (vLLM); the HTTP benchmark `lab/pk1/http_bench.py` (32 clients, 16 short prompts of about 33
  tokens and 16 documents of about 444, 32-token answers, greedy). Other GPUs, models, widths or workloads are not
  covered. Pods differ by several percent for both engines: only same-pod comparisons count.
- Same run, other figures: 8 clients IronMule 195.6 against vLLM 189.4 tok/s; first-token p50 at 32 clients
  IronMule 2,321 ms, vLLM 1,823 ms (vLLM hands out first tokens earlier). Factual check 18/18 in every run of both.
  Against the previous IronMule default on the same pod (PS1 on, PK6 off): +0.9 % at 32, +1.0 % at 8 clients.
- Reference correction: pod 36's "R" had PS1 **off**, pod 37's "R" had PS1 **on**. The comparable previous default
  is pod 36's S (1.092 against vLLM). Pod 36 R's 1.056 and pod 37 R's 1.090 are different configurations, not a
  measure of pod-to-pod spread.

## Current default build

Active by default (the configuration measured as K on pod 37):

| switch | default | what it does | off |
| --- | --- | --- | --- |
| PS1 | on | one native pass for a packed prefill (rectangular ubatches merged; attention per rectangle) | `IRONMULE_PS1=0` |
| ps1a | always built | ggml-alloc allocates PS1's per-layer attention output at its first reader (removes the 5 -> 18.7 GB compute-buffer growth and its 0.38 s reallocations) | patch set; no switch |
| PK6 | on | prefill: SwiGLU and fused add + RMS norm skip f32 writes only fp16-copy matmuls would read | `IRONMULE_PK6=0` |
| PS1 warm-up | off (`0`) | start-up merged prefill; no longer needed with ps1a | `IRONMULE_PS1_WARM=2` turns it on |
| DH2 | off | token delivery on a thread during the next native call; no measured gain | `IRONMULE_GGML_DH2=1` |
| SV5 | off | a join's requests rendered as one batch; -15.7 ms in 5/8 rounds, failed the rule | `IRONMULE_SV5=1` |

Other defaults that the measured build relied on (from the code, `ironmule/ggml_engine.py` unless named):

| switch | default | note |
| --- | --- | --- |
| `IRONMULE_ADM1`, `IRONMULE_ADM2` (`ironmule_product/http_server.py`, `serving_policy.py`) | on | admission without file reads; logs written behind (pod 34) |
| `IRONMULE_GGML_PX1`, `IRONMULE_GGML_PX2` | on | kept-prefix evening and its measured cost decision (PS1 keeps every prefix) |
| `IRONMULE_GGML_BF1` | on | numpy batch fill for calls of 64+ rows |
| `IRONMULE_GGML_PREFIX_REUSE`, `IRONMULE_GGML_COLLECTIVE`, `IRONMULE_PILOT` | on | prefix reuse; speculation; the per-step decision network |
| `IRONMULE_PK2` / `IRONMULE_CUBLAS_MIN_ROWS` | auto (engine measures; 128 on A100) | dequantise + cuBLAS from that many rows |
| `IRONMULE_W4` | on | W4R decode kernels, tuned at the first start |
| `IRONMULE_PK3`, `IRONMULE_PK4`, `IRONMULE_PK5`, `IRONMULE_FU1`, `IRONMULE_GR3`, `IRONMULE_GR4` (native) | on | fp16 copy reuse / producer-written copy / fusions / CUDA graphs |
| `IRONMULE_GGML_PK1C`, `IRONMULE_GGML_EM1`, `IRONMULE_KG_QKV` | off | experiments kept opt-in |
| `IRONMULE_TL1`, `IRONMULE_TL2`, `IRONMULE_PK6_LOG`, `IRONMULE_PS1_LOG`, `IRONMULE_GGML_TRACE` | unset | diagnostics only |
| `IRONMULE_LAB_SWITCH` | unset | JSON file of switches flipped inside one server (paired A/B); keys ADM1, ADM2, PX2, BF1, PS1, PK6, SV5, DH2 |

## Where the code is

- `ironmule/ggml_engine.py`: the GGUF engine on llama.cpp (packed prefill, PX1/PX2, BF1, PS1/PK6 switches, DH2,
  PS1 warm-up `ps1_warm_up`, pass-cost measurement).
- `ironmule_product/worker.py` (worker process, rendering incl. SV4/SV5, joins), `backend.py` (worker protocol),
  `service.py` (scheduler, groups, joins), `http_server.py`, `serving_policy.py` (ADM1/ADM2), `lab_switch.py`.
- `tools/build_ggml.py`: fetches llama-cpp-python 0.3.16 (llama.cpp 4227c9b, 2025-08-14), applies the patch sets in
  order and builds. Base (CPU and CUDA): `fa1 row1 am1 tk1 tl1 ps1 tl2 ps1a`; CUDA adds
  `gr1 kg1 kg5 pk2 gr2 pk3 gr3 fu1 pk4 pk5 em1 gr4 qk1 pd1 tl2c pk6`.
- Patch sets of this work (`native/ggml_patches/`):
  - `ps1/`: llama-batch.h, llama-kv-cache-unified.{h,cpp}, llama-graph.{h,cpp}, llama-context.cpp; merge of up to 16
    rectangles, per-segment masks and K/V views, FA outputs as views of one tensor, concat fallback
    (`IRONMULE_PS1_CONCAT=1`), graph capacity 24 nodes a tensor.
  - `ps1a/ggml-alloc.c.diff`: leafs named `ps1_all-<layer>` are allocated at their first reader.
  - `pk6/ggml-cuda.cu.diff`: reader index per graph, skip decision, abort on a PK3 miss of a skipped tensor.
  - `tl2/`, `tl2c/`: diagnostics (graph build vs allocation, CPU time, buffer and CUDA pool growth with host time).
- Kernel library for decode: `native/kernels/w4mma.cu`, generated variants `ironmule/kernel_gen.py`.
- Tests touched: `tests/engine/test_build_ggml.py`, `test_ggml_devices.py`, `test_ggml_engine.py` (defaults, PX/PS1,
  DH2), `test_lab_switch.py`, `test_product_worker.py` (SV5).

## Reproduction

Pod setup (as in every session; nothing on the pod is stored in the repository):

```sh
# on a fresh A100 pod with the image above, the repository at /work/IronMule
JOBS=28 bash lab/rp/setup.sh 80     # pip install, models (pinned SHA-256), tokenizer, vLLM 0.11.0 venv, CUDA build
```

`lab/rp/setup.sh` downloads `Qwen/Qwen3-32B-GGUF` Q4_K_M (revision pinned in `lab/portable_gpu/fetch.py`), the AWQ
checkpoint (`lab/portable_gpu/fetch_awq.py`) and the Qwen3-32B tokenizer, installs `vllm==0.11.0
transformers==4.57.1` in `/opt/vllm`, and runs `tools/build_ggml.py` for compute capability 80. Weights stay on the
pod. Pod control from a workstation: `lab/rp/rp.py` (`put/get/run/job/log/kill`) with the token agent `lab/rp/agent.py`.

Server and benchmark (the first start runs the quality gate and W4R tuning, about 20 minutes; later starts 85 s):

```sh
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 python -m ironmule_cli serve \
  --gguf /models/Qwen3-32B-Q4_K_M.gguf --state-dir /work/st --agent on --no-thinking \
  --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002
python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
  --widths 32 --repeats 3 --label K1 --timeline tl_K1_32.jsonl > http_K1_32.jsonl
python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
  --widths 8 --repeats 2 --label K1q --quality > http_K1q_8.jsonl
```

The whole session as run (engine check, paired waves, fresh starts, vLLM, verdict): `bash lab/pk6/a100_pk6.sh e`,
then `d`, then `h` (rules in the file header; verdict `python lab/adm2/verdict.py final DIR --final K --ref R`).

Local tests (no GPU; CPU build of MLX and llama.cpp):

```sh
pip install -e ".[cpu,dev]"
python -m pytest -m "not integration"
python -m pytest tests/engine/test_build_ggml.py tests/engine/test_ggml_engine.py tests/engine/test_lab_switch.py tests/engine/test_product_worker.py
```

The full non-integration suite passed locally at c33a34a's code (exit 0). PK6 is CUDA-only and is compiled and checked only
on a GPU pod (`lab/ps1/check.py` with `IRONMULE_PK6=0/1`, logits must be equal).

## Evidence

| step | LAB entry | scripts | raw data | commits |
| --- | --- | --- | --- | --- |
| ADM1/ADM2/BF1/PX2, +6.4 % | [pod 34](LAB.md#2026-11-25-a100-pod-34-adm1-and-adm2-kept----3121-against-2934-toks-at-32-clients-64--10--not-reached) | `lab/adm1/`, `lab/adm2/` | `lab/pk1/results/pod34/` | d371f98, 11f00e4 |
| PS1 / DH2 first measured, +7.7 % | [pod 35](LAB.md#2026-11-25-a100-pod-35-ps1-measured--90-ms-a-wave-in-one-process-outliers-over-http-rejected-by-the-rule-dh2-no-gain-77--against-vllm) | `lab/ps1/` | `lab/pk1/results/pod35/` | e9c7b12, f507892 |
| slow PS1 prefills attributed, ps1a, +9.2 % | [before 36](LAB.md#2026-11-25-before-pod-36-what-the-slow-ps1-prefills-have-in-common-tl2-and-the-start-up-warm-up-of-the-full-micro-batch), [pod 36](LAB.md#2026-11-25-a100-pod-36-ps1s-slow-prefills-attributed-and-removed-ps1a-ps1-on-by-default-92--against-vllm) | `lab/ps1/a100_ps1b.sh`, `growth.py`, `d1_report.py`, `slow_calls.py` | `lab/pk1/results/pod36/` | cdd45ad, 605bfef, 02b285b |
| PK6 / SV5, +10.02 % | [before 37](LAB.md#2026-11-25-before-pod-37-what-sets-the-median-wave-and-two-small-levers-pk6-sv5), [pod 37](LAB.md#2026-11-25-a100-pod-37-pk6-kept----3046-against-2769-toks-at-32-clients-1002--the-10--reached-by-its-rule) | `lab/pk6/a100_pk6.sh`, `engine_ab.py` | `lab/pk1/results/pod37/` | 0d6d69d, c33a34a |

Pod 37's logits files of the GPU check (2 x 19 MB) are not in the repository; their comparison (max |difference|
0.0) is in `lab/pk1/results/pod37/pk6_e.log`.

## Open points

1. **No robust margin.** ([#53](https://github.com/Tobayko/IronMule/issues/53)) 1.1002 with starts at 1.105 and 1.095. A repeat on another pod may land below 1.10. The
   next lever must bring tens of milliseconds a wave, measured with the same rule.
2. **Slower first wave after a start.** ([#54](https://github.com/Tobayko/IronMule/issues/54)) 105 ms over the target line in pod 37 (admission ends at 51-60 ms instead
   of 25-30; 16-24 ms more between the batch decision and the prefill call; prompt rows without a kept prefix). It
   does not set the median today but costs margin.
3. **Decode step.** ([#56](https://github.com/Tobayko/IronMule/issues/56)) 31 calls of 31.3 ms against vLLM's about 23 ms; the matmul kernels at 32 rows (gate/up, down)
   are the known lever (pod 28 profile).
4. **PK6 cache miss.** ([#55](https://github.com/Tobayko/IronMule/issues/55)) If a reader of a skipped tensor misses PK3's fp16 copy, the process aborts
   (`GGML_ABORT("PK6: ...")`) instead of computing on unwritten memory: correct but it ends the worker. It did not
   happen on pod 37 (189 skips a graph, no abort). A reader in another graph split would not be seen by the index;
   with full GPU offload the model is one split. A fallback that recomputes the f32 values (or disables PK6 for the
   graph) is not implemented.
5. **Wave end.** 28-48 ms after the last decode call until the last response (32 done frames handled about one a
   millisecond, mostly thread hand-offs); no small fix found.

## Approaches already tried (do not repeat unchanged)

- Allocator growth by 1.5x for PS1 (`ggml-alloc.c`): worker SIGSEGV at the first request (pod 35 run 2), withdrawn.
- PS1 start-up warm-up (one merged call of the full micro-batch): removed the slow calls but the 18.7 GB buffer made
  the start-up gate's W4R tuning fail with out_of_memory (pod 36); superseded by ps1a.
- Chaining PS1's segment outputs only (views releasing `all`): no effect on the buffer growth (pod 36 D1); the cause
  was the leaf pre-allocation.
- `CUDA_MODULE_LOADING=EAGER`: no effect on the slow prefills (pod 36).
- DH2 (token delivery off the critical path): no gain (pod 35).
- SV5 (batched rendering of joins): -15.7 ms in 5 of 8 paired rounds, failed the rule (pod 37).
- Speculative decoding at 32 clients: no reserve (verification of 64 rows 1.86 x a 32-row step; pod 17).
- QK1 grouped q/k/v at 32 rows: at most 0.4 ms a step (pod 28), opt-in.
- A cuBLAS algorithm / workspace choice for the prefill GEMMs: estimated 30-60 ms (pod 32), never measured.

## Resource status (last confirmed)

- Pod 36 (`o3mzkxnw6v9kwv`) and pod 37 (`p7w3phuua2ih9q`) deleted; `get-pod` returned "not found" after each delete.
- Local watchdog scripts stopped and the scheduled reminders deleted after each pod.
- Cost per pod from the session records: pod 36 about 1.45 h, about 3.2 USD; pod 37 about 0.62 h, 1.36 USD.
  Account balances are not given here: they were not re-checked for this handover.
