# Roadmap: from a research project to a product

Recorded 2026-11-15 at the owner's request, after a review of the repository (code, tests, documents and the
benchmarks in [LAB.md](LAB.md)). The marks were given before the suites were run; the audit below ran them. This is a plan and an assessment. Nothing
here is a measured result unless it says so, and nothing here changes current behaviour.

**What "done" means.** A stranger installs IronMule with one command, repeats the README's numbers on their own
machine with one command, and uses it from the client they already have.

## Read this first: the review looked at an older main

The review, the audit and the micro-benchmarks below were made on a checkout of `main` at `2e72c17` (2026-11-15).
`main` moved 47 commits while they were written; it was merged into this branch on 2026-11-16. Checked against
`main` at `821fe50`:

| Point below | State on main |
| :-- | :-- |
| The two searches in the engine loop that grow with the context (Phase 4, step 1) | Done as ST1 (LAB.md): an n-gram index per request and a prefix trie in the shared pool, tokens identical. The micro-benchmarks below describe the old code |
| A GPU kernel for 4-bit weights (Phase 4, step 3) | Done as KG2 (LAB.md): generated kernels, one request on a T4 1.27x, on by default on GPUs. vLLM AWQ's 63.7 tokens per second (MG2 v6) is the number still to reach |
| Is the CPU bound by memory bandwidth (Phase 4, step 4) | Answered as ST2 (LAB.md) |
| The Pilot's decision time | 0.17-0.19 ms in the running server since it keeps `/proc` open (0.39 ms before) |
| The 14 failing tests | Fixed on main as well; main's version of those files is kept here. A Linux CI job now runs `tests/engine`, `tests/agent` and the claims files |
| "One flagship" (Phase 1) | The owner's goal of 2026-11-16 (BACKLOG, "The main goal"): IronMule decides on the CPU, the served model's main target is the GPU, and the CPU serving path stays |

Since then (2026-11-17):

- **macOS is out for now** (owner decision, AGENTS.md). CI has no macOS job; every one had been cancelled at its
  start since 2026-11-15, so the Rust crates and the wheel had not been checked for days. CI on Linux now also runs
  the Rust crates, the wheel in a clean environment, the quick start, and `tests/learning`, `tests/runtime` and
  `tests/evidence`. Not covered until macOS returns: Metal and the real-model suite on Apple Silicon.
- **The README rows of the audit's claims table are corrected**: what the Pilot changes on a GPU, the "one GPU"
  label, and the 32-slot caution under the top table. `--gguf`'s help was already corrected on main.
- **On main meanwhile** (LAB.md): the draft model on a T4, one request 1.23x (DM2); KV defragmentation, no gain
  (KVD1); one KV stream per request, 1.75-1.87x the pool at 32 requests at once and half with arrivals, in work
  (KG3).

Still open: requests held until a run goes idle, sampling, the one-command install, CI for the GGUF engine with a
real model, and vLLM AWQ's lead at 8 and 32 requests on one GPU.

## Where the project stands

| Area | Mark (of 10) | Why |
| :-- | --: | :-- |
| Measurement | 8 | Every number names its machine, model and comparison; losses are written down (vLLM AWQ ahead, GR1 no gain); answers are checked against a reference. Many headline numbers are one run per engine. |
| Engineering | 7 | The shared prefix pool (SP1), PAD1 and the Pilot under foreign load are own ideas with a measured effect. The loop that drives them is Python, with two searches that grow with the context. |
| Tests | 6 | About 1,700 tests, but the GGUF engine behind the headline numbers does not run against a real model in CI: CI is macOS only, and those tests skip without `IRONMULE_TEST_GGUF` and llama.cpp's bindings. |
| Differentiation | 6 | A real niche: CPU servers with many users and foreign load. On one GPU vLLM AWQ leads. The foreign-load multiplier compares against llama-server at its default thread count. |
| Product | 4 | Greedy decoding only (`temperature`, `top_p`, `stop`, tools and logprobs are not implemented), `llama-cpp-python` is not a declared dependency, and the fastest path is not one command away. |
| Focus | 4 | Three backends (MLX, MLX on CUDA, ggml), four Rust crates outside the hot path, an evidence framework of about 9,700 lines and a backlog of about 6,900 lines, with one maintainer. |

Overall 6 of 10. The largest risk is breadth, not speed: the lead rests on patches against a fast-moving
llama.cpp that one person rebases.

## Measured on the way (2026-11-15)

Micro-benchmarks of the Python side of one engine step, on a 4-vCPU cloud container (Intel Xeon at 2.1 GHz,
Python 3.13, numpy 2.5), best of 3 repeats. They time single functions on synthetic token lists, not the server;
an end-to-end gain from fixing them is not measured yet.

| What | Cost |
| :-- | --: |
| `speculation.draft()` with no match, context of 300 / 2,000 / 8,000 tokens | 0.06 / 0.40 / 1.62 ms per request and step |
| `GgmlEngine._common`, 2,000 shared tokens against 32 pool entries | 2.63 ms per waiting request (the loop runs it in `run()` and again in `SharedPool.place`) |
| the same comparison with numpy | 0.08 ms |
| filling the ctypes batch for a 2,000-token prompt | 0.56 ms |
| argmax over 151,000 logits | 11 microseconds per row |
| the Pilot's network in numpy / reading `/proc` | 5 / 26 microseconds |

Read: the Pilot and the argmax cost nothing worth moving. The n-gram search and the prefix comparison are the two
places where Python's time grows with the context and the pool.

## Audit: targets not met and outliers (2026-11-15)

Found by reading the README against [LAB.md](LAB.md) and [BACKLOG.md](BACKLOG.md), and by running the suites
(`pip install -e ".[cpu,dev]"` on the container above, Linux, no GPU, no model). The failing tests are dealt with
(see Cleanup below); the claims are not corrected yet.

**Tests.** `tests/engine tests/agent tests/learning tests/runtime tests/evidence tests/claims -m "not integration"`:
2,219 passed, 80 skipped, 14 failed. CI runs only `tests/engine`, `tests/agent` and two claims files, so none of
the 14 is seen there.

| Failing | Count | Cause |
| :-- | --: | :-- |
| `tests/evidence/test_friday_evidence.py` | 7 | `experiments/legacy_h1h2_summaries_v1.json` is not in this repository; `friday_evidence/registry.py` still points at it, so the legacy import is broken here as well |
| `tests/runtime/test_kernel_registry.py` | 4 | needs Metal, has no platform skip |
| `tests/runtime/test_wired_fraction_native_memsize.py` | 2 | macOS `sysctl` behaviour, has no platform skip |
| `tests/learning/test_requalification.py` | 1 | needs the cached Gemma 3 4B snapshot, is not marked `integration` |

**Claims against the project's own measurements.**

| Claim | Where | What the measurements say |
| :-- | :-- | :-- |
| The agent "decides in microseconds" | AGENTS.md, AGENT.md | 0.38-0.5 ms per step in the running server (PI1). In isolation here: 30 microseconds, 80 with a `/proc` read on every call. The README's reason (reading `/proc`) explains about 50 microseconds, not 0.4 ms; the rest is unexplained. |
| The Pilot "never changes what the model answers" | README, `pilot.py` | Shown for threads on a CPU (4, 3, 2, 1). On a GPU the Pilot's draft switch and running cap change the rows of a step, and LAB.md (MG2 v6, SG1 v3) records that multi-row CUDA kernels round differently: 47-50 of 50 answers the same, long answers differ. |
| "Without a draft model it is level with llama.cpp" (one request, T4) | README | Level on one repeated prompt (SG1 v2). On four different prompts 8 % behind: 43.5 against 47.3 (SG1 v3). |
| 1.09x llama-server with its own draft model | README | The same binary and setting differs by 5-11 % from run to run on that GPU (SG1: 56.6 and 53.8; 29.7 and 26.7), so 1.09x is inside the spread. |
| One user: 10.1 against 8.0 tokens per second; first word 0.33 s against 113 s | README, top table | CMP1 ran `llama-server` with 32 slots, which pays for every slot in each step (CMP1's own caution). Sized to the load it was level for one request (ROW1 round 5: 10.4 against 9.6-10.0) and made 27.3 / 40.9 against 50.4 / 71.3 at 8 / 16. The README does not carry that caution. Between sessions the same one-user numbers moved about 10 % (IronMule 10.1 and 9.33, llama-server 8.0 and 7.23). |
| "8 requests, one GPU: best llama.cpp server setup" | README, GPU table | The best llama-server arm used both GPUs (layer split or one server per GPU). The label is wrong, to IronMule's disadvantage. |
| Quality at least llama.cpp's (ratio at most 1.000; 1.005 for llama.cpp's own options) | BACKLOG, main goal | FA1x on Gemma 3 1B: 1.0024 [1.0004, 1.0044]. It passes 1.005, but FA1x is IronMule's kernel, and the interval lies wholly above 1.000. |
| Tokens per second come first | BACKLOG, main goal | The shipped Pilot (v4 seed 10) was picked on reward: 52,253 dreamed tokens against 52,650 for the weights before and 52,708 for seed 9 (0.8 % fewer), with less waiting. On the real machine it was not slower. |
| `--gguf` is "CPU only" / "on a machine without a GPU" | `cli.py` help, HTTP.md | The README and MG2 serve GGUFs on GPUs (`--gpus all --gguf`). |

**Read from the code, not reproduced.** One `GgmlEngine.run()` lasts as long as requests keep arriving. Its
`served` list, the bridge's `arrived` map and the worker's `streams` keep every request until the run goes idle,
and the agent's traffic memory (`observe_request`) is fed only then. Under load without a gap, memory grows and
the agent learns nothing new.

**This document's own error, corrected above.** The first version proposed measuring Q8_0 against Q4_K_M on the
CPU; ROW1 round 4 had already done so (Q4_K_M is slower there).

## Cleanup (2026-11-16)

Owner: remove what is not needed, do not archive it. Done on that day:

- 17 files in `docs/assets` that no file referenced (5.8 MB), and the Qwen3.6 routing figure that no document
  showed, with its evidence file and its function in `tools/make_figures.py` (18 figures remain and match).
- `tools/gg0_probe.py` and `tools/cs1_probe.py`, which nothing referenced.
- [BACKLOG.md](BACKLOG.md): 6,944 lines to about 690. The goal, the open work and Tier 0 stay; every answered
  entry is one line.
- The 14 failing tests now skip with their reason where their precondition is absent (Metal, macOS, the
  research repository's legacy file) or are marked `integration`; after the merge of main its own fixes of the
  same files are kept: 2,300 passed, 88 skipped, none failed. CI still
  does not run `tests/learning`, `tests/runtime` and `tests/evidence`.

Not removed, because tests or AGENTS.md hold them; each is a decision for the owner:

| What | Size | Why it may go |
| :-- | :-- | :-- |
| `tools/product_*.py`, their specs `docs/PROD*.md` and their tests | 5,232 + 805 + 1,158 lines | Harnesses of the September product validation on the MLX path; four read raw files that are not in this repository, and `product_bounded_bench.py` hashes a `BACKLOG.md` at the root that does not exist |
| `tools/cuda_probe.py`, `kaggle_whole_machine.py`, `tpu_probe.py`, `tpu_step_probe.py` and their tests | 4,509 + 2,761 lines | Kaggle harnesses of closed entries (HW3, KB1-KB3, TP1) |
| `ironmule/adaptive.py`, `ironmule/opt_adapter.py` | 865 lines | Imported by no product module, only by their tests |
| `native/optimizer` | 7,259 lines of Rust | Used through `live_learning.py` and `service.py`; outside the path the README's numbers come from |
| The MLX serving engines beside the GGUF engine (`tuned`, `dense`, `rust2`) | most of `ironmule/` | Phase 1's decision: one flagship |

## The phases

Phases 1 to 3 come first. More speed alone moves the project by about one mark; usable and checkable moves it
further.

### Phase 1: focus

- One flagship: GGUF on CPUs and GPUs. MLX becomes an optional extra, and `mlx` leaves the required dependencies.
- Split the backlog: closed and rejected entries go to the research repository; the open ones stay here.
- Each of the four Rust crates either takes a job in the hot path or moves to the research repository.

### Phase 2: usable

- Sampling (`temperature`, `top_p`, `seed`), `stop`, and JSON output through grammars.
- One-command install: a wheel or container image with llama.cpp built with IronMule's patches.
- A compatibility test in CI: the OpenAI client and a few common tools against the server.

Bar: from install to the first answer in under five minutes without reading the documents.

### Phase 3: checkable

- A Linux CI job with a small GGUF, so the engine behind the headline numbers runs against a real model on every
  commit.
- Fuzz tests for the prefix pool's reference counts and for speculation: the same tokens with and without them.
- `ironmule bench`: one command that compares IronMule and llama-server on the user's machine and prints the
  README's table, with several runs and their spread.
- A fair arm under foreign load: llama-server with its thread count set by hand.

Bar: every number in the README can be repeated with one command.

### Phase 4: faster

1. Algorithms in the loop (tokens stay identical): an incremental n-gram index per request in
   `speculation.draft()`, and numpy or a radix tree for the prefix comparison, scanned once per request.
2. The engine loop as a small C++ shim against `libllama`: no ctypes, no `llama-cpp-python`, no contention with
   the HTTP threads, and llama.cpp's sampler chain for Phase 2.
3. A GPU kernel on the Marlin principle (4-bit weights, FP16 arithmetic on tensor cores, arXiv 2408.11743) for
   the quantized matrix products, as a ggml patch beside `native/ggml_patches`. Kill criterion: one request on a
   T4 gains less than 15 %.
4. CPU: 4-bit formats. Already measured (LAB.md, ROW1 round 4): on this CPU Q4_K_M is compute-bound and slower
   than Q8_0 for one request (13.5 against 20 tokens per second) although it reads fewer bytes, and both 4-bit
   formats are worse at more rows. The open work there is the kernel for q4_K and the Q6_K output matrix, not a
   switch of format.

Bar: level with vLLM AWQ for one request on a T4, and still clearly ahead of llama.cpp for many requests on a CPU.

Not planned: a rewrite in Go (cgo calls and garbage collection cost, and the hot path stays C++ and CUDA).

Ideas from outside, not measured here: Marlin reports up to 4x per layer and 2.8x end to end in vLLM; learned
draft heads of the EAGLE family need training per model and stop paying below an acceptance of about 0.5 and at
large batches; published CPU results put AVX-512's and AMX's gain mostly in prompt processing; TensorRT-LLM is
reported 30-70 % above llama.cpp on consumer GPUs, with an engine build of over an hour.

Further candidates: more controls for the Pilot (thread pinning, the CPU/GPU layer split, the prompt batch size),
the prefix pool saved to disk across restarts, a quantized KV cache behind the quality gate, and choosing the
quantization that fits the memory beside the wanted sequences.

### Phase 5: keep the lead

- Send patches upstream where llama.cpp will take them; what it takes needs no more rebasing.
- Make the Pilot the core: more controls and learning while serving. That is the part a flag in llama-server
  does not copy.
- Win a second maintainer. Phases 2 and 3 are what lets someone join.

## Prompt for an agent that starts Phase 4

```text
Goal: IronMule is clearly ahead of llama.cpp in tokens per second and closes the gap to vLLM AWQ on a GPU, at the
same answer quality. Follow this order and measure after every step.

Direction:
- Algorithms first, then language, then kernels. No rewrite on suspicion.
- Compute belongs in the kernels; Python only steers. What runs per token must not grow with the context.
- Measure every change against llama-server on the same machine and model, check the facts against the
  reference, and note the numbers with the machine in docs/LAB.md.

Step 1 (now, tokens stay identical):
- speculation.draft(): an incremental index per request (the last occurrence of every 2- and 3-gram) instead of
  a search over the whole context in every step.
- GgmlEngine._common and SharedPool.place: numpy or a radix tree instead of a Python loop; the prefix scan runs
  once per request.
- One test each that compares the old and the new implementation on random inputs; tests/engine stays green.

Step 2 (an experiment in lab/, measure first, then decide):
- GPU: profile one decode step on a T4. Try an INT4xFP16 kernel on the Marlin principle (arXiv 2408.11743) for
  the quantized matrix products as a ggml patch beside native/ggml_patches. Stop if one request gains less than
  15 %.
- CPU: the 4-bit formats are compute-bound on the measured Xeon (LAB.md, ROW1 round 4: Q4_K_M 13.5 against
  Q8_0 20 tokens per second for one request). Profile the q4_K and Q6_K kernels before anything else there.

Step 3 (only after asking):
- The engine loop as a C++ shim against libllama, with the sampler chain for temperature and top_p.
- An EAGLE-style draft head for one model, compared with the draft model (DM1).
- A Linux CI job with a small GGUF so the GGUF engine's real-model tests run.

Do not: rewrite in Go, add dependencies without a measurement, start Kaggle runs without approval.
```
