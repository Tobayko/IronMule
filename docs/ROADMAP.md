# Roadmap: from a research project to a product

Recorded 2026-11-15 at the owner's request, after a review of the repository (code, tests, documents and the
benchmarks in [LAB.md](LAB.md); the test suites were read, not run). This is a plan and an assessment. Nothing
here is a measured result unless it says so, and nothing here changes current behaviour.

**What "done" means.** A stranger installs IronMule with one command, repeats the README's numbers on their own
machine with one command, and uses it from the client they already have.

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
4. CPU: measure whether generation is bound by memory bandwidth (tokens per second over threads, Q8_0 against
   Q4_K_M). If it is, work goes into moving fewer bytes, not into more SIMD.

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
- CPU: measure whether generation is bound by memory bandwidth (tokens per second over threads, Q8_0 against
  Q4_K_M). If it is, aim at moving fewer bytes, not at more SIMD.

Step 3 (only after asking):
- The engine loop as a C++ shim against libllama, with the sampler chain for temperature and top_p.
- An EAGLE-style draft head for one model, compared with the draft model (DM1).
- A Linux CI job with a small GGUF so the GGUF engine's real-model tests run.

Do not: rewrite in Go, add dependencies without a measurement, start Kaggle runs without approval.
```
