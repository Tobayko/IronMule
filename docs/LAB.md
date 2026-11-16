# Lab notebook

Short, dated notes on experiments that live in `lab/`. Each entry says what was tried, the measured numbers,
the machine and model, what it was compared against, and what follows. Lab code is not shipped; nothing here is
a product claim until its answers were checked against a reference engine on the same model.

## Thesis: rows are free

For a model that does not fit in cache, one decode pass is limited by how fast the weights stream from memory,
not by arithmetic. On a Sapphire-Rapids-class CPU with AMX, the int8 multiply-accumulate hardware should be able
to process many activation rows (concurrent requests, speculation positions of one request, or prompt tokens)
in nearly the time of one row, as long as the weights are read once per pass. llama.cpp pays 2.5-3x the time of
one row for 8 rows on this machine, so if a kernel makes extra rows nearly free, throughput under load, prefill
and exact speculation all rise above llama.cpp. ROW1 tests whether that holds on this VM.

## ROW1

Machine for all entries unless stated: 4 vCPU KVM guest, Intel Sapphire-Rapids-class Xeon at 2.1 GHz
(AMX int8/bf16, AVX-512 VNNI), 2 MiB L2 per core, 260 MiB L3, 15 GB RAM, THP in madvise mode. Reference engine:
llama.cpp commit 4227c9b, Release CPU build, Q8_0 GGUFs. Code: `lab/row1/`.

### 2026-11-11 E0: the machine's physics

Programs: `lab/row1/bw.c` (AVX-512 streaming read of a 2 GiB buffer, best of 5 passes), `lab/row1/peak.c`
(register/L1-resident `tdpbssd` with four independent C tiles; `vpdpbusd` with 12 independent accumulators).

| Measure | 1 thread | 2 threads | 4 threads |
| --- | --- | --- | --- |
| Streaming read, 4 KiB pages (GB/s) | 12.9 | 22.6 | 44-56 (three runs: 56.1, 44.9, 52.7, 44.3) |
| Streaming read, MADV_HUGEPAGE, 2 GiB AnonHugePages confirmed (GB/s) | 13.9 | 24.9 | 46-50 |
| AMX `tdpbssd` int8 (MAC/s) | 2.44e12 | 5.18e12 | 1.04e13 |
| AVX-512 VNNI `vpdpbusd` (MAC/s) | 2.2e11 | 5.6e11 | 1.1e12 |

One thread gets only about a quarter of the bandwidth; four threads are needed to stream. Huge pages make no
clear difference to a plain streaming read (run-to-run noise on this VM is about 10-20%). AMX works at full rate
under KVM: about 9.5x VNNI per thread, about 1,250 MACs per cycle at 2.1 GHz.

Consequence for Qwen3-1.7B Q8_0 (about 1.8 GB of weights, about 1.7e9 MACs per row and pass): a pass bound by
50 GB/s takes about 36 ms. In 36 ms four threads can do 3.7e11 AMX MACs (about 220 rows) or 4e10 VNNI MACs
(about 23 rows). On paper, rows are free up to well past 48 with AMX and up to about 20 with VNNI.

### 2026-11-11 E1: llama.cpp against the physics

`lab/row1/rows_llama.py`: one `llama_decode` call (4 threads, llama-server's polling thread pool, logits for every
row), median of 9 calls after one warm call, after a 32-token prompt in each of 32 sequences (KV 256 cells per
sequence, f16, no flash attention). "seqs" = R sequences with one new position each; "window" = R consecutive
positions of one sequence (removed again after each call). Default load (mmap on, no THP) versus `use_mmap=False`
with `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (AnonHugePages 3.3 GB on 1.7B and 1.9 GB on 0.6B confirmed in
smaps_rollup; the default load had 0).

Qwen3-1.7B Q8_0, ms per call (default / huge pages):

| R | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| seqs | 48.0 / 44.8 | 86.1 / 67.8 | 96.3 / 73.9 | 109.6 / 90.5 | 158.1 / 122.2 | 268.2 / 233.0 |
| window | 76.7* / 47.2 | 81.5 / 71.6 | 86.6 / 77.4 | 96.9 / 105.2 | 129.4 / 124.0 | 238.3 / 248.2 |

Qwen3-0.6B Q8_0, ms per call (default / huge pages):

| R | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| seqs | 19.8 / 18.8 | 34.8 / 29.9 | 39.1 / 33.9 | 47.9 / 43.1 | 67.7 / 65.9 | 135.6 / 111.0 |
| window | 27.5* / 18.2 | 35.3 / 31.1 | 39.9 / 35.1 | 47.1 / 40.9 | 66.7 / 62.9 | 109.2 / 109.1 |

(*) outliers of a noisy VM; the R=1 seqs figure is the same work. At R=1 llama.cpp reads the 1.7B weights at about
40 GB/s, so it is bandwidth-bound. Beyond that it is compute-bound at a low rate: R=32 on 1.7B is 32 x 1.7e9 MACs in
233-268 ms = 2.0-2.3e11 MAC/s, 2% of the AMX peak measured above. The step from R=1 to R=2 alone costs 1.5-1.8x.

Where the time goes (`lab/row1/prof_llama.py` on a copy of llama.cpp with per-op timers in
`ggml_graph_compute_thread`, per thread, averaged over 8 calls; Qwen3-1.7B, default load):

| R | call | Q8_0 mul_mat compute | Q8_0 barrier wait | f16 mul_mat (attention KQ/KQV) | everything else |
| --- | --- | --- | --- | --- | --- |
| 1 | 49.9 ms | 39.6 ms | 5.1 ms | 1.0 ms | about 3 ms |
| 8 | 96.7 ms | 73.5 ms | 7.3 ms | 6.1 ms | about 8 ms |
| 32 | 241.3 ms | 163.7 ms | 21.6 ms | 27.2 ms | about 25 ms |

Qwen3-0.6B: R=1 19.5 ms (Q8_0 13.9 + 1.7 wait), R=8 43.0 ms (26.7 + 1.9; f16 attention 6.5), R=32 121.7 ms
(59.4 + 7.6; f16 attention 25.8; SET_ROWS 7.5). So about 75-85% of a call is the Q8_0 matmul including the waits at
its end. Which kernel: this build (GGML_NATIVE on a CPU with AMX) puts the Q8_0 weights in ggml's AMX extra buffer
("AMX model buffer size" in the load log, every weight "amx repack"ed). For M=1 it runs the AVX-512 VNNI
`tinygemm_kernel_vnni` in `ggml/src/ggml-cpu/amx/mmq.cpp`; for M>1 it runs `tinygemm_kernel_amx` on 32x32 output
blocks with 32-wide K steps, after thread 0 alone quantizes all activation rows. Not llamafile sgemm, not vec_dot.
The AMX kernel reaches only 2% of the tile peak; the attention part (f16 KV, every sequence attends its full 256-cell
window) is the second cost and grows linearly with rows.

Huge pages (no mmap plus glibc hugetlb=1) give llama.cpp 5-25% on most rows here, at the edge of the noise:
1.7B one request 20.8 -> 22.3 tokens/s, 8 rows 73 -> 88 tokens/s; 0.6B one request 50.5 -> 53.2, 8 rows 167 -> 186.

### 2026-11-11 E2: our AMX kernel on the real weights

`lab/row1/gemm.c` (built as `libgemm.so`, driven by `lab/row1/e2_pass.py`). A "pass" is all 197 Q8_0 matrices of the
model in forward order (28 x q, k, v, o, gate, up, down, then the tied 151,936 x 2,048 output matrix; 1.72e9 weights
for Qwen3-1.7B, 0.60e9 for Qwen3-0.6B), each split over 4 pinned threads by pairs of 16-column output tiles, with
a spin barrier after each matrix; activations are int8 noise (values do not change the time). Weights repacked so
each thread streams its share strictly sequentially (n-tile pairs interleaved), software prefetch 16 KiB ahead
into L2. Only matmuls: no attention, norms, rope or activation quantization.

- Variant A: exact Q8_0 semantics. K steps of 32, per step `tdpbssd` into fresh C tiles, tile store, then
  float scale by weight f16 scale x activation scale and accumulate in AVX-512 registers. Same bytes as Q8_0.
- Variant B: one float scale per weight row (requantized from the Q8_0 values) and per activation row; int32
  accumulation over the whole K, scaled once at the end. 1 byte per weight.

ms per pass, best / median of three alternating runs, 4 threads. "huge" = `MADV_HUGEPAGE` on the packed weights
(AnonHugePages 3.3 GB / 0.66 GB confirmed); "4K" = `MADV_NOHUGEPAGE`.

Qwen3-1.7B:

| R | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A huge | 48.0 / 48.5 | 43.3 / 45.8 | 47.8 / 49.6 | 57.8 / 64.1 | 64.6 / 70.5 | 91.4 / 93.8 | 146.3 / 152.6 |
| A 4K | 46.7 / 48.6 | 45.6 / 48.0 | 48.7 / 51.1 | 61.4 / 66.3 | 70.1 / 70.2 | 108.6 / 119.9 | 148.3 / 148.7 |
| B huge | 40.1 / 41.4 | 38.5 / 40.3 | 38.9 / 47.1 | 41.1 / 41.8 | 41.9 / 45.4 | 51.5 / 53.4 | 72.9 / 81.3 |
| B 4K | 42.2 / 45.6 | 40.8 / 48.4 | 41.5 / 50.7 | 42.4 / 54.7 | 42.6 / 52.3 | 56.0 / 69.4 | 65.1 / 76.2 |
| llama.cpp Q8_0 matmul incl. its waits (E1) | 44.7 | | | 80.8 | | 185.3 | |

Qwen3-0.6B:

| R | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A huge | 18.1 / 18.7 | 17.3 / 17.3 | 17.4 / 18.3 | 20.7 / 21.5 | 23.9 / 24.3 | 33.9 / 36.2 | 58.2 / 58.7 |
| B huge | 14.6 / 14.9 | 14.4 / 14.8 | 14.7 / 15.2 | 15.1 / 15.2 | 15.8 / 16.7 | 20.4 / 20.7 | 30.0 / 30.8 |
| B 4K | 15.0 / 15.3 | 15.3 / 15.3 | 15.0 / 15.9 | 15.1 / 15.9 | 17.4 / 18.3 | 20.8 / 21.3 | 29.5 / 32.1 |
| llama.cpp Q8_0 matmul incl. waits (E1) | 15.6 | | | 28.6 | | 67.0 | |

Rows are close to free with variant B: 16 rows cost 1.05-1.1x one row, 32 rows 1.3-1.4x, 48 rows 1.8-2x
(B at R=1 is at 42-47 GB/s, the machine's streaming bandwidth). Variant A is bound by its per-32 float
post-processing: 16 rows 1.35x, 32 rows 1.9x. Against llama.cpp's own Q8_0 matmul time (E1): A is 1.3x faster at
8 rows and 2x at 32; B is 2x at 8 rows and 3.6x at 32. Huge pages: no reliable difference (within noise).
The VM is noisy: single runs of the same configuration varied between 40 and 68 ms.

Accuracy (`lab/row1/e2_acc.py`, Qwen3-1.7B, 16 real tokens; activations = real embedding rows -> RMSNorm x the
layer's norm weight, and SiLU(gate) x up computed in float for ffn_down; not hidden states of a full forward).
Relative Frobenius error against float64 math on the dequantized Q8_0 weights:

| tensor | llama.cpp Q8_0 arithmetic (numpy emulation) | ours A | ours B | A vs llama |
| --- | --- | --- | --- | --- |
| blk.0 attn_q / ffn_down | 4.5e-3 / 8.5e-3 | 4.5e-3 / 8.5e-3 | 1.7e-2 / 4.1e-2 | 1.3e-7 / 2.4e-7 |
| blk.14 attn_q / ffn_down | 6.8e-3 / 9.7e-3 | 6.8e-3 / 9.7e-3 | 2.0e-2 / 6.2e-2 | 1.5e-7 / 2.5e-7 |
| blk.27 ffn_up / ffn_down | 1.0e-2 / 6.9e-3 | 1.0e-2 / 6.9e-3 | 7.8e-2 / 6.9e-2 | 1.3e-7 / 2.6e-7 |
| output | 1.7e-3 | 1.7e-3 | 5.7e-3 | 1.2e-7 |

A equals llama.cpp's arithmetic to 1e-7 (summation order). B is 3-10x worse. `lab/row1/e2_decomp.py` splits
B's error: per-row int8 weights alone give 5e-3 to 1.3e-2 (close to llama's total), per-row int8 activations
give 1e-2 to 7.8e-2 (the dominant part, from outlier channels). With 16-bit activations (hi and lo int8 halves,
numpy emulation) B's error falls to the weight-only figure, 5e-3 to 1.3e-2, i.e. 1-2x llama's. That variant
("B16") costs B's time at 2R rows: 8 rows about 42 ms, 16 rows about 52 ms, 24 rows about 73 ms on 1.7B.

### 2026-11-11 Step 4: variant A inside llama.cpp

Patch (`lab/row1/ggml_patch/row1_kern.h` and `mmq.cpp.diff` against `ggml/src/ggml-cpu/amx/mmq.cpp` of llama.cpp
4227c9b, applied to a copy of the tree): at load, every Q8_0 weight in the AMX buffer is packed a second time into
variant A's layout (huge-page mmap; ggml's own layout stays for its M=1 VNNI kernel); `ggml_backend_amx_mul_mat`
runs variant A for 2 <= M <= 48, after ggml's own Q8_0 quantization of the activations. `ROW1_GGML=0` turns it
off in the same build. Load: no mmap, glibc hugetlb=1.

Answers (`lab/row1/e4_check.py`, Qwen3-1.7B, 4 questions decoded together, 40 greedy tokens): both settings give
the same correct answers (Paris; 8 legs; William Shakespeare; 17 x 3 = 51). Two of four continuations are
token-identical over 41 tokens, two diverge after the answer (token 19 and 22, wording only), as expected from
a different float summation order.

ms per `llama_decode` call (rows_llama.py), two runs each, off -> on:

| model, rows | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.7B seqs run 1 | 47.1 -> 74.0* | 75.5 -> 84.3 | 79.0 -> 92.0 | 99.2 -> 104.6 | 116.3 -> 105.7 | 217.1 -> 156.2 |
| 1.7B seqs run 2 | 46.1 -> 46.6 | 76.4 -> 64.0 | 93.2 -> 62.2 | 105.1 -> 94.7 | 140.9 -> 135.3 | 260.0 -> 176.2 |
| 1.7B window run 1 | 45.4 -> 44.1 | 73.4 -> 57.2 | 78.7 -> 63.7 | 92.7 -> 81.9 | 128.1 -> 102.2 | 266.0 -> 157.0 |
| 1.7B window run 2 | 74.9* -> 49.6 | 91.0 -> 59.9 | 85.5 -> 66.2 | 86.5 -> 92.2 | 114.6 -> 113.4 | 221.7 -> 175.5 |
| 0.6B seqs run 1 | 18.5 -> 17.7 | 31.9 -> 24.7 | 37.9 -> 28.3 | 58.2 -> 38.4 | 78.6 -> 55.6 | 141.7 -> 99.6 |
| 0.6B window run 1 | 22.4 -> 18.2 | 33.4 -> 28.3 | 36.2 -> 31.5 | 52.6 -> 47.0 | 75.8 -> 56.1 | 120.3 -> 97.5 |

(*) noise; R=1 does not use the patch. Per-op profile at 1.7B (prof_llama.py, off -> on): Q8_0 matmul plus its
waits 77.2 -> 70.1 ms at 8 rows and 176.9 -> 114.9 ms at 32 rows; attention (f16 mul_mat) stays 6.5 ms and
24-27 ms; the rest of the call about 10 ms and 30 ms. So at 32 rows the patch gives 1.3-1.7x per call, at 8 rows
5-15%, at 2-4 rows 1.2-1.3x on the window. The exact variant A is limited by its per-32 scaling, and beyond
16 rows attention and the other ops are a third of the call.

What follows: rows are close to free only with long int32 accumulation (variant B family). Exact Q8_0 semantics
cannot get there without a cheaper per-block scale step. The next experiment is B16 (per-row int8 weights,
16-bit activations as two int8 rows) in this same hook, run through the answer check and a perplexity
comparison against stock llama.cpp, plus attention for many rows (unified KV, only the filled cells).
No ledger entry was written for ROW1 (lab work; the research repository was not used).

### 2026-11-11 ROW1 round 2, part 1: B and B16 in the ggml hook

`lab/row1/ggml_patch/` (header plus diff against `ggml-cpu/amx/mmq.cpp` of 4227c9b), `ROW1_GGML=0|A|B|B16`, read
at load (which second packing is built) and at run time. B and B16 now serve every M including 1; rows above 48
kernel rows run in chunks of 32 with the weight-pair outer loop, so a weight block is read from RAM once and then
from L2 for every further chunk. Activation quantization for B/B16 is split over all threads (A still uses ggml's
thread-0 quantization). B16 splits each activation row (scale amax/32639) into high = (v+128)>>8 and low = v-256*high,
both int8; worst-case int32 sum per kernel row for K=6144 is 127*128*6144 = 1.0e8, far inside int32; high and low
are combined in float after the accumulation. Resident: A's second copy is the Q8_0 size (1.83 GB for 1.7B,
0.64 GB for 0.6B), B/B16's is 1 byte per weight plus a float per row (1.72 GB, 0.60 GB), beside ggml's own copy.

### 2026-11-11 ROW1 round 2, part 2: answers

Stock = the same patched build with `ROW1_GGML=0`. Perplexity: the project's fixed text, 16 chunks of 512 tokens,
each one 512-row decode (`lab/row1/ppl.py`, GgmlEngine.chunk_nll; tokenized with the GGUF's tokenizer). Facts:
`lab/row1/facts.py`, 50 questions, greedy, `/no_think`, at most 48 tokens, decoded 8 at a time.

| Model | Variant | Perplexity | Ratio to stock | Facts correct | Right/wrong differs from stock | Answer text identical to stock | Time of the 16 x 512 scoring |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B | stock | 32.156 | 1 | 48/50 | - | 50 | 33.3 s |
| 0.6B | A | 32.151 | 0.9998 | 48/50 | 0 | 42 | 29.6 s |
| 0.6B | B | 33.004 | 1.0264 | 47/50 | 1 | 18 | 22.2 s |
| 0.6B | B16 | 32.348 | 1.0060 | 48/50 | 0 | 38 | 22.2 s |
| 1.7B | stock | 21.619 | 1 | 47/50 | - | 50 | 65.2 s |
| 1.7B | A | 21.639 | 1.0009 | 48/50 | 1 | 41 | 50.9 s |
| 1.7B | B | 22.158 | 1.0249 | 48/50 | 1 | 16 | 25.1 s |
| 1.7B | B16 | 21.523 | 0.9956 | 48/50 | 1 | 34 | 31.7 s |

The fact differences: on 1.7B stock spent its 48 tokens writing out 2^10 as a product and never reached 1024; A, B
and B16 answered 1024. On 0.6B, B's answer to 1000 - 357 was cut by the token limit. Stock misses on both models:
capital of Japan (0.6B says Osaka), "hardest natural substance" (hedges). A and B16 keep the quality (perplexity
within 0.1% / 0.6%); B costs 2.5% perplexity from its 8-bit per-row activations and is dropped from part 3.

### 2026-11-11 ROW1 round 2, part 3: speed end to end

Stock = the patched build with `ROW1_GGML=0`. B is out (part 2). Drivers: `lab/row1/rows_llama.py` (now 48 sequences),
`lab/row1/serve.py` (llama-server of the patched copy, `-t 4 -np N -c N*2048 --temp 0 --no-webui`, N streaming chat
requests at once, 64 tokens each, different short story prompts; one warm-up block, median of 3 blocks), summary
`lab/row1/summ3.py`. Raw files in the session scratchpad (`row1/p3`).

ms per `llama_decode` call, R sequences one new position each (default load, single run of 9 calls, median):

| Model | Variant | R=1 | 2 | 4 | 8 | 16 | 32 | 48 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.7B | stock | 46.7 | 76.9 | 104.1 | 113.6 | 160.4 | 232.1 | 466.3 |
| 1.7B | A | 46.2 | 82.1 | 88.0 | 96.7 | 106.9 | 164.5 | 258.0 |
| 1.7B | B16 | 48.1 | 52.2 | 58.3 | 69.0 | 81.4 | 125.9 | 168.7 |
| 0.6B | stock | 19.4 | 30.1 | 33.7 | 52.5 | 70.6 | 114.3 | 192.2 |
| 0.6B | A | 20.6 | 26.3 | 30.7 | 56.8 | 81.4 | 115.4 | 144.1 |
| 0.6B | B16 | 23.7 | 22.0 | 25.6 | 31.8 | 47.0 | 102.8 | 113.7 |

Serving, total tokens/s over the block's wall time, mean time to first token in brackets:

| Model | Setting | N=1 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- | --- |
| 1.7B | original unpatched llama-server | | 56.2 (0.77 s) | | |
| 1.7B | stock | 17.2 (0.06) | 59.9 (0.63) | 85.6 (0.77) | 89.4 (1.94) |
| 1.7B | stock, --no-mmap + hugetlb | 18.6 (0.06) | 64.8 (0.65) | 88.5 (1.35) | 97.1 (2.02) |
| 1.7B | A | 17.1 (0.08) | 71.0 (0.60) | 96.5 (1.16) | 125.5 (1.64) |
| 1.7B | B16 | 18.2 (0.08) | 86.6 (0.44) | 124.2 (0.71) | 151.4 (1.00) |
| 1.7B | B16, --no-mmap + hugetlb | 17.2 (0.06) | 91.7 (0.43) | 129.7 (0.78) | 162.7 (1.02) |
| 1.7B | stock, -fa | | | 98.8 (1.25) | 104.1 (1.83) |
| 1.7B | B16, -fa | | | 148.7 (0.57) | 189.3 (0.95) |
| 0.6B | original unpatched llama-server | | 108.1 (0.40) | | |
| 0.6B | stock | 38.2 (0.03) | 108.0 (0.38) | 145.9 (0.67) | 162.2 (0.95) |
| 0.6B | A | 43.6 (0.03) | 117.9 (0.34) | 155.3 (0.58) | 175.3 (0.79) |
| 0.6B | B16 | 38.1 (0.03) | 145.9 (0.18) | 183.8 (0.28) | 203.5 (0.65) |

B16 against stock: 1.7B 1.45x at 8 requests, 1.45x at 16, 1.7x at 32 (1.8x with -fa on both, 2.1x for B16 -fa
against stock default); first tokens 30-50% sooner. 0.6B 1.35x / 1.26x / 1.25x. The patched stock binary matches the
original llama-server at N=8 (59.9 vs 56.2 and 108.0 vs 108.1).

One request alone, 128 tokens, tokens/s (median of 3): 1.7B stock 18.1, stock + huge pages 20.0, B16 + huge pages
19.5, A + huge pages 17.7; 0.6B stock 45.3, stock + huge pages 46.5, B16 + huge pages 41.2, A + huge pages 40.9. As
expected one row gains nothing: it is bandwidth-bound already; B16 at one row costs two kernel rows and is level on
1.7B, 11% behind on 0.6B.

### 2026-11-11 ROW1 round 2, part 4: prefill

One request, prompt cut from the fixed text, `n_predict` 1, prompt cache off; llama-server's reported prompt
tokens/s, median of 3 (default batch: `-b 2048 -ub 512`; second figure with `-b 2048 -ub 2048`):

| Model | Variant | 512 tokens | 2,000 tokens |
| --- | --- | --- | --- |
| 1.7B | stock | 145 / 159 | 127 / 109 |
| 1.7B | A | 207 / 214 | 175 / 138 |
| 1.7B | B16 | 350 / 342 | 243 / 173 |
| 0.6B | stock | 310 / 308 | 219 / 175 |
| 0.6B | A | 392 / 398 | 263 / 178 |
| 0.6B | B16 | 544 / 515 | 317 / 216 |

B16 prefills 2.4x (512) and 1.9x (2,000) faster on 1.7B, 1.75x / 1.45x on 0.6B. One 2,048-row ubatch is slower than
four of 512 on every variant (the f16 attention over the long window grows). The longer prompt loses more because
attention is untouched.

### 2026-11-11 ROW1 round 2, part 5: what costs now

Per-op timers (prof_llama.py, 1.7B, B16, huge pages), per call: R=1 50.2 ms (Q8_0 39.3 + 5.4 wait); R=8 76.5 ms
(Q8_0 45.3 + 11.8 wait, f16 attention 6.6, set_rows 1.7, rest about 4); R=32 147.8 ms (Q8_0 68.0 + 22.7 wait,
f16 attention 23.7, set_rows 6.6 + 2.6 wait, norms, rope, softmax, glu about 12). At 32 rows the matmul compute is
46% of the call, the barrier waits around it 15%, attention 16%. Next: (1) the barrier waits (imbalance between
threads of the matmul and its quantization phase; the KV-window attention also waits); (2) attention, where -fa
alone already gives B16 +25% at 32 requests; (3) the kernel itself: 32 rows in B16 are 64 kernel rows,
64 x 1.72e9 MACs in 68 ms = 1.6e12 MAC/s, 16% of the AMX peak, so a better large-M loop order still has room.

### 2026-11-11 ROW1 round 3, part 1: waits, large M, B8s

Changes in the hook (B, B16, B8s paths): (1) work is handed out from one atomic counter in units of pair groups
(about 8 units per thread) instead of a fixed split, after activation quantization on all threads
(`ROW1_DYN=0` restores the fixed split); (2) rows above one chunk are cut into row blocks whose packed activations
fit in 768 KiB of L2 (`ROW1_RBKB`), and work units are (row block, pair group) in row-block order, so the
activation block stays in L2 while the weights stream over it; (3) new variant B8s (`ROW1_GGML=B8S`, `ROW1_SEG`):
B's weights, int8 activations with one scale per row and K-segment, int32 accumulation within a segment.

Barrier waits (per-op timers, 1.7B, B16, huge pages, ms per call, fixed split -> dynamic units):
R=1 call 55.1 -> 49.0 (Q8_0 wait 8.2 -> 2.7); R=8 79.3 -> 63.3 (wait 12.1 -> 2.3); R=32 151.7 -> 124.2 (Q8_0
compute 71.1 -> 66.4, wait 19.3 -> 3.3; f16 attention without -fa 26.5; total waits of the call 29.7 -> 10.8).
The waits were imbalance inside the matmul, not ggml's per-op barrier; no fused q/k/v call was tried, since the
remaining Q8_0 wait is 3 ms. The plain rows tables of the same two settings varied by up to 40% between runs.

Row blocks for prefill (B16, 1.7B, llama-server default batch, prompt tokens/s, 3 runs): 512 tokens 359/364/337
without -> 411/385/378 with; 2,000 tokens 241/233/249 -> 253/255/258. Profile of one 512-token decode (1.7B, ms):
stock Q8_0 2,476 + 384 wait, f16 attention 444, softmax 104; B16 Q8_0 794 + 9 wait, f16 attention 470, softmax 109.
B16's matmul runs 1,024 kernel rows x 1.72e9 MACs in 0.79 s = 2.2e12 MAC/s, 21% of AMX peak (from 16% at
32 rows before). A further loop order (B tiles held across two row chunks, or K-sliced C accumulation) was not
built; the f16 attention is now 60% of B16's matmul time in a 512-token prefill.

B8s answers (perplexity ratio to stock, facts, answers whose right/wrong differs from stock):

| Model | Segment | Ratio | Facts | Differs |
| --- | --- | --- | --- | --- |
| 0.6B | 128 | 1.0104 | 47/50 | 1 (1000 - 357) |
| 0.6B | 256 | 1.0105 | 47/50 | 1 (1000 - 357) |
| 0.6B | 512 | 1.0184 | 47/50 | 1 (1000 - 357) |
| 1.7B | 128 | 0.9980 | 47/50 | 0 |
| 1.7B | 256 | 1.0012 | 46/50 | 1 (insect legs) |
| 1.7B | 512 | 1.0084 | 47/50 | 0 |

B8s fails the 1.005 bound on 0.6B at every segment size (B16 was 1.006 there); on 1.7B segment 128 passes. Speed
on 1.7B (segment 128) below: slower than B16 at every row count, because its per-segment float step costs more than
B16's second kernel row. B16 stays the variant.

Speed on 1.7B with dynamic units and row blocks, huge pages (`--no-mmap`, glibc hugetlb=1). Rows table, ms per call:

| Variant | R=1 | 8 | 16 | 32 | 48 |
| --- | --- | --- | --- | --- | --- |
| stock | 49.1 | 100.3 | 134.0 | 247.9 | 410.3 |
| B16 | 47.3 | 63.9 | 78.9 | 130.4 | 173.4 |
| B8s (128) | 48.8 | 67.6 | 89.2 | 135.9 | 189.0 |

Serving, llama-server with `-fa --no-mmap` and hugetlb, total tokens/s (mean first-token time), median of 3 blocks:

| Variant | N=8 | N=16 | N=32 |
| --- | --- | --- | --- |
| stock -fa | 65.5 (0.78 s) | 103.5 (0.98 s) | 108.9 (1.91 s) |
| B16 -fa | 107.1 (0.31 s) | 163.0 (0.64 s) | 216.0 (0.81 s) |
| B8s (128) -fa | 96.9 (0.36 s) | 146.2 (0.66 s) | 189.8 (0.99 s) |

B16 is now 1.64x / 1.57x / 1.98x stock with -fa (round 2: 1.50x / 1.82x at N=16 / 32), first tokens 2.4x sooner.

Prefill again (prompt tokens/s, median of 3; 512 / 2,000 tokens): B16 default 385 / 255; B8s 366 / 233; with -fa
stock 138 / 59 and B16 318 / 76, i.e. llama.cpp's flash attention on this CPU makes long prompts much slower and
should stay off for prefill.

### 2026-11-11 ROW1 round 3, part 2: vocabulary shortlist (offline only)

Hidden states: the output matmul's input, captured by the hook (`ROW1_DUMP`) while scoring the fixed text with the
stock path (8,196 positions per model). Exact = float32 on the dequantized Q8_0 output matrix (151,936 rows).
`lab/row1/shortlist.py` (all positions) and `lab/row1/shortlist2.py` (PCA fitted on the first half, tested on the
other 4,098). Recall = share of positions whose exact argmax is in the sketch's top k. Full matrix: 330 MB (1.7B),
165 MB (0.6B).

| Model | Sketch | Size MB | k=16 | 64 | 256 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.7B | int4 per row | 155.9 | 0.9999 | 1.0 | 1.0 | 1.0 |
| 1.7B | sign bits x row mean | 39.2 | 0.875 | 0.953 | 0.988 | 0.998 |
| 1.7B | int2 per row | 78.1 | 0.402 | 0.527 | 0.634 | 0.732 |
| 1.7B | rotation by W's top singular vectors, d=512 int8 | 78.4 | 0.159 | 0.213 | 0.273 | 0.337 |
| 1.7B | PCA of hidden states, d=256 int8 | 39.5 | 0.904 | 0.966 | 0.991 | 0.998 |
| 1.7B | PCA d=512 int4 | 39.5 | 0.956 | 0.990 | 0.997 | 1.0 |
| 1.7B | PCA d=512 int8 | 78.4 | 0.965 | 0.992 | 0.9985 | 1.0 |
| 0.6B | PCA d=256 int8 | 39.5 | 0.943 | 0.985 | 0.996 | 0.9995 |
| 0.6B | PCA d=512 int4 | 39.5 | 0.988 | 0.998 | 0.9998 | 1.0 |
| 0.6B | PCA d=512 int8 | 78.4 | 0.990 | 0.998 | 0.9998 | 1.0 |

W's own singular directions are useless (top 512 of 2,048 hold 42% of its energy); the hidden states' principal
directions work (top 512 hold 97% / 98% of their variance). Certified rate (Cauchy-Schwarz bound on the dropped
dimensions plus rounding): 0 for every sketch and k except 0.005 (0.6B, d=512 int8, k=1024); the bound is far too
loose to prove anything. No sketch reaches recall 1.0 at k <= 256, so by the round's rule the shortlist was not
built into the hook. What it would give: PCA d=512 int4 + 1,024 exact rows reads about 42 MB instead of 330 MB,
about 16% of a 1.7B pass (at most about +19% tokens/s for one request), uncertified (no miss in 4,098 positions).

### 2026-11-11 ROW1 round 3, part 3: copy-from-context drafts

`lab/row1/drafts.py`, `ironmule.speculation.draft` (2-3 token suffix match), share of all positions where the
first 1 / 2 / 3 drafted tokens are right. Fixed text (truth = the text, 8,192 positions; same tokenizer for both
models): draft available 7.8%, right 1.0% / 0.2% / 0.06%. Greedy fact answers (truth = the stock answers, context =
prompt + answer so far): 1.7B (1,460 positions) available 17.3%, right 9.5% / 5.8% / 3.7%; 0.6B (1,057) available
17.4%, right 8.9% / 4.4% / 2.1%. Copy drafts are rarely right on prose and short answers; a speculation step on this
kernel would need a better drafter (n-gram tables from a corpus or a small model) to pay.

### 2026-11-11 ROW1 round 4, part 1: one request at 4 bit (stock llama.cpp)

Files: `unsloth/Qwen3-1.7B-GGUF` Q4_0 and Q4_K_M. In both the tied output matrix (`token_embd`) is Q6_K and stays in
the plain CPU buffer (ggml's AMX buffer refuses it); everything else goes to ggml's AMX buffer (1,002 MiB for Q4_0,
1,068 MiB for Q4_K_M, beside 244 MiB of CPU buffer). Patched build with the hook off (it touches only Q8_0 here),
`--no-mmap` and hugetlb.

| File | 1 request, 128 tokens (tok/s) | call at R=1 | R=8 | R=16 | R=32 | bytes per pass | overall GB/s at R=1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q8_0 (round 2/3, for comparison) | 20.0 | 47-49 ms | 64-100 | 79-134 | 130-248 | 1.83 GB | 37-39 |
| Q4_0 | 20.3 / 21.1 / 20.4 | 40.5 ms | 131.6 | 175.9 | 316.4 | about 1.31 GB | 32 |
| Q4_K_M | 13.2 / 13.7 / 13.5 | 69.1 ms | 170.5 | 224.5 | 358.5 | about 1.38 GB | 20 |

Per-op profile at R=1 (ms per call, compute + wait): Q4_0: q4_0 matmuls (193 nodes) 16.8 + 3.3, the Q6_K output
matrix (1 node) 13.5, q4_1 (3 nodes) 0.5, f16 attention 1.0. Q4_K_M: q4_K (168 nodes) 32.0 + 8.0, q6_K (29 nodes:
half the ffn_down/attn_v and the output) 24.4 + 2.3. Kernels: ggml's AMX extra buffer handles q4_0, q4_1, q4_K and
q6_K in `ggml-cpu/amx/mmq.cpp` (M=1: `tinygemm_kernel_vnni`, M>1: `tinygemm_kernel_amx`); the output matrix in the CPU
buffer runs ggml's generic Q6_K x Q8_K `vec_dot` loop.

Reading: Q4_0's layers are bandwidth-bound already (about 1.0 GB in 20 ms, 50 GB/s), but its Q6_K output matrix is
compute-bound (256 MB in 13.5 ms, 19 GB/s) and takes a third of the call. Q4_K_M is compute-bound throughout
(20 GB/s): one request is slower than with Q8_0 (13.5 vs 20 tok/s) although it reads 25% fewer bytes. At more rows
both 4-bit formats are much worse than Q8_0 (R=8 Q4_0 132 ms against Q8_0 B16 64 ms). So there is room: for one
request mainly on Q4_K_M and in the Q6_K output matrix, and for every format at R > 1.

### 2026-11-11 ROW1 round 4, part 2: Qwen3-4B Q8_0 (`Qwen/Qwen3-4B-GGUF`)

Stock `-fa` against B16 `-fa`, both `--no-mmap` with hugetlb; context capped at 32,768 tokens (KV for 32 x 2,048
would not fit beside two weight copies in 15 GB). Quality: perplexity stock 17.619, B16 17.596 (ratio 0.9987);
facts 49/50 both (both miss 1000 - 357), no answer differs in right/wrong.

Serving, total tokens/s (mean first-token time), 3 blocks:

| Variant | N=1 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- |
| stock -fa | 9.0 / 8.2 / 8.3 (0.13-0.15 s) | 30.6 / 32.5 / 32.0 (1.5-1.9 s) | 52.2 / 55.6 / 58.7 (1.0-2.7 s) | 52.7 / 52.9 / 56.9 (3.4-5.2 s) |
| B16 -fa | 9.6 / 9.1 / 8.6 (0.10-0.12 s) | 54.3 / 57.1 / 55.7 (0.55-0.66 s) | 84.3 / 87.7 / 86.2 (0.72-1.29 s) | 65.9 / 57.0 / 9.9 (4-10 s) |

B16 is 1.75x stock at N=8 and 1.55x at N=16 (1.7B: 1.64x and 1.57x). At N=32 B16 collapsed: ggml's own AMX copy
(4.3 GB) plus our packed copy (4.0 GB) plus 4.8 GB of KV leave no room, and the machine reclaims pages. Prefill
(prompt tokens/s, 3 runs, no -fa): 512 tokens stock 66 / 68 / 71, B16 152 / 164 / 167 (2.4x); 2,000 tokens stock
58 / 55 / 52, B16 101 / 108 / 104 (1.9x).

### 2026-11-11 ROW1 round 4, part 3: Hadamard rotation for one kernel row per row (offline)

`lab/row1/capture.py` captures the real inputs of 9 matmuls per model (layers 2, 14, 26: attn_q, ffn_up, ffn_down;
one 512-token chunk of the fixed text, stock path; last 256 rows used); `lab/row1/rotation.py` computes relative
Frobenius error against float64 on the dequantized Q8_0 weights. Rotation R = random signs x block Hadamard (block
2048 on 1.7B, 1024 on 0.6B), weights W R requantized per row to int8, activations x R quantized per row.

| Tensor (1.7B) | llama Q8_0 | B | B16 | rotated B | rotated B16 | max/std of x, before / after |
| --- | --- | --- | --- | --- | --- | --- |
| blk.2.attn_q | 7.3e-3 | 3.9e-2 | 6.9e-3 | 8.3e-3 | 6.1e-3 | 22.5 / 3.4 |
| blk.2.ffn_up | 2.1e-2 | 1.2e-1 | 1.6e-2 | 2.0e-2 | 1.5e-2 | 26.8 / 3.1 |
| blk.2.ffn_down | 1.0e-2 | 7.9e-2 | 1.1e-2 | 1.3e-2 | 9.0e-3 | 30.1 / 3.8 |
| blk.14.attn_q | 8.0e-3 | 4.1e-2 | 6.9e-3 | 8.7e-3 | 6.4e-3 | 22.8 / 3.4 |
| blk.14.ffn_up | 8.8e-3 | 3.6e-2 | 9.1e-3 | 1.2e-2 | 8.4e-3 | 14.4 / 3.5 |
| blk.14.ffn_down | 1.1e-2 | 5.7e-2 | 1.3e-2 | 1.4e-2 | 9.8e-3 | 18.9 / 3.9 |
| blk.26.attn_q | 1.0e-2 | 3.7e-2 | 9.1e-3 | 1.1e-2 | 8.2e-3 | 15.7 / 3.5 |
| blk.26.ffn_up | 1.1e-2 | 4.7e-2 | 8.9e-3 | 1.2e-2 | 8.6e-3 | 19.2 / 3.4 |
| blk.26.ffn_down | 9.7e-3 | 5.5e-2 | 1.1e-2 | 1.3e-2 | 9.2e-3 | 33.1 / 3.8 |

0.6B (same layers): llama 6.9e-3 to 1.6e-2, B 2.8e-2 to 8.4e-2, B16 6.9e-3 to 1.3e-2, rotated B 9.1e-3 to 1.6e-2,
rotated B16 6.7e-3 to 1.2e-2 (`$SP/row1/p5/rotation_0.6B.txt`). The rotation flattens the activations (max/std
from 13-33 to 3-5) and cuts B's error 3-6x, but rotated B stays 1.1-1.5x above B16 (and 1.0-1.7x above llama's
own Q8_0 error). By the round's rule (build only at B16's error or better) it was not built. Rotated B16 is the
most accurate of all (below llama's Q8_0 on every tensor of 1.7B) but costs B16's two kernel rows.

### 2026-11-11 ROW1 round 4, part 4: attention

Code path (llama.cpp 4227c9b, CPU): without -fa, KQ and KQV are `GGML_OP_MUL_MAT` with an f16 src0 (the K view and
the transposed V cache); `ggml_compute_forward_mul_mat` first tries llamafile's `tinyBLAS` with the f32 activations
(no kernel for f16 x f32), converts the activations to f16, and runs `tinyBLAS<16, __m512, __m512, ggml_fp16_t,
ggml_fp16_t, float>`, which converts f16 to f32 and uses f32 FMAs, one call per head (16 heads x 28 layers). With
-fa it runs `GGML_OP_FLASH_ATTN_EXT`, the CPU flash kernel, which walks one query row at a time over the KV.
Measured (1.7B, B16 matmuls, huge pages, per-op timers):

| Setting | 32-row decode call | attention part | 512-token prefill: attention, matmul |
| --- | --- | --- | --- |
| f16 KV, no -fa | 127.8 ms | f16 mul_mat 25.8 ms | 459 ms, 802 ms |
| f32 KV (`kv_f32`), no -fa | 141.6 ms | f32 mul_mat 27.0 ms | 558 ms, 690 ms |
| f16 KV, -fa | 119.6 ms | FLASH_ATTN_EXT 17.0 + 4.8 wait | 798 + 288 wait, 724 ms |

f32 KV is no faster (the f32 sgemm path is not better than the f16 one here). Flash attention helps a decode call
with many rows (attention 26 -> 22 ms incl. waits) and hurts prefill badly (459 -> 1,086 ms), and the KV layout
(V transposed or not) is fixed per context, so llama.cpp cannot switch per batch. In prefill the non-flash path
does 1.5e10 MACs in 0.46 s = 3.3e10 MAC/s, about 12% of the 4-core f32 FMA peak; AMX-BF16 or AVX512-FP16 would have
room there, but that is a new attention kernel, not a small patch, so none was built in this round.

### 2026-11-12 ROW1 round 4, part 1b: our kernels on the 4-bit files

Built in the hook: (1) A4, exact Q4_0 semantics (`ROW1_GGML=A` on a Q4_0 file): weights repacked at load to
nibble rows that unpack with four AVX-512 ops per 128 B straight into the B tile layout in L1, Q8_0 activations from
ggml's own quantization, variant A's per-32 float step; used for M >= 2 (at M = 1 it was slower than ggml's
bandwidth-bound VNNI path: 47.6 against 41.6 ms per call). (2) `ROW1_OUT=1`: the big output matrix of a 4-bit GGUF
(the tied Q6_K `token_embd`) is dequantized at load with ggml's `to_float`, requantized per row to int8 and served
by the B16 kernel for every M; ggml's own AMX copy of it is then not written. (3) Under B/B16/B8s ggml's AMX copy
of each Q8_0 weight is no longer written either (it is never read), which saves its full size: Qwen3-1.7B B16
process RSS 3.62 GB against 3.93 GB stock with mmap on, i.e. our copy replaces ggml's.

Qwen3-1.7B, `--no-mmap` + hugetlb. One request 128 tokens (tok/s, 3 runs), rows table (ms per call), perplexity
ratio to stock (16 x 512 fixed text), facts:

| File | Setting | 1 request | R=1 | R=8 | R=16 | R=32 | ppl ratio | facts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q4_0 | stock | 22.7 / 21.9 / 23.2 | 49.7 | 112.9 | 158.5 | 290.3 | 1 (25.419) | 48/50 |
| Q4_0 | output B16 | 31.0 / 30.7 / 27.1 | 28.6 | 77.2 | 116.1 | 223.5 | 1.0000 | 48/50 |
| Q4_0 | A4 + output B16 | 30.9 / 31.1 / 31.5 | 26.8 | 53.6 | 82.1 | 146.2 | 0.897 (see below) | 48/50 |
| Q4_K_M | stock | 13.6 / 14.9 / 15.4 | 61.5 | 126.0 | 173.2 | 410.6 | 1 (21.969) | 48/50 |
| Q4_K_M | output B16 | 17.3 / 17.4 / 17.7 | 51.9 | 111.7 | 135.9 | 250.8 | 0.9968 | 48/50 |

No fact answer differs in right/wrong from stock in any setting. Profile of one call, Q4_0, A4 + output B16: R=1
28.5 ms (q4_0 layers 17.8 + 1.9 wait, output 5.3, from 13.5-14.3 stock), R=8 55.6 ms (layers 31.7 + 3.9, output
5.9), R=32 155.3 ms (layers 75.0 + 20.7 wait, output 11.4; A4 still uses the fixed per-thread split).

A finding: stock llama.cpp's AMX Q4_0 kernel for M > 1 (`tinygemm_kernel_amx`, the path every prompt and every
batched decode takes) loses quality. `lab/row1/nll_check.py` scores the first 3 chunks two ways: stock M=512
perplexity 28.94, stock one token per call (M=1, VNNI) 24.56, our A4 at M=512 24.63. So the 0.897 ratio is stock's
batched path being worse, not ours being better: A4 matches ggml's own M=1 arithmetic. On Q4_0, stock prefill and
batched serving through ggml's AMX buffer give worse answers than its single-token path on this CPU.

So on 4-bit files there was room for one request: 1.37x on Q4_0 (22.6 -> 31.2 tok/s, now 1.56x the Q8_0 file's
20.0) and 1.17x on Q4_K_M, almost all from the output matrix; 2.1x at 8 rows on Q4_0. Q4_K_M's layers (q4_K,
compute-bound at about 20 GB/s) would need their own kernel: super-blocks of 256 with 8 sub-blocks of 32, a 6-bit
scale and a 6-bit min per sub-block, f16 d and dmin per super-block; dot = d * sum_s sc_s (x_s . q_s) - dmin *
sum_s m_s sum(x_s), i.e. variant A's per-32 step with integer sub-block scales plus a correction from per-32
activation sums. Not built. A per-row 4-bit format of our own (B-style) would read 0.5 byte per weight but per-row
4-bit weights carry about 8x the rounding error of per-row int8 (round 3's int4 sketch), so it is not promising
without a rotation; not built.

2026-11-12, Qwen3-4B again with ggml's unused AMX copy no longer written (B16 resident: our 4.0 GB copy only),
same settings (`-fa --no-mmap`, hugetlb, context cap 32,768), B16 serving tokens/s (mean first-token time):
N=1 11.9 / 12.6 / 11.2 (0.09-0.10 s); N=8 70.5 / 70.8 / 65.8 (0.60-0.66 s); N=16 93.5 / 98.9 / 104.9 (0.85-1.24 s);
N=32 129.2 / 127.9 / 122.3 (1.55-1.99 s). Against stock -fa (above): 1.4x at N=1, 2.2x at N=8, 1.8x at N=16,
2.4x at N=32, first tokens 2.5x sooner at N=8. On 1.7B the same comparison gave 1.0x / 1.64x / 1.57x / 1.98x, so
the factor grows with model size, mostly because stock's batched Q8_0 path degrades more on the larger matrices.
The N=1 gain is new; it was not re-checked against a stock run in the same session.

### 2026-11-12 ROW1 round 5, part 1: the kernels under IronMule's product

The product's ggml engine loads libllama from `LLAMA_CPP_LIB_PATH`, so pointing it at the patched build with
`ROW1_GGML=B16` puts the kernels under `ProductService` + worker + `GgmlEngine` unchanged. Product changes needed for
more than 8 requests in one batch (defaults unchanged; engine tests and `tests/engine -m "not integration"` green):
`IRONMULE_GGML_SEQS` / `IRONMULE_GGML_SEQ_CTX` / `IRONMULE_GGML_PROMPT_BUDGET` in `ironmule/ggml_engine.py` (its
warm-up now decodes the many-sequence prompt in pieces of the batch size), and the worker's stream width and the
client's batch capacity for the ggml engine follow `IRONMULE_GGML_SEQS` (`ironmule_product/worker.py`, `backend.py`;
without that the service admitted requests in waves of 8: N=32 first tokens 6.4 s). Hook fix: the packed copies are
unmapped when ggml frees the AMX buffer (`amx.cpp.diff`); before, the gate's tuning child, which loads the model
once per setting, ran out of memory.

Driver `lab/row1/product.py` (gate, then one worker; per N one warm-up block and 3 blocks of N streamed requests at
once, 64 tokens, `serve.py`'s prompts; then the 50 facts through the service). Speculation on (product default),
32 sequences. Reference: the original unpatched `llama-server` (4227c9b) with `-fa --no-mmap`, hugetlb, `-np N`.

Gate (`ironmule.ggml_engine.gate`, run under B16, so it compares settings against B16's own default): passes on
both models, chosen setting flash attention; ratio 0.99983 (1.7B), 0.99943 (4B). Against the hook off, B16's
perplexity on the same text and chunks was 0.9956 (1.7B, round 2) and 0.9987 (4B, round 4) of stock's.

Total tokens/s (mean first-token time), 3 blocks:

| Model | Setting | N=1 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- | --- |
| 1.7B | llama-server -fa | 21.9 / 21.8 / 21.7 (0.05 s) | 72.3 / 60.9 / 69.4 (0.45-0.71 s) | 102.3 / 99.7 / 106.2 (0.6-1.2 s) | 107.1 / 109.2 / 112.2 (2.0-2.4 s) |
| 1.7B | IronMule + B16, 32 seqs | 23.4 / 23.6 / 23.3 (0.08 s) | 124.6 / 141.7 / 136.1 (0.29-0.42 s) | 172.8 / 184.4 / 202.8 (0.51-0.53 s) | 213.0 / 210.7 / 224.6 (1.0-1.1 s) |
| 4B | llama-server -fa (ctx cap 32,768) | 9.6 / 9.9 / 9.7 (0.11 s) | 33.0 / 34.1 / 34.0 (1.2-2.2 s) | 52.5 / 58.0 / 58.6 (1.4-2.8 s) | 56.5 / 63.3 / 60.2 (2.2-5.1 s) |
| 4B | IronMule + B16, 32 seqs x 1,024 | 10.9 / 11.0 / 11.0 (0.18-0.21 s) | 64.2 / 63.4 / 62.2 (0.75-0.89 s) | 69.2 / 74.9 / 77.4 (1.4-1.55 s) | 94.7 / 96.0 / 101.4 (2.5-2.8 s) |

1.7B: 1.08x / 1.98x / 1.80x / 1.96x of llama-server's best setting, first tokens 1.6-2.2x sooner under load (one
request 0.08 against 0.05 s). 4B: 1.13x / 1.88x / 1.31x / 1.60x; at N=16/32 the product falls behind what B16 gave
through llama-server (round 4: 99 / 128 tok/s), probably the engine's per-step Python work (sampling, joins,
speculation bookkeeping) at 4B's longer steps; not profiled. Facts through the product: 48/50 (1.7B), 49/50 (4B);
the 4B run used the Qwen3-1.7B snapshot as the registered model for its tokenizer (Qwen3's tokenizer is the same;
no 4B snapshot is on disk and the bridge's tokenizer and end-token checks passed).

### 2026-11-12 ROW1 round 5, part 4: the lossy batched Q4_0 path in llama.cpp

Cause, found by reading 4227c9b's `ggml/src/ggml-cpu/amx/mmq.cpp`: in `tinygemm_kernel_amx` (the non-K-quant
template), the branch for a full 32-row block (`M == 2 * TILE_M`) unpacks the second 16-column weight tile from the
first block: `unpack_B<TB>(Tile1, B_blk0);` where `B_blk1` is meant (the non-unpack branch right below loads
`B_blk1` correctly). Only types that need unpacking go through it (`do_unpack`: the 4-bit Q4_0 and Q4_1); Q8_0 and
the K-quants are unaffected. Effect: whenever a matmul has at least 32 activation rows (prompt processing, batched
decode of 32+ rows), every second group of 16 output columns of each 32-row block is computed with the wrong
weights. That matches round 4's numbers: perplexity 28.9 at M=512 against 24.6 one token per call on Qwen3-1.7B
Q4_0, while Q4_K_M showed no such gap (25.32 / 25.29 this round).

Current llama.cpp: the master branch's `mmq.cpp` (fetched from raw.githubusercontent.com on 2026-11-12; the GitHub
API and history pages were not reachable from this session, so the fixing commit and its date are not known) reads
`unpack_B<TB>(Tile1, B_blk1);`, i.e. it is fixed there. Master also quantizes the activations on all threads
(`parallel_for_ggml` around `from_float`) and handles batched (3-D) matmuls. Note to pass on: "llama.cpp builds
with GGML_NATIVE on AMX CPUs (Sapphire Rapids and later) up to at least commit 4227c9b (August 2025; also what
llama-cpp-python 0.3.16 vendors) compute wrong results for Q4_0/Q4_1 weights whenever a matmul has 32 or more rows
(prompt processing, large batches): `tinygemm_kernel_amx` unpacks `B_blk0` instead of `B_blk1` for the second
tile. Fixed on current master. Symptom: worse perplexity at batch > 32 than one token per call (28.9 vs 24.6 on
Qwen3-1.7B Q4_0)."

### 2026-11-12 ROW1 round 5, part 5: where the product loses to llama-server at 4B, N=16/32 (stub, no llama.cpp)

Round 5 part 1 left a gap: at N=16/32 the product with B16 made 74 / 96 tok/s on Qwen3-4B, the same kernels under
`llama-server` 99 / 128. Suspected: the engine's per-step Python work. `lab/row1/step_overhead.py` runs
`GgmlEngine.run` unchanged with llama.cpp replaced by a no-op stub (random logits over 151,936 tokens, 40-token
prompts, 64 tokens each, this session's container: same CPU family, 2.3 GHz). Python per step: 0.02 ms (N=1, no
speculation), 0.29 (N=8), 0.48 (N=16), 0.89 ms (N=32; argmax over 32 rows alone 0.74 ms), against steps of roughly
250-330 ms at 4B, N=32. So the Python loop is not the gap (under 0.5 %). Two things found instead:
- *Staggered starts.* `PROMPT_BUDGET` = 96 (GG2, tuned on SmolLM2 with 8 sequences) lets only 2 of these 40-token
  prompts (4 of the about 22-token prompts `serve.py` sends) start per step, so N=32 needs 79 (71) steps for 64 tokens
  each instead of 64, and the early steps carry few decode rows. `llama-server` takes every waiting prompt into one
  call up to its batch. Expected effect on the 4B gap at N=32: about 64/71 = 0.90 of the possible throughput, so
  about a third of the measured 0.75; round 5 part 1 did not set `IRONMULE_GGML_PROMPT_BUDGET`.
- *Speculation is off at N >= 16.* `StepPolicy(MAX_ROWS = 16)` gives a step `16 - N` draft rows, none at N = 16 or
  32, but `draft()` still searches every request's context each step (about 0.1 ms per step at N=32, harmless).
Not measured: the rest of the gap (KV layout `kv_unified`, the worker's per-token streaming). Next: the product at
4B with `IRONMULE_GGML_PROMPT_BUDGET=1024` against round 5 part 1's numbers; needs this session's llama.cpp build.

### 2026-11-12 ROW1 round 5, part 6: the prompt budget, measured (Qwen3-4B Q8_0, B16, this session's container)

This session's container: 4 vCPU Sapphire-Rapids-class Xeon at 2.3 GHz (round 5 part 1 ran at 2.1 GHz), llama.cpp
4227c9b rebuilt with the ROW1 patch, `-fa --no-mmap` and hugetlb on the server, 32 sequences x 1,024 tokens in the
product, speculation on, gate (cached, B16): flash attention, ratio 0.99943. `lab/row1/part5_p6.sh`, `serve.py`'s
prompts (about 22 tokens), 64 tokens, 3 blocks; tokens/s and mean first token:

| Setting | N=1 | N=8 | N=16 | N=32 | facts |
| --- | --- | --- | --- | --- | --- |
| llama-server -fa, B16 | 10.4 / 10.0 / 10.1 (0.09-0.11 s) | 63.7 / 64.6 / 56.9 (0.67-0.76 s) | 90.2 / 97.7 / 98.7 (0.60-1.28 s) | 135.7 / 146.3 / 134.6 (1.21-1.74 s) | - |
| product, budget 96 (GG2) | 9.6 / 9.7 / 9.6 (0.15-0.16 s) | 63.8 / 61.6 / 63.9 (0.62-0.66 s) | 93.0 / 98.4 / 97.2 (0.92-1.05 s) | 126.4 / 128.8 / 131.8 (1.86-2.02 s) | 49/50 |
| product, budget 1,024 | 10.0 / 9.8 / 10.1 (0.16 s) | 62.9 / 64.4 / 64.1 (0.68-0.71 s) | 100.6 / 104.8 / 103.7 (1.17-1.18 s) | 137.6 / 140.3 / 145.7 (2.04-2.12 s) | 49/50 |

Round 5 part 1's gap (0.75x of the server at N=16/32) did not reproduce here: with GG2's budget the product is at
0.93x-1.01x of the server, and taking every waiting prompt into one call adds 7-9 % at N=16/32 (1.08x / 1.01x of the
server) for first tokens 0.1-0.2 s later. The engine's budget now defaults to one call's size (`IRONMULE_GGML_PROMPT_BUDGET`
still sets a number). First tokens stayed 0.06 s later than the server's even at N=1: part 7.

Two setup faults found on the way, both in this image only: `friday_evidence/identity.py` rejected mlx_lm's files
because `site-packages` is a symlink to `dist-packages` here (now resolved before the check), and the worker runs
`python -I`, which does not see packages pip put in `~/.local` (jinja2, needed by the bindings).

### 2026-11-12 ROW1 round 5, part 7: first tokens, and prefix reuse in the engine

`lab/row1/ttft_probe.py`, one request (22-token prompt) at a time: the engine alone in process takes 0.14 s to its
first token, through the product 0.13 s, so the service and worker add nothing; the server's 0.09-0.10 s comes from
`cache_prompt` (its default): a slot keeps a finished request's KV and a new request computes only what follows the
longest shared prefix, and `serve.py` repeats its prompts every block. The engine now does the same
(`IRONMULE_GGML_PREFIX_REUSE`, on by default): a finished request's KV stays in its sequence (every token that went
through the model), a new request takes the free sequence with the longest shared prefix, positions after it are
removed, and at least the prompt's last token is computed; a cache that refuses a partial removal falls back to a full
prompt. `lab/row1/prefix_check.py`, 8 chat prompts with a shared system prompt (93-94 tokens), one at a time, twice,
48 tokens: first token **0.359 s without reuse, 0.143 s with it** (0.130 s in the second round), 1,296 of 2,235 prompt
tokens reused, and **24 of 24 answers token-identical** with reuse on and off. Tests: three stub tests and one on the
real GGUF in `tests/engine/test_ggml_engine.py`.

### 2026-11-12 ROW1 round 5, part 8: the new defaults end to end against llama-server (same kernels)

`lab/row1/part5_p8.sh`, same machine, model, kernels and settings as part 6; the product now with prefix reuse and the
one-call budget (both defaults). Tokens/s per block and mean first token; the server's plain row is part 6's (same
session). "System" adds one shared system prompt of about 90 tokens to every request, on both sides.

| Workload | Setting | N=1 | N=8 | N=16 | N=32 | facts |
| --- | --- | --- | --- | --- | --- | --- |
| plain | llama-server -fa, B16 | 10.2 (0.10 s) | 61.7 (0.72 s) | 95.5 (0.93 s) | 138.9 (1.51 s) | - |
| plain | product | 10.0 / 10.2 / 10.4 (0.11 s) | 66.8 / 64.8 / 64.2 (0.22-0.25 s) | 116.9 / 117.1 / 111.5 (0.24-0.34 s) | 156.2 / 159.0 / 160.3 (0.31-0.56 s) | 49/50 |
| system | llama-server -fa, B16 | 9.1 / 9.5 / 9.7 (0.11-0.12 s) | 55.1 / 54.8 / 54.7 (0.29-0.37 s) | 82.0 / 80.0 / 82.5 (0.38-0.45 s) | 111.4 / 111.2 / 105.8 (0.53-0.66 s) | - |
| system | product | 9.8 / 10.0 / 10.2 (0.11-0.12 s) | 58.6 / 57.9 / 56.1 (0.24-0.27 s) | 83.5 / 84.5 / 80.9 (0.26-0.36 s) | 113.3 / 114.2 / 112.4 (0.41-0.49 s) | 50/50 |

Plain: **1.00x / 1.06x / 1.21x / 1.14x** the server's tokens/s, first tokens 3.1x / 3.3x / 3.3x sooner under load
(equal alone). System prompt: 1.06x / 1.05x / 1.02x / 1.03x, first tokens 1.3x-1.4x sooner. Before today the
product trailed the same kernels under the server at N=32 (0.93x, part 6). The engine picks the free sequence whose
kept KV matches best; the server's first tokens under load suggest its slot choice reuses less (not examined).
Answers: facts 49-50/50 as before. Next: the sequence count is the largest lever left (N=32 gives 2.4x N=8) and is
still set by hand (`IRONMULE_GGML_SEQS`, default 8); the engine should measure and pick it within the machine's memory.

### 2026-11-13 SEQ1: the engine chooses its sequence count itself

Owner, 2026-11-13: "let the engine choose the sequence count itself, like an agent". Built (`choose_sequences`,
`measure_widths`, `decide_width`, `fitting_widths`, `memory_bytes` in `ironmule/ggml_engine.py`; `ironmule serve
--engine ggml` runs it after the gate unless `IRONMULE_GGML_SEQS` is set; `/health` shows it; AGENT.md describes it).
Rule, fixed before the product run: of 8, 16, 32, 64, the widths whose KV cache at the promised request length fits
beside the model and a reserve (a tenth of memory, at least 1.5 GiB); measured one context at a time on one loaded
model; doubled while a doubling buys at least 10 %; remembered per GGUF, settings, length, engine and machine.

Found on the way: this container's cgroup v1 limit is 14.3 GB while `/proc/meminfo` says 16.9 GB, and the first try
was killed for memory (it also held the 8-sequence context open while it built the 32 one). `memory_bytes` now reads
the cgroup's limit and counts its reclaimable file cache as free (the GGUF just read sits there as 4.3 GB of active
cache); each measured context is freed before the next.

The choice on this machine (Qwen3-4B Q8_0, B16, flash attention; KV cell 147,456 bytes):

| Request length | Widths that fit | Measured tokens/s at full load | Chosen | Time |
| --- | --- | --- | --- | --- |
| 2,048 (default) | 8, 16 | 71.4 / 123.9 | 16 | 57 s |
| 1,024 | 8, 16, 32 | 72.5 / 114.9 / 172.9 | 32 | 92 s |

Through the product (`lab/row1/seq1_p9.sh`, `serve.py`'s prompts, 64 tokens, 3 blocks, request length 2,048),
tokens/s and mean first token:

| Width | N=1 | N=8 | N=16 | N=32 | facts |
| --- | --- | --- | --- | --- | --- |
| 8 (the old default) | 10.3 / 11.0 / 10.6 (0.10-0.11 s) | 70.6 / 69.0 / 72.0 (0.20-0.28 s) | 65.6 / 65.6 / 67.0 (4.5 s) | 64.4 / 63.6 / 65.7 (12.3-12.6 s) | 49/50 |
| 16 (chosen by the engine) | 10.5 / 11.3 / 10.2 (0.09-0.10 s) | 70.5 / 71.1 / 71.0 (0.11-0.12 s) | 113.7 / 114.7 / 110.7 (0.31-0.33 s) | 115.0 / 115.2 / 108.5 (5.2-5.6 s) | 49/50 |

**1.72x at 16 requests and 1.76x at 32**, first tokens 14x sooner at 16, the same alone and at 8. Part 8's 1.14x over
`llama-server` at N=32 used 32 sequences of 1,024 tokens, set by hand; at the default length the engine now picks 16
by itself, since 32 sequences of 2,048 tokens do not fit this container. Next: let a request's length decide its
sequence's room (llama.cpp's unified KV cache shares one pool), so the engine need not trade width for length.

### 2026-11-13 KV1: where a step's time goes at longer contexts, and two ways to more width that do not work here

Same container as SEQ1 (4 vCPU Sapphire-Rapids-class Xeon, cgroup limit 14.3 GB), Qwen3-4B Q8_0, llama.cpp 4227c9b
with the ROW1 patch (B16 on), `lab/kv1/`. Question: SEQ1 is limited by KV memory at the default request length of
2,048 (16 sequences fit, 32 do not); can the width grow without changing the model?

- *8-bit KV cache (llama.cpp's `-ctk q8_0 -ctv q8_0`, engine option `kv_q8`).* Half the bytes per cell (78,336 against
  147,456), so 32 sequences fit: 148.6 tok/s at 32 against 96.7 at 16 (f16) at full load. **It fails the quality gate**:
  perplexity ratio to llama.cpp's default 1.0026, 95 % interval [0.9990, 1.0082], above the owner's bound of 1.005
  (16 chunks). It does not serve; the option stays in the engine for models where it passes.
- *One shared KV pool (`kv_unified`), 32 sequences in the cells of 16.* On the CPU every sequence's attention then runs
  over every used cell of the pool: a step at a 64-token context 0.292 s (16 split sequences: 0.143 s), at 512 tokens
  1.388 s (0.429 s). Rejected on this CPU.
- *The finding on the way:* at 16 sequences a full step took 0.143 s at a 64-token context and 0.429 s at 512 tokens,
  and 0.845 s at 1,024 (llama.cpp's flash attention). At chat-like contexts the step is dominated by attention, not by
  the weights; the setting the gate's speed test picks (flash attention, measured at 64 tokens) is the slower one at
  long contexts: without flash attention 0.417 s at 16 x 1,024 against 0.845 s, but 0.213 s against 0.148 s at
  16 x 64. Two follow-ups: FA1 (a faster attention kernel) and AG10 (the agent chooses the attention by the traffic).

### 2026-11-13 FA1: a flash-attention kernel for f16 KV on AVX-512 (llama.cpp patch)

llama.cpp's CPU flash attention walks the keys one at a time per query head: an f16 dot product, an `expf` and an f16
multiply-add of V per key, with V*softmax accumulated in f16, and every KV head read again for each query head that
shares it. FA1 (`native/ggml_patches/fa1/fa1_kern.h`, a 5-line hook in `ggml-cpu/ops.cpp`; `FA1=0` turns it off) does
the same attention with the query heads of a KV head together, keys in blocks of 32 (one running-max update and
rescale per block, exponentials vectorized with ggml's own `ggml_v_expf`), and f32 accumulation. Taken only for f32 Q,
f16 K/V, head sizes 64-256, no ALiBi, no sinks, a mask shared by all heads.

Correctness (`fa1_check.py`-style, four sequences of 700/300/50/5 tokens and a decode step, 16 logit rows): against
llama.cpp without flash attention as the reference, llama.cpp's flash attention differs by up to 0.491 (mean 0.0188)
in a logit, FA1 by up to 0.205 (mean 0.0091); every argmax the same. Gate (`lab/fa1/gate.py`, the numeric gate's 16
chunks through the multi-position decode): FA1's perplexity ratio to llama.cpp's default **1.00001 [0.99993, 1.00009]**
(llama.cpp's flash attention 0.99943). Decode step at full load (median of 3 x 8 steps, `lab/kv1/attn.py`):

| Sequences x context | llama.cpp flash attention | FA1 | llama.cpp without flash attention |
| --- | --- | --- | --- |
| 1 x 64 | 0.099 s | 0.093 s | 0.116 s |
| 1 x 1,024 | 0.139 s | 0.122 s | 0.114 s |
| 16 x 64 | 0.148 s | 0.157 s | 0.213 s |
| 16 x 256 | 0.229 s | 0.208 s | 0.259 s |
| 16 x 1,024 | 0.845 s | **0.481 s** (1.76x) | 0.417 s |

At 16 x 1,024 FA1's attention part still moved only about 7.5 GB/s of KV: compute-bound per key (one horizontal sum
per key and head). **FA1 v2** converts a block's K rows to f32 once and scores 16 keys per head in 16 accumulators,
summed across lanes together (`fa1_hsum16`, checked against scalar sums); logits and gate ratio unchanged (1.00001
[0.99993, 1.00009]). Steps, same probe:

| Sequences x context | llama.cpp flash attention | FA1 v2 | llama.cpp without flash attention |
| --- | --- | --- | --- |
| 1 x 64 | 0.099 s | 0.097 s | 0.116 s |
| 1 x 1,024 | 0.139 s | **0.104 s** | 0.114 s |
| 16 x 64 | 0.148 s | **0.134 s** | 0.213 s |
| 16 x 256 | 0.229 s | **0.154 s** | 0.259 s |
| 16 x 1,024 | 0.845 s | **0.314 s** (2.69x) | 0.417 s |

FA1 v2 is now the fastest of the three at every measured size, so the choice of attention that KV1 left to the agent
is settled on this CPU in FA1's favour (the agent keeps the choice for machines where it is not).

### 2026-11-13 FA1 end to end: IronMule against stock llama-server with a long system prompt

`lab/fa1/e2e.sh` (stock servers) and `lab/row1/product.py` (IronMule), same container, Qwen3-4B Q8_0, every request
with the same 479-token system prompt (`lab/fa1/system.txt`, a library's house rules) and a short story request, 64
tokens, one warm-up and 3 blocks of N streamed requests. Stock = the same llama.cpp build (4227c9b) with ROW1 and FA1
off (`ROW1_GGML=0 FA1=0`), `llama-server -np N -c N*2048 -t 4`, with and without `-fa`. IronMule = the product
(service, worker, ggml engine with B16 and FA1 v2, flash attention as its gate chose, 16 sequences as SEQ1 chose, prefix
reuse, speculation). Tokens/s per block (mean first token):

| N | llama-server -fa | llama-server | IronMule (FA1 v1) | IronMule (FA1 v2) |
| --- | --- | --- | --- | --- |
| 1 | 6.5 / 7.4 / 7.1 (0.13-0.15 s) | 7.6 / 7.9 / 7.6 (0.13 s) | 9.7 / 9.1 / 9.3 (0.11 s) | 8.7 / 9.0 / 9.8 (0.11-0.13 s) |
| 8 | 22.9 / 22.8 / 12.1* (0.38-0.67 s) | 28.1 / 26.7 / 27.1 (0.43-0.71 s) | 41.6 / 41.8 / 44.1 (0.37-0.49 s) | 48.4 / 47.4 / 47.5 (0.27-0.28 s) |
| 16 | 31.0 / 32.7 / 17.9* (0.74-0.83 s) | 41.2 / 41.4 / 40.2 (0.67-0.73 s) | 54.3 / 56.7 / 56.7 (0.55-0.78 s) | 66.6 / 67.5 / 68.4 (0.44-0.47 s) |

*Third block disturbed by a test run of this session on the same CPU; the other two stand.

Against the better stock server (without `-fa` at this context): **1.19x / 1.75x / 1.65x** tokens per second with
FA1 v2, first tokens 1.1x / 1.6-2.6x / 1.5x sooner. Against `llama-server -fa`: 1.31x / 2.09x / 2.12x.

Answers. With the library system prompt the model often declines general questions ("ask the staff at the desk"):
facts 36/50 through IronMule, 38/50 through stock `llama-server` (`lab/fa1/facts_server.py`), the misses being such
refusals on both. Without a system prompt: 49/50 on both.

### 2026-11-13 FA1 on other families (Gemma 3, Llama 3.2, Mistral), this container

`lab/fa1/families.py`: logits of four sequences (700/300/50/5 tokens) and a decode step against llama.cpp without
flash attention; perplexity on the numeric gate's text tokenized by the GGUF itself (16 chunks); full-load decode
steps at 8 sequences. ROW1 off (these are not Q8_0-only kernels' models; B16 serves Q8_0 weights only).

| GGUF | gate ratio of FA1 to llama.cpp's default | logit error vs no flash: FA1 / llama.cpp flash (max, mean) | 8 x 64: no flash / llama.cpp flash / FA1 | 8 x 1,024: no flash / llama.cpp flash / FA1 |
| --- | --- | --- | --- | --- |
| Gemma 3 1B it Q8_0 (head 256, sliding window) | 1.0008 [0.9989, 1.0026] | 1.57, 0.113 / 1.88, 0.138 | 0.060 / 0.057 / **0.048 s** | 0.068 / 0.069 / **0.062 s** |
| Llama 3.2 1B Instruct Q8_0 (head 64) | 1.00001 [0.99998, 1.00005] | 0.026, 0.002 / 0.114, 0.010 | 0.046 / **0.038** / 0.041 s | 0.073 / 0.099 / **0.062 s** |
| Mistral 7B Instruct v0.3 Q4_K_M | 0.9989 [0.9955, 1.0017] | 0.79, 0.060 / 1.43, 0.064 | 0.414 / 0.419 / **0.360 s** | 0.519 / 0.751 / **0.486 s** |

Every argmax agreed. FA1 passes the gate's bound (1.005) on all three and is closer to the non-flash reference than
llama.cpp's flash attention on each. Note on the owner's stricter reading for IronMule's own kernels (2026-11-10:
perplexity not above llama.cpp's default, ratio at most 1.000): FA1's point ratio is 1.00001 on Qwen3-4B and Llama 3.2
and 1.0008 on Gemma 3 1B, inside noise but above 1.000; under the 2026-11-11 reading for llama.cpp's own options (flash
attention is one; within 1.005) it passes. The product's gate applies the latter. **Owner, 2026-11-13: "FA1 ist in
Ordnung"** -- FA1 counts as the implementation of llama.cpp's flash-attention option and is held to its bound (1.005).

### 2026-11-13 AG10 S: the ggml engine's agent in simulation, traffic that changes regime

`training/ag10_sim.py`, 3 seeds x 3 phases x 60 windows of 32 arrivals (an arrival every 0.25 s on average, the engine
busy): short chats (prompts 20-120 tokens), long-context requests (1,200-1,700), short again. True step costs from
KV1's measurement on this CPU (16 and 8 sequences, flash attention or not); the agent's start values off by up to
15 %; 3 % noise on every observed step. Tokens/s of the agent's choice against the fixed incumbent (SEQ1's choice,
16 sequences with flash attention) and against the best fixed configuration in hindsight:

| Phase | agent / incumbent | agent / best fixed (which) |
| --- | --- | --- |
| short | 1.00 / 1.00 / 1.00 | 1.00 (w16-fa) |
| long | 1.196 / 1.200 / 1.197 | 0.985 / 0.988 / 0.985 (w16, no flash) |
| short again | 0.979 / 0.961 / 0.972 | same (w16-fa) |

Every seed switched to llama.cpp's matmul attention a few windows into the long phase and back in the third. Run 1
(before the fix) did not come back: the served configuration's least-squares fit, seeing only full steps at long
contexts, could not tell rows from attended tokens apart and drifted where it had no data, and the trial was judged on
traffic from before the switch (0.86 of the best in the third phase). Fixed: the startup measurement's points are folded
into every model at each idle call, and a trial is judged only on the traffic since its switch. On this CPU with FA1 v2
the choice is moot (FA1 is fastest at every size, FA1 above); the agent keeps it for machines where it is not.

### 2026-11-13 AG10 live: `ironmule serve --agent on --gguf` on this container, and two defects it showed

`ironmule serve --state-dir ... --model Qwen/Qwen3-1.7B --gguf Qwen3-4B-Q8_0.gguf --agent on` (B16 and FA1 v2 in the
library), driven over HTTP by waves of 12 chat requests (48 tokens, every second one with the 479-token system
prompt), 10 s idle between rounds. Startup: gate passed with flash attention (FA1, ratio 1.00001), SEQ1 chose 16
sequences, AG10 got four configurations (8/16 sequences x flash attention on/off).

First run, AG10 version 1: at its first idle spell it switched to 8 sequences with flash attention, and the next waves
took 17-33 s instead of 9 s. Two causes, both fixed in version 2: (1) only the served configuration learned what a
prompt costs (3.3 s per 1,000 tokens here), the others had no prefill price and dreamed it free; the width measurement
now times the prefill (`prefill_per_1k`) and it is an anchor; the traffic memory now keeps the prompt tokens actually
computed after prefix reuse. (2) A configuration fitted on live steps was compared with ones known only from the
startup measurement, and the live machine was slower (factor 1.12-1.57 here, falling as the live model filled); a
configuration is now priced by its start model times a live factor until it has 200 live steps. Regression tests in
`tests/agent/test_ggml_agent.py`.

Version 2, three rounds of 5 waves: 183 requests, none failed; every idle decision "stay" (16 sequences with FA1, which
is the fastest configuration at these contexts per FA1 v2's table); waves 7.3-11.4 s (35-54 tok/s, the first wave of a
fresh server 22 s with its cold prefix cache). `/health` shows the agent under `ggml.agent`. A switch under real traffic
whose regime changes is still to be shown on hardware (simulated in AG10 S).

### 2026-11-13 SP1: one KV pool with shared prefixes on the CPU (FA1u), this container

Owner, 2026-11-13: clearly beat llama.cpp, Ollama, LM Studio, vLLM and SGLang. llama.cpp (and so Ollama and LM Studio,
which run it) gives every parallel slot its own cells and reuses a slot's own last prompt only; with a system prompt
every request holds and reads its own copy. vLLM and SGLang share prefix pages (paged attention, RadixAttention) on
GPUs. SP1 does that on llama.cpp's CPU path:

- *FA1u* (`native/ggml_patches/fa1/fa1_kern.h`): several query tokens of one stream together (a unified cache, or a
  prompt), each key block's K and V converted once for all of them, scoring 16 keys at a time where most are visible and
  key by key where few are, the visible keys only into V. A block that 16 sequences see is read once per step.
- *SharedPool* (`ironmule/ggml_engine.py`): llama.cpp's unified cache; a request takes the longest prefix any running or
  finished sequence holds via `llama_memory_seq_cp` (cells gain a sequence id, nothing is copied), reserves only its
  prompt and answer limit, cells are counted by reference, finished sequences stay as a cache and are evicted least
  recently used; a prompt that shares 64+ more tokens with one starting in the same step waits one step to share it.

Correctness (`share_check.py`, 16 sequences = 400 shared + 20 own tokens and a step): logits against split sequences
without flash attention max 0.057 apart (FA1 on split sequences: 0.056), every argmax the same.

Decode step at full load (`lab/fa1/share_speed.py`, Qwen3-4B Q8_0, 480-token shared prefix + 80 own tokens):

| Sequences | split, FA1 | shared pool, FA1u | | setup (prefill) split / shared |
| --- | --- | --- | --- | --- |
| 16 | 0.218 s (73.4 tok/s) | 0.158 s (101.0 tok/s) | 1.38x | 33.5 s / 7.9 s |
| 32 | 0.364 s (88.0 tok/s) | 0.271 s (118.0 tok/s) | 1.34x | 64.8 s / 13.3 s |

Through the engine's run loop (`lab/fa1/share_run.py`, 16 chat requests with the 479-token library system prompt, 48
tokens, speculation on, then a second wave of 16 new questions):

| | split (prefix reuse per sequence) | shared pool |
| --- | --- | --- |
| cold burst | 10.1 tok/s, first token 17.5 s | **51.8 tok/s, first token 2.7 s** (5.1x) |
| second wave | 43.1 tok/s, first token 1.39 s | **66.3 tok/s, first token 1.11 s** (1.54x) |
| pool cells used afterwards | 16 x ~560 per sequence | 1,789 for all |

30 of 32 answers token-identical with the split run (the two others differ by batching order). The bridge serves the
pool when FA1u is in the library (`ironmule_fa1_version`), the gate's own check of the unified setting passes, and the
layers are on the CPU; it runs twice SEQ1's width at once in the same memory.

### 2026-11-13 SP4: collective speculation (code, not measured yet)

`ironmule/speculation.py` `CollectiveIndex`: every 2-4-gram of recent answers (all users) maps to the tokens that
followed; a request without a long own match drafts from it in its own acceptance bucket (100 + n), so the step policy
learns whether it pays. Kept tokens are always the model's own next tokens. On by default in the ggml engine
(`IRONMULE_GGML_COLLECTIVE=0` turns it off).


### 2026-11-13 CMP1: IronMule against llama.cpp, Ollama and LM Studio on this CPU, same GGUF

Owner, 2026-11-13: clearly beat llama.cpp, Ollama, LM Studio, vLLM and SGLang (MLX left out for now). This container
(4 vCPU Sapphire-Rapids-class Xeon, 14.3 GB cgroup limit), Qwen3-4B Q8_0 (the same GGUF file for every engine),
`lab/cmp/bench.py` through each engine's OpenAI-compatible chat API, every answer re-tokenized with Qwen3's tokenizer,
one engine at a time, one server configuration for all N (as a deployment would run it). Setups (`lab/cmp/run_all.sh`):
IronMule `ironmule serve --agent on --gguf` (FA1/FA1u + ROW1 B16 in the library, SEQ1 width 16, shared pool of 32
running requests, AG10, SP4); llama.cpp 4227c9b stock (`ROW1_GGML=0 FA1=0`) `llama-server -np 32 -c 32768 --jinja`,
with and without `-fa`; Ollama 0.40.16 with its own llama.cpp, `OLLAMA_NUM_PARALLEL=32` (its runner took about 12 GB
and was killed for memory twice; a second run with 16 slots was killed once at N=32), `OLLAMA_CONTEXT_LENGTH=1024`;
LM Studio llmster 0.0.27, runtime llama.cpp avx2 2.66.0 (its Linux CPU runtime has no AVX-512 build), `--parallel 32
--context-length 32768 --gpu off`; LM Studio chose 1 thread on this container by itself, so it was run again with
`llm.load.llama.cpuThreadPoolSize = 4`, and only partly (it was too slow for the whole set: burst with the system
prompt 4.8 / 8.4 / 7.2 tok/s at N = 1 / 8 / 16 with first tokens 1.5 / 11.9 / 33.2 s, from its log; the rest below).
Results in `lab/cmp/results/`.

Burst with the 479-token system prompt (`burst-sys`), 64 tokens, median of 3 blocks: tokens/s (mean first token)

| Engine | N=1 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- |
| ironmule | 10.7 (0.19 s) | 50.4 (0.71 s) | 71.3 (1.19 s) | 87.5 (2.30 s) |
| llama-server | 7.3 (0.80 s), IronMule 1.47x | 20.1 (3.22 s), IronMule 2.51x | 28.1 (5.70 s), IronMule 2.54x | 28.9 (11.31 s), IronMule 3.03x |
| llama-server-fa | 6.2 (0.90 s), IronMule 1.72x | 16.9 (3.99 s), IronMule 2.99x | 19.8 (7.40 s), IronMule 3.61x | 21.6 (14.72 s), IronMule 4.04x |
| ollama | 7.3 (0.87 s), IronMule 1.46x | 18.9 (3.86 s), IronMule 2.67x | 7.6 (7.05 s), IronMule 9.33x | 7.8 (14.45 s), IronMule 11.23x |
| ollama-16 | 6.8 (0.87 s), IronMule 1.57x | 17.7 (4.32 s), IronMule 2.85x | 22.7 (6.58 s), IronMule 3.14x | 21.8 (24.50 s), IronMule 4.01x |
| lmstudio | - | - | - | - |

Burst without a system prompt (`burst`): tokens/s (mean first token)

| Engine | N=1 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- |
| ironmule | 10.7 (0.15 s) | 58.7 (0.53 s) | 84.9 (0.92 s) | 113.2 (1.58 s) |
| llama-server | 7.6 (0.83 s), IronMule 1.42x | 8.9 (5.92 s), IronMule 6.60x | 14.1 (8.38 s), IronMule 6.01x | 54.4 (8.20 s), IronMule 2.08x |
| llama-server-fa | 7.3 (0.78 s), IronMule 1.48x | 8.2 (5.61 s), IronMule 7.13x | 12.7 (8.97 s), IronMule 6.68x | 60.3 (7.66 s), IronMule 1.88x |
| ollama | 8.8 (0.67 s), IronMule 1.22x | 10.9 (4.31 s), IronMule 5.39x | 17.9 (6.46 s), IronMule 4.74x | 61.1 (7.89 s), IronMule 1.85x |
| ollama-16 | 8.3 (0.66 s), IronMule 1.29x | 17.0 (3.35 s), IronMule 3.45x | 60.2 (4.06 s), IronMule 1.41x | 61.3 (11.84 s), IronMule 1.85x |
| lmstudio | 5.2 (1.49 s), IronMule 2.05x | 13.3 (9.28 s), IronMule 4.42x | - | - |

One request alone (256 tokens), open-loop Poisson arrivals at 0.5/s for 90 s (system prompt, 64 tokens), the 50 facts:

| Engine | single: decode tok/s, first token | Poisson: tok/s, first token p50 / p90, end to end p90 | facts |
| --- | --- | --- | --- |
| ironmule | 10.1, 0.13 s | 18.8, 0.33 / 0.38 s, 8.6 s | 49/50 |
| llama-server | 8.0, 0.83 s | 7.0, 113.02 / 140.17 s, 202.2 s | 49/50 |
| llama-server-fa | 8.6, 0.71 s | 6.4, 124.23 / 150.58 s, 227.7 s | 49/50 |
| ollama | - | 7.3, 84.16 / 91.78 s, 189.9 s | - |
| ollama-16 | 8.6, 0.70 s | 11.4, 47.06 / 57.21 s, 90.8 s | 49/50 |
| lmstudio | 6.1, 1.35 s | - | 49/50 |

Reading: with many users IronMule serves 2.5-4x the tokens per second of `llama-server` and 2.7-4x Ollama's with a
shared system prompt, first tokens 4-6x sooner; at 0.5 requests/s every llama.cpp-based server fell behind its queue
(first token p50 47-124 s) while IronMule answered in 0.33 s (p90 0.38 s); alone it decodes 1.17-1.66x faster with a
first token 5-10x sooner. Answers: 49/50 on every engine. Two cautions. (1) llama.cpp's servers with 32 slots pay for
all slots in every step (`llama-server`'s log: 0.44-0.52 s a step with 8 requests running, without a system prompt), which
makes their N=8/16 rows low; sized to the load (`-np N`, FA1 end-to-end entry above) `llama-server` made 27.3 / 40.9
tok/s at N=8/16 with the system prompt, still 0.54x / 0.57x of IronMule here. (2) One machine, one run per engine.
vLLM: no CPU wheel on PyPI, triton-cpu behind a blocked host, the source build stopped for disk and memory at 194 of 416
steps; vLLM and SGLang are measured on GPUs instead (MG2 harness, Kaggle T4, once the weekly quota is back).

### 2026-11-13 CMP1, multi-turn chats: IronMule against llama-server

Same setups as CMP1 (`bench.py --workloads multiturn`). Each of N users holds a 3-turn conversation: every turn sends the
479-token system prompt, the earlier turns and a new question (64-token answers); a user's next turn starts when its
answer is in, users run at once. Total tokens/s and mean first token per turn:

| Users | IronMule | llama-server | first token, turns 1 / 2 / 3: IronMule | llama-server |
| --- | --- | --- | --- | --- |
| 8 | 38.6 tok/s | 8.7 tok/s (**4.4x**) | 2.67 / 0.79 / 0.71 s | 40.07 / 3.24 / 3.64 s |
| 16 | 47.8 tok/s | 13.0 tok/s (**3.7x**) | 0.72 / 3.06 / 6.53 s | 43.09 / 4.77 / 6.91 s |

No failed request on either. Open: at 16 users IronMule's later turns wait about as long for a first token as
llama-server's (a returning turn's own new tokens queue behind running steps while 32 may run); a turn that continues
its own kept sequence could be admitted ahead of fresh conversations.

### 2026-11-13 RL-MATH: a roofline of this CPU, and FA1x (attention on the AMX matrix unit)

Owner, 2026-11-13: "approach it mathematically, maybe there is something left". Measured: streaming read 47.4 GB/s with
4 threads (`lab/row1/bw.c`, huge pages; 11.6 / 22.6 GB/s with 1 / 2). Qwen3-4B Q8_0 reads 4.28 GB of weights per decode
step (the output matrix is the tied embedding). AMX int8 peak about 1,024 MACs/cycle/core x 2.3 GHz x 4 = 9.4e12 MAC/s;
AMX bf16 half of that; AVX-512 FP32 about 2.9e11 FLOP/s.

| Quantity | Model | Measured | Share of the bound |
| --- | --- | --- | --- |
| one request: step floor = weights / bandwidth | 4.28 GB / 47.4 GB/s = 90 ms (11.0 tok/s) | 99 ms (10.1 tok/s) | 92 % |
| per extra row of a step (B16: two int8 passes) | 2 x 4.0e9 MAC / 9.4e12 = 0.85 ms | 2.4 ms (SEQ1: 8 -> 16 rows) | 35 % of AMX peak |
| attention per row at a 560-token chat context, 32 rows | 165 M MAC in FP32 on the vector units | about 3 ms | about a third of FP32 peak |

So: (1) one user alone is at 92 % of what the memory can deliver; only more tokens per weight read (speculation) can
add there. (2) The row term is at a third of AMX peak (B16 kernel efficiency, ROW1's ground). (3) The attention of a
busy chat step runs on the vector units while the matrix unit, ten times faster in bf16, idles.

**FA1x** (`native/ggml_patches/fa1/fa1_kern.h`, on by default with FA1/FA1u, `FA1X=0` turns it off) moves (3) to AMX:
- *Accuracy by an error-compensated split:* bf16 alone (8-bit mantissa) put prefill logits up to 2.7 off. Each operand
  x = hi + lo with hi = bf16(x), lo = bf16(x - hi) (exact for f16 K and V, about 16 bits for f32 Q and P), and every
  product as hi*hi + hi*lo + lo*hi (lo*lo, about 2^-16, dropped): the Ozaki-style split, three AMX passes, fp32
  accumulation. Logits then within 0.06 (shared-pool decode) and 0.26-0.28 (prefill) of the non-flash reference, as
  close as FA1u (0.06 / 0.21); every argmax the same.
- *Layout by orientation:* v1 computed S = Q K^T and spent ten times the AMX work re-laying K into tiles each block (it
  was slower than FA1u). v2 computes S^T = K Q^T: K stays in its row layout as the A tile, Q^T is laid out once per
  unit, the online softmax runs down the columns of S^T 16 rows per vector, P is gathered to rows, O += P V with V rows
  interleaved in pairs by one permute.
- *Density routing:* a block's AMX cost is about fixed, the vector path's grows with the visible (row, key) pairs; in a
  unified pool the other sequences' cells make most blocks sparse (each key seen by one sequence's rk2 rows). A block
  with at least `FA1X_DENSE` = 480 visible pairs goes to AMX (the shared prefix), the rest to FA1u's per-pair path.
  480 / 1,500 / 3,000 measured within noise; 480 kept.

Decode step, Qwen3-4B Q8_0, shared pool, 480-token shared prefix + 80 own tokens (`share_speed.py`), and the setup
(prefill of the prefix and the own tokens):

| Sequences | FA1u | FA1x | setup FA1u / FA1x |
| --- | --- | --- | --- |
| 16 | 0.163 s | 0.160-0.175 s (same within noise) | 7.5 s / 5.6-5.9 s |
| 32 | 0.258 s (124 tok/s) | **0.221 s (144.5 tok/s)**, 1.17x | 12.7 s / 10.2-10.6 s (1.2x) |

Gate: Qwen3-4B 1.00002 [0.99994, 1.00010]; Llama 3.2 1B 1.00011 [0.99931, 1.00101]; Gemma 3 1B 1.0024 [1.0004, 1.0044]
(passes 1.005, but less margin than FA1's 1.0008; Gemma 3 1B is the most sensitive of the three: llama.cpp's own flash
attention moves its logits by up to 3.1, FA1x by 3.8). The product gates every GGUF at its start, so a model where FA1x
fails falls back.

Through the product (`run_all.sh`, `burst-sys`, same machine and setup as CMP1): N=8 / 16 / 32 **52.7 / 82.4 / 113.3
tok/s**, first token 0.59 / 0.90 / 1.88 s, against CMP1's 50.4 / 71.3 / 87.5 tok/s and 0.71 / 1.19 / 2.30 s: 1.05x /
1.16x / 1.29x. Against CMP1's `llama-server` at N=32: 3.9x the tokens per second.

Next by the same arithmetic: (2) the row term (B16 at 35 % of AMX peak: two int8 passes per row; an exact
error-compensated int8 split like the one above costs the same two passes, so the gain must come from the kernel's
efficiency, ROW1); (1) for one user, speculation: rows cost 2.4 ms against a 99 ms step, so verifying k drafts costs
about 2.4 % each, and with per-token acceptance a the expected tokens per step are (1 - a^(k+1)) / (1 - a): a draft source
with a = 0.6 at k = 7 would give 2.4 tokens for 1.17 steps, 2.1x; prompt lookup's a on free text is far lower, which is
why tree drafts (several candidates per position, cheap as shared-pool sequences) are the next mathematical lever.

### 2026-11-14 DM1: a draft model for one or two requests, stopped by an optimal rule (engine numbers; product run pending)

RL-MATH put one request at 92 % of the memory bound and a verification row at 2-3 % of a step: one user can only gain
by more tokens per read of the weights. Target Qwen3-4B Q8_0, draft Qwen3-0.6B Q8_0 (same tokenizer), greedy, 8 prompts
(story, explanation, code, list, JSON, summary, instructions), 128 tokens, B16 + FA1 in the library.

*Probe* (`lab/dm1/probe.py`, raw decode calls, host at 2.3 GHz): plain 12.9-13.2 tok/s. Fixed draft length k = 3 / 5 / 7:
20.6 / 19.5 / 17.6 tok/s (1.56x / 1.48x / 1.33x); a draft call 15-17 ms, a verify step 82-89 ms. The optimal myopic
stopping rule (draft on while q x mean chance / (call + row) beats the step's rate so far; chance from the draft's
top-1 probability through a calibration learned online): **21.9-22.0 tok/s, 1.71x**, 2.8 tokens per step, acceptance
0.66. A first form of the rule that priced the already computed draft instead of the next one made 1.49-1.56x.
Learned calibration (acceptance by confidence decile): 0.03, 0.14, 0.28, 0.30, 0.40, 0.45, 0.53, 0.74, 0.77, 0.97.
8 of 8 answers token-identical to plain decoding in every run.

*In the engine* (`ironmule/draft_model.py`, `GgmlEngine(draft_gguf=...)`, `lab/dm1/engine_run.py`, shared pool,
collective speculation off for the comparison), tokens/s plain / with the draft model:

| Requests at once | plain | draft model | |
| --- | --- | --- | --- |
| 1 | 13.67 | **22.16** | 1.62x |
| 2 | 23.78 | **32.37** | 1.36x |
| 4 | 44.35 | 42.96 | 0.97x (it drafts only while at most 2 run) |
| 8 | 79.06 | 76.57 | 0.97x |

32 of 32 answers token-identical. With four draft slots and no limit on the load, 4 and 8 requests lost (42 and 60-65
tok/s): steps with drafts made 33-45 tok/s at 3-7 running requests where plain steps make 36-70, although the cost
model predicted about +25 % at 4. Not explained yet (draft calls of 2-3 rows measured 27-32 ms there against 16.5 ms
for one row). So the draft model rests above `IRONMULE_GGML_DRAFT_MAX_RUNNING` = 2, and a measured check stops it at
any load where steps with drafts were slower than plain ones. Product: `ironmule serve --gguf BIG --draft-gguf SMALL`.
The session's container moved to another host (2.1 GHz) during this entry; the product numbers follow on that host
with their own plain baseline.

### 2026-11-14 PI1: the Pilot, a trained decision network for every engine step, and the busy-server collapse

*The collapse* (`lab/pi1/threads.py`, Qwen3-4B Q8_0, 4-vCPU Xeon, B16 + FA1, H busy foreign processes). llama.cpp
splits each matmul across its threads and waits at a barrier for the slowest; a thread that shares its core with a
foreign process holds everyone up. Seconds of a one-row step:

| foreign | 1 thread | 2 threads | 3 threads | 4 threads |
| --- | --- | --- | --- | --- |
| 0 | 0.414 | 0.184 | 0.139 | **0.108** |
| 1 | 0.353 | 0.186 | **0.136** | 2.423 |
| 2 | 0.344 | **0.194** | 1.103 | 5.480 |
| 3 | 0.343 | 0.333 | 4.959 | 5.656 |

Stock `llama-server` (all cores, `-np 8`, `lab/pi1/busy_cmp.sh`, workload `quick`) under the same load: 7.23 / 0.24 /
0.13 tok/s for one user with 0 / 1 / 2 foreign processes (first token 0.83 / 23.5 / 45.9 s), 23.3 / 1.41 / 0.68 tok/s for
eight (first token 3.7 / 54 / 149 s).

*The Pilot* (`ironmule/pilot.py`, weights `ironmule/pilot_weights.npz`): an MLP 20 -> 128 -> 128 -> four heads
(threads as a fraction of the usable cores, running cap, prompt tokens per step, drafting), trained from scratch in a
dreamed world of continuous batching whose step times follow the measured costs and this collapse, randomised over
cores, bandwidth, model size, traffic, bursts and foreign processes; imitation of "threads = free cores" first, then
PPO. The world and the recipe stay private (open weights only).

*In the engine* (`lab/pi1/real.py`, shared pool cap 32, collective speculation off, 32 tokens, one block per cell;
`default` = all 4 threads, no Pilot), tokens/s:

| foreign | default, 1 | Pilot, 1 | default, 8 | Pilot, 8 | Pilot's threads |
| --- | --- | --- | --- | --- | --- |
| 0 | 9.74 | 8.98 | 61.3 | 53.1 | 4 |
| 1 | 0.30 | **6.76** | 2.51 | **50.8** | 3 |
| 2 | 0.18 | **5.30** | 1.45 | **38.5** | 2 |

The idle row's gap is the first measured cell after startup: alternating on one engine (`ab.py`, three rounds, 32
tokens) the Pilot and no Pilot made 10.21 / 9.83 / 9.79 against 9.34 / 9.75 / 9.86 tok/s for one request and 60.9 /
68.0 / 72.6 against 70.8 / 67.6 / 71.6 for eight. Answers were token-identical at 4, 3, 2 and 1 threads (one and four
requests). Decision time in the server 531-558 microseconds per step (the MLP about 20; the rest is the `/proc` read,
slowed by the engine's polling threads), 0.5 % of a step.

*Through the product* (`lab/pi1/busy_cmp.sh`, `ironmule serve --gguf --agent on`, workload `quick`, against the
`llama-server` numbers above), tokens/s and first token:

| foreign | llama-server, 1 | IronMule, 1 | llama-server, 8 | IronMule, 8 |
| --- | --- | --- | --- | --- |
| 0 | 7.23, 0.83 s | **9.33, 0.27 s** | 23.3, 3.7 s | **58.8, 0.54 s** |
| 1 | 0.24, 23.5 s | **7.07, 0.31 s** | 1.41, 54 s | **46.2, 0.34 s** |
| 2 | 0.13, 45.9 s | **5.46, 0.29 s** | 0.68, 149 s | **35.2, 0.68 s** |

With collective speculation on (the default), the same run made 9.66 / 21.8 / 22.4 for one request: `quick` repeats its
prompts, so from the second load level on SP4 drafted from the answers it had already served. That is real for
repeated questions but not the Pilot's doing, hence the table without it. `/health` showed `ggml.pilot` deciding 4 / 3 /
2 threads with 0.0 / 1.03 / 1.98 foreign cores seen. In every measured cell the other three heads kept the engine's defaults (running cap at the pool's
limit, no prompt budget, drafting allowed): at these loads only the thread count changed anything.

*Load that comes and goes* (`real.py`, `REAL_HOGS=dyn`: one foreign process switched on or off every 6 s, set at
each block's start, 3 blocks of 32 tokens): Pilot 7.0 / 38.2 tok/s for 1 / 8 requests, defaults 0.32 / 6.25. The
switch is read at block starts, so the slower arm spent a different share of its time with the process on; a rough
check that the Pilot follows a changing load, not a clean ratio.

*Training.* Two versions, each imitation (3,000 steps, thread accuracy 0.998) and 150 PPO iterations on 1,024 dreamed
servers on this CPU (about 33 minutes). The first world showed no CPU load at all without a foreign process; the real
machine shows 0.0-0.2 cores of background, and the first weights gave up a core from 0.2 foreign cores on. The world
now adds 0-0.35 cores of harmless background to what the Pilot sees; the shipped weights keep all 4 threads up to 0.5
foreign cores and choose 3 at 0.8-1.0 and 2 from 1.5. Dream evaluation (240 s, 1,024 machines, reward per server):
Pilot -1436, defaults -2251, the hand rule "threads = free cores" -1417; PPO's best checkpoint came after 25 iterations
(-493.6 against -508.9 after imitation on the 120 s evaluation), later iterations did not improve on it. Also found
on the way: `ironmule serve --gguf` timed out at its 120 s worker start once the start measured the Pilot's costs (105 s
here); the ggml worker now has the 600 s the dense plan has.

*Next:* a GPU-trained version (Kaggle, 8,192 servers, 400 iterations, two seeds) against these weights; GPU servers
(MG2), where the Pilot's thread head does not act and only cap, budget and drafting remain.

### 2026-11-14 MG2 v3: the first GPU numbers (Kaggle T4 x2, Qwen3-4B)

`lab/mg2/kaggle_gpu.py` at 43d78f5 on a Kaggle "GPU T4 x2" session (CUDA build of llama.cpp 4227c9b for sm_75, an hour
of the session). Qwen3-4B Q4_K_M for llama-server and IronMule's engine (`ironmule_*`: per-sequence KV, speculation
from the context, Pilot active; one GPU or one engine per GPU), Qwen3-4B in float16 for vLLM; 128 tokens of a short
story per request, greedy, N requests at once, median of 3 blocks after a warm-up, tokens/s and mean first token:

| arm | 1 request | 8 requests | 32 requests |
| --- | --- | --- | --- |
| llama-server, one GPU | 53.8, 0.022 s | 146.4, 0.137 s | 504.0, 0.389 s |
| llama-server, split over both | 64.4, 0.019 s | 188.1, 0.097 s | 453.9, 0.334 s |
| llama-server, one per GPU | 36.7, 0.028 s | 172.2, 0.135 s | 735.0, 0.275 s |
| vLLM, one GPU (float16) | 22.1, 0.048 s | 185.0, 0.092 s | 473.7, 0.153 s |
| **IronMule, one GPU** | **110.4, 0.016 s** | 118.1, 0.038 s | **683.9, 0.039 s** |
| **IronMule, one per GPU** | **110.3, 0.016 s** | 177.7, 0.035 s | **958.5, 0.028 s** |

Facts: 49 of 50 through llama-server and through the engine (same answers checked), 50 of 50 for vLLM's float16
model. *Correction (v4):* the one-request column is not a fair comparison: the blocks repeat their prompts and the
engine's collective speculation (SP4) drafted from the block before (v4: 315 of 328 drafts accepted); v5 turns SP4
off. At 8 and 32 requests the engine drafted nothing, so those columns stand. IronMule is 1.7-2x the best llama-server arm for one request and 1.3x for 32 (958.5 against 735.0 for both
setups one per GPU), with first tokens about 4x sooner at 8 and 10x sooner at 32 than llama-server in the same setup. At 8 requests on one GPU it is behind llama-server
(146.4) and vLLM (185.0): 8 rows cost almost as long as 8 single steps, which the CPU does not show (8 story requests
there: 62-68 tok/s against 10 for one, per-sequence or shared KV alike). Not measured: SGLang (its `outlines_core`
has no wheel for Kaggle's Python 3.13), Gemma 3, Mistral 7B and Llama 3.1 8B (llama-server could not load them: vLLM's
engine process outlived its killed parent and kept GPU 0's memory). v4 (30dcc74) kills the Python servers' process
groups and waits for the memory, installs SGLang into a 3.12 venv, adds vLLM with Qwen3-4B-AWQ (4-bit), adds the engine
with one shared KV pool, records the engine's step times by rows, and reuses this build (a private dataset) instead of
building again.

### 2026-11-14 PI1 (continued): Kaggle-trained weights, part-time foreign load, new shipped weights

*Kaggle training.* The private kernel `ironmule-pilot-train` (two T4s, one seed each, 8,192 dreamed servers, 400 PPO
iterations after imitation, 67 minutes after 3 h 15 min in the queue). On the same dream evaluation as the first
weights (240 s, 1,024 machines, seed 12345): seed 1 -1406.7, seed 0 -1424.3, first weights -1436.3, the hand rule
"threads = free cores" -1416.9. On the real machine seed 1 chose the same threads (4 / 3 / 2 with 0 / 1 / 2 busy
processes) at the same speed within noise; it differed only in giving up a core from 0.5 foreign cores instead of 0.8.

*Part-time load* (`lab/pi1/half.py`, one foreign process busy a fraction of every 20 ms, Qwen3-4B Q8_0, step seconds):

| busy | 3 threads, 1 row | 4 threads, 1 row | 3 threads, 8 rows | 4 threads, 8 rows | seen at 4 threads |
| --- | --- | --- | --- | --- | --- |
| 0 % | 0.134 | **0.113** | 0.161 | **0.137** | 0.07-0.09 |
| 25 % | **0.130** | 0.139 | **0.152** | 0.184 | 0.0-0.22 |
| 50 % | **0.133** | 0.217 | **0.162** | 0.239 | 0.23-0.32 |
| 75 % | **0.145** | 0.305 | **0.156** | 0.337 | 0.44-0.51 |
| 100 % | **0.143** | 1.723 | **0.175** | 1.053 | 0.95-0.97 |

A process busy a quarter of the time already makes 4 threads the wrong choice, and the reader sees about half of a
part-time process while our threads starve it. The dream world had only processes that were on or off and treated up
to 0.35 cores of background as harmless, so every version kept 4 threads far too long. World v3 (private): half the
foreign processes are part-time (duty 0.1-0.9, a 4-thread step slower by 0.9-2.5x per core of partial excess, seen at
40-70 % while starved), background 0-0.12 cores. Kaggle seed 1, fine-tuned 100 iterations in it on this CPU (29
minutes; best checkpoint after 75), is the new `ironmule/pilot_weights.npz`.

*In the engine* (`real.py`, shared pool, SP4 off, 2 blocks of 32 tokens per cell; tokens/s for 1 / 8 requests):

| load | defaults | first weights | Kaggle seed 1 | shipped now |
| --- | --- | --- | --- | --- |
| idle | 9.17 / 61.0 | 9.08 / 59.5 | 9.26 / 58.7 | 9.39 / 60.0 |
| 25 % busy | 6.99 / 44.2 | 6.57 / 46.4 | 7.01 / 46.0 | 6.96 / 46.5 |
| 50 % busy | 5.20 / 35.4 | 5.32 / 36.0 | 7.39 / 50.2 | **7.44 / 51.1** |

The new weights chose 3 threads at 50 % (and at 25 % for one request); with whole processes they chose 3 / 2 threads
for 1 / 2 busy processes (7.49 / 45.2 and 5.68 / 35.0 tok/s), the same as before. Decision time 377-416 microseconds.
In world v3 (same evaluation): new weights -1415.0 (8.9 % more tokens than the defaults, 36 % less waiting), Kaggle seed
1 -1411.1, first weights -1427.6, hand rule -1420.2, defaults -2035.9. A from-scratch Kaggle run in world v3 is queued.

### 2026-11-14 MG2 v4: four models, vLLM 4-bit, and why 8 requests are slow on a T4

`lab/mg2/kaggle_gpu.py` at 30dcc74, Kaggle T4 x2 (the cached build was not found: built again, 59 minutes; 1 h 54 min
in all). Same workload as v3. Tokens/s, median of 3 blocks:

| Qwen3-4B | 1 request | 8 requests | 32 requests |
| --- | --- | --- | --- |
| llama-server, one GPU / split / one per GPU | 55.2 / 64.2 / 49.2 | 153.1 / 190.6 / 225.3 | 520.0 / 471.7 / 796.8 |
| vLLM float16 / AWQ 4-bit, one GPU | 28.1 / 65.7 | 192.7 / **431.1** | 514.6 / 892.5 |
| IronMule, one GPU / one per GPU | (146.0 / 144.5, SP4) | 150.5 / 242.4 | 747.6 / **1037.1** |
| IronMule shared pool, one GPU / one per GPU | (143.7 / 141.2, SP4) | 150.5 / 251.4 | 730.8 / **1059.1** |

32 requests, one engine or server per GPU: Gemma 3 4B 1042 against 704 (llama-server), Mistral 7B 713 against 642,
Llama 3.1 8B 667 against 576; first tokens 0.03-0.05 s against 0.2-0.3 s. Facts: Qwen3 49 / 49 (llama-server /
IronMule), Mistral 49 / 49, Llama 3.1 49 / 49, vLLM 50 (both), Gemma 3 48 / **23**: the engine got Gemma's prompts
without BOS (llama-server's `/apply-template` leaves it out, llama-server adds it when it tokenizes); Gemma 3 is out
until v5 checks it again. SGLang: `outlines-core` 0.1.26 needs a Rust compiler on Python 3.12 too. The one-request
cells (in brackets) are SP4 drafting from the previous block's answers, not a comparison.

*8 requests.* The engine's step prices by rows (Qwen3, one GPU): 1 row 19.6 ms, 4 rows 33 ms, 5-8 rows 50-52 ms, 32
rows 41 ms. llama.cpp's CUDA backend runs quantised matmuls of up to 8 tokens as matrix-vector kernels whose cost
grows with every token, and larger ones as matrix-matrix kernels; on a T4 a step of 8 costs more than one of 32.
vLLM AWQ at 8 requests (431) shows what the GPU can do there. PAD1 (74026876) measures the price of 1-16 tokens at
startup on a GPU and fills a step of 5-8 tokens with verified guesses up to the cheaper size (answers unchanged:
token-identical on the CPU with a forced table, per-sequence and shared pool). v5 measures it.

*PI1, from scratch in world v3* (Kaggle, two T4s, 8,192 servers, 400 iterations, 77 minutes). In world v3 (same
evaluation as above) seed 0 scored -1398.4, better than the shipped weights (-1415.0); but on the real machine's
readings (`half.py`: foreign load seen and step time at 4 threads) it keeps 4 threads with one core half busy, where 3
were measured 1.4x faster in the engine. The dream world gives the part-time case too little weight for the reward
to prefer it; the shipped weights stay. Next: weight the part-time case in the world by what servers show, or train
on recorded server traces.

### 2026-11-14 MG2 v5: PAD1 on T4, Gemma 3 fixed, fair one-request numbers

`lab/mg2/kaggle_gpu.py` at 74026876 (cached build: 8 s; 75 minutes in all), SP4 off, BOS added, PAD1 on (measured
at startup: Qwen3-4B, one sequence, 1 token 20.6 ms, 8 tokens 54.1 ms, 9-16 tokens 27.3-28.1 ms, so 3-8 tokens are
filled to 9-10). Tokens/s, median of 3 blocks:

| | Qwen3-4B | Gemma 3 4B | Mistral 7B | Llama 3.1 8B |
| --- | --- | --- | --- | --- |
| 1 request: best llama-server (split) | 64.2 | 60.1 | 48.6 | 44.4 |
| 1 request: llama-server / IronMule, one GPU | 52.9 / 46.1 | 49.6 / 47.4 | 25.3 / 23.7 | 22.9 / 22.5 |
| 8 requests: best llama-server | 193.5 | 195.2 | 128.0 | 119.1 |
| 8 requests: IronMule shared pool + PAD1, one GPU | **270.7** | **263.6** | **175.1** | **157.7** |
| 8 requests: the same, one per GPU | **288.0** | **272.1** | **177.1** | **161.6** |
| 8 requests: IronMule per-sequence KV + PAD1, one GPU | 94.7 | 99.3 | 57.1 | 53.8 |
| 32 requests: best llama-server (one per GPU) | 786.0 | 713.2 | 614.1 | 544.1 |
| 32 requests: IronMule shared pool, one per GPU | **1026.9** | **1007.3** | **679.1** | **633.2** |
| first token at 32: llama-server / IronMule (one per GPU) | 0.245 / 0.029 s | 0.239 / 0.032 s | 0.405 / 0.037 s | 0.346 / 0.041 s |

vLLM on one GPU (Qwen3-4B): float16 26.7 / 190.8 / 500.7, AWQ 4-bit 61.7 / 403.0 / 865.0 for 1 / 8 / 32 requests,
ahead of IronMule on one GPU at 8 (271) and 32 (714); facts 50 / 50. Facts through llama-server / the engine:
Qwen3 49 / 49, Gemma 3 47 / 48, Mistral 49 / 49, Llama 3.1 49 / 49. SGLang: not installed (the install's output went
to no log; v6 logs it).

Read: PAD1 with one unified KV cache is a real gain at 8 requests: 1.32-1.40x the best llama-server arm on all four
models. With per-sequence caches it is a loss (llama.cpp splits a step whose sequences carry different token counts
into several passes), so PAD1 now runs only with a unified cache. Without SP4's repeated-prompt drafts IronMule is
slightly behind llama-server for one request on one GPU (46.1 against 52.9 on Qwen3) and behind its layer split over
two GPUs (64.2): one request is bound by the per-token step, where IronMule has nothing extra yet. The product does
not yet use the shared pool on GPUs (it is gated to CPUs, where FA1u serves it); v6 checks the pool's padded answers
on the GPU first.

## 2026-11-15 SGLang dropped

Owner, 2026-11-15: SGLang is left out for good, since it would not install on Kaggle (v4-v5: `outlines-core` has no
wheel for Python 3.13 and needs a Rust compiler on 3.12). The comparison set is now llama.cpp, Ollama, LM Studio and
vLLM. The harness (`lab/mg2/kaggle_gpu.py`) and `lab/cmp` no longer have an SGLang arm. MG2 v6 had already started with
the arm, so its SGLang result is ignored.

## 2026-11-15 MG2 v6: the shared pool's padded answers on a GPU

Kaggle T4 x2, commit c1d4f54, the same arms as v5 plus a check of the answers: the 50 facts through the plain engine
and through the shared pool with PAD1's guesses (8 at a time), token for token against each other.

| | Qwen3-4B | Gemma 3 4B | Mistral 7B | Llama 3.1 8B |
| --- | --- | --- | --- | --- |
| facts: llama-server / engine / pool + PAD1 | 49 / 49 / 49 | 47 / 48 / 48 | 49 / 49 / 49 | 49 / 49 / 49 |
| pool answers token for token the same as the engine's | 50 / 50 | 50 / 50 | 49 / 50 | 47 / 50 |
| PAD1 guesses in the pool's fact steps | 176 | 124 | 135 | 243 |
| 8 req, one GPU: llama-server / pool + PAD1 (tok/s) | 152.8 / **272.7** | 144.8 / **260.7** | 76.0 / **171.8** | 74.5 / **159.5** |
| 32 req, one per GPU: llama-server / pool | 777.3 / **1021.9** | 700.8 / **997.4** | 612.4 / **679.5** | 547.2 / **629.0** |
| 1 req, one GPU: llama-server / engine | 54.9 / 47.2 | 49.6 / 45.9 | 26.7 / 26.3 | 23.5 / 23.2 |

The best llama-server arm at 8 requests was its layer split or one server per GPU (193.3, 193.6, 126.8, 118.2), so the
pool on one GPU is 1.35-1.41x the best of them, as in v5. Read: PAD1 changes a few answers in their wording (CUDA's
kernels for 9 and more rows round differently from the 1-8-row ones) and no fact; the pool serves on GPUs from now on
(`ironmule serve` opens it where its perplexity gate passes, as on CPUs). vLLM AWQ again 63.7 / 419.2 / 860.5 for
1 / 8 / 32 requests. SGLang's server exited after its install (dropped for good, see above).

## 2026-11-15 SG1: one request on one T4

Question: why one request through the engine (MG2 v5: 46.1 tok/s on Qwen3-4B) is slower than through llama-server
(52.9). Kaggle T4, one GPU, 3 blocks of 128 tokens after a warm-up, the same build as MG2. GR1 (CUDA graphs on GPUs
older than Ampere, `native/ggml_patches/gr1`) was meant to be measured too, but its rebuild failed: the cached build
has no `ggml-cuda.cu` to patch, so every `_g1` arm ran the stock library and is a repeat of its `_g0` twin.

| tok/s, median of 3 | Qwen3-4B | Gemma 3 4B | Mistral 7B | Llama 3.1 8B |
| --- | --- | --- | --- | --- |
| llama-server -t 4 (twice) | 54.1, 53.1 | 47.7, 45.9 | 29.7, 26.7 | 27.0, 24.0 |
| engine, the Pilot's threads (twice) | 54.1, 52.8 | 44.7, 44.8 | 26.3, 28.9 | 24.9, 26.2 |
| engine, 1 / 2 / 4 threads | 51.7 / 49.2 / 45.9 | 46.7 / 46.7 / 45.5 | 28.8 / 27.7 / 28.0 | 24.9 / 25.2 / 24.9 |
| engine without speculation | 48.7 | 46.4 | 27.4 | 25.0 |

Answers: every arm's tokens are the same as the engine's default arm; facts 49, 47, 49, 49 through both. Read: the
gap of v5 was mostly the GPU's own spread. The same binary and setting differs by up to 3 tok/s from one run to the
next (Mistral's llama-server 29.7 and 26.7), and one request through the engine is within that spread of
llama-server (Qwen3 54.1 against 54.1). Threads make no difference above it with every layer on the GPU. One request
on a T4 is bound by llama.cpp's own kernels, about 18.5 ms per Qwen3-4B token, against 15.7 ms for vLLM's AWQ
kernels with CUDA graphs. Next: GR1 with a fresh CUDA build (about an hour of the session), then speculation that
fills a step to 9-16 tokens, which a T4 computes in 27-28 ms against 20.5 ms for one token (MG2 v5's padding table).

## 2026-11-15 SG1 v2: GR1 (CUDA graphs on a T4) measured, no gain

Kaggle T4, a fresh CUDA build of 4227c9b with GR1 applied before cmake (58 min), one request on GPU 0, graphs off
(`_g0`) and on (`IRONMULE_CUDA_GRAPHS=1`, `_g1`) alternating twice (a, b), 3 blocks of 128 tokens each.

| tok/s, mean of a and b | Qwen3-4B | Gemma 3 4B | Mistral 7B | Llama 3.1 8B |
| --- | --- | --- | --- | --- |
| llama-server, graphs off / on | 55.2 / 54.2 | 47.1 / 46.7 | 29.9 / 28.1 | 27.3 / 25.0 |
| engine (Pilot), graphs off / on | 55.6 / 54.2 | 46.6 / 46.0 | 28.6 / 29.7 | 25.9 / 27.0 |
| engine without speculation, graphs on | 49.9 | 46.3 | 28.8 | 26.6 |

Tokens: every `_g1` arm the same as its `_g0` twin; facts 49, 47, 49, 49 either way, 50 of 50 answers the same.
Read: no gain from graphs on a T4. llama-server is 0.4-2.3 tok/s slower with them, the engine between -1.4 and +1.1,
which is inside the drift from a to b (Qwen3: 56.6 to 53.8 for the same llama-server). One request on a T4 is bound
by the kernels themselves, not by launching them. GR1 stays in `native/ggml_patches` as opt-in and is not turned on.
The engine is level with llama-server (55.6 against 55.2 on Qwen3-4B). Next: a draft model (SG1 v3).

## 2026-11-15 SG1 v3: one request on a T4 with a draft model (DM1)

Kaggle T4, the stock path of SG1 v2's build (graphs off), one request at a time on 4 different story prompts, 3 blocks
of 4 x 128 tokens after a warm-up, the order llama-server, llama-server `-md` (its own speculative decoding with the
same draft GGUF, `-ngld 99`, its defaults), the engine, the engine with `--draft-gguf` (DM1), twice (a, b).

| tok/s, mean of a and b | Qwen3-4B + Qwen3-0.6B | Llama 3.1 8B + Llama 3.2 1B | Gemma 3 4B + Gemma 3 1B |
| --- | --- | --- | --- |
| llama-server | 47.3 | 23.2 | 45.3 |
| llama-server -md (draft model) | 49.6 | 28.2 | 37.1 |
| engine | 43.5 | 22.9 | 45.1 |
| engine + DM1 | **54.3** | **30.6** | 44.4 (refused, see below) |
| DM1 drafts accepted | 1,073 of 1,519 | 1,740 of 2,255 | -- |

Answers: facts 49 / 49 (Qwen3) and 49 / 49 (Llama 3.1) with and without the draft model through both engines, 50 of 50
fact answers token for token the same with DM1 (llama-server -md: 48 of 50 on Qwen3). The long story answers differ in
places with DM1 and with `-md` alike: verifying several tokens in one step takes CUDA's multi-row kernels, which round
differently from the one-row ones. Gemma 3 1B was refused by the engine: its vocabulary (262,144) is not Gemma 3 4B's
(262,208, image tokens included); llama-server took it and lost 18 %. Read: with a draft model the engine is 1.15x
llama-server on Qwen3-4B and 1.32x on Llama 3.1 8B for one request on one T4, and 1.09x llama-server with its own draft
model on both. vLLM AWQ (63.7 tok/s on Qwen3-4B, MG2 v6) is still ahead. Without the draft model the engine was 8 %
behind llama-server on these four different prompts (level on one repeated prompt in v2).

## 2026-11-15 PI1: a new Pilot from scratch (world v4), shipped

Two changes after the from-scratch v3 runs kept 4 threads beside a half-busy process: world v4 dreams more servers
like the measured one, and the warm start copies a teacher that knows the dream's true foreign load instead of the
visible-load rule ("threads = free cores"), which cannot see a starved half-busy process. With only the first change,
two seeds from scratch still kept 4 threads at every part-time load on the real readings; with both, two seeds chose 3
(like the weights before). Each run took 38-39 minutes on this 4-vCPU Xeon, two at once. The world and the recipe stay
private.

| | reward (dream, v4) | tokens | wait s | real readings: idle / 25 / 50 / 75 / 100 % |
| --- | --- | --- | --- | --- |
| defaults | -2546.9 | 46,448 | 23,838 | 4 / 4 / 4 / 4 / 4 |
| hand rule (threads = free cores) | -1453.9 | 52,195 | 13,362 | -- |
| shipped before (local5) | -1434.4 | 52,650 | 13,234 | 4 / 3 / 3 / 3 / 3 |
| v4 seed 9 | -1436.1 | 52,708 | 13,270 | 4 / 3 / 3 / 3 / 3 |
| **v4 seed 10 (shipped)** | **-1407.7** | 52,253 | **13,119** | 4 / 3 / 3 / 3 / 3 |

On the real machine (engine, `lab/pi1/real.py`, Qwen3-4B Q8_0, shared pool, 64 tokens; tok/s for 1 / 8 requests):

| foreign load | defaults | shipped before | v4 seed 10 |
| --- | --- | --- | --- |
| none (two alternating runs each) | 10.27, 9.85 / 66.3, 64.3 | 9.38, 9.48 / 63.8, 62.1 | 9.75, 9.61 / 63.7, 64.8 |
| one busy process | 0.31 / 2.41 | 7.19 / 50.8 | 7.45 / 52.0 (3 threads) |
| one process busy half of every 20 ms | 5.26 / 38.4 | 7.54 / 51.4 | 7.65 / 53.8 (3 threads) |
| two busy processes | -- (stopped: under 1 tok/s) | -- | 5.78 / 36.9 (2 threads) |
| two busy processes, 16 tokens, one block | 0.19 / 1.46 (4 threads) | -- | 6.11 / 30.9 (2 threads) |

Read: seed 10 is at least as good as the weights before in every row and replaces them as `ironmule/pilot_weights.npz`.
Idle, separate processes put it 1.6-3.8 % below the defaults at the same 4 threads; the same process with the Pilot
attached and detached in turn (4 rounds each, `e.pilot = None`) shows no cost: 10.04 against 10.05 tok/s for one
request and 65.6 against 65.0 for eight (decisions 0.38 ms a step), so that gap was the spread between processes.
With two busy processes the Pilot is 32x the defaults for one request and 21x for eight (short answers, one block).

## 2026-11-15 KG1 v1: generated Q4_K CUDA kernels against llama.cpp's, on a T4

Goal (owner, 2026-11-15): kernels the agent writes and tunes for each machine, as one more lever beside threads and
batching. First step: `lab/kg1/gen.py` generates one CUDA kernel per parameter set for llama.cpp's Q4_K weights times
float32 activations (lanes per 256-weight block L, rows per warp R, warps per block W, tokens per call B: 105
variants), `lab/kg1/kaggle_kg1.py` runs each on the real Q4_K weights of Qwen3-4B Q4_K_M on a Kaggle T4 against
llama.cpp's own kernel (`ggml_mul_mat` on the CUDA backend, `lab/kg1/ref_ggml.c`) and a float64 reference that
decodes the format element by element. 8 weight copies per timing (no cache reuse), 50 repetitions.

| shape (K x N), tokens | llama.cpp us (GB/s) | best variant us (GB/s) | variant |
| --- | --- | --- | --- |
| 2560 x 1024, 1 / 2 / 4 / 8 | 20.9 (71) / 22.1 / 33.1 / 54.2 | 20.8 / **15.2** / **20.8** / **35.5** | L16R1W4, L32R1W4, L32R1W2, L16R4W2 |
| 2560 x 9728, 1 / 4 / 8 / 16 | 71.4 (196) / 140.4 / 251.7 / 107.3 | 88.6 (158) / 158.5 / 286.5 / 626.7 | L8R2W2, L16R4W8, ... |
| 9728 x 2560, 1 / 4 / 8 / 16 | 62.0 (226) / 113.9 / 243.3 / 109.1 | 89.3 (157) / 143.9 / 280.3 / 581.7 | L8R4W8, L16R4W8, ... |

Errors against float64: every variant 1.4-2.4e-7, llama.cpp 4.8-5.6e-3 (1-8 tokens) and 1.1-1.5e-2 (16), because it
rounds the activations to 8 bits first. All Q4_K matmuls of one token (the model's counts per shape): llama.cpp 9.69
ms at 1 token, the best variants 12.05 ms (0.80x); 0.71x / 0.87x / 0.90x at 2 / 4 / 8 tokens, 0.21x at 16, where
llama.cpp switches to its tensor-core kernel. Read: llama.cpp's one-token kernel reads up to 226 GB/s, 70 % of the
T4's 320 GB/s; the float32 variants win on the small 2560 x 1024 matrices and lose on the large ones, and from 4 tokens
they are bound by float32 arithmetic. llama.cpp's own small-batch path is weakest at 4-8 tokens (55-57 GB/s on the
large matrices, slower than its 16-token path). v2 adds variants with 8-bit activations and `__dp4a` (four
multiply-adds per instruction) for that range.

## 2026-11-15 KG1 v3: the library tuned on a T4, but the engine ran without the hook

Kaggle T4, `lab/kg1/kaggle_e2e.py` at 404cc14. The tuning worked: `ironmule/kernel_gen.py` compiled 288 variants
(50 s), measured them on the five Q4_K shapes of Qwen3-4B for 1-8 tokens (124 s) and built a 40-entry library; the
fastest variants were mostly L8 with R 1-2 at 1-4 tokens and R 2-8 at 5-8 tokens (per shape: 2560 x 9728 67.1 /
85.4 / 129.1 us at 1 / 4 / 8 tokens; 9728 x 2560 68.5 / 79.1 / 112.6 us). The patched build failed: `tools/build_ggml.py`
found no CUDA::cuda_driver on Kaggle (fixed: it now falls back to the driver library), and MG2's build that ran after it
extracted the source afresh, so the engine served without the kg1 hook. Its own measurement shows that: steps with the
library "on" and "off" cost the same (18.7 / 34.0 / 54.4 ms at 1 / 4 / 8 tokens either way). Measured anyway, on one
GPU, Qwen3-4B Q4_K_M, tok/s, two alternating passes:

| requests | llama-server | engine (shared pool + PAD1) |
| --- | --- | --- |
| 1 | 53.0, 48.7 | 49.1, 49.9 |
| 2 | 94.7, 86.6 | 90.8, 86.5 |
| 4 | 119.1, 112.5 | 148.3, 147.2 |
| 8 | 138.7, 137.0 | 278.2, 278.4 |

Facts 49 / 50 for every engine arm. v4 builds with MG2's build and the patches applied right before cmake (as SG1 v2
did for gr1), keeps the build and the model outside `/kaggle/working`, and adds both engines with PAD1 off.

## 2026-11-15 PI1 on a second machine: a Kaggle CPU session

The shipped Pilot (from scratch, world v4) on a different CPU than it was checked on: a Kaggle CPU session, Intel Xeon
@ 2.20GHz (family 6 model 79, Broadwell: AVX2, no AVX-512), 2 cores with 2 threads each, Qwen3-4B Q4_K_M, the engine
built with `tools/build_ggml.py` (53 min), `lab/pi1/kaggle_cpu.py` at 7245a82 (`lab/pi1/real.py`: 64-token answers,
16 with two busy programs; then `lab/pi1/ab_idle.py`). tok/s for 1 / 8 requests:

| foreign load | defaults (4 threads) | Pilot | its threads |
| --- | --- | --- | --- |
| none | 6.30 / 8.97 | 6.27 / 8.95 | 4 |
| one busy process | 1.60 / 3.31 | **3.72 / 5.31** | 2 |
| one process busy half of every 20 ms | 4.93 / 8.12 | 4.83 / 8.10 | 4 |
| two busy processes | 1.09 / 1.88 | **4.09 / 2.91** | 2 |

Idle, one process, the Pilot attached and detached in turn (4 rounds): 6.42 / 8.83 against 6.36 / 9.23; decisions
0.51 ms a step. Read: on a machine it never saw, with hyper-threads, it is 2.3x / 1.6x the defaults beside one busy
process and 3.8x / 1.5x beside two, and equal idle and beside a half-busy process (which costs this machine only 2-22 %
at 4 threads, so keeping 4 is right here). Beside one busy process it chose 2 threads, where the 4-vCPU Xeon chose 3:
here a foreign process on one hyper-thread also slows its sibling. The 8-request cases at 2 busy processes are
slow either way (first tokens about 21 s): two cores of an older CPU are not enough for eight 4B requests.

## 2026-11-16 PI1 on a second machine: the Kaggle-trained Pilot

The shipped Pilot (Kaggle seed 21, trained from scratch on two T4) on the same second machine as the entry above: a
Kaggle CPU session, Intel Xeon @ 2.20GHz (Broadwell, AVX2), 2 cores with 2 threads each, 31 GB, Qwen3-4B Q4_K_M, the
engine built with `tools/build_ggml.py` (58 min, patches fa1 and row1), `lab/pi1/kaggle_cpu.py` at 68d25f7. tok/s for
1 / 8 requests, first token in seconds in brackets:

| foreign load | defaults (4 threads) | Pilot | its threads | the weights before |
| --- | --- | --- | --- | --- |
| none | 6.10 / 8.65 | 5.99 / 8.63 | 4 | 6.27 / 8.95 (4) |
| one busy process | 1.12 / 2.72 | **4.75 / 7.21** | 3 | 3.72 / 5.31 (2) |
| one process busy half of every 20 ms | 4.53 / 7.65 | **4.92** / 7.67 | 3 | 4.83 / 8.10 (4) |
| two busy processes | 0.88 / 1.68 | **3.93 / 2.89** | 2 | 4.09 / 2.91 (2) |

First tokens: one busy process 1.12 / 3.96 s against 3.50 / 8.12 s with the defaults; two busy processes 0.34 / 21.9 s
against 1.90 / 22.1 s. Decisions 0.55-0.61 ms a step. Idle, the Pilot attached and detached in turn in one process (4
rounds): 5.97 / 8.29 against 6.05 / 8.84; its 8-request rounds rose from 7.84 to 8.60 (the first round lowest), so
the idle cost is between 0 % (the separate runs above) and 6 %. Read: on a machine it never saw, it is 4.2x / 2.7x the
defaults beside one busy process and 4.5x / 1.7x beside two, 9 % faster for one request beside a half-busy process and
equal idle. Against the weights before it chooses 3 threads beside one busy process (1.3x / 1.4x faster there) and
the same elsewhere within a few percent. Eight 4B requests beside two busy processes stay slow for both (first tokens
about 22 s).

## 2026-11-16 PI1 decision time: /proc reread through open files

The Pilot's decision took about 0.4 ms a step in the running server, against 15 us in a loop. Split on the 4-vCPU
Xeon, 60 ms between decisions (as between engine steps): opening and reading `/proc/stat` and `/proc/self/stat` 190-250
us, the features 50 us, the network 60 us, the four argmaxes 26 us; after a step Python's code is out of the caches.
Two changes, the same decisions as before on 6,000 random inputs (machines, costs, loads, queues):

- `CpuLoad` keeps both files open and rereads them with `pread` (about 50 us instead of 190 us);
- the range check and the argmaxes in plain Python on the 20 features and the 21 logits.

Tried first and dropped: reading the load in a background thread every 50 ms. Decisions took 0.114 ms instead of 0.40,
but idle with 8 requests the engine made 37.9 / 38.0 tok/s against 39.7 / 39.7 in two rounds: its wake-ups push a
llama.cpp thread off its core.

Real engine, `lab/pi1/real.py`, Qwen3-4B Q8_0, shared pool, 64-token answers, tok/s for 1 / 8 requests, two rounds
(new, old, defaults in that order):

| foreign load | new | old | defaults |
| --- | --- | --- | --- |
| none | 9.28 / 39.62, 9.67 / 39.14 | 9.37 / 40.40, 9.34 / 40.64 | 9.39 / 40.76, 9.91 / 40.57 |
| one busy process | 7.13 / 31.60, 7.44 / 32.28 | 7.29 / 31.35, 7.12 / 32.96 | 0.33 / 2.07, 0.33 / 2.06 |
| one process busy half of every 20 ms | 7.17 / 31.33, 7.60 / 32.98 | 7.26 / 31.05, 7.36 / 32.22 | 5.29 / 22.30, 5.26 / 22.76 |
| decision | 0.184-0.192 ms | 0.386-0.401 ms | - |

Idle, 8 requests, in mixed order (old new def new old def def old new): new 40.16 / 40.47 / 38.53, old 40.27 / 39.22 /
38.90, defaults 41.60 / 39.58 / 40.04 tok/s. Read: a decision takes half the time (0.19 ms); the speed is the same as
before within the run-to-run spread (38.5-41.6 idle), and the 2-3 % idle gap of the first table was the run order.
Beside a busy process the Pilot stays 15-22x the defaults, beside a half-busy one 1.4x. Shipped.

## 2026-11-16 PI2: Pilot version 2, five prompt budgets, dream worlds v5 and v6

Version 2 (`ironmule/pilot.py`, `VERSION = 2`): five prompt budgets (32, 128, 512, 2,048 tokens, none) instead of four,
and the shared pool now honours a set budget (before, it took every waiting prompt that fit). Measured first, budget
alone, 4-vCPU Xeon, Qwen3-4B Q8_0, 8 requests, two rounds: first token 1.26 / 1.07 s with budget 32 against 1.89 / 1.34 s
without one when idle, 0.87 / 0.89 s against 1.42 / 1.35 s beside two busy programs, at the same tokens per second; 16
requests idle 58.9 / 60.5 tok/s against 63.4 / 62.7.

- *World v5, trained from scratch on two Kaggle T4 (43 min, seed 30 of two):* beside two busy programs it chose 2
  threads and budget 32, the first token of 8 requests 0.61 s against 0.94 s with the version-1 weights (-35 %), the
  same tok/s. Idle it was 4-11 % slower: it stayed at 3 threads on 40-50 % of the steps. Not shipped.
- *Why:* traced step by step. With all 4 cores taken, the load reader sees 0.00-0.02 foreign cores on an idle machine;
  at 3 threads it sees 0.11-0.12 (a tenth of the steps 0.14-0.19), the daemons of a quiet server now running on the
  free core. Beside a process busy 25 % of the time it sees 0.29 at 3 threads. World v5 had the daemons at any thread
  count, so a free core's reading looked like a part-time process and kept the Pilot there.
- *World v6:* the daemons are seen only on a free core, at the measured level. Three runs on this CPU (from scratch,
  and two fine-tunes of the v5 weights, 1,024 dreamed servers, 100 rounds, 16-47 min) fixed either the idle case or
  kept budget 32, not both. Replayed on the recorded real features (no engine): the from-scratch run stayed at 4 idle
  threads in 88-97 % of the steps (v5: 50-65 %), the fine-tunes 50-71 %.
- *World v6 on two Kaggle T4 (47 min, from scratch, seeds 70/71; 8,192 dreamed servers, 200 rounds):* both keep 4 idle
  threads in 88-98 % of the replayed steps and give up a core beside the 25 % process in 79-97 %; neither uses the small
  budgets (in the dream they move the reward by about 4 %). Seed 70 (best in the dream: reward -303 against -307 for
  "threads = free cores" and -1118 for the defaults) shipped.

Real engine, `lab/pi1/real.py`, 64-token answers, tok/s for 1 / 8 / 16 requests (first token of 8 in brackets), mean of
two rounds in alternating order. Version 1 ran from d1b1897 (its decision 0.39 ms, the pread change came later); the
defaults and v5 are from the same day's earlier rounds:

| foreign load | v6 (shipped) | version 1 | v5 | defaults |
| --- | --- | --- | --- | --- |
| none | 10.09 / 36.02 / 59.13 (1.47 s) | 9.28 / 35.62 / 58.73 (1.59 s) | 8.90 / 33.59 / 58.79 (1.82 s) | 10.14 / 36.21 / 56.83 (1.43 s) |
| one busy process | 7.38 / 28.90 / 48.43 (0.73 s) | 7.83 / 28.08 / 45.95 (0.71 s) | 7.35 / 28.91 / 46.62 (0.71 s) | 1.59 / 6.65 / 12.26 (2.09 s) |
| two busy processes | 5.56 / 20.72 / 34.44 (0.94 s) | 5.33 / 19.92 / 34.11 (0.94 s) | 5.65 / 20.67 / 34.28 (0.61 s) | 0.29 / 1.75 / 3.27 (6.11 s) |
| one process busy 25 % of every 20 ms | 6.90 / 27.96 / 46.39 (0.71 s) | 7.61 / 28.03 / 46.52 (0.70 s) | - | - |
| one process busy 50 % of every 20 ms | 7.64 / 28.24 / 46.56 (0.72 s) | 7.92 / 29.14 / 48.84 (0.65 s) | 7.69 / 29.56 / 48.54 (0.66 s) | 5.90 / 23.05 / 37.96 (0.81 s) |

Read: v6 is on par with version 1 (+0.1 % tok/s over the 15 cells, rounds differ by up to 10 %), ahead idle, behind for
one request beside the 25 % process (6.90 against 7.61: there it keeps 4 threads on some steps). Its decision takes 0.17
ms. The 35 % sooner first token of v5 beside two busy programs is not in v6; a world in which the small budgets matter
more, or a reward that weighs them, is what would bring it back.

Second machine (the Kaggle CPU session of the PI1 entries: Xeon @ 2.20GHz, 2 cores with 2 threads each, Qwen3-4B
Q4_K_M, engine build 62 min, `lab/pi1/kaggle_cpu.py` at 7d09702 with the v6 weights, 1.8 h in all). tok/s for 1 / 8
requests, first token in seconds in brackets:

| foreign load | defaults (4 threads) | v6 | its threads |
| --- | --- | --- | --- |
| none | 5.13 / 7.82 (1.21 / 8.45 s) | 5.02 / 7.34 (1.22 / 9.35 s) | 4 |
| one busy process | 1.36 / 3.00 (2.80 / 6.82 s) | **4.25 / 6.49** (1.26 / 4.28 s) | 3 |
| one process busy half of every 20 ms | 4.33 / 7.11 (0.83 / 1.97 s) | **4.66** / 7.18 (0.86 / 1.94 s) | 3 |
| two busy processes | 0.57 / 1.81 (1.27 / 23.6 s) | **2.75 / 2.53** (0.53 / 24.7 s) | 2 |

Idle, attached and detached in turn in one process (4 rounds): 5.30 / 7.55 against 4.98 / 7.41, so the 6 % gap of the
separate 8-request runs above is round-to-round spread. Decisions 0.28-0.32 ms a step (0.55-0.61 ms before the pread
change). This session's machine was slower than the one of the version-1 run (idle defaults 5.13 / 7.82 against 6.10 /
8.65), so the rows are compared within the session: 3.1x / 2.2x the defaults beside one busy process, 4.8x / 1.4x
beside two, 8 % faster for one request beside a half-busy process, equal idle. Against each session's idle defaults
v6 keeps 83 % / 83 % beside one busy process, version 1 kept 78 % / 83 %: on par, as on the first machine.

## 2026-11-16 PI3 probe: what the Pilot can decide on a T4

Kaggle T4 (one of two), `lab/pi3/kaggle_gpu_probe.py` at 6ad9ab2, MG2's CUDA build (no generated kernels), Qwen3-4B
Q4_K_M with every layer on the GPU, shared pool (16 x 2,048 cells, cap 32), PAD1, 47 minutes. Each arm is a decision
source swapped into the same engine; every prompt is new to the engine (a first run showed a long prompt prefilled
once and then reused by every later arm). The T4 runs faster in its first minutes (55 against 42 tok/s for one
request in the first cell), so arms alternate.

A. Host CPU load (4 vCPUs), tok/s for 1 / 8 / 32 requests at once, 64 tokens, mean of 2 rounds:

| busy processes | 1 thread | 2 threads | 4 threads | v6 (4 threads) |
| --- | --- | --- | --- | --- |
| none | 49 / 265 / 624 | 44 / 247 / 598 | 45 / 246 / 599 | 43 / 248 / 596 |
| one | 42 / 236 / 576 | 42 / 237 / 580 | 42 / 237 / 578 | 43 / 237 / 575 |
| two | 42 / 230 / 564 | 42 / 228 / 563 | 43 / 231 / 565 | 42 / 231 / 567 |
| three | 42 / 237 / 561 | 42 / 234 / 565 | 42 / 236 / 563 | 42 / 233 / 560 |
| one, half of every 20 ms | 43 / 238 / 578 | 44 / 240 / 575 | 43 / 238 / 577 | 43 / 236 / 571 |

B. Arriving requests, first token counted from arrival, mean of 3 rounds: "chat" (48 short prompts, Poisson 4/s, answers
64-192 tokens) and "mixed" (48 prompts in clumps of 4 at 3/s, 30 % of them a 1,364-token document, answers 32-160).
tok/s, first token mean / median / 90th percentile in seconds:

| arm | chat | mixed |
| --- | --- | --- |
| defaults (cap 32, every waiting prompt) | 260, 0.69 / 0.14 / 2.53 | 89, 7.70 / 6.87 / 13.09 |
| v6 | 269, 0.32 / 0.14 / 1.03 | 90, 7.45 / 6.42 / 13.26 |
| prompt budget 128 | 258, 0.53 / 0.15 / 1.66 | **90, 4.13 / 1.57 / 12.63** |
| prompt budget 512 | 252, 0.58 / 0.15 / 1.77 | **90, 4.11 / 1.73 / 11.18** |
| cap 16 | 249, 2.34 / 2.43 / 5.26 | 86, 9.46 / 8.59 / 16.69 |
| cap 8 | 160, 10.0 / 9.98 / 19.1 | 79, 12.4 / 11.8 / 23.2 |
| cap 4 | 83, 27.9 / 26.6 / 53.0 | 51, 25.3 / 24.2 / 47.0 |
| 1 thread | 260, 0.45 / 0.15 / 1.36 | 90, 7.80 / 7.12 / 13.47 |

C. With a draft model (Qwen3-0.6B Q8_0, DM1), tok/s for 1 / 8 requests, 2 rounds: defaults 57.1, 38.3 / 271, 218;
drafts off 39.4, 38.4 / 215, 214; v6 (drafts on) 45.7, 46.8 / 215, 212. The draft model's own stop rule settles
after its first minutes (its first round 57 tok/s, later 38-47), so these differences are its state, not the arm.

Answers: 49 of 50 facts with the defaults and with v6, 48 of 50 fact answers token for token the same.

The engine's steps (38,983 logged): one row 24 ms, 2 rows 34, 3 rows 40, 8 rows 59 (unpadded), 9-16 rows 40-50 (PAD1),
17-31 rows 72-81 and 32 rows 49 ms (median, attended tokens 0.5-3.5k); a prompt of 540 tokens alone 320-460 ms.
The Pilot's startup measurement: one row 18.4 ms, each further row 4.3 ms, 1,000 prompt tokens 494 ms, attention below
what it can measure. v6 saw these as features outside the GPU servers it dreamed (each further row 0.002-0.02 of a
one-row step, 1,000 prompt tokens 0.3-3 of it; measured 0.23 and 27).

Read: on a GPU the host's threads and its CPU load do not matter (within 2 %, even three busy processes on four
vCPUs); the running cap must stay at its maximum (cap 16 already costs first tokens); the prompt budget is the decision
that pays: with documents among chats, 128 or 512 prompt tokens per step halve the mean first token (7.70 to 4.11 s,
median 6.87 to 1.73 s) at the same tokens per second, because short prompts no longer wait for a step that prefills
several documents at once. v6 keeps every budget open on a GPU: its dream world priced a GPU prompt about ten times too
low and let prompts in oldest first, where the engine takes the shortest first. Next: dream world v7 with these costs
and the engine's order, a Pilot trained from scratch in it, and a check on the T4. Seen in passing: 17-31 rows cost 1.5x
what 32 rows cost, which looked like an engine gain of its own (padding them to 32 as PAD1 pads 4-8 to 9); PAD2 below
found it is not.

## 2026-11-16 PI3: GPU weights for the Pilot (dream world v7)

Dream world v7 (private) takes the probe's T4 costs into its GPU servers: each further unpadded row 0.06-0.45 of a
one-row step, 1,000 prompt tokens 8-90 of it, attention 0.0002-0.05 of it, PAD1's step staircase (a step of r rows
costs min(1 + u (r - 1), P1 + s2 (r - 16)) one-row steps), a quarter of them the measured T4 class; documents among
chats (in half the servers 5-45 % of the prompts are 6-50x longer); waiting prompts let in as the engine does (one
passed over 8 times first, then the shortest), where v6 took the oldest first; GPU servers 45 % of the dream instead
of 20 %. Its replay of the probe's mixed traffic: prompt budget 512 3.15 s to the first token against 4.37 s with
every prompt let in (-28 %; the T4: -46 %), the same tokens per second; small caps lose there as on the T4.

Trained from scratch on Kaggle (two T4, seeds 80 and 81, 8,192 dreamed servers, 180 PPO iterations, 41 minutes,
private kernel `ironmule-pilot-train` v6). Dream reward per server over 240 s (tokens, wait s, first token s):

| | all servers | GPU servers only (120 s) |
| --- | --- | --- |
| defaults | -1122.9 (16,189, 17,699, 16.1) | -62.5 (16,418, 857, 2.48) |
| hand rule (threads = free cores) | -150.5 (16,867, 1,888, 5.23) | -- |
| prompt budget 512 held | -- | -59.0 (16,460, 861, 2.40) |
| **seed 80** | **-142.9** (16,899, 1,948, 5.05) | -63.1 (16,352, 884, 2.49) |
| seed 81 | -151.6 (16,831, 1,896, 5.34) | -66.3 (16,411, 898, 2.60) |

On the probe's measured T4 costs seed 80 keeps the cap at 32 and a prompt budget of 128 in every state tried (idle,
8-30 running, chats and documents waiting); seed 81 keeps every budget open, as v6 did.

Seed 80 on the CPU (this 4-vCPU Xeon, `lab/pi1/real.py`, Qwen3-4B Q8_0, two rounds in alternating order against the
shipped v6 weights, tok/s for 1 / 8 requests, first token of 8 in brackets, its threads per round):

| foreign load | seed 80 | v6 |
| --- | --- | --- |
| none | 10.05 / 37.99 (1.19 s), 4 / 4 | 10.18 / 36.70 (1.55 s), 3-4 / 4 |
| one busy process | 5.43 / 22.71 (0.94 s), 3 / 2 | **7.78 / 30.34** (0.71 s), 3 / 3 |
| two busy processes | 5.64 / 21.34 (0.53 s), 2 / 2 | 5.64 / 22.14 (0.53 s), 2 / 2 |
| one process busy 25 % of every 20 ms | 7.01 / 27.91 (0.40 s), 3 / 4 | 7.42 / 30.28 (0.40 s), 3 / 3 |
| one process busy 50 % of every 20 ms | 6.18 / 28.43 (0.46 s), 3 / 3 | **8.12 / 31.45** (0.36 s), 3 / 3 |

Read: idle, its budget of 128 brings the first of 8 tokens 23 % sooner (1.19 against 1.55 s), but its threads are not
steady: beside one busy process it took 3 threads in one round and 2 in the other (3.84 against 7.90 tok/s for one
request there), beside the 25 % process 4 threads in the second round; 9 % below v6 over the ten cells. So the
Pilot now ships two weight files: `pilot_weights.npz` (v6) where the layers run on the CPU, `pilot_weights_gpu.npz`
(seed 80) where they run on a GPU; there the threads do not matter (probe, part A) and its cap and budget decisions
are what the T4 check below measures.

The T4 check: Kaggle T4, `lab/pi3/kaggle_gpu_probe.py` in `PI3_MODE=check` at 1b7dbbb (seed 80 shipped there as the
engine's weights, the same file as `pilot_weights_gpu.npz`), 27 minutes, the probe's setup. The defaults ran first in
the first round, while the T4 still ran at its higher starting clock (53.8 against 36.1 tok/s for one request), so the
first round is left out below (all rounds: the same direction, mixed 86.6 against 84.8 tok/s and 5.11 against 10.13 s).
Blocks of requests at once, 128 tokens, tok/s (mean first token): defaults 2 rounds, seed 80 3 rounds.

| requests at once | defaults | seed 80 (cap 32, budget 128) |
| --- | --- | --- |
| 1 | 36.2 (0.048 s) | 36.0 (0.048 s) |
| 8 | 205.9 (0.169 s) | 206.5 (**0.145 s**) |
| 32 | 490.0 (0.623 s) | 486.7 (**0.429 s**) |

Arriving requests, rounds 2-4 (the arm order rotated each round), tok/s and first token mean / median / 90th
percentile in seconds:

| arm | chat | mixed (30 % documents) |
| --- | --- | --- |
| defaults | 238.6, 0.68 / 0.16 / 2.53 | 75.4, 11.76 / 13.28 / 19.70 |
| **seed 80** | 228.5, 0.83 / 0.17 / 3.14 | **80.3, 5.22 / 1.35 / 16.95** |
| v6 | 233.9, 0.72 / 0.17 / 2.82 | 75.5, 12.05 / 13.59 / 20.46 |
| prompt budget 512 held | 228.8, 0.80 / 0.17 / 3.24 | 76.9, 5.80 / 1.93 / 19.24 |

It decided cap 32, budget 128, 4 threads, drafts on in all 5,904 steps (v6: budget open in all 2,285). With the draft
model (2 rounds; tok/s for 1 / 8 requests and chat): defaults 54.4, 34.7 / 251, 199 / 325, 296; seed 80 36.0, 36.0 /
203, 208 / 312, 297 (the defaults' first round again at the starting clock). Answers: 49 of 50 facts with the defaults
and with seed 80, 49 of 50 answers token for token the same.

Read: with documents among chats the first token comes 2.3x sooner (median 9.8x) and 6 % more tokens per second come
out; for 8 and 32 requests at once the first token comes 14 % and 31 % sooner at the same tokens per second (budget 128
lets three short prompts start per step instead of all 32 in one long step); one request is unchanged. For short chats
alone seed 80 and a held budget of 512 are 4 % below the defaults, inside the spread of the rounds (the same arm 207-255
tok/s), so whether a budget costs anything there is not settled. The GPU weights ship (BACKLOG PI3 has the kill rule's
reading). Next on a GPU: padding 17-31 rows to 32 (PAD2 below: dropped).

## 2026-11-16 PAD2: filling 17-30 rows to 32 on a T4, dropped

Built from the probe's serving steps (17-30 rows 74-88 ms, 31-32 rows 48-61 ms): PAD1's startup measurement went on
to 32 tokens on a GPU, and its table would fill a step to 32 where that was measured at least 5 % cheaper. Kaggle T4,
`lab/pi3/kaggle_gpu_probe.py` in `PI3_MODE=pad2` at c23739a, 16 minutes, the shipped GPU Pilot deciding in both arms.
The startup measurement (one sequence, t tokens, ms): 1-8 tokens 19.3-49.8, 9-16 25.8-26.9, 17-24 29.4-31.8, 25-32
34.1-36.2: no step down after 16, so the table stayed PAD1's (4-8 to 10) and both arms ran the same steps (the same
guesses: 14 in the facts, 68-250 per stream). Blocks of requests at once, tok/s for 20 / 24 / 28 (3 rounds): 386 / 413 /
431 and 341 / 404 / 430, the first arm of the first round again at the T4's starting clock (466 against 335 for 20);
a step 58 / 59 / 65 ms, against 66 ms for 32 in the check before. 49 of 50 facts either way.

Read: a step's width is not what made the probe's 17-30-row steps dear; those steps came from arriving requests,
whose contexts in the shared pool are uneven, while the 32-row steps came mostly from blocks. Dropped (the engine
change reverted, the harness mode kept). Open: why such steps cost up to 1.5x a block's step of the same width.

## 2026-11-16 KVD1: the attended KV range on a T4, no gain, kept off

Hypothesis from PAD2: every step attends over the cells up to the highest one in use (llama.cpp `used_max_p1`), and
the shared pool's 32,768 cells fill from wherever llama.cpp's head stands, so after many requests a step reads most of
the cache. Two levers, both off unless set: llama.cpp's defragmentation threshold (`IRONMULE_GGML_DEFRAG`) and a budget
for the cells finished sequences keep as a prefix cache (`IRONMULE_GGML_KEEP`). On this CPU neither changed a step (255
ms for 8 rows fresh, 231 after 40 kept prompts; 287 with defragmentation). Kaggle T4, `lab/pi3/kaggle_gpu_probe.py` in
`PI3_MODE=kvd` at e2da4dd, 18 minutes: per arm its own engine, 8 requests fresh, then 60 kept 1,364-token prompts, then
8 and 32 requests and both arrival patterns; two rounds, the order reversed. tok/s (round 1, round 2):

| arm | 8 fresh | 8 after the fill | 32 after the fill | chat | mixed | cells in use |
| --- | --- | --- | --- | --- | --- | --- |
| base | 289, 237 | 230, 212 | 525, 495 | 260, 302 | 84.6, 58.5 | 24,027, 26,389 |
| defrag 0.1 | 253, 254 | 225, 226 | 502, 498 | 264, 299 | 84.9, 58.1 | 22,757, 26,486 |
| defrag 0.1, keep a quarter | 252, 253 | 225, 223 | 494, 490 | 260, 300 | 94.6, 60.9 | 7,596, 7,953 |

Facts 49 of 50 in every arm, the 50 fact answers token for token the same as the base's in both others.

Read: after the fill the steps are equally dear in all three arms, though the budget cut the cells in use from about
25,000 to 7,700; so a step's attended range is not what makes it dear on a T4, and defragmentation does nothing
measurable either. Mixed traffic with the budget: 94.6 against 84.6 tok/s in the first round, 60.9 against 58.5 in
the second (all arms lost a third between the rounds), inside the T4's spread. Both levers stay off; the code stays
for a machine where they pay. The fresh-to-filled drop (about 11 % in every arm) and the 1.5x of uneven serving steps
remain unexplained.

## 2026-11-15 ST1: per-step Python that no longer grows with the context (tokens identical)

Owner's plan (2026-11-15): algorithms first, then language, then kernels; what runs per token must not grow with the
context. Three changes, each with a test against the old implementation on random inputs:

- prompt lookup: `speculation.NgramIndex` per request (the last start of every 2- and 3-gram, updated by the new
  tokens) instead of `draft()` scanning the whole context every step; the prompt is indexed once per request;
- the shared pool's longest common prefix: a token trie (`ggml_engine._PrefixTrie`, ties to the earliest sequence as
  before) instead of comparing the prompt with every held sequence; `_common` compares 64 tokens per slice in C;
- the engine no longer builds `ids + tokens` per request and step for drafting, SP4 and the pool's sync.

`lab/st1/bench.py` on the 4-vCPU Xeon (Python 3.13), microseconds:

| context tokens | draft() scan | NgramIndex | the old ids + tokens copy |
| --- | --- | --- | --- |
| 512 | 154 | 0.76 | 2.3 |
| 2,048 | 646 | 0.83 | 9.2 |
| 8,192 | 2,662 | 0.78 | 35.6 |
| 32,768 | 10,474 | 0.77 | 173 |

| held sequences x tokens | old scan (token loop) | scan with the slice compare | trie |
| --- | --- | --- | --- |
| 8 x 2,048 | 429 | 72 | 58 |
| 32 x 2,048 | 1,776 | 297 | 61 |
| 64 x 4,096 | 7,066 | 1,156 | 122 |

Tokens: the old and the new engine gave the same tokens for 8 of 8 prompts on Qwen3-4B Q8_0 (shared pool and
per-sequence caches; the draft counts differ run to run for both, as the step policy prices steps by measured time).
End to end through the HTTP API (`lab/cmp/run_all.sh`, burst-sys = requests sharing a system prompt, same state, so
the same settings), tok/s, medians of 3 blocks:

| | burst-sys 8 | burst-sys 32 | one request | facts |
| --- | --- | --- | --- | --- |
| llama-server (stock, -np 32) | 21.3 | 28.5 | 8.8 | 49 / 50 |
| IronMule before | 47.4 | 74.5 | 9.86 | 49 / 50 |
| IronMule ST1 | 50.2 | 76.2 | 9.5 | 49 / 50 |

On this CPU a step costs 100-130 ms, so the saved milliseconds are within the spread at these contexts; they matter
where steps are short (a T4 step is 19-28 ms, where a 32,768-token scan cost 10 ms per request) and contexts long.
The first run of this comparison was not like for like: the new engine's fresh start chose a float32 KV cache on
tuning noise (scores 25.9 against 24.6 at a 64-token context) and served 61 / 7.05 tok/s; the autotune now ranks f32
KV first only 8 % ahead (`rank_settings`, ENGINE_VERSION 5).

## 2026-11-15 ST2 (CPU): is decode bound by memory bandwidth?

`lab/st2/cpu_bw.py` on the 4-vCPU Xeon (Sapphire Rapids: AMX, AVX512-VNNI), Qwen3-4B as Q8_0 (4.28 GB) and Q4_K_M
(2.50 GB), one decode step of 1 or 8 rows at a 64-token context, median of 10; "GB/s" is the weight bytes a step reads
over its time. The machine's memory: 14.7 / 30.2 / 33.5 GB/s read with 1 / 2 / 4 processes (numpy sum, 512 MB each).

| threads | Q8_0 1 row tok/s (GB/s) | Q8_0 8 rows | Q8_0 ROW1 off, 1 / 8 rows | Q4_K_M 1 row (GB/s) | Q4_K_M 8 rows |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.84 (12.2) | 19.5 | 2.71 / 12.9 | 2.48 (6.2) | 10.3 |
| 2 | 5.88 (25.2) | 35.8 | 5.28 / 23.7 | 4.70 (11.7) | 19.1 |
| 3 | 8.42 (36.0) | 57.3 | 7.01 / 34.1 | 6.26 (15.6) | 26.9 |
| 4 | 9.50 (40.6) | 63.1 | 9.21 / 44.5 | 7.24 (18.1) | 33.5 |

Read: one request with Q8_0 is memory bound: at 4 threads it reads 40 GB/s, above what the numpy probe reached, and
the fourth thread adds only 13 % (ROW1's AMX kernels change nothing there: 9.50 against 9.21). Q4_K_M moves 58 % of
the bytes and is slower (7.24 tok/s): it is bound by its decoding (18 GB/s, still rising with threads), not by memory.
With 8 rows compute decides (ROW1 1.42x). So for one request the lever is fewer bytes at Q8_0's compute speed: a 4-bit
weight read at 40 GB/s would allow up to about 16 tok/s; llama.cpp's Q4_K kernels on this CPU reach 7.2. A 4-bit
format changes the answers and goes through the quality gate. Not more SIMD for Q8_0's one-row case.

## 2026-11-15 ST2 (GPU): MQ1, Q4_K repacked for warp streaming, on a T4

`lab/mq1/kaggle_mq1.py` at f2128a9, Kaggle T4 (SG1 v2's llama.cpp build for llama.cpp's own kernels), Qwen3-4B
Q4_K_M. MQ1 (`lab/mq1/mq1.cu`): the Marlin principle (arXiv 2408.11743) as far as a Turing GPU allows (no cp.async):
the Q4_K quants reordered once so that lane j of a warp reads sub-block j's 16 bytes (512 contiguous bytes per warp
and iteration), the same 4.5 bits a weight, the block headers beside them, float32 activations as they are. The
repacking decodes to exactly the weights gguf-py's dequantize gives (max difference 0.0, checked locally). Rule fixed
before the data: on to an end-to-end run only if one request's step would be at least 15 % shorter.

| Q4_K shape (K x N, count) | llama.cpp us (GB/s), 1 token | MQ1 best us (GB/s) | variant W/R/U |
| --- | --- | --- | --- |
| 2560 x 1024 (54) | 24.6 (60) | **16.1** (91) | 4/4/2 |
| 2560 x 4096 (36) | 62.8 (94) | **39.4** (150) | 4/4/1 |
| 2560 x 9728 (72) | 103.0 (136) | **77.2** (182) | 4/4/1 |
| 4096 x 2560 (36) | **29.7** (199) | 35.6 (166) | 4/4/1 |
| 9728 x 2560 (18) | **62.4** (225) | 77.5 (181) | 4/4/1 |

Errors against float64: MQ1 2.4-3.1e-7, llama.cpp 4.8-5.5e-3. Q6_K (llama.cpp only): 4.06 ms a token, the output
matrix 2.07 ms of it. At 2 and 4 tokens MQ1 is 0.3-0.7x llama.cpp (float32 arithmetic per row). All Q4_K matmuls of
one token: llama.cpp 13.19 ms, the faster of the two per shape 10.04 ms. The engine's step was not measured on this
GPU (its probe printed nothing; llama-server did not start at the end, which stopped the run); with the 18.7 ms one
request's step measured on other T4 sessions (KG1 v3, SG1) the projection is 18.7 to 15.5 ms, +20 %; with this
session's slower llama.cpp matmuls (13.2 + 4.1 ms of matmuls alone) it is about +16-18 %. llama.cpp's own times differ
between T4 sessions by up to 1.8x on one shape (2560 x 4096: 34.5 us in KG1 v2, 62.8 us here), so only an
end-to-end run on one session decides. By the rule: on to the end-to-end run, MQ1 for one token only.

## 2026-11-15 KG2 (ST2): the generated kernel library end to end on a T4, one request 1.27x

Kaggle T4, `lab/kg1/kaggle_e2e.py` at the KG2 commit: llama.cpp 4227c9b built with gr1 and the kg1 hook (57 min),
`ironmule/kernel_gen.py` tuned 288 `__dp4a` variants and 18 MQ1 variants on the five Q4_K shapes of Qwen3-4B Q4_K_M
(137 s) and built a 40-entry library, then each arm at 1 / 2 / 4 / 8 requests (128 tokens, one warm-up block and 3
blocks, greedy, SP4 off), two alternating passes. Rule fixed before the data (ST2): one request at least 15 % faster.

| tok/s, mean of 2 passes | llama-server | engine (pool + PAD1) | engine + library | engine, PAD1 off | engine + library, PAD1 off |
| --- | --- | --- | --- | --- | --- |
| 1 request | 46.6 | 44.9 | **57.0** | 44.5 | 57.0 |
| 2 requests | 79.2 | 80.5 | **92.7** | 77.7 | 93.9 |
| 4 requests | 102.5 | 143.4 | **153.0** | 104.7 | 139.8 |
| 8 requests | 127.1 | 272.4 | **277.1** | 132.6 | 203.0 |

The engine's own measurement at startup, one sequence: a step of 1 / 2 / 4 / 8 tokens 16.6 / 18.7 / 27.9 / 36.4 ms
with the library and 18.7 / 22.9 / 36.0 / 55.9 ms without; it kept the library for every count from 1 to 8. Facts 49
of 50 with and without it; the 50 fact answers token for token the same at one request, 47 of 50 at eight. Read:
the rule holds (1.27x the engine without the library, 1.22x llama-server, for one request on one T4), so the server
now tunes and loads the library on a GPU by itself (`ironmule serve --gguf`, cached per GPU, nvcc and model shapes;
the gate runs with it and is keyed by it; IRONMULE_KG=0 turns it off). The gain comes from the `__dp4a` variants:
MQ1 was correct (2.4-3.9e-7 against float64) but 7-22 % slower than the best of them on every shape (2560 x 1024:
14.3 against 11.7 us; 9728 x 2560: 73.7 against 68.5), so no MQ1 kernel entered the library. With PAD1 off the
library alone gives 1.53x at 8 requests; with PAD1 the two overlap (9-16-token steps use llama.cpp's tensor cores).
vLLM AWQ on a T4 (MG2 v6): 63.7 / 419 / 861 tok/s at 1 / 8 / 32 requests, still ahead.

## 2026-11-15 PI1: the Pilot trained from scratch on Kaggle (world v4), shipped for its stability

The same recipe as the CPU run before it (world v4, the warm start from the teacher that knows the dream's true load,
then PPO), from random weights, on Kaggle: two T4 GPUs, one seed each (20, 21), 8,192 dreamed servers, 200 PPO
iterations, 49 minutes (private kernel `ironmule-pilot-train` v3; the world and the recipe stay private).

| | reward (dream, v4) | tokens | wait s | real readings: idle / 25 / 50 / 75 / 100 % |
| --- | --- | --- | --- | --- |
| shipped before (v4, CPU, seed 10) | -1407.7 | 52,253 | 13,119 | 4 / 3 / 3 / 3 / 3 |
| Kaggle seed 20 | -1428.8 | 52,654 | 13,250 | 4 / 3 / 3 / 3 / 3 |
| **Kaggle seed 21** | -1418.5 | **52,828** | 13,159 | 4 / 3 / 3 / 3 / 3 |

On the real machine (4-vCPU Xeon, Qwen3-4B Q8_0, `lab/pi1/real.py`, 64 tokens), the two alternating twice, tok/s for
1 / 8 requests, with the threads each ended on:

| foreign load | Kaggle seed 21 (a, b) | shipped before (a, b) |
| --- | --- | --- |
| none | 9.13 / 63.7 (4), 9.11 / 62.6 (4) | 9.35 / 64.3 (4), 9.45 / 61.0 (4) |
| one busy process | 7.12 / 49.3 (3), 7.40 / 49.2 (3) | 7.34 / 49.3 (3), **4.19 / 36.0 (2)** |
| one process busy half of every 20 ms | 7.15 / 50.2 (3), 7.27 / 51.1 (3) | 7.19 / 51.7 (3), **5.84 / 39.6 (4)** |

Read: in the dream the two are within noise (seed 21 1.1 % more tokens, 0.3 % more waiting). On the machine the
weights before chose differently in their second pass on readings near a boundary (2 threads beside a busy process,
whose reading came out 1.07 / 0.96; 4 beside a half-busy one read as 0.18 / 0.29), and lost 20-43 % there; seed 21 chose
3 both times. Seed 21 replaces them as `ironmule/pilot_weights.npz`: the same speed where both chose alike (idle 9.1
against 9.4 for one request at the same 4 threads, within the runs' spread), and the same choice in both passes.
Beside two busy processes (16-token answers, one block) seed 21 chose 2 threads: 5.04 / 26.0 tok/s for 1 / 8 requests
(the defaults: 0.19 / 1.46 in the run of the weights before).
