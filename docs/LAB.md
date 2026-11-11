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
