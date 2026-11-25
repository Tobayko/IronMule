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

## 2026-11-16 DM2: the draft model on a T4, its rate check fixed

The draft model's stop rule pauses drafting for PROBE steps where steps with drafts measured slower than plain ones.
Their cost included the draft model's catch-up on a new request's prompt, paid once per request, so a stream of new
requests turned drafting off (PI3 probe: 35.3 tok/s measured with drafts for one request; its first round 57 against 39
without). Catch-up seconds are now kept out of that check. On this CPU (Qwen3-4B Q4_K_M with Qwen3-0.6B Q8_0, 12
requests one after another with new 200-400-token prompts, 64 tokens): 365 drafts in 768 tokens instead of 139, 7.3
tok/s measured with drafts against 7.0 plain; overall 4.62 against 4.64 tok/s (prompts dominate on a CPU, and the draft
model reads them too). On a GPU, `ironmule serve` now names a draft model of the same family measured on a T4 (Qwen3,
Llama 3.1) with the command that would download it, and downloads nothing.

Kaggle T4, `lab/pi3/kaggle_gpu_probe.py` in `PI3_MODE=dm2` at 12af3c0, 10 minutes, each arm on its own fresh engine, two
rounds in reversed order. (The harness fetched `pilot_weights.npz` only, so here and in PAD2 and KVD1 the CPU weights
decided on the GPU; the same in both arms. Fixed for later runs.) tok/s, round 1 and round 2:

| | with the draft model | without |
| --- | --- | --- |
| 1 request (4 x 128 tokens) | **60.2, 52.2** | 45.2, 43.8 |
| 16 requests one after another, new prompts | **53.4**, 40.7 | 40.2, 40.2 |
| 8 at once | 265.7, 260.8 | 264.2, 262.9 |
| 32 at once | 604.7, 602.3 | 599.8, 591.8 |
| chat arrivals | 329.5, 324.8 | 332.5, 335.0 |
| mixed arrivals | 73.0, 115.7 | 72.8, 115.8 |

Answers: 49 of 50 facts with and without it, the 50 fact answers token for token the same. 1,955 and 852 drafts, 74 %
and 79 % accepted. Read: one request 1.26x on average (1.33x and 1.19x); 8 and 32 requests and both arrival patterns
the same within 2 % (the draft model rests beyond 2 running requests, by design). In the second round the check
measured 24.2 tok/s with drafts for one request and paused drafting, so the requests one after another gained 1 %
there against 33 % in the first; it tried again only every 200 steps. PROBE is now 25.

Confirmation at cfa3049 (PROBE 25, the GPU weights now fetched and deciding), 10 minutes, the same design: one request
60.7 / 46.2 with the draft model against 46.1 / 44.0 without (round 1 / round 2), one after another 53.9 / 39.8 against
39.2 / 38.9, 8 and 32 requests and both arrival patterns the same within 2 %; 49 of 50 facts, 48 of 50 fact answers the
same; drafts measured at 56.3 tok/s for one request (no pause). Read: in both runs the draft model won most where it
ran first in the session (+32-37 %), where the T4 still runs at its higher starting clock (the probe: 55 against 42
tok/s in its first cell), and less where it ran second (+19 % and +5 %). So the gain for one request lies between
about 5 % and 20 %, not settled; a run that warms the T4 first and orders the arms with, without, without, with
settles it.

The settling run at 982f9bc (`PI3_WARM_S=180`, `PI3_ABBA=1`), 23.5 minutes: 3 minutes of 8-request blocks first (41
blocks, the last at 218 tok/s), then four rounds in the order with, without / without, with / without, with / with,
without, each arm on a fresh engine, the GPU weights deciding. tok/s per round and the mean:

| | with the draft model | without | ratio |
| --- | --- | --- | --- |
| 1 request (4 x 128 tokens) | 51.8, 53.6, 55.2, 55.7: **54.1** | 45.0, 43.7, 43.5, 44.3: 44.1 | **1.23x** |
| 16 requests one after another, new prompts | 48.3, 49.8, 48.6, 48.3: **48.8** | 40.6, 40.0, 41.3, 41.7: 40.9 | **1.19x** |
| 8 at once | 261.4 | 263.8 | 0.99x |
| 32 at once | 598.1 | 598.1 | 1.00x |
| chat arrivals | 341.9 | 342.2 | 1.00x |
| mixed arrivals | 89.5 | 89.7 | 1.00x |

1,910-2,009 drafts per round, 74-76 % accepted; 49 of 50 facts with and without it, 49 of 50 fact answers token for
token the same. Read: with the draft model one request on a T4 makes 1.23x the tokens per second and a stream of new
requests one at a time 1.19x, in every one of the four pairs (+15 % to +28 %), whichever arm ran first; from 8 requests
on the same as without it (it rests beyond two). DM2 ships: the rate check without the catch-up, PROBE 25, and
`ironmule serve` naming the draft model on a GPU.

## 2026-11-17 KG3: what a GPU step with many requests pays for

The owner chose a kernel project for more tokens per second at 8-32 requests on a GPU (BACKLOG PI3). First where the
time goes. From the PAD2 run: a serving step of 20 sequences cost 54 ms of llama.cpp's decode against about 59 ms in
all (the engine's own Python about 5 ms), and 20 tokens of one sequence cost 30 ms at startup. Kaggle T4,
`PI3_MODE=span` at bba69b0, 7 minutes: plain decode steps (median of 24) of n sequences with 200 tokens each, their
cells low in a fresh cache, high after 60 kept long prompts, and after clearing the cache, with and without
defragmentation, two rounds. ms per step (the four runs' range):

| sequences | cells low | cells high | cleared |
| --- | --- | --- | --- |
| 8 | 49-66 | 62-64 | 61-65 |
| 20 | 39-44 | 46.2-46.7 | 43.4-44.4 |
| 32 | 47-55 | 57-61 | 54-60 |

Defragmentation changed nothing. 8 sequences cost more than 20 (8 tokens use llama.cpp's vector kernels, 9 and more
its tensor cores; in serving PAD1 fills 4-8 to 10). Read: cells high in the cache cost about 7 % at 20-32 sequences;
the larger part is that 20 sequences of 200 tokens cost 43.5 ms where 20 tokens of one cost 30. In this llama.cpp
(4227c9b) a GPU step with more than one query token runs the `mma_f16` flash-attention kernel over every cell up to
the span with the mask, with no skip of masked blocks, so each request's token also reads every other request's
cells. Across 120,000 logged T4 serving steps, steps of 20-32 rows cost about 50 ms up to 2-3k attended tokens and
72-83 ms beyond. Next: one KV stream per request (each token reads only its own cells), measured against the pool
without a rebuild (`PI3_MODE=streams`).

`PI3_MODE=streams` at 6d578e6, 12.5 minutes, each arm on a fresh engine, two rounds in reversed order, the GPU Pilot
deciding: the shared pool (16 x 2,048 cells), the pool without speculation, and one KV stream per request (32 x 1,024
cells, the engine's per-sequence mode) without speculation. tok/s, mean of the two rounds:

| | pool | pool, no speculation | one stream per request, no speculation |
| --- | --- | --- | --- |
| 8 requests, ~420-token prompts | **82.4** | 60.0 | 57.1 |
| 20 requests, ~420-token prompts | 60.4 | 55.0 | **79.6** |
| 32 requests, ~420-token prompts | 42.0 | 39.6 | **73.4** |
| 32 short requests at once | 301.4 | 308.5 | **564.3** (first token 0.44 against 0.71 s) |
| chat arrivals | **305.6** | 306.8 | 148.6 |

49 of 50 facts in every arm. Read: with one stream per request a token reads only its own cells, and 32 requests at
once make 1.75-1.87x the pool's tokens per second; with requests arriving it makes half. The likely reason is llama.cpp's
split of a call with one KV stream per sequence: sequences with different token counts in one call (a new prompt beside
running requests, or uneven drafts) go into a pass per count. The engine now gives a new prompt its own call and every
stream the same number of drafts in that mode (the pool is unchanged); 8 of 8 answers on this CPU the same as the
pool's with requests arriving mid-run. (The prompt budget of 128 the GPU Pilot holds lets in one 420-token prompt per
step, which is why those blocks are slow in every arm.)

KG3 v2 at 9cbd5a2 (prompts in their own call, equal drafts per stream; `PI3_STREAMS_V=2`), 19.4 minutes: two
minutes of warm-up, four rounds (pool, streams / streams, pool / streams, pool / pool, streams), one stream per request
with 32 x 2,048 cells and speculation. tok/s, mean of four rounds, streams against the pool: 8 requests of ~420-token
prompts 63.3 against 82.6 (0.77x), 20 requests 80.2 against 73.9 (1.08x), 32 requests 75.1 against 52.2 (1.44x), 32
short requests at once 473 against 348 (1.36x), mixed arrivals 88.3 against 79.8 (1.11x, first token 2.4 against 5.6
s), chat arrivals 154 against 320 (0.48x); 49 of 50 facts either way. The logged steps show why chat lost: with streams
a step of 32 rows cost 46 ms but one of 10 rows 209 ms and of 5 rows 103 ms. llama.cpp keeps a call over KV streams in
one pass only while its sequence ids run consecutively (`split_equal(..., sequential=true)`); once a request ends, the
running ones leave gaps and every step splits into a pass per run. The engine now fills such a gap with placeholder
rows for the free id (the BOS token past what its stream holds, no logits, removed after the step) and gives a new
request the lowest free id unless a kept prefix of 64 tokens or more is elsewhere; on this CPU 8 of 8 answers with
requests arriving mid-run are the pool's.

The first T4 run of that (at db6130f) stopped after its first arm: a new request that took the free id in the middle
was prefilled in its own call in the same step, after the placeholder for that id had been planned at position 0, and
llama.cpp refused the row ("inconsistent sequence positions"). The placeholders are now planned right before the
step's call, after its prompt calls, at the position llama.cpp reports for the stream (`llama_memory_seq_pos_max`);
the stub test now refuses rows as llama.cpp does and fails on the earlier code. On this CPU, with a request taking
the free id between two running ones: 4 of 5 answers token for token the pool's, the fifth "The Rhine" against
"Rhine" in its fifteenth token (a step of a different width rounds differently, as PAD1's do on a GPU).

KG3 v3 at 63e6e13 (the same design as v2), 16 minutes, complete. tok/s, mean of four rounds, one stream per request
against the pool (each round's numbers within a few per cent of each other):

| | one stream per request | shared pool | ratio |
| --- | --- | --- | --- |
| 8 requests, ~420-token prompts | 67.7 | **84.3** | 0.80x |
| 20 requests, ~420-token prompts | **89.4** | 75.7 | **1.18x** |
| 32 requests, ~420-token prompts | **92.4** | 54.5 | **1.70x** |
| 32 short requests at once | **567.8** (first token 0.72 s) | 385.6 (0.60 s) | **1.47x** |
| chat arrivals | **354.1** (first token 0.07 s) | 332.9 (0.11 s) | **1.06x** |
| mixed arrivals | **129.6** (first token 2.04 s) | 85.0 (5.25 s) | **1.53x** |

49 of 50 facts either way; 47 of the 50 fact answers token for token the pool's (a step of a different width rounds
differently). A step of 32 streams cost 44 ms and no step split any more. Read: one KV stream per request is 1.18-1.70x
the pool from 20 requests on and 1.53x with documents among arriving chats, 1.06x for chats alone; the pool stays ahead
at 8 requests, where PAD1 fills a step to 10 rows (the tensor cores), which needs the one pool. Next: the same
filling for streams (every stream verifies the same number of guesses), then streams as the GPU default.

PAD3 and the check at 70ece6b, 18.6 minutes (the same design, 4 and 8 requests and 8 short ones at once added).
tok/s, mean of four rounds, one stream per request against the pool:

| | one stream per request | shared pool | ratio |
| --- | --- | --- | --- |
| 4 requests, ~420-token prompts | 68.5 | 69.8 | 0.98x |
| 8 requests, ~420-token prompts | **81.6** | 78.2 | 1.04x |
| 8 short requests at once | **234.6** (first token 0.17 s) | 194.4 (0.15 s) | **1.21x** |
| 20 requests | **88.7** | 69.1 | **1.28x** |
| 32 requests | **89.6** | 50.4 | **1.78x** |
| 32 short requests at once | **545.8** (first token 0.73 s) | 343.3 (0.65 s) | **1.59x** |
| chat arrivals | **343.9** (first token 0.07 s) | 319.4 (0.15 s) | 1.08x |
| mixed arrivals | **123.6** (first token 2.07 s) | 80.0 (6.47 s) | **1.54x** |

About 2,300 PAD3 guesses per round; 49 of 50 facts either way, 49 of 50 fact answers token for token the pool's. Read:
with PAD3 one stream per request is at least level with the pool everywhere measured (4 requests inside the spread)
and 1.28-1.78x from 20 requests on; the first token of a block of requests comes 0.02-0.08 s later (its prompts run
in their own calls). So on a GPU `ironmule serve` now gates this setting (flash attention, one KV stream per request)
beside the pool's and serves with it where it passes; `IRONMULE_GGML_STREAMS=0` keeps the pool. The serve path itself
was not run on a GPU here; the engine was, through the harness.

## 2026-11-17 KG3 end to end: `ironmule serve` on a T4, and SEQ1's width broken since PI1

`lab/pi3/kaggle_serve.py` at 758ec73 (a first try stopped at its first start: the harness had copied MG2's build where
its libraries could not find each other), Kaggle T4, Qwen3-4B Q4_K_M, the repository installed, `ironmule setup`,
four starts of `ironmule serve --gguf` (streams, pool, pool, streams), chat requests through `/v1/chat/completions`
(128 tokens), 15 minutes. tok/s for 1 / 8 / 32 requests at once: streams 45.6 / 243.9 / 260.6 and 47.4 / 244.8 /
267.0; pool 45.4 / 266.0 / 433.1 and 42.8 / 259.7 / 426.6; 49 of 50 facts in every start. So the served streams
were 0.61x the pool at 32 requests. `/health` showed why: every start served 8 sequences, "the KV size could not be
read; 8 sequences as before", so the streams took 8 requests at once and the pool its cap of 16. SEQ1's KV-size probe
opens an engine of 256 cells, and since PI1 an engine opens the Pilot, whose cost measurement decodes past 256 tokens:
the probe failed on every machine and every server since then served 8 sequences. Fixed: SEQ1's children run without
the Pilot, an engine of fewer than 512 cells does not open it, and `ironmule serve` keeps the pool where the streams
would get fewer than 16 sequences. On this CPU (Qwen3-0.6B Q8_0, the streams' settings) SEQ1 now measures 8 / 16 / 32
sequences at 179 / 286 / 319 tok/s and chooses 32.

Again on the T4 at 639de29 (SEQ1 fixed), 9 minutes, the same design. The streams' starts measured 8 / 16 / 32
sequences and served 32; the pool's starts (llama.cpp's default setting, which the gate kept on this T4) reported "the
width measurement failed; 8 sequences as before" and served 8 sequences with a cap of 16 (that child's error was not
recorded; on this CPU the same measurement with the default setting works). tok/s, mean of two starts each:

| `ironmule serve` | one stream per request (32) | shared pool (8, cap 16) | ratio |
| --- | --- | --- | --- |
| 1 request | 46.3 | 43.6 | 1.06x |
| 8 requests at once | 238.3 | **260.0** | 0.92x |
| 32 requests at once | **518.4** | 422.2 | **1.23x** |

49 of 50 facts in every start. Read: through the server the streams default makes 32 requests 1.23x what the pool
serves today and 8 requests 0.92x; one request the same. The pool here has 16,384 cells against 32,768 in the harness,
so its steps read fewer cells (the harness's pool at 32 short requests: 343-386 tok/s). Open: the pool's width
measurement with llama.cpp's default setting on a GPU, and the streams' 8-request step through the server.

## 2026-11-17 KG4: where a GPU step of 8-32 requests goes, and a Q4_K tensor-core kernel for 1-32 tokens

Owner's choice (2026-11-17): the kernel project for more tokens per second at 8 and 32 requests.

Probe (`lab/kg4/kaggle_prof.py` at 43c15e0, Kaggle T4, Qwen3-4B Q4_K_M, the engine as `ironmule serve` runs it on a
GPU: one KV stream per request, 32 sequences, speculation off, 2.6 minutes). A step of one token per request, ms (tok/s):
1 request 18.1 (55), 8 requests 49.3 (162), 16 requests 30.9 (518), 32 requests 46.1 (695) at context 300; at context
1,000 52.7 / 37.3 / 60.9 ms for 8 / 16 / 32. Logits for every row or for one row: the same within noise (the copy and
the argmax on the host cost about 3 ms at 32 rows, hidden behind the GPU). CUPTI kernel profile, GPU busy 92-98 % of
the wall time: at 8 rows llama.cpp's `mul_mat_vec_q<Q4_K, 8>` takes 41.7 of 57.4 ms (so 8 rows cost more than 16, which
the engine's padding works around); at 32 rows `mul_mat_q` takes 20.7 ms on Q4_K, 6.6 on Q6_K and 2.2 in stream-k
fix-ups, flash attention 10.5 ms (context 300). Q4_K at 32 tokens reads its weights at about 97 GB/s of the T4's 320.

`lab/kg4/kmma.cu` v1 (at 0514e0f, `lab/kg4/kaggle_kmma.py`, 23 minutes): Q4_K x 8-bit activations on Turing's tensor
cores (`mma.sync m8n8k16 s8`), activations in fragment order staged in shared memory, the weights read once per row;
its layout emulated in numpy first (exact against plain dequantisation). Against `ggml_mul_mat` on the same data,
all Q4_K matmuls of one step weighted by how often the model has each shape, the faster of the two per shape:

| tokens | llama.cpp | with kmma v1 | kmma error / llama.cpp's (float64) |
| --- | --- | --- | --- |
| 1 | 9.89 ms | 9.42 | 5.2e-3 / 5.2e-3 |
| 8 | 29.13 ms | **11.36** | 5.4e-3 / 5.4e-3 |
| 16 | 16.44 ms | 13.80 | 5.4e-3 / 1.2e-2 |
| 24 | 17.60 ms | 15.83 | 5.4e-3 / 1.2e-2 |
| 32 | 20.11 ms | 17.44 | 5.4e-3 / 1.2e-2 |

Read: correct everywhere and half llama.cpp's error from 16 tokens on. At 8 tokens 2.6x; at 16-32 tokens 8-16 %, the
2560 x 1024 shape 2.5x, the large ones even. The kernel reads 160-170 GB/s at 1-8 tokens (each 144-byte block was
loaded and then computed, a barrier per block) and 87-112 GB/s at 32, where turning every sub-block's int32 into
float (a quarter-rate instruction on Turing) costs about as much as the reads. v2: the next block's weights loaded
while the current one is computed, activations staged 2-8 blocks per barrier, their scales as half2 (as llama.cpp's
q8_1), and int32 to float through an exact integer-and-float add. `ironmule/kernel_gen.py` (KG_VERSION 3) tunes it
beside the dp4a and MQ1 kernels and serves up to 32 tokens through the kg1 hook; the engine measures 16 / 24 / 32 at
startup for 9-32. Next: the end-to-end run (`lab/kg4/kaggle_e2e.py`).

End to end (`lab/kg4/kaggle_e2e.py` at b147c66, Kaggle T4, KG2's llama.cpp build with the kg1 hook, 8.6 minutes): the
library tuned in 189 s (160 entries; at 8 tokens the tensor-core kernel won every shape, at 1 token the dp4a ones of
KG2), the engine (one KV stream per request, 32 sequences, padding and speculation as shipped, Pilot off) with the
library ("kg", the engine's own choice: it kept 1-8, 16, 24 and 32) and with its mask at 0 ("off"), alternating, two
rounds. Q4_K matmuls of one step in isolation, llama.cpp (v1 run) against the v2 entries: 29.1 / 16.4 / 17.6 / 20.1 ms
against 11.0 / 11.8 / 13.6 / 14.7 at 8 / 16 / 24 / 32 tokens. In the engine:

| | kg | off | ratio |
| --- | --- | --- | --- |
| step, 8 requests (ctx 300 / 1,000) | 29.3 / 33.2 ms | 58.5 / 60.5 ms | 2.0x / 1.8x |
| step, 16 requests | 32.0 / 39.4 ms | 34.0 / 40.4 ms | 1.06x / 1.03x |
| step, 32 requests | 49.1 / 60.5 ms | 50.2 / 63.4 ms | 1.02x / 1.05x |
| block, 1 request, 128 tokens | 57.8 tok/s | 45.1 | 1.28x |
| block, 8 requests | **280.3** | 250.6 | **1.12x** |
| block, 32 requests | **641.7** | 602.2 | **1.07x** |

Facts 49 of 50 either way, 48 of the 50 answers token for token the same. Read: shipped (the engine keeps the library
only where its own startup measurement is at least 3 % faster). The raw 8-request step halves, but the engine already
padded 8 rows to a cheaper count, so blocks of 8 gain 1.12x; at 32 requests the isolated kernels would save 5.4 ms of
a 50 ms step and the engine sees 1-3 ms: unexplained so far (clock under sustained load, the extra quantize and
memset launches per matmul, or attention and Q6_K growing their share), the next profile's question. Against the
library shipped before (KG2: 1-8 tokens) the new part is 8-32 requests; one request is KG2's gain.

Profile (`lab/kg4/kaggle_prof2.py` at e822e46, T4, 4.5 minutes; the engine's steps of one token per request, context
300, the library on for every count or its mask at 0, CUPTI over 8 steps). Steps: 8 requests 27.0 against 53.8 ms, 32
requests 44.8 against 47.5 ms (2.0x and 1.06x). Per step at 32 requests: the Q4_K matmuls with their activation
quantization 18.2 ms through the library against 21.9 through llama.cpp (`mul_mat_q` 19.4, stream-k fix-ups 1.7,
quantization 0.8), so 3.7 ms of the 5.4 the isolated kernels promised arrive: the library's quantization costs 1.5 ms
against llama.cpp's 0.8, and the kernels run about 12 % slower inside the step than alone. The rest of the 32-request
step is not Q4_K: flash attention 10.5 ms, Q6_K 6.6 ms (`mul_mat_q`), the logits' copy to the host 1.6 ms, norms,
rope and the like about 3 ms; the GPU is busy 91 % of the wall time either way. At 8 requests llama.cpp's Q6_K kernel
(`mul_mat_vec_q<Q6_K, 8>`) takes 8-12 ms of the step. Next: the same tensor-core kernel for Q6_K (its weights, 0.7 GB,
could be read in about 2.4 ms) and a cheaper activation quantization.

Q6_K on the tensor cores (KG_VERSION 4, at fc13ca2, `lab/kg4/kaggle_e2e.py` as above, 9.7 minutes; the layout emulated
in numpy first, exact). The 210-byte Q6_K blocks are read with 2-byte loads, every 16-weight chunk of the tensor cores
lies inside one scale group (the activations in a second lane order), and the two chunks of a 32-value group are
scaled in int32. Against llama.cpp's Q6_K matmuls of one step (the KG4 v1 run): 5.61 against 7.30 ms at 8 tokens (the
output matrix 2.76 against 3.90 ms), but 5.61 against 3.97 at 1 token and 6.60 / 8.11 / 8.51 against 5.25 / 5.83 / 6.45
at 16 / 24 / 32, where llama.cpp's own tensor-core kernel is better; error 5.2-5.4e-3 against float64 on every
variant. In the engine (the library's one mask then served Q4_K and Q6_K together, so the engine kept 1-8 and 16 and
dropped 24 and 32 for both): blocks of 8 requests 322.7 tok/s (the run before, Q4_K alone: 280.3), 32 requests 631.4
against 630.2 without the library, 1 request 56.4 against 46.0; 49 of 50 facts either way and 49 of the 50 answers
token for token the same. The off arm here kept the padding measured with the library on, so its 8-request blocks
(141.9) are not the engine's own (250.6 in the run before). Fixed since: Q6_K has its own mask and the engine keeps
it only for the token counts where Q4_K and Q6_K together beat Q4_K alone and the step without the library; the
harness measures each arm's own padding.

Confirmed at 5972eac (T4, 9.7 minutes, each arm on its own padding table): the engine kept the library's Q4_K kernels
for 1-8, 16, 24 and 32 tokens and its Q6_K kernels for 3 and 5-8 (Q4_K and Q6_K together 23.3 against 27.0 ms at 8
tokens, 41.4 against 36.5 at 32). Blocks of 128 tokens, mean of two rounds: 1 request 56.3 against 41.2 tok/s
(1.37x), **8 requests 310.4 against 249.2 (1.25x)**, 32 requests 626.9 against 602.2 (1.04x); steps of one token per
request at context 300: 26.4 against 60.8 ms at 8 (unpadded), 32.8 against 34.5 at 16, 49.4 against 50.7 at 32. 49 of 50
facts either way, 48 of the 50 answers token for token the same.

## 2026-11-17 PI4 (not run): would a GPU Pilot trained anew on the KG4-era costs decide better?

After KG4 the T4's step costs changed (8 rows no longer dearer than 16, one row 17-19 ms, 8 / 16 / 32 rows 1.35 / 1.63
/ 2.38 of it, 1k attended tokens about 0.028 of it with one KV stream per request). Before spending GPU quota on a new
training from scratch, the dream world was given exactly these costs for every GPU server (v8, private, CPU only, no
quota), with the prompt budget ignored as the engine does since PB1. 1,024 dreamed servers, 120 s of traffic each:
engine defaults 10,932 tokens, 825 request-seconds of waiting, first token 3.26 s; the shipped GPU weights 11,047 /
823 / 3.18 s; drafts off 10,684 / 749 / 3.08 s; a cap of 16 10,036 / 1,019 / 3.74 s; a cap of 8 8,643 / 1,607 / 5.21 s.
Read: on such a GPU the decisions that are left (the cap, the draft) are already at their best with the shipped
weights; a new training could gain about 1-2 %, under the T4's own spread between runs. The owner chose not to train
(2026-11-17); the GPU weights stay.

## 2026-11-17 PB1: the GPU Pilot's prompt budget with one KV stream per request

Another session read low tokens per second for bursts of medium prompts on the T4 and suspected the GPU Pilot's prompt
budget (128 tokens a step) of costing 4-5x. `lab/pi3/kaggle_gpu_probe.py` (`PI3_MODE=budget`) at 5abba8d, Kaggle T4,
Qwen3-4B Q4_K_M, the shipped GPU engine (one KV stream per request, PAD3), 120 s warm-up, two rounds, 9.9 minutes.
Arms: the GPU Pilot, the Pilot off (every waiting prompt that fits), a fixed budget of 512 (cap 32). Workloads: blocks
of 8 / 20 / 32 medium prompts (about 400 tokens, a one-sentence answer asked, at most 128), 32 short chats, mixed arrivals (chats and documents), chats arriving.
tok/s, round 1 / round 2:

| Workload | Pilot | Pilot off | budget 512 |
| --- | --- | --- | --- |
| 8 medium | 88.1 / 81.8 | **89.6 / 84.3** | 84.0 / 80.9 |
| 20 medium | 95.2 / 86.9 | **98.7 / 92.9** | 93.5 / 87.1 |
| 32 medium | 96.4 / 89.4 | **102.7 / 98.3** | 96.2 / 88.6 |
| 32 chats | 591.8 / 570.8 | **610.5 / 591.3** | 604.7 / 599.0 |
| mixed arrivals | 118.5 / 162.6 | 121.7 / 158.0 | 120.0 / 161.6 |
| chats arriving | 366.2 / 318.0 | 365.0 / 318.1 | 353.9 / 319.4 |

49 of 50 facts in every arm. The suspicion is refuted: the medium blocks are slow in tokens per second because their
answers are short and the prompts dominate, for every arm alike. The budget itself cost 2-9 % on the blocks and gained
nothing on arrivals. Shipped: with one KV stream per request on a GPU each prompt already gets its own call, so the
engine no longer holds the Pilot's prompt budget there (as the "Pilot off" arm); the shared pool and the CPU keep it.
`IRONMULE_GGML_PROMPT_BUDGET` still sets one by hand.

Confirmed on the T4 at a1ac392 (the change merged), the same harness and design, 9.6 minutes. The "pilot" arm now runs
the shipped path; the "budget 512" arm, a fixed decision passed as a Pilot, now also has its budget ignored, so it
differs from the Pilot only in its fixed cap of 32. tok/s, round 1 / round 2:

| Workload | Pilot (shipped) | Pilot off | budget 512 (ignored) |
| --- | --- | --- | --- |
| 8 medium | 90.0 / 83.3 | 86.7 / 81.8 | 85.1 / 86.8 |
| 20 medium | 97.9 / 91.8 | 97.3 / 91.5 | 94.6 / 97.2 |
| 32 medium | 102.3 / 95.8 | 102.4 / 96.4 | 102.3 / 103.5 |
| 32 chats | 602.9 / 571.6 | 596.9 / 571.6 | 591.3 / 592.2 |
| mixed arrivals | 119.3 / 156.5 | 119.2 / 160.5 | 115.0 / 160.8 |
| chats arriving | 349.2 / 318.6 | 354.8 / 321.8 | 352.6 / 318.7 |

49 of 50 facts in every arm. The Pilot now equals the Pilot-off engine on the blocks (32 medium: 102.3 against 102.4,
before the change 96.4 against 102.7), first token with mixed arrivals 2.29 / 2.02 s against 2.41 / 2.02 s.

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

### 2026-11-18 — Local Pilot thread-weight replay on Linux CPU

One functional run used SmolLM2-135M-Instruct Q4_K_M, three sequences of 512 tokens and plain greedy decode on a
shared Linux runner. The real low-load criterion excluded this runner (2.55 foreign cores at observation). For
this smoke only, its load sensor was held at zero to exercise the probes. Starting from an 80-token filler
context, 1, 2 and 3 threads each returned the same eight generated tokens. A separate 12-token request also
matched before and after replay. The search comparison retained its baseline and installed no weight. Default-on
use, live throughput and first-token latency still need representative A/B measurements under natural traffic.
This was a functional check. The pass and rollback paths also have deterministic unit fixtures.

### 2026-11-18 — AG10: a real HTTP server through two changes in prompt length

Linux runner (AMD EPYC 9V74, cgroup quota 4 CPUs), SmolLM2-135M-Instruct Q4_K_M, model SHA-256
`2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`, patched CPU llama.cpp 4227c9b.
Both servers used the same default numeric settings and an initial width of 8. Pilot, shared KV and speculation
were off in both to isolate AG10. One warmup, then 48 requests each for short chats, document prompts, short chats
again (mean prompt tokens 50, 1,085, 52). The output limit was 16. Synthetic record numbers varied; the eight
capital questions repeated. Each 4.5 s gap between waves allowed the real AG10 idle loop to run. The throughput
below counts reported output tokens per active wave second; p95 is time from HTTP request to first content.

| Concurrent | Phase | Agent / fixed tok/s | Agent / fixed p95 first token (s) |
| ---: | --- | ---: | ---: |
| 8 | short | 85.56 / 59.75 | 0.67 / 1.22 |
| 8 | documents | 4.72 / 3.33 | 15.75 / 21.62 |
| 8 | short again | 88.27 / 57.01 | 0.61 / 1.11 |
| 16 | short | 97.53 / 80.50 | 1.17 / 1.45 |
| 16 | documents | 4.20 / 4.80 | 40.47 / 28.72 |
| 16 | short again | 45.85 / 84.68 | 3.21 / 1.33 |

At 8, the agent kept w8 in all 13 decisions. Another CPU job overlapped the fixed arm; the ratios do not measure
a benefit from learning. At 16, fixed ran first, then agent with the same width calibration. The quality gate
revision changed between these two startups and both chose the same default options. The run had one sample
per arm. The agent served w8 through the first document wave, switched to w64 after request 65, and served two
document waves and the first two returning short waves there. Its trial requires 64 further completions
(`2 * WINDOW`) before it can judge a reversal. It reverted after request 129, and served the last short wave at
w8. With 16-way traffic, warm-up included, that is the earliest possible evaluation of this trial. While the
dream predicted a gain, the measured document and returning phases had lower throughput and higher p95 latency.
A candidate for a future experiment is an earlier live regression check at a load change. The run does not isolate
the cost of width alone; the normal agent and fixed serving paths also differ in admission capacity.

At 16, all 144 paired streamed texts matched by SHA-256 (143/144 at 8). Neither arm returned bare city names.
Checking the timed output hashes against the correct full-sentence answers verified 138/144 in every arm.
Four separate replay samples confirmed the format, e.g. "The capital of France is Paris." At 16, all six
remaining document answers were replayed and matched their recorded hashes. They described Austria without
naming Vienna before reaching the token limit. These repeated questions do not establish general model quality.

To repeat after exporting the bindings and library path printed by `tools/build_ggml.py`, use fresh result directories:

```sh
python lab/ag10/run_pair.py "$GGUF" run-8 --requests 48 --width 8 --lines 96 --idle 4.5
python lab/ag10/run_pair.py "$GGUF" run-16 --requests 48 --width 16 --lines 96 --idle 4.5 \
  --first fixed --gate-cache run-8/fixed-state/ggml-gate
```

## 2026-11-18 Historical MLX adapter trial and T4 preparation

On Linux CPU, SmolLM2-135M-Instruct was expanded to dense float32 from the MLX 8-bit snapshot at revision
`0f0d9b8218915bc34d401e1a340b8c049d300d5e`. With one OpenBLAS thread, four rank-four LoRA updates on
synthetic examples and fusion took 3.81 seconds. Frozen challenger, incumbent and repeated incumbent each
scored 12/18 on the fresh QG1 v3 challenge (seed 4314256273567495177): 4/6 arithmetic, 2/6 extraction and
6/6 facts. The candidate failed the required improvement, so no revision was promoted and the later perplexity
test was skipped. Earlier strict-format grading gave 0/18 on a different challenge; inspection found correct
answers expressed as complete sentences. The v3 grading rule was revised after that observation. Other CPU tests
overlapped the v3 run, so its measured answer latencies are unsuitable for a speed claim. The owner subsequently
confirmed that learning applies to hardware execution. The independent QG1 checks now assess execution choices
with a fixed language-model checkpoint.

The experimental T4 attention kernel, its probe and the patched upstream translation unit compiled locally
for SM75 with CUDA 12.6. Numerical execution on a physical T4, the complete step comparison and the HTTP
comparison remain open. The [KG5 run package](../lab/kg5/README.md) caps a combined job at ten minutes of
workload time and records intermediate results on rejection or timeout.


### 2026-11-18 — From-scratch hardware model, two concurrent decodes

On the same Linux CPU and SmolLM2-135M-Instruct Q4_K_M, two simultaneous requests produced nine observations
in the same 64-token context band. The load sensor read approximately three foreign cores. This probe used its
unmodified readings. Two paired eight-step decode probes with two sequences each produced 16 matching tokens per
pair. One CPU thread took 0.858 and 0.756 of two threads' times. These were search observations from a shared
runner: they created eleven coefficients from zero in the local model database. No serving weight was qualified by
these two comparisons. The restart, learning from losses, hardware transfer of suggestions, local qualification and
rollback paths also have deterministic tests across multiple concurrency and foreign-load regimes.

### 2026-11-18 — Earlier AG10 withdrawal and recovery after a failed live switch

Two further pairs used the HTTP transition protocol above on the same 4-vCPU EPYC runner and SmolLM2-135M GGUF.
Each arm began at w8, with fresh agent state and the same cached width measurements. Each wave had 16 requests.
The new withdrawal compares worker busy periods with similar prompt and output lengths, request counts and
computed prompt tokens. One window must contain at least eight completions; worker first-token time starts at
admission. The final guard also bins the median inter-arrival spacing. A slower candidate withdraws if it misses
85 % of incumbent throughput or exceeds 125 % of its mean first-token time. Up to the last three matched windows
per configuration are compared by median; these observations can withdraw or block a candidate, not qualify one.

| Arm order and change | Phase | Agent / fixed output tok/s | Agent / fixed p95 HTTP first-token time (s) |
| --- | --- | ---: | ---: |
| Agent first, early withdrawal only | short | 93.67 / 73.98 | 1.25 / 1.51 |
| | documents | 3.77 / 4.56 | 40.14 / 30.84 |
| | short again | 15.19 / 75.54 | 12.17 / 1.48 |
| Fixed first, early withdrawal and fresh-traffic recovery | short | 79.64 / 89.23 | 1.42 / 1.32 |
| | documents | 3.68 / 4.23 | 42.13 / 32.97 |
| | short again | 50.77 / 78.68 | 2.17 / 1.54 |

In both pairs, the agent switched from w8 to w64 after 65 requests and reverted after 81 (16 new completions,
previously 64). In the first pair the w64 document window delivered 0.729x the w8 window's worker throughput
and 1.421x its mean worker first-token time. The old dream still proposed w32 after 97 requests and w64 after
129. Both trials reverted after a single returning-chat wave, with a high cost to those waves.

The final guard holds the recovered incumbent until at least 64 fresh completions in consecutive windows match the
current request mix. The next trial must qualify on those fresh arrivals. The second pair saw a 0.684x worker
throughput ratio and 1.268x worker first-token ratio on the w64 document trial. The remaining document wave stayed
at w8, then the mix changed, restarting the evidence count. All three returning-chat waves stayed at w8 (16, 32, 48
matching completions). Each pair had 143/144 identical streamed responses, and 138/144 correct factual answers on
both arms. The quicker recovery and absence of further switches were observed in the real server. The final
adaptive arm was slower end to end than the fixed arm in all three phases; the test has one sample per arm.

A synthetic replay with two seeds and 12 windows per phase gave agent/fixed throughput ratios of
1.000 / 1.195 / 0.897 and 1.000 / 1.177 / 0.858 for short/document/short traffic. This simulator supplies
step costs and arrivals but no request-level completion timestamps. It exercises the existing dream decisions;
the real withdrawal was checked above. Deterministic tests also cover a permanent move to singleton requests
(64 new completions unlock a fresh dream), a continually alternating mix (512 new completions expire the hold),
and an idle tick without new arrivals.

### 2026-11-18 — Short repeated prompts on the preferred GGUF stream

During a subsequent CPU test with SmolLM2-135M-Instruct Q4_K_M, a repeated 16-token prompt reused zero tokens
from its preferred KV stream. The 64-token threshold for moving a request to a different stream was also being
applied to the stream the scheduler had already chosen. After limiting that threshold to stream changes, the
same request reused 15 prompt tokens and produced the same output as the fresh request. The HTTP timings above
precede this change. Its end-to-end speed impact has not been measured.

### 2026-11-18 — GGUF sequence widths on Runpod, including the cost of a load transition

The [portable width experiment](../lab/portable_gpu/README.md) compares one and four sequence
slots on the same GGUF with greedy decoding. Each workload has a separate calibration and
holdout prompt set, two repetitions per width, alternating arm order, and cleared KV between
batches. One or four requests arrive together. Throughput counts generated tokens over engine
wall time. First-token latency is the median across requests, then the median across repetitions.
The GPU must execute a device-local arithmetic graph and the loader must confirm every model
layer was offloaded. A comparison qualifies only when both widths copy every expected first
word and generate identical full token sequences. The six-token answer limit and synthetic
copy task bound the scope of the results.

| GPU, model and held-out workload | Width 1 output tok/s | Width 4 output tok/s | Width 1 first token | Width 4 first token |
| --- | ---: | ---: | ---: | ---: |
| A100 SXM 80 GB, SmolLM2-135M, four 32-token requests | 250.74 | 391.56 | 44.53 ms | 21.99 ms |
| A100 SXM 80 GB, Qwen2.5-0.5B, four 256-token requests | 168.57 | 261.05 | 72.39 ms | 36.70 ms |
| H100 80 GB, Qwen3-4B, four 32-token requests | 164.15 | 288.93 | 65.91 ms | 27.03 ms |
| H100 80 GB, Qwen3-4B, four 1,024-token requests | 49.71 | 51.95 | 276.11 ms | 248.86 ms |

On the H100, Qwen3-4B served a fresh short request, a burst of four 1,024-token requests,
then another short request. The calibration rule selected widths 1, 4, 1. Each switch rebuilt
the native context and warmed it while the new wave waited. Both trials returned identical,
correct answers. The two fixed-width runs each took 551 ms. The two switching runs took
650 and 648 ms. Median first-token latency on the returning short request rose from about
11 ms to 63 ms. These measurements include reconfiguration at arrival. An idle server may
absorb some rebuild time when the next request arrives later.

The two arms use IronMule's own GGUF engine with the same pinned llama.cpp kernel build.
These measurements quantify the effect of sequence width inside that engine. Stock HTTP
serving and vLLM require their own end-to-end comparisons.

A second H100 experiment exercised the controller itself, with measured context creation
and release times on Qwen3-4B. The two policies received the same 47 calibration requests
before 54 fresh requests per arm, including sustained concurrency after the document burst.
The harness called the real idle decision between waves and charged the decision and any
physical rebuild to arriving requests. This was a single ordered comparison.
The original policy switched once and delivered 125.21 tokens/s with a median first-token
latency of 188.86 ms. The cost-aware challenger stayed at width 1 and delivered 103.49
tokens/s with 268.46 ms median first-token latency. Both policies produced the same token
IDs and stop reasons and copied all 54 answers correctly. The challenger lost 17.35% in
throughput on this trace. Its short-horizon estimate missed the subsequent sustained
traffic. The original policy remains the serving default.

**32B models.** The same experiment used two monolithic Q4_K_M files on the A100 SXM
80 GB and RTX 6000 Ada 48 GB. Both models loaded all 65 layers
on each tested device. The following rows are held-out medians for four simultaneous
requests. First-token measurements follow the same per-request timing boundary.

| Model, GPU | Input tokens per request | Width 1 → 4, output tok/s | Width 1 → 4, first token | Quality |
| --- | ---: | ---: | ---: | --- |
| Qwen3-32B, A100 SXM | 32 | 33.81 → 51.34 | 324.56 → 145.51 ms | Exact, correct |
| Qwen3-32B, A100 SXM | 1,024 | 5.67 → 5.74 | 2,519.53 → 2,406.09 ms | Exact, correct |
| Qwen3-32B, RTX 6000 Ada | 32 | 34.61 → 68.52 | 302.97 → 97.19 ms | Exact, correct |
| Qwen3-32B, RTX 6000 Ada | 1,024 | 7.73 → 8.18 | 1,796.81 → 1,649.01 ms | Later tokens differed; excluded |
| Qwen2.5-32B-Instruct, A100 SXM | 32 | 33.92 → 52.30 | 322.76 → 141.39 ms | Exact, correct |
| Qwen2.5-32B-Instruct, A100 SXM | 1,024 | 5.90 → 6.06 | 2,420.54 → 2,281.51 ms | Exact, correct |
| Qwen2.5-32B-Instruct, RTX 6000 Ada | 32 | 34.54 → 68.22 | 304.42 → 98.36 ms | Exact, correct |
| Qwen2.5-32B-Instruct, RTX 6000 Ada | 1,024 | 8.04 → 8.62 | 1,721.66 → 1,560.91 ms | Exact, correct |

All four calibrations selected width 4 for four short requests. Qwen3-32B on A100
selected width 1 for four document requests. On RTX 6000 Ada, calibration selected 4
for the same long shape but generated-token agreement failed on fresh targets. A device
or model name alone would have missed that behavior.

**Coverage and learning.** Twenty-one completed comparisons covered 20 GPU/GGUF combinations
on A40, A100 SXM, H100, L40S, RTX 4090 and RTX 6000 Ada. The nine models ranged from
SmolLM2-135M through Qwen2.5-32B-Instruct and Qwen3-32B. Repeated H100/Qwen3-4B
measurements count as separate runs. Of 168 paired workload measurements, 83
calibration pairs entered the fits, 81 holdout pairs passed quality and four pairs
were rejected. For the GPU-type transfer test, 16 folds had both an eligible source
GPU and target data. On 61 quality-matched target pairs the log-ratio mean absolute
error was 0.068 for throughput and 0.099 for first-token latency. Predicting equal
costs gave 0.224 and 0.326, respectively. Blind transfer selected the wider setting on
three additional holdout cases whose outputs failed the quality gate. Local remeasurement
remains part of the serving path.

### 2026-11-18: GGUF chat template rendering on the CPU

The HTTP chat path compiled its Jinja template on each request. Reusing a compiled template
for the same model keeps request messages, template variables and output separate. Seven
paired batches each rendered 24 distinct requests per arm, alternating which version went
first. Each request produced the same rendered string in both versions. Median warm time
per render fell from 11.400 to 0.024 ms for the Qwen3-32B template and from 1.568 to
0.010 ms for SmolLM2-135M on the Linux cloud worker. These times cover prompt rendering
alone. End-to-end throughput and time to first visible token are measured through the
HTTP servers in the [comparison workflow](COMPARE.md).

### 2026-11-18: Qwen3-0.6B on two EPYC vCPUs against upstream llama-server

On two pinned virtual cores of an AMD EPYC 9V74 cloud worker, a frozen checkout
(`1223ae5`) served Qwen3-0.6B Q4_K_M through the normal agent and a separate stock
llama-server built from upstream `4227c9b`. All six starts verified the same GGUF,
input token IDs and CPU placement. The timed workload combined short requests and
document summaries with an eight-token output limit and 512 context tokens. A
separate 18-question holdout at each load and repeat allowed up to 32 output tokens.
Three paired repetitions at each load alternated which server went first.

| Clients | IronMule output tok/s | llama-server output tok/s | IronMule p95 TTFT | llama-server p95 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.67 | 3.69 | 2,954 ms | 3,962 ms |
| 4 | 4.95 | 3.52 | 6,014 ms | 8,463 ms |

The table gives the median of the three per-round measurements. IronMule delivered
higher throughput in all six pairs. At one client, its p95 first-token time regressed
22.4% in the first round. Across the 108 scored answers IronMule scored 13 and stock
scored 10. At four clients in round two, IronMule missed one answer that stock got right.
Some reference rounds scored zero in an entire category. The comparison therefore
did not qualify. The hardware controller found no concurrent performance deficit
requiring its local admission experiment. The initial gate took 1,302 seconds in a
separate startup. Once cached, the complete six-server comparison took 297 seconds.
[Redacted measurement and per-question scores](../evidence/comparison/qwen3-06b-cpu-frozen-20261118.json)

### 2026-11-18: repeated GGUF verification during startup

The GGUF serving path obtains a content revision, binds quality results to the file and
binds batch-width results to the file. In one process it can reuse the SHA-256 of a
large regular file while its device, inode, size, modification and change times agree.
The initial scan checks the opened descriptor and the named file before accepting its
digest. Each new process performs its own verification.

Five alternating paired runs on the Linux cloud CPU read the 484,220,320-byte
Qwen3-0.6B Q4_K_M GGUF from a warm OS file cache. Three consecutive identity checks
took 1.044 s in the preceding implementation versus 0.360 s with one cold process-level
verification followed by two lookups. Every check returned the pinned model SHA-256.
This measurement covers file verification alone.

### 2026-11-18: Qwen3-0.6B on NVIDIA L40S against llama.cpp and vLLM

All three HTTP servers ran sequentially on the same 48 GB L40S with the same Q4_K_M
GGUF and offline Hugging Face tokenizer. The stock binary was built from llama.cpp
`4227c9b`. IronMule used upstream CUDA GGML kernels with its own agent and scheduler.
vLLM 0.11.0 used its normal GGUF serving configuration. All nine server starts verified
placement and the rendered token IDs. The 27 cells covered 1, 8 and 32 simultaneous
clients, three rotated repetitions and a 32-token output limit. Values are medians
of the three observations per engine and load.

| Clients | IronMule tok/s | llama.cpp tok/s | vLLM tok/s | p95 first visible token, IronMule / llama.cpp / vLLM |
| ---: | ---: | ---: | ---: | --- |
| 1 | 376.1 | 371.6 | 314.4 | 20.2 / 18.0 / 45.7 ms |
| 8 | 928.7 | 716.8 | 772.2 | 93.2 / 83.8 / 198.6 ms |
| 32 | 1,247.6 | 1,123.7 | 1,285.1 | 392.4 / 329.3 / 566.0 ms |

IronMule's throughput led at one and eight clients. vLLM led at 32. At 32 clients
the total holdout questions answered correctly across rounds were 7/54 (IronMule),
11/54 (llama.cpp) and 8/54 (vLLM). Some reference categories registered zero correct
answers. Both competitor verdicts failed the acceptance criteria. The controller
selected a local experiment targeting first-token latency at 32 clients, then
deferred it because it could not reproduce the serving worker's native configuration.
[Redacted per-cell measurements and verdicts](../evidence/comparison/qwen3-06b-l40s-20261118.json)

### 2026-11-18: Qwen3-32B on NVIDIA L40S, identical GGUF

The same physical L40S hosted each production HTTP server in turn. The models, token
IDs, 2,048-token context, generation cap, request order within a cell and device
placement were verified in each arm. Each concurrency was measured in three rotations.
The default vLLM 0.11.0 GGUF backend was used. Each server produced 162 graded answers
across the nine cells. Every answer was correct under this short-answer protocol.

| Clients | IronMule output tok/s | llama.cpp output tok/s | vLLM GGUF output tok/s | p95 first visible token, IronMule / llama.cpp / vLLM |
| ---: | ---: | ---: | ---: | --- |
| 1 | 38.11 | 28.07 | 15.28 | 206 / 227 / 1,872 ms |
| 8 | 112.32 | 67.96 | 25.30 | 1,016 / 1,107 / 8,168 ms |
| 32 | 174.46 | 130.42 | 27.22 | 3,957 / 4,527 / 32,720 ms |

Values are medians. In each of the nine individual cells against each reference,
IronMule exceeded the 3% throughput margin, preserved request rate and quality and
had no higher p95 time to first visible token. The controller found no verified
concurrent deficit at these loads. The result applies to the bounded Q4_K_M GGUF
protocol; AWQ serving uses a different artifact and kernel path.
[Full report and per-question scores](../evidence/comparison/qwen3-32b-l40s-20261118.json)

### 2026-11-18: Qwen3-0.6B on GeForce RTX 4090

IronMule, stock llama.cpp `4227c9b` and vLLM 0.11.0 each served the same Q4_K_M
file on one RTX 4090. Each server completed nine cells with verified device placement
and rendered input IDs. The load and quality protocol matched the L40S small-model
test. The table reports medians over the three repetitions.

| Clients | IronMule output tok/s | llama.cpp output tok/s | vLLM GGUF output tok/s | p95 first visible token, IronMule / llama.cpp / vLLM |
| ---: | ---: | ---: | ---: | --- |
| 1 | 392.13 | 378.08 | 295.56 | 24.6 / 22.1 / 58.3 ms |
| 8 | 907.21 | 660.76 | 733.68 | 109.4 / 86.6 / 220.4 ms |
| 32 | 1,176.00 | 853.03 | 1,188.59 | 425.0 / 333.6 / 646.8 ms |

IronMule answered 27/162 short questions correctly, stock llama.cpp 26/162, and
vLLM 25/162. Some answers previously correct on the reference were lost, and some
reference categories scored zero. The comparison missed the acceptance criteria
against both reference servers. [Server measurements](../evidence/comparison/qwen3-06b-rtx4090-20261118.json)

In a separate on-device replay, IronMule selected the first-token latency gap against
llama.cpp at 32 clients and tested a 33-token prefill budget. Four alternating
candidate/incumbent pairs produced identical completions, with 128/128 graded answers
correct on both sides of this scheduling holdout. The candidate improved p95 first-token
time by 11.7%, 8.7%, 15.7% and 2.5%. Its throughput gains were 6.7%, 6.8%, 9.5% and
0.7%. The last pair missed the required 3% gain. IronMule recorded the trial and
retained its previous setting. [Autonomous holdout](../evidence/comparison/qwen3-06b-rtx4090-holdout-20261118.json)

### 2026-11-18: Qwen3-32B GGUF and native AWQ on one L40S

This second test measured two independently deployable quantizations on one physical
48 GB GPU. IronMule used Qwen3-32B Q4_K_M GGUF. vLLM 0.11.0 used the official
Qwen3-32B-AWQ checkpoint and resolved AWQMarlin with float16. Both received identical
rendered token IDs and followed the same performance and short-answer protocol.
All nine verified pairs matched on output lengths and stop reasons. Each server solved
162/162 graded short answers.

| Clients | IronMule GGUF output tok/s | vLLM AWQ output tok/s | IronMule p95 first visible token | vLLM p95 first visible token |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 37.91 | 32.92 | 221 ms | 203 ms |
| 8 | 107.36 | 143.24 | 1,056 ms | 905 ms |
| 32 | 165.02 | 225.11 | 4,258 ms | 3,462 ms |

Values are medians of three alternating repetitions. IronMule led in tokens/s at a single
client. The native vLLM deployment led in tokens/s at 8 and 32 clients and had a lower
median p95 time to first visible token at all three loads. The ratios describe complete
deployment configurations, including differences in weight representation and kernels.
This report is external evaluation data. [Per-round measurements and paired comparisons](../evidence/comparison/qwen3-32b-l40s-native-awq-20261118.json)

The distribution of first-token latency also matters. At 32 clients the median request
received its first visible token in 0.95 s with IronMule and 2.33 s with vLLM AWQ.
IronMule's p95 was 4.26 s and vLLM's was 3.46 s. All 32 answers finished in 6.21 s
and 4.55 s respectively. Each figure is the median of three repetitions. IronMule's
first-token latency deficit appeared in the slow end of the distribution. Distinguishing
queuing, prompt processing and token generation requires timings inside the server.

### 2026-11-18: own-traffic admission learning on an AMD EPYC CPU

The production IronMule server at `268911b` served SmolLM2-135M Q4_K_M with
`--agent on` on two CPU cores. The isolated product Python had no vLLM installation.
Neither comparison server was available on its executable search path. Ten waves of
four real HTTP requests produced 40/40 responses and 320 output tokens. From five
recurring native workload observations the idle worker selected a prompt admission
budget of 84 tokens. The driving client supplied requests and watched the service.

Across four alternating, held-out pairs the candidate delivered 4.7%, 11.5%, 7.6%
and 12.8% higher throughput with lower p95 first-token times in each pair. Both
settings returned 16/16 correct copy answers. The third pair differed on its full
output tokens or stop reasons, so IronMule retained the incumbent and recorded the
attempt. The whole run took 61.7 seconds with the existing native qualification
cache. [Product provenance, request counts, incumbent and candidate](../evidence/self-admission/smollm2-135m-cpu-20261118.json)

### 2026-11-18: own-traffic admission learning on a GeForce RTX 4090

The unmodified production server at `268911b` served Qwen3-0.6B Q4_K_M on an
independently verified physical RTX 4090. The isolated interpreter and executable
search path excluded vLLM and llama-server. Ten bursts of four HTTP requests
produced 40 responses and 320 output tokens. The 64-sequence, 2,048-token context
profile recorded nine own workload observations. The worker deferred the experiment.
No candidate/incumbent pairs or new rules were recorded. This run captured GPU
placement and the decision. The exact reason was absent from this first export.
[RTX 4090, 2,048-token profile](../evidence/self-admission/qwen3-06b-rtx4090-deferred-20261118.json)

### 2026-11-18: measuring standalone GGUF prompt processing

IronMule can process a prompt in its own native call when it cannot share the current
batch. The previous step observer timed the combined batch and missed those separate
calls. This hid prompt-processing work from the controller's cost model. The observer
now records each prompt chunk when it finishes. It also classifies placeholder rows
between active GPU streams as decode work.

On a two-thread AMD EPYC CPU, SmolLM2-135M Q4_K_M served short, long and short prompt
waves at concurrency 8 and 32. The six waves totaled 120 requests. An independent hook
counted 114 native decode calls in each run. The old observer recorded 18. The fixed
observer recorded all 114, including the 96 standalone calls that processed 4,223
prompt tokens. All 360 generated tokens and all stop reasons matched between revisions.
The experiment verifies measurement coverage. Its timings apply to this CPU workload.
[Paired measurements](../evidence/runtime/smollm2-prefill-observer-cpu.json) · [Reproduction script](../lab/portable_gpu/prefill_observer_probe.py)

### 2026-11-18: Qwen3-32B prefill observations and controller decisions on an L40S

Both engine revisions ran Qwen3-32B Q4_K_M on the same 48 GB NVIDIA L40S with all
65 layers on CUDA. At 8 and 32 concurrent requests, an independent hook counted 114
native calls across six waves. The previous observer recorded 18. The current observer
recorded all 114, including 96 standalone calls that processed 4,223 prompt tokens.
All 120 requests produced identical output tokens and stop reasons. An additional
eight-request wave exercised gaps between GPU streams. Each of two affected calls
had seven decode rows, including three placeholders that the old observer counted as
prompt tokens. [GPU placement and call counts](../evidence/runtime/qwen3-32b-l40s-prefill-observer.json)

On the same GPU, four fresh instances of the production width controller served two pairs
of native replays in alternating revision order. Each instance learned from 48 warmup
requests, then served 96 held-out copy tasks in short and document bursts. Both revisions
used the same calibration of widths 8 and 32 measured on that GPU. The harness called
the controller between waves and included that decision time in the receiving wave.
Each wave allowed up to three output tokens. This measures a different workload from
the 32-output-token HTTP server benchmark above.

The additional observations increased the number of live samples per instance from
108 to 240. In the final decision of each replay the controller estimated width 32 could
deliver 3.17% and 4.18% higher throughput after the fix. The lower bounds of those
estimates were 2.22% and 2.91%. Both fell below its 3% switching threshold, and every
instance served at width 8 throughout. The previous revision delivered 33.760 output
tokens/s over the held-out waves and the current revision 33.746. Each revision solved
192/192 copy tasks, with identical output tokens and stops across the 12 paired waves.
The controller fitted its cost parameters in memory during each replay. The measured
controller probe is available at commit `2dce04c`.
[Controller decisions and paired timings](../evidence/runtime/qwen3-32b-prefill-controller-l40s.json) · [Reproduction script](../lab/portable_gpu/controller_pair.py)

### 2026-11-18: repeating the GPU tests with Qwen3-0.6B on an A40

An NVIDIA A40 ran Qwen3-0.6B Q4_K_M with all 29 layers on CUDA. The same six waves
again produced 114 native calls. The old observer saw 18 and the revised observer saw
114. All 360 output tokens and stop reasons matched across the revisions. Both calls
that bridged gaps between GPU streams counted all seven rows as decode work after the
fix. [A40 placement and native call counts](../evidence/runtime/qwen3-06b-a40-prefill-observer.json)

Each controller instance accumulated 240 live samples after the fix, up from 108. All
four instances retained width 8. Before and after the fix, each revision solved all
192 held-out copy tasks. Output tokens and stops matched in all 12 paired waves.
Pooled throughput decreased from 195.56 to 192.49 output tokens/s (1.57%) on this
workload. In the first paired repetition it went from 195.03 to 192.27 and in the
second from 196.10 to 192.71. For the document bursts the median per-wave p95
first-token time increased from 492.0 to 498.5 ms. These measurements cover two
repetitions of a short-output workload. The model and the GPU both differ from the
L40S experiment. [A40 controller decisions and paired timings](../evidence/runtime/qwen3-06b-prefill-controller-a40.json)

### 2026-11-18: GGUF admission with a long answer and a waiting burst

The HTTP scheduler used to start a new group after its 32nd lifetime member, even when
31 members had already finished and the native engine had free sequences. IronMule now
allows up to one additional group-width of waiting requests to run beside the last
member. Only requests already waiting when the group drops to its last member qualify.
The concurrent width stays at most 32. The group retains at most 32 requests' maximum
output-token allowance (262,144 tokens). The worker also hands an oversized initial
GGUF group to the native pending queue, which fills only available sequences.

On an AMD EPYC CPU, the public HTTP test started one long answer and sent 31 requests
with varying prompt lengths and output limits. After eight short answers finished it
sent another seven. Fresh workers used the same SmolLM2-135M Q4_K_M, eight sequences,
2,048-token context, two inference threads and checked standard GGUF settings. Each
cell delivered 39 requests and 161 output tokens. Other jobs shared the host quota.

| Fresh worker pair | Baseline / final tok/s | Baseline / final last-wave p95 first visible text | Last-wave requests finished before the long answer |
| --- | --- | --- | --- |
| Final then baseline | 44.89 / 46.51 | 1.905 / 1.716 s | 0 / 7 |
| Baseline then final | 39.26 / 50.34 | 2.430 / 1.465 s | 0 / 7 |

36 and 38 of the 39 response texts matched their paired baseline byte for byte.
The prompts exercise timing and carry no semantic score. A separate late-arrival test
waited for all 31 short HTTP responses before sending the seven. Both revisions put
the seven in a new group, as intended. The long answers ended at different token
counts in that pair. A worker-level test with 16 initial requests verified that all
eight waiting rows finished before the long row in both candidate runs, against zero
at baseline. All four worker runs generated 97 tokens over 57 steps. Their total
wall times varied in both directions, and 15 of 16 token streams matched.

An earlier unrestricted refill produced 53.76 and 60.10 tok/s against 72.42 and 73.74
at baseline in an equal-token test with different cache histories. Waiting for the
last member improved the early-burst trials, but doing so for newly arriving requests
still lost throughput in both late-arrival trials on fresh workers. These observations
led to the final backlog restriction. The measurements cover this CPU configuration.
[All trials, source identities and group counts](../evidence/runtime/gguf-tail-refill-cpu.json) · [HTTP workload](../lab/ag10/http_refill.py) · [Worker workload](../lab/ag10/worker_refill.py)

### Controller training, forecasts and a completed width trial

Two REINFORCE runs each used 32,000 simulated training episodes. Both policies were
evaluated on 384 held-out episodes. Increasing the scheduling head from 14 to 30
weights reduced the learned policy's simulated spike cost by 6.01% and increased its
mean time to first token from 8.32 to 11.15 seconds. The larger head remains experimental.
[Dispatch policy](../evidence/runtime/rl4-h9-holdout.json) · [Width policy](../evidence/runtime/rl6-h9-width-holdout.json)

On 576 held-out SmolLM2-135M CPU requests, conditioning the controller's dream on
previous workload shapes reduced mean absolute log2 prediction error from 1.009692
to 0.976098. In a follow-on live test the controller selected eight slots, served 64
requests and reverted to four after 94 native step observations. Including the cost
of switching, those two trial blocks ran at 32.21 output tokens/s. Fixed widths four
and eight served them at 93.11 and 68.05. All arms answered all 64 copy requests
correctly. On the subsequent 384 requests the controller retained width four. Each
arm answered 383 correctly.
[Workload, checkpoints and raw results](../training/README.md#serving-the-proposed-width-and-testing-its-rollback)

### 2026-11-18: held-out tool-result conversations beside document jobs

The paired Qwen3-32B deployment runner can now replay a small mixed trace after
its standard timings. The trace starts two document extractions, then interleaves
eight short lookups and four two-turn agent conversations when the first visible
document output arrives. A second turn follows its actual first response and
replays a canonical tool-call history with a user-role text envelope containing
the tool result. Each deployment uses the same inputs regardless of what its
first reply said. The harness grades the first command, the final returned
value and the completed two-turn workflow. It also checks every document's
ordered extraction and every short lookup.

On a local 135M SmolLM2 GGUF with two CPU threads, the functional probe verified
native prompt lengths and streamed all 18 responses in 7.25 s. All eight short
lookups began while a document was running. None of the 18 responses met the
strict output contract. One first turn produced the requested command followed
by extra prose, and its second turn repeated the command rather than the route.
The local run establishes client behavior and carries no GPU comparison.
[Synthetic responses, stops, class timings and prompt digest](../evidence/runtime/heldout-mixed-smollm2-cpu.json) · [Paired and local invocation](../lab/portable_gpu/README.md)

### Qwen3-32B HTTP latency under load on the L40S

One new round served the same tokenized performance and factual requests as the
earlier native vLLM AWQ deployment test. All verified IronMule cells used Q4_K_M
and 32 separate KV streams. The vLLM values below come from the corresponding
seed and round on an earlier L40S allocation. Each timed response generated 32
tokens. Each load level also passed all 18 factual checks.

| Concurrent clients | IronMule output tok/s | Earlier vLLM AWQ tok/s | IronMule / vLLM p95 first visible text (ms) | IronMule p95 queue / dispatch to first token (ms) |
|---:|---:|---:|---:|---:|
| 1 | 37.74 | 32.91 | 240 / 203 | 1 / 235 |
| 8 | 113.52 | 143.29 | 1,148 / 905 | 10 / 1,128 |
| 32 | 161.13 | 224.34 | 4,406 / 3,506 | 41 / 4,320 |

The queue and worker percentiles describe separate distributions. At 32 clients,
the worker's dispatch-to-first interval dominates the delay. The grouped-prompt
deployment selected a shared KV pool during startup and produced zero verified
grouped measurements. The shared-pool trial ended with a runtime error before
it completed prompt verification. Neither arm qualified for selection.
[Server timings and attempted configurations](../evidence/comparison/qwen3-32b-l40s-prefill-http.json) · [Verified IronMule cells](../evidence/runtime/qwen3-32b-l40s-prefill-selection.json) · [Archived matched requests and vLLM results](../evidence/comparison/qwen3-32b-l40s-native-awq-20261118.json)

### Qwen3-32B CUDA kernels on a second L40S

With all 65 layers on the GPU, three matched native-engine rounds at each width
compared the stock kernel, the selected KG4 tensor-core layout and a cuBLAS
path for large Q4_K/Q6_K prefill matrices. Each arm used the same pinned Q4_K_M
weights. CUDA graphs were disabled so per-wave counters could confirm dispatch.
The engine processed alternating short and document prompts with 32 output tokens.
The native timings exclude HTTP transport and production scheduling.

| Native path | 8 clients, median output tok/s | 32 clients, median output tok/s | Correct factual answers |
|---|---:|---:|---:|
| Stock | 94.04 | 168.28 | 18/18 |
| KG4 | 106.54 | 165.60 | 18/18 |
| Large-prompt cuBLAS | 76.80 | 121.46 | 18/18 |

KG4 increased throughput by 13.38% at eight clients and reduced it by 1.59% at 32
(medians of paired ratios). The predeclared rule for one global configuration
required a gain at one width and no loss at the other. Both candidates therefore
ended at the native screening stage. In a separate probe using the original CUDA
wheel, two stock runs agreed. Grouped processing changed one of eight short
continuations and three of the 32 mixed requests. All four cold 444-token prompts
matched in an isolated run and shared a single 1,776-row native call.
[Native timings, dispatch counts and quality results](../evidence/runtime/qwen3-32b-l40s-kernel-screen.json)

### Follow-up for the Qwen3-32B L40S kernel

The earlier native screen already enabled KG4 only for Q4_K matrix products with
six, seven or eight input rows. Its local probe measured 41.36, 41.73 and
42.16 ms for those steps, against 47.51, 55.56 and 56.41 ms on stock. A
32-request wave still made 384 KG4 submissions at each of those three row
counts. A typical eight-request wave made 384, 384 and 10,752, respectively.
These counts motivated a separate test of the request load surrounding each
matrix product. [Original native screen](../evidence/runtime/qwen3-32b-l40s-kernel-screen.json)

The experimental selector remeasures those three row counts on the target L40S.
It retains a row count only if its median step time improves by at least 3%.
It enables the resulting mask while the current group has six to eight active
and pending requests. If the group ever exceeds eight, stock kernels handle
its remaining requests through the last straggler. A new group can reactivate
the mask on the resident model. A change in the measured KV settings or
sequence capacity also returns dispatch to stock. Q6_K stays on stock.

The opt-in trial uses a pinned kernel and disabled CUDA graphs. Its native arms
bracket the candidate with stock runs on held-out eight- and 32-request waves.
The trial records native submissions, first-token latencies, answers and
teacher-forced loss at each enabled width. A native pass unlocks the same
bracketing comparison through the HTTP service, which checks the serving
worker's own layer placement. [Native study](../lab/kg4/scoped_native.py) · [HTTP study](../lab/kg4/scoped_http.py)

### 2026-11-19: Qwen3-32B on an A100 SXM4 80 GB against vLLM AWQ (PK1, PK2, W4R)

Machine: RunPod secure cloud, one NVIDIA A100-SXM4-80GB (driver 595.112, CUDA 12.8 toolkit), 16 vCPU.
Model: Qwen3-32B Q4_K_M GGUF (pinned, 19.76 GB) for IronMule; Qwen3-32B-AWQ (pinned) for vLLM 0.11.0, which
resolved `awq_marlin`, float16, chunked prefill 2048, CUDA graphs. Workload: `compare.performance_prompts`
(half 33-token, half 444-token prompts), 32 output tokens, greedy; 8 requests one after another at 1 client,
8 at once at 8 clients, 32 at once at 32 clients; two repeats (`lab/pk1/http_bench.py`).

vLLM AWQ over HTTP: 51.8 / 182.6 / 271.8 tok/s at 1 / 8 / 32 clients, p95 first token 208 / 873 / 3,045 ms,
18/18 facts at every width ([results](../lab/pk1/results/a100-vllm-awq.jsonl)).

Where IronMule lost (CUPTI kernel traces through `lab/pk1/profile_engine.py`; ncu counters are locked in the
container): stock llama.cpp `4227c9b` on this GPU prefills at 1,150-1,230 tok/s (MMQ) and decodes 23.5 / 76.6 /
61 ms per step at 1 / 8 / 32 sequences, about 0.3-0.8 TB/s of the 1.6 TB/s a plain read stream reaches here.

What was changed, all in the engine and chosen on the machine:
- PK1, packed prefill: every prompt that starts in a step goes into one native call in ascending sequence order,
  so llama.cpp splits it into few rectangular micro-batches; micro-batch up to 8192 rows (`IRONMULE_GGML_UBATCH`).
- PK2 (`native/ggml_patches/pk2`): prompt matmuls from a row threshold on through dequantization + cuBLAS with
  fp32 accumulation. The engine times MMQ against cuBLAS on a packed burst at start and keeps the faster threshold
  if the teacher-forced likelihood stays within 1.005: on this A100 2.05-2.26x a burst, likelihood ratio 1.0007.
  Prefill 2,618 tok/s at 2048-row micro-batches, 2,953 at 8192 (the per-micro-batch Q4_K dequantization fell from
  430 to 112 ms of 2.4 s); the cuBLAS GEMMs themselves run at 253 TFLOPS.
- W4R (`native/kernels/w4mma.cu`, `ironmule/w4_kernels.py`): IronMule's own decode kernels for Q4_K/Q6_K x 1-32
  tokens on tensor cores. The weights are repacked once into mma-fragment tile order (a warp reads 2 KB contiguous per
  super-block, lane pieces interleaved), the tensor cores multiply 1024 + q exactly and scale and minimum are applied
  per 32 weights in fp32. Every variant is checked against a float64 decode of the format (relative error 2e-4) and
  timed on the model's own matrices; the engine keeps the library per token count where its step is faster.
  The way there, measured on the 25600x5120 gate matrix (74 MB, 46 us at the read ceiling): GGUF row-major tiles
  through shared memory 73-102 us at one token; the same with x loads removed 52 us, which showed the activation loads
  (not the arithmetic, not the mma) were the cost; tile-order repacking with register prefetch 59 us; interleaved
  pieces and a per-warp cp.async ring 55.7 / 57 / 123 us at 1 / 8 / 32 tokens, against 62.6 / 70.5 / 139 us for the
  KG kernels and about 46 us for a perfect read.

Engine-level (`lab/pk1/engine_bench.py`, same prompts, no HTTP), tok/s at 1 / 8 / 32:

| Arm | 1 | 8 | 32 | decode ms/step at 8 / 32 |
| --- | ---: | ---: | ---: | ---: |
| default engine (no packing) | 35.0 | 90.1 | 111.0 | 34 / 63 |
| PK1 + PK2, micro-batch 8192 | 37.7 | 124.3 | 215.0 | 47 / 64 |
| + KG library | 55.6 | 143.4 | 229.1 | 32 / 57 |
| + W4R library | 52.6 | 146.7 | 229.3 | 31 / 56 |

The 1-client figures move with the prompts (n-gram speculation accepts more on some): 42-68 tok/s across repeats.
Read: prefill is now faster than vLLM's (2.7 s against about 3.0 s for the 32-request burst); decode steps are not
(31 against about 17 ms at 8 sequences, 56 against about 23 ms at 32). IronMule is behind vLLM AWQ by about 20 %
at 8 and 16 % at 32 clients and level at 1 client on this GPU. What follows: the decode step, where the remaining
cost is the kernels at 8-32 tokens (0.5-1.3 TB/s), small per-layer ops and launch gaps without CUDA graphs, and the
19 MB logits copy per 32-row step (2.3 ms).

Production server over HTTP, same prompts, seeds and client as the vLLM run (`ironmule serve --agent on` with the
tuned W4R library, PK1, PK2 auto, 8192-row micro-batches; the quality gate passed with the generated kernels at a
perplexity ratio of 0.994 against llama.cpp's default, interval 0.985-1.001; SEQ1 chose 64 sequences):

| Clients | IronMule tok/s (repeat 0 / 1) | vLLM AWQ tok/s | p95 first token IronMule / vLLM | facts |
| ---: | ---: | ---: | ---: | --- |
| 1 | 44.5 / 83.4 | 51.8 / 51.9 | 285 / 208 ms | 18/18 both |
| 8 | 140.2 / 144.5 | 183.1 / 182.1 | 972 / 870 ms | 18/18 both |
| 32 | 214.3 / 214.1 | 273.0 / 270.6 | 2,986 / 3,045 ms | 18/18 both |

IronMule is level or ahead at one client (speculation decides by prompt: -14 % and +61 %), 22 % behind at 8 and
21 % behind at 32 clients. The owner's minimum (vLLM's speed) is not reached at 8 and 32 clients on this GPU.
[IronMule results](../lab/pk1/results/a100-ironmule-w4-http.jsonl) · [W4R tuning record](../lab/pk1/results/a100-w4-tuning.json)
Pod time 4.5 h, 9.92 USD. Next: decode steps at 8-32 rows (kernels at 1.3 TB/s for 1-8 tokens and 0.5 TB/s at 32,
capture-safe W4R so CUDA graphs can stay on, an on-GPU argmax instead of the 19 MB logits copy).

### 2026-11-19, second A100 session: dequant-once W4R, interleaved Q6_K, AM1, graphs kept on

Same model, workload and harness; a second A100-SXM4-80GB (RunPod secure, US-KS-2). vLLM 0.11.0 AWQ on this pod:
52.0 / 183.9 / 273.7 tok/s at 1 / 8 / 32 clients (18/18 facts), within 0.7 % of the first pod
([results](../lab/pk1/results/a100-vllm-awq-pod2.jsonl)).

Changes since the first session:
- W4R variants that dequantize each weight once to fp16 (q x S - m) and reuse it for all 8-token tiles, instead of
  scaling per tile in fp32; at 32 tokens the 25600x5120 gate matrix went from 123 to 93.5 us.
- Q6_K tiles interleaved like Q4_K: the Q6_K down matrix 82 / 86 / 122 us at 1 / 8 / 32 tokens (was 93 / 94 / 165).
- W4R is capture-safe (inside a CUDA graph capture it serves only matrices it has already repacked), so CUDA graphs
  stay on for one-row steps.
- AM1 (`native/ggml_patches/am1`): greedy steps copy only each output row's argmax from the GPU, computed by
  `ggml_argmax` over the logits' memory, instead of 600 KB per row; checked against the host argmax at start.
- GR2 (`native/ggml_patches/gr2`, opt-in): CUDA graphs for multi-row decode steps. Same tokens, no measurable gain
  (38.6 against 38.8 ms per 32-row step); left off.
- A draft model (Qwen3-0.6B Q4_K_M, DM1 widened to 8 sequences) lost everywhere (104 against 157 tok/s at 8 clients)
  and stays off.

Engine-level, tok/s at 1 / 8 / 32 (two repeats), decode ms per step at 8 / 32:

| Arm | 1 | 8 | 32 | decode 8 / 32 |
| --- | ---: | ---: | ---: | ---: |
| first session's W4R | 42.4 / 62.8 | 144.6 / 148.8 | 230.1 / 228.5 | 31 / 56 |
| dequant-once + AM1 + graphs on | 48.6 / 79.5 | 155.9 / 161.7 | 258.2 / 258.0 | 26 / 40 |
| + interleaved Q6_K | 46.6 / 80.0 | 157.0 / 163.0 | 261.9 / 261.7 | 25.7 / 38.6 |

[Engine arms](../lab/pk1/results/a100-engine-arms-pod2.jsonl). Against vLLM on this pod: -10 % / +54 % at one client
(prompt-dependent speculation), -13 % at 8, -4 % at 32 clients.

Production server over HTTP on the second pod (`ironmule serve --agent on`, the pod's own tuned W4R table including
the Q6_K output matrix, PK1 with a 20 ms arrival-gathering window from idle, PK2 auto, AM1, CUDA graphs on; the
quality gate passed at a perplexity ratio of 0.994; SEQ1 chose 32 sequences), same client, prompts and seeds:

| Clients | IronMule tok/s (repeat 0 / 1) | vLLM AWQ tok/s, same pod | p95 first token IronMule / vLLM | facts |
| ---: | ---: | ---: | ---: | --- |
| 1 | 50.2 / 85.1 | 51.9 / 52.0 | 290 / 208 ms | 18/18 both |
| 8 | 147.8 / 150.7 | 184.2 / 183.7 | 964 / 863 ms | 18/18 both |
| 32 | 238.5 / 238.8 | 273.8 / 273.6 | 3,002 / 3,013 ms | 18/18 both |

Without the gathering window repeat 0 at 32 clients was 225.4 (the first HTTP arrivals were prefilled before the rest
of the burst had come in). Against vLLM AWQ: level-to-ahead at one client (-3 % / +64 % by prompt), -19 % at 8 and
-13 % at 32 clients. The server is 8-9 % behind the engine on the same pod (238 against 262 at 32 clients): the packed
prefill takes about 3.0 s through the service against 2.7 s in the engine; the cause is not yet located.
[HTTP results](../lab/pk1/results/a100-ironmule-http-pod2.jsonl) · [W4R tuning](../lab/pk1/results/a100-w4-tuning-pod2.json)
Next: the 0.3 s service-path prefill gap, then decode steps at 8 rows (25.7 ms against vLLM's about 17).

### 2026-11-19, third A100 session: server intervals, v9 kernels for 9-32 tokens, token writer thread
Machine: RunPod A100 SXM4 80 GB (secure cloud), same software stack and Qwen3-32B Q4_K_M GGUF as the second session;
compared against that session's vLLM 0.11.0 AWQ numbers (vLLM was not reinstalled on this pod).

- The server now records each request's queue, dispatch-to-first-token and first-to-end intervals
  (`lab/pk1/http_bench.py` reads them from `/v1/ironmule/live`). They locate the earlier "server prefill gap": all 32
  requests of a burst land in **one** packed prefill (first-token times within 10 ms of each other), and that prefill
  takes 2.8-3.0 s, about what vLLM needs for the same 7,632 prompt tokens (its last first token comes at 3.0 s).
  The remaining gap is the decode step: about 40 ms per 32-row step in the server against vLLM's 23-26 ms.
- Backend client (`ironmule_product/backend.py`): requests that join a running group are forwarded every 2 ms instead
  of on the 50 ms cancel tick while the worker is silent. The intervals showed no split burst on this workload, so
  this is a latency guard, not a measured gain.
- FP fused-activation variants (no prep kernel, 1-8 tokens) lost in tuning (e.g. 69 against 55 us at one token for
  the 5120x25600 matrix); the tuner never picks them.
- Kernel profile (CUPTI, engine, 16 decode steps): at 32 rows 25.2 of 34.3 ms kernel time per step are W4R matmuls,
  2.0 ms prep_x, 0.9 ms reduce_split, 2.6 ms attention; at 8 rows 15.7 ms matmul and 6 ms of small kernels; the GPU
  is idle 2.5-3 ms per step (graph build, readback, Python).
- W4R v9 (`w4d`, `w4d6` in `native/kernels/w4mma.cu`): v7's barrier-free register prefetch with dequantize-once
  arithmetic, one or two 16-row tiles per warp, x fragments through L1 one sub-block ahead, and the sub-block scales
  and minimums formed in half2 arithmetic (1024 + s trick, exact, one rounding as before) instead of int-to-float
  conversions. Tuned on the pod, us per call, old best -> v9:

| Matrix (K x N, Q4_K) | 16 tokens | 32 tokens |
| --- | ---: | ---: |
| 5120 x 25600 (gate, up) | 74.5 -> 59.0 | 94.8 -> 78.6 |
| 25600 x 5120 (down) | 79.7 -> 70.8 | 108.4 -> 99.7 |
| 5120 x 8192 (q) | 35.4 -> 32.8 | 46.7 -> 44.7 |
| 8192 x 5120 (o) | 32.2 -> 30.6 | 46.9 -> 40.9 |

  Q6_K v9 barely helps (output matrix at 32 tokens 565 -> 547 us; down 93 -> 92 at 16, none at 32).
- Token events of a grouped run are encoded and flushed by a writer thread in the worker (`_emit_soon`); every other
  event drains it first, so the protocol order is unchanged.

Production server over HTTP (`ironmule serve --agent on`, same settings as the second session; gate ratio 0.994;
startup width measurement 876 -> 929 tok/s at 32 sequences, 566 -> 611 at 16), three repeats:

| Clients | before (repeat 0 / 1) | v9 + writer (repeat 0 / 1 / 2) | vLLM AWQ (2nd pod) | facts |
| ---: | ---: | ---: | ---: | --- |
| 1 | 50.8 / 86.9 | 50.5 / 86.8 / 88.8 | 52.0 | 18/18 |
| 8 | 148.8 / 138.6 | 148.5 / 153.9 / 155.0 | 184 | 18/18 |
| 32 | 232.2 / 235.1 | 247.4 / 247.2 / 247.5 | 274 | 18/18, 18/18, 17/18 |

The 17/18 is a grader format miss, not a wrong fact: on that seed the model answers "33 * 6" with "200 - 2 = 198 ...
**198**" instead of the number alone, identically at 1, 8 and 32 clients (the one-token path is unchanged), so the
new kernels do not cause it. Against vLLM: -10 % at 32 clients (was -13 %), -16 % at 8, level-to-ahead at one client
by prompt. [before](../lab/pk1/results/a100-ironmule-http-pod3-before.jsonl) ·
[v9 without half2 scales](../lab/pk1/results/a100-ironmule-http-pod3-w4d.jsonl) ·
[v9](../lab/pk1/results/a100-ironmule-http-pod3-w4d-h2.jsonl). Pod time 1.18 h, 2.59 USD; total RunPod spend of this
work 14.74 of the 20 USD the owner allowed.
Next: decode steps. At 32 rows the matmuls alone (about 23 ms) are near vLLM's whole step; v9 reaches about 0.95 TB/s
at 32 tokens against 1.3 TB/s at one token. Then the per-step small kernels (prep_x deduplicated for q/k/v and
gate/up, which needs a per-graph epoch from llama.cpp to be safe; reduce_split folded into the kernel; rms_norm) and
the 2.5-3 ms the GPU waits per step.

### 2026-11-19, fourth A100 session: registers, SASS and constants of the W4R kernels
Question from the owner: were PTX, SASS and the driver optimised? Before this session the kernels used inline PTX
only for `mma.sync` and `cp.async`, built with plain `nvcc -O3 -arch=sm_80`; nobody had read their SASS or register
counts, and no driver or GPU setting was touched (clock, persistence and L2 policy were never tried).

- Registers without a GPU: NVRTC and `ptxas -v` from pip (`nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-nvcc-cu12`) on the
  device code. No kernel spills. The v9 variants the tuner picks at 32 tokens (two tiles per warp) use 236-245
  registers per thread, 8 warps per SM; one-token v7 uses 125.
- v10 (`w4e`): v9's arithmetic with a per-warp cp.async ring in shared memory, 71-106 registers (140 with two tiles).
  Twice the warps did not help: slower than v9 on most shapes (5120x25600 at 32 tokens 99.4 against 78.6 us), faster
  only on 5120x8192 at 16 and 32 tokens (39.5 / 48.8 against 40.3 / 54.4 us, same run). Occupancy is not the limit.
- SASS (`cuobjdump -sass` on the pod): about 300 of v9's 2,400 instructions per two-tile super-block were
  `IMAD.U32 R, RZ, RZ, UR4`: a PRMT takes one immediate, ptxas made the data constant 0x64 the immediate and rebuilt
  the selector 0x4140 / 0x4342 in a register before every PRMT (an `asm mov` of 0x64 was folded by ptxas). A 0x64 that
  ptxas cannot prove (`0x63 + blockDim.y`) makes the selector the immediate: v9 2,368 -> 2,072 instructions, IMAD
  438 -> 144. v7 converted every 6-bit scale with a quarter-rate I2F (64 per super-block); `byte_f` puts the byte into
  the mantissa of 2^23 and subtracts 2^23 (exact, PRMT + FADD at full rate).
- Precomputing 64-bit x row pointers made the SASS longer (more LEA/IADD3.X); reverted.
- A/B on the same A100, two interleaved rounds, best us per call, before -> after the constant and I2F changes:
  1 token -1 to -2 %, 8 tokens -1.4 to -2.6 %, 16 tokens -0.4 to -3.6 %, 32 tokens -1.6 to -3.8 % on all Q4_K shapes
  except 8192x5120 at 32 tokens (43.7 -> 49.6 us, +13 %, not explained). 12 % fewer instructions bought 1-4 %, so
  instruction issue is not the main limit either; the remaining suspect is memory latency with one super-block of
  prefetch per warp. No HTTP run: the expected 1-2 % is inside its run-to-run spread.
Pod time 0.41 h, 0.89 USD; total 15.63 of 20 USD.
Next for the kernels: deeper prefetch at 32 tokens (two super-blocks ahead within v9's registers, or a hybrid of v9
registers and a short cp.async ring), measured per shape on the A100; driver-side L2 persistence for activations.

### 2026-11-19, fifth A100 session: the 5120-row matrices at 32 tokens (o and down)
Correction to the fourth session's note: the +13 % outlier was the o matrix (K 8192 -> N 5120), not down; in this
session the same old variant measured 48.9 and 40.5 us in two rounds, so that outlier is run-to-run noise.
The matrices with 5120 output rows are the least efficient at 32 tokens (o about 40-48 us against about 16 us at
1.5 TB/s, down about 99 us against 49 us). Tried, A100 SXM4, best us per call, two rounds each:
- In-block K-split variants of v9 (one tile per 4-16 warps, shared-memory reduction, no reduce kernel): slightly better
  at 16 tokens (down Q4 68.5 -> 66.2, down Q6 91.3 -> 86.3), worse at 32 (down Q4 104 against 99). The GPU's
  fill (80 blocks for 108 SMs in the chosen variant) is not the limit.
- v10 with three or four tiles per warp (more x reuse per load): slower (down Q4 at 32 tokens 121 against 99).
- Diagnostic: the same kernels with every x read folded onto two super-blocks (L1-resident, wrong values) took down
  Q4 at 32 tokens from 99.3 to 86.5 us: re-reading the 1.6 MB of x from L2 for every tile costs about 13 %.
- v11 (`w4f`, `w4f6`): v9's register-prefetched weights with each super-block's x staged once per block in shared
  memory (cp.async, double buffered) and read by all the block's tiles. Down Q4 at 32 tokens 99.1 -> 92.0 us
  (variant VF(4,4,1), grid K-split 4, both rounds); Q6 down 121.8 -> 111.6-117.1. Worse on o, gate and at 16 tokens,
  where the tuner keeps the older variants.
- A grid K-split of 8 helped the older variants as well (down Q4 91.8, Q6 118.0 at 32 tokens); the tuner now tries
  1, 2, 4 and 8 (`ironmule/w4_kernels.py`).
Expected effect per 32-row decode step: about 0.4 ms of about 38 ms (1 %), inside the HTTP run-to-run spread, so no
HTTP run. Pod time 0.54 h, 1.18 USD; total 16.82 of 20 USD.

### 2026-11-19, sixth A100 session: every winner combined, with and without GR2
Arm "all": PK1 packing with the gathering window, PK2 auto, AM1, CUDA graphs for one-row steps, the token writer
thread, the 2 ms join forwarding, and a full W4R tuning on the pod with every variant so far (v7-v11, the SASS constant
and byte_f changes, in-block K-splits, grid K-split up to 8). Arm "all_gr2": the same plus CUDA graphs for decode
steps up to 32 rows (`IRONMULE_CUDA_GRAPHS_ROWS=32`). Left out on evidence: the draft model, FP prep variants.
Same pod, same client, prompts and seeds, three repeats, `ironmule serve --agent on`:

| Clients | all (r0 / r1 / r2) | all_gr2 (r0 / r1 / r2) | third session's best | vLLM AWQ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50.8 / 86.2 / 88.4 | 50.9 / 86.2 / 88.9 | 50.5 / 86.8 / 88.8 | 52.0 |
| 8 | 148.5 / 151.8 / 151.5 | 148.8 / 152.7 / 153.0 | 148.5 / 153.9 / 155.0 | 184 |
| 32 | 246.9 / 249.0 / 247.1 | 249.0 / 250.5 / 245.2 | 247.4 / 247.2 / 247.5 | 274 |

Facts 18/18 everywhere except repeat 2 at 32 clients (17/18, the known "33 * 6" format answer).
The startup width measurement rose to 996 tok/s at 32 sequences (929 in the third session, 876 at its start), and a
32-row decode step through the server fell to about 35.4 ms (first-to-end 1,097 ms over 31 steps; 38.3 ms in the third
session). The HTTP throughput did not move: with 32-token answers about 73 % of the 32-client wall time is the packed
prefill (first tokens at 2.96-3.06 s here against 2.90-2.93 s in the third session, a different pod), so the decode
gain (about 90 ms per wave) is the size of the prefill's pod-to-pod spread. GR2 again made no measurable difference.
Pod time 0.52 h, 1.14 USD; total 17.96 of 20 USD.

### 2026-11-20, seventh A100 session: the prefill, a split burst (PK1-B) and shared fp16 activations (PK3)
Owner raised the RunPod budget by 5 USD (25 in total). With 32-token answers the 32-client wave is about 73 % prefill,
so the prefill came first.
- Engine prefill profile (32 prompts x 240 tokens = 7,680 rows, one packed call, PK2 cuBLAS): 2.56 s of kernels;
  GEMMs 1,922 ms (about 249 TFLOPS, 80 % of the A100's fp16 peak), fp32 -> fp16 activation conversion 194 ms,
  Q4_K/Q6_K dequantisation 113 ms, SwiGLU, norms, add, RoPE, attention and KV writes about 330 ms, GPU idle 5 ms.
  Without PK2 (llama.cpp's MMQ kernels) the same prefill takes 6.0 s.
- A one-line timing log per packed prefill call inside the running server (pod-only, not committed) showed why the
  server's first tokens came at about 3.0-3.1 s: a 32-request burst was split, `rows=474 prompts=2 s=0.358` followed by
  `rows=7155 prompts=30 s=2.565 since_run=0.463`. The HTTP burst reached the engine over about 50 ms and the 4 ms quiet
  gap of the gathering window closed after the first two prompts.
- PK1-B (`ironmule/ggml_engine.py`): once the window holds two requests it waits for a 25 ms gap, up to 150 ms
  (`IRONMULE_GGML_GATHER_BURST_*`); a single request still starts after 4 ms. Afterwards every 32-request wave was one
  call (`rows=7608 prompts=32`, started 28 ms into the run); one-client first tokens unchanged (290 ms).
- PK3 (`native/ggml_patches/pk3`, applied after gr2): a graph epoch; within one graph the cuBLAS matmuls that read the
  same src1 node share one fp16 copy, and the W4R library learns each src1 node (`ironmule_kg_src_node`) and skips
  prep_x for q/k/v and gate/up. Keys include the node, so another tensor at a reused address never matches (a pure
  address key would be unsafe for models whose o projection has the hidden size as input). Conversions in the
  7,680-row prefill 445 -> 254 calls, 194 -> 145 ms; the whole prefill 2.595 -> 2.576 s.
Same pod, same client, prompts and seeds; before = the code of the sixth session:

| Clients | before (r0 / r1) | PK1-B + PK3 (r0 / r1 / r2) | vLLM AWQ |
| ---: | ---: | ---: | ---: |
| 1 | - | 51.3 / 86.9 / 91.1 | 52.0 |
| 8 | - | 147.2 / 150.3 / 150.0 | 184 |
| 32 | 237.3 / 235.0 | 245.7 / 246.4 / 248.4 | 274 |

Facts 18/18 in all nine runs. At 32 clients: first token 3.10-3.19 -> 3.00-3.03 s, first-to-end 1,170 -> 1,090 ms,
+4.6 % throughput on this pod. Across pods the 32-client numbers stay at 245-250 tok/s: this pod's baseline was lower.
Open: the server's packed prefill takes 2.74 s against 2.58 s for the same rows in the engine profile (both with PK3);
the cause is not located. Pod time 1.04 h, 2.27 USD; total 20.23 of 25 USD.

### 2026-11-20, maintainer study: where vLLM's 32-client advantage comes from
Owner's note: the maintainers may study vLLM to find IronMule's bottlenecks; the agent (AG1 controller, Pilot) does not
see any of this and decides only from its own observations. vLLM 0.11.0, the same Qwen3-32B AWQ, A100 SXM4 80 GB,
same launch as before (`--max-model-len 2048 --max-num-seqs 32`), recorded with vLLM's torch profiler over one 32-client
wave of the same HTTP client. vLLM on this pod: 51.7 / 181.6 / 268.6 tok/s at 1 / 8 / 32 clients.
Its log: awq_marlin kernels, chunked prefill with max_num_batched_tokens=2048, FlashAttention backend, torch.compile
(Triton fusions), CUDA graphs PIECEWISE for mixed prefill-decode steps and FULL for decode steps. Prefix caching is on
by default: a wave repeating an earlier seed was mostly served from cache (1.56 s of kernels), so the analysis below uses a
fresh seed; the benchmark's prompts share only the chat template and a few words within a wave.
- Whole fresh wave: about 3.67 s of kernels for 7,632 prompt tokens plus 32 x 32 answers, about what IronMule's GPU
  does for the same wave (2.58 s prefill + 31 x 35 ms decode). The prefill GEMMs alone are slower in vLLM (Marlin at
  large batch, 2.63 s) than IronMule's PK2 cuBLAS path (1.92 s GEMM + 0.26 s conversion and dequantisation).
- The difference is the decode step at 32 rows (median of 34 steps; IronMule from its own profiles):

| per decode step | vLLM, 1 row | vLLM, 32 rows | IronMule, 32 rows |
| --- | ---: | ---: | ---: |
| weight GEMMs (4-bit, + fp16 output matrix) | 11.6 + 0.9 ms | 12.0 + 0.9 ms | about 19-21 ms (W4R) |
| attention (+ KV write) | 1.2 ms | 1.5 ms | about 2.6 ms |
| norms, activation, residual, RoPE | about 1.1 ms (fused Triton) | about 1.3 ms | about 6-7 ms incl. prep_x, reduce_split |
| GPU idle inside the step | about 0 (full CUDA graph) | about 0 | about 3 ms |
| total | 15.2 ms | 16.3 ms | about 35 ms |

  Marlin reads AWQ's 4.25-bit weights at about 1.35-1.4 TB/s and costs the same at 32 rows as at one; W4R reads Q4_K_M's
  4.85 bits at about 1.3 TB/s at one row but slows to about 0.95 TB/s at 32 (one-row IronMule decode is faster than
  vLLM's: about 11.6 against 15.2 ms).
Consequences for the work (maintainer side): (1) a 32-row matmul that keeps one-row speed, either a Marlin-style Q4_K
kernel (weights shuffled for lop3 dequantisation, activations shared through shared memory per block, a multi-stage
cp.async pipeline, in-kernel reduction) or Q4_K converted once into Marlin's layout with float zero points behind the
quality gate; (2) fused norm, residual, activation and RoPE kernels and a full CUDA graph for multi-row decode steps
(about 8-9 ms of IronMule's step against about 2.8 ms); (3) decode attention over one KV stream per sequence (2.6 ms
against 1.5 ms). IronMule's prefill path stays. Pod time 0.23 h, 0.50 USD; total 20.73 of 25 USD.

### 2026-11-20: correction to the vLLM study, and MQ1 (Marlin on Q4_K) measured
Correction: the vLLM steps read above as "32 rows" (1,160 kernels, 12.0 ms of Marlin) ran Marlin's eight-row variant
(`m_block_size_8`), so they were steps of at most eight rows. In the fresh-seed trace the 32-row decode steps are the
7,680 calls of `Marlin<256, 2, 16, 4>` (30 steps x 64 layers x 4 fused matmuls): 491 ms, **16.4 ms of 4-bit GEMM per
32-row step**, plus about 0.8 ms for the fp16 output matrix and about 2.5-3 ms for everything else. vLLM's 32-row step is
thus about 20 ms against IronMule's about 35 ms: the GEMMs differ by about 3-4 ms (AWQ carries about 12 % fewer bytes
than Q4_K_M), everything else by about 12 ms (fused norm, residual and activation kernels, a full CUDA graph, attention).
The fusion and graph work (point 2) is the larger lever, not the matmul.

MQ1 (`native/kernels/mq1`, commit 4a10665): vLLM's Marlin kernel, vendored unchanged apart from its torch includes,
as one more W4R tuner variant. Q4_K is converted once on the GPU into u4 values (exact), fp16 scales and fp16 float
zero points per 32-weight group (Marlin's HQQ path, group 32 instantiated by IronMule), then Marlin's repack. Correct
everywhere (relative error 8.3e-4 to 9.3e-4 against float64, bound 4e-3), but slower than W4R on every Q4_K shape of
Qwen3-32B on the A100 except one tie, best microseconds per call:

| shape (K x N) | 1 token W4R / MQ1 | 8 tokens | 16 tokens | 32 tokens |
| --- | ---: | ---: | ---: | ---: |
| 5120 x 25600 (gate, up) | 55.1 / 63.4 | 64.4 / 75.2 | 77.1 / 92.2 | 90.3 / 95.7 |
| 25600 x 5120 (down) | 59.0 / 64.1 | 70.7 / 72.3 | 79.3 / 82.1 | 105.0 / 104.0 |
| 8192 x 5120 (o) | 25.2 / 29.2 | 30.9 / 35.7 | 34.3 / 39.3 | 45.8 / 53.9 |
| 5120 x 8192 (q) | 26.9 / 28.9 | 31.8 / 35.7 | 38.7 / 39.4 | 48.1 / 48.4 |
| 5120 x 1024 (k, v) | 10.8 / 27.3 | 11.8 / 27.5 | 15.5 / 37.6 | 20.9 / 88.2 |

Q4_K's per-32 scale and minimum add about 24 % of metadata to Marlin's weight stream (AWQ: one scale and zero per 128),
and MQ1 adds an fp32 -> fp16 and an fp16 -> fp32 conversion kernel per call. MQ1 stays as a tuner variant (it is only
chosen where measured faster and within the bound); it is not a step towards vLLM's decode on Q4_K_M.
[MQ1 tuning](../lab/pk1/results/a100-mq1-tuning.json). Pod time 0.17 h, 0.36 USD; total 21.10 of 25 USD.

### 2026-11-22, A100: why the GPU idled 3.8 ms per decode step, and GR3
Host-side split of a 32-row decode step (engine, A100, Qwen3-32B, W4R table of the pod): Python around the call
0.1 ms; `llama_decode` 33.6 ms and synchronous; GPU kernel time 30.0 ms. llama.cpp's graph reuse is active (turning it
off costs 0.9 ms). Per step 1,895 `cudaLaunchKernel` calls (12.3 ms of host time under the profiler) and no CUDA graph,
even with GR2 at 32 rows: a diagnostic build showed that llama.cpp switched CUDA graphs off for the context's whole life
during the engine's warm-up (a 1-node and a 2,246-node graph alternated, four consecutive "updates"), before the first
served step, so one-row steps had no graphs either. Once graphs were kept, the first capture of a new graph recorded
llama.cpp's MMQ kernels (55.8 ms per 32-row step), because the W4R library may not allocate inside a capture.
GR3 (`native/ggml_patches/gr3`, after pk3; `IRONMULE_GR3=0` restores llama.cpp): graphs of fewer than 8 nodes stay out
of CUDA graphs, too many consecutive updates pause graphs for 32 calls instead of disabling them, and a changed graph
runs once uncaptured before it is captured (then with the W4R kernels). Decode step, wall time with the profiler on:

| rows | GR3 off | GR3 on (graphs up to 32 rows) |
| ---: | ---: | ---: |
| 32 | 33.9 ms | 33.0 ms (1 graph launch instead of 1,895 kernel launches) |
| 8 | 23.3 ms | 23.1 ms |
| 1 | 22.5 ms | 21.8 ms |

Without the profiler the engine ran 785 32-row steps in 25 s (31.8 ms per step). About 3.5 ms per 32-row step remain
between wall time and GPU time with graphs on; that host time is not located (py-spy cannot attach in the container).
Quality gate: the gate's tuning child (`autotune`, several contexts in one process) ran out of the A100's 80 GB with
MQ1's extra weight copies (the pod's table had chosen MQ1 for three shapes on noise-level ties); without them it passed
with GR3 on and one-row graphs. MQ1 is now opt-in (`IRONMULE_MQ1=1`). Graphs up to 32 rows (`IRONMULE_CUDA_GRAPHS_ROWS=32`)
were not validated through the gate: the one served run with them failed the gate's tuning with out_of_memory, but that
server had re-tuned its own table with the pod's older library that still offered MQ1, so the cause is not separated;
they stay opt-in and no HTTP numbers exist for GR3. Pod time 1.27 h, 2.78 USD; total 23.87 of 25 USD.
Next: the server A/B for GR3 (one-row graphs by default, 32-row graphs opt-in) on a fresh pod with the opt-in MQ1;
then point 2's fusions (add + RMS norm + mul, RoPE + KV write, prep_x into the norm) and the remaining host time.

### 2026-11-22, A100: GR3 served, FU1, 32-row CUDA graphs through the gate -- level with vLLM at 32 clients
One pod (RunPod A100 SXM4 80 GB), the product path end to end (`ironmule serve --agent on`, its own W4R tuning and
quality gate in a fresh state), same HTTP client, prompts and seeds as all earlier runs, two repeats, facts graded:

| Clients | GR3 on, one-row graphs | GR3 off | FU1 + GR3 + graphs up to 32 rows | vLLM AWQ (2nd pod) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50.0 / 87.4 | 51.6 / 88.3 | 53.1 / 91.2 | 52.0 |
| 8 | 151.9 / 157.3 | 152.3 / 146.8 | 158.8 / 160.9 | 184 |
| 32 | 269.1 / 264.7 | 254.6 / 275.3 | **268.1 / 280.6** | 274 |

Facts 18/18 in every run. The quality gate passed with FU1 and multi-row graphs (perplexity ratio 0.9935, interval
0.9858-1.0001). Startup width measurement at 32 sequences: 1,015 tok/s (GR3 on) and 1,024 tok/s (FU1 + 32-row graphs).
GR3 on against off is inside the run-to-run spread on HTTP (32 clients 266.9 against 264.9 mean). At 32 clients the
last arm averages 274.3 tok/s, level with vLLM's 274 on the second pod (its second repeat +2.4 %); at 8 clients
IronMule stays about 13 % behind, at one client level to well ahead depending on the prompt.
- 32-row CUDA graphs through the gate: a fresh-state server with `IRONMULE_CUDA_GRAPHS_ROWS=32` again failed the gate's
  tuning with out_of_memory although MQ1 was already opt-in: multi-row graphs in a process that opens several contexts
  hold too much memory. The gate's children and SEQ1's measuring children now run with one-row graphs (they do not
  change tokens); the serving worker uses 32-row graphs, set by `serve` on Ampere+.
- FU1 (`native/ggml_patches/fu1`, after gr3): a residual ADD, its RMS_NORM and the weight MUL in one kernel, in
  llama.cpp's arithmetic order. Engine test with 32 real prompts and 48 greedy steps: the same tokens (hash
  d008af8928708bd7) with FU1 off/on and with one-row/32-row graphs; per 32-row step 32.96 -> 32.50 ms (FU1),
  32.96 -> 30.89 ms (32-row graphs), 30.74 ms with both.
Pod time 1.03 h by the session's ledger (RunPod's billing for the day was not complete); total 26.14 of 35 USD.

### 2026-11-22, A100: PK4 served, and vLLM on the same pod
Another RunPod A100 SXM4 80 GB pod. PK4 (`native/ggml_patches/pk4`, after fu1): the fused norm and a new SWIGLU kernel
write the fp16 copy that the next cuBLAS prompt matmul would convert, into PK3's slot under PK3's key. Engine, 7,680-row
packed prefill: conversions 254 -> 65 calls (146 -> 28 ms), prefill 2.573 -> 2.485 s; greedy tokens of 32 real prompts
over 48 steps unchanged. Also on this pod: v9 variants with a minimum of blocks per SM (fewer registers, spills) lost at
32 tokens and won the 5120 x 25600 matrices at 16 tokens (61.2 -> 57.1 us).
Served, `ironmule serve --agent on` with FU1 and 32-row graphs, fresh state (gate passed, ratio 0.9940), three repeats,
then vLLM 0.11.0 (Qwen3-32B-AWQ, AWQ-Marlin, same launch as before) on the same pod, same client, prompts and seeds:

| Clients | IronMule, PK4 on | IronMule, PK4 off | vLLM AWQ, same pod |
| ---: | ---: | ---: | ---: |
| 1 | 52.8 / 88.0 / 91.5 | 52.5 / 88.5 / 89.8 | 51.6 / 51.8 / 51.8 |
| 8 | 153.2 / 153.9 / 156.1 | 151.7 / 154.5 / 155.3 | 180.8 / 180.8 / 180.1 |
| 32 | 255.2 / 252.3 / 250.0 | 250.0 / 248.6 / 245.4 | 266.9 / 264.4 / 266.3 |

Facts 18/18 in every run of both servers. PK4 adds about 1.8 % at 32 clients (first tokens about 80 ms earlier).
Against vLLM on the same machine: -5.0 % at 32 clients, -14.5 % at 8, +2 % to +77 % at one client. This corrects the
"level at 32 clients" of the previous entry, which compared IronMule on one pod with vLLM measured on another: this
pod ran IronMule about 7 % and vLLM about 3 % slower than the earlier ones, so cross-pod comparisons carry several
percent of machine difference. Comparisons against vLLM are to be made on the same pod from now on.
[IronMule PK4 on](../lab/pk1/results/a100-ironmule-http-pod12-pk4on.jsonl) ·
[off](../lab/pk1/results/a100-ironmule-http-pod12-pk4off.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod12.jsonl).
Pod time 0.92 h, 2.01 USD; total 28.15 of 35 USD.

### 2026-11-22, A100: where a decode step's time goes, by width
Another RunPod A100 SXM4 80 GB pod, Qwen3-32B Q4_K_M, engine with the served settings (PK4, FU1, graphs up to 32
rows), the pod's own W4R tuning. Decode step wall time without a profiler (median of 40 steps, context 240):
18.8 / 20.8 / 23.6 / 30.8 ms at 1 / 8 / 16 / 32 rows. Kernel span of a step (CUPTI): 18.8 / 20.6 / 21.8 / 28.4 ms
with 0.07-0.13 ms of gaps inside, so with GR3 the GPU idles only before the graph launch (input setup, about 0.3 ms at
one row and 0.8 ms at 32) and after the last kernel (readback and Python, 0.1-0.7 ms): about 2.3 ms per 32-row step,
almost nothing at one row. Reading all argmax rows of a step in one call instead of one per row changed the step by
at most 0.2 ms and was not kept. Kernels per step:

| per step | 1 row | 32 rows |
| --- | ---: | ---: |
| W4R matmuls (449 calls) | 14.45 ms | 21.08 ms |
| flash attention + fixup (+ KV max) | 1.33 ms | 2.50 ms |
| prep_x and reduce_split (321 / 481 calls) | 0.76 ms | 1.95 ms |
| FU1 add + norm + mul (128 calls) | 0.92 ms | 0.96 ms |
| q/k norm, RoPE, KV write, SwiGLU | 1.18 ms | 1.78 ms |
| total | 18.7 ms | 28.4 ms |

The matmuls read the weights at about 1.37 TB/s at one row (vLLM's Marlin on AWQ: about 1.35-1.4) but at about 0.94
TB/s at 32 rows (Marlin about 1.2). The Q6_K matrices of Q4_K_M (half the down and v projections, the output matrix)
cost about 0.7 ms of a 32-row step against Q4_K, so requantising them would not close the gap. What follows: the
32-row matmul is the lever for +10 % over vLLM (about 6 ms per step at Marlin's 32-row efficiency); then the small
kernels (about 7 ms against vLLM's 2.5-3) and the 2.3 ms of host time around the graph.
[W4R tuning of this pod](../lab/pk1/results/a100-w4-tuning-pod13.log). Pod time 0.4 h, 0.88 USD; total 29.03 of 35 USD.

### 2026-11-23, A100: where the 32-client wave's time goes, and W8A8 prompt matmuls (I81, rejected)
Owner: the whole remaining RunPod balance for the +10 % goal. Two RunPod A100 SXM4 80 GB pods, Qwen3-32B Q4_K_M.
- Server trace (`IRONMULE_GGML_TRACE`, one line per native call; HTTP 8 and 32 clients, two repeats): 262.5 / 263.9
  tok/s at 32 clients and 155.9 / 162.6 at 8 on this pod. A 32-client wave is one packed prefill call (7,600 rows,
  2.64-2.68 s) and 31 decode calls of about 27-32 ms; between calls the server loses 0.83 ms (median). At 8 clients
  the prefill call takes 0.83-0.90 s for about 1,900 rows.
- The benchmark's prompts are 16 short (21 tokens) and 16 long (432 tokens). With one KV stream per sequence llama.cpp
  splits a call into rectangular micro-batches of consecutive sequence ids at equal token counts: two passes when the
  short prompts hold the lowest ids (the server sorts them so), one pass per long prompt when short and long
  alternate (a lab script did that: 6.4 s instead of 2.46 s). Engine prefill of the real mix, sorted: 2.46 s (32) and
  0.81 s (8; 0.69 s for the same rows at one equal length). Kernels of the 32-prompt prefill: fp16 GEMMs 1.71 s of
  2.43 s, Q4_K / Q6_K dequantisation 0.22 s (once per pass), SwiGLU 0.10 s, attention 0.08 s.
- int8 against fp16 tensor-core GEMM on the prefill shapes (torch `_int_mm` against fp16 matmul, 7,680 rows):
  1.47-1.59x, 315-386 TOPS against 214-249 TF.
- I81 (`lab/i81`, a patch after pk4, not in the build): `IRONMULE_I8=2` quantised PK2's dequantised weights per
  output row and the activations per token to int8 and multiplied on the int8 tensor cores; `=1` only rounded both
  operands. Perplexity on the gate's 16 chunks against llama.cpp's MMQ path, ratio and 95 % interval (bound: upper
  end below 1.005), and engine prefill of the benchmark mix:

| path | perplexity ratio | prefill 32 / 8 clients |
| --- | --- | ---: |
| llama.cpp MMQ (reference) | 1.000 | 5.76 / 1.50 s |
| PK2 fp16 cuBLAS (served) | 1.004 [0.998, 1.011] | 2.44 / 0.81 s |
| int8 rounding, fp16 multiply | 1.133 [1.047, 1.268] | 2.75 / 0.94 s |
| int8 everywhere | 1.090 [1.041, 1.137] | 2.13 / 0.78 s |
| int8, down projection on fp16 | 1.022 [1.014, 1.030] | 2.21 / 0.78 s |

  Per-token int8 activations cost 2-9 % perplexity: rejected (the 0.23-0.31 s it saves per 32-client wave would have
  been about +6-8 %). Activation-side fixes (per-channel smoothing, outlier channels on fp16) were not tried.
What follows: the prefill GEMMs already run at about 80 % of the fp16 peak and the decode step at the bandwidth of
its kernels, so the remaining levers are small: dequantisation once per pass (0.22 s), the server's prefill and decode
host time, and the gather window. Pod time about 0.75 h; balance 13.56 USD after these two pods.
[I81 measurements](../lab/i81/a100-i8-ppl.log)

### 2026-11-23, A100: I82 (smoothed int8 prompt matmuls, rejected) and PK5 (kept)
Owner: keep going until +10 % over vLLM. One more RunPod A100 SXM4 80 GB pod (about 1.4 USD), Qwen3-32B Q4_K_M,
engine-level, perplexity on the gate's 16 chunks against llama.cpp's MMQ path (bound: upper 95 % end below 1.005),
prefill of the 32-client benchmark mix. `lab/i82` holds the patch, scripts and logs.
- Locating the int8 error (operands rounded to int8 levels, fp16 multiply): weights per row alone 1.052
  [0.997, 1.165], activations per token alone 1.079 [1.030, 1.131]; both add error.
- Per-channel smoothing (s = sqrt(max|x|) / sqrt(max|w|), activation maxima recorded on README + RUNTIME text): a first
  run gave 0.939 [0.893, 0.989], better than the reference, which did not reproduce: with the calibration inside the
  engine, as a server would run it, smoothed int8 gave 1.086 [1.037, 1.131] (unsmoothed 1.090). Rejected; the 0.939
  is recorded as not reproduced and was not acted on.
- int8 straight from the Q4_K blocks (no fp16 copy): prefill 1.97 s against 2.33 s, but 1.155 with one chunk at
  3.34 against 2.16 nats; not investigated further once the int8 path as a whole had failed.
- PK5 (`native/ggml_patches/pk5`, after pk4): the Q4_K -> fp16 dequantisation of PK2's prompt matmuls with eight
  contiguous values and one 16-byte store per thread, the same float arithmetic as llama.cpp's kernel. Per-chunk
  perplexity identical with and without (bit-identical output); prefill of the benchmark mix 2.425 -> 2.345 s
  (32 clients) and 0.81 -> 0.76 s (8 clients). On by default, `IRONMULE_PK5=0` off.
Balance after this pod: 12.20 USD.
[I82 logs](../lab/i82/a100-i82-ppl.log)

### 2026-11-23, A100: speculative decoding with a batch draft model (SD1), and vLLM on the same pod with PK5
Owner: option 2 (bigger work for +10 %), all doors open. One RunPod A100 SXM4 80 GB pod (pod 17), Qwen3-32B Q4_K_M.
- Model of a 32-client wave (32-token answers): wall = prefill + 31 decode steps + idle. GPU work with PK5 about
  2.35 s prefill + 31 x 28.4 ms; for +10 % over vLLM the wave has to fit in 3.49 s (pod 12 numbers).
- Draft model acceptance, teacher-forced on the benchmark's prompts (k = 1, Qwen3-0.6B Q4_K_M for Qwen3-32B):
  0.660-0.676 at 32 clients (two seeds), 0.629-0.684 at 8. Step costs (`lab/sd1/sd_probe.py`): 32 rows 30.3 ms,
  64 rows 56.4 ms as two 32-row W4R calls (`IRONMULE_W4_SPLIT64`) or 81 ms on MMQ; 8 rows 20.7 ms, 16 rows 23.4 ms;
  a draft-model step of 32 / 8 rows 7.0 / 4.6 ms. At 32 clients a draft costs more than it saves (the slowest of 32
  requests still needs about 22 steps); at 8 clients it can pay once the draft step is cheap.
- Found on the way: the W4R library's token mask is one per process, and a second engine (the draft model) measured
  and overwrote it, turning W4R off for the served model (32-row step 55.7 instead of 30.3 ms). A draft engine now
  leaves the mask alone (`kernel_mask=False`); DM1's GPU results before this fix were measured with that fault.
- SD1 (`ironmule/draft_model.py` `BatchDrafter`, `IRONMULE_GGML_SD=k`, off by default): one draft per request for
  all running requests in one draft call, two tokens read per request and step. Engine waves at 8 clients:
  165.5 / 180.6 / 165.7 tok/s with SD1 (acceptance 0.64-0.70) against 176.6 / 184.3 / 178.6 without; with
  multi-row CUDA graphs on, one SD1 run produced a draft token of -1 (not located). Not a gain yet.
- Server trace on this pod: a 32-request burst reaches the server over 50 ms, the packed prefill then takes
  2.30-2.35 s (PK5), the 31 decode steps about 1.0 s; the server-side wave ends 3.45-3.51 s after the first request.
- Same pod, same client, prompts and seeds, three repeats, IronMule `serve --agent on` with PK5 against vLLM 0.11.0
  AWQ-Marlin:

| Clients | IronMule (PK5) | vLLM AWQ | difference |
| ---: | ---: | ---: | ---: |
| 1 | 51.3 / 95.0 / 100.0 | 51.3 / 52.7 / 51.8 | 0 to +93 % |
| 8 | 165.7 / 167.4 / 169.8 | 190.2 / 197.3 / 194.6 | -13.6 % |
| 32 | 293.8 / 291.6 / 291.1 | 295.5 / 291.3 / 290.9 | -0.1 % (292.2 against 292.6) |

  Facts 18/18 in every run except one IronMule run at 32 clients (17/18; the result file keeps only text hashes, so
  whether it is the known grader format miss is not checked). This pod ran both servers faster than pod 12 (vLLM
  292 against 266 at 32 clients). IronMule is level with vLLM at 32 clients on the same machine, behind at 8, and
  the +10 % goal is not reached.
[IronMule](../lab/pk1/results/a100-ironmule-http-pod17-pk5.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod17.jsonl).
Balance after this pod: 10.32 USD.

### 2026-11-23, A100: the server's own time per wave (SV1-SV4), and two more same-pod comparisons
Owner: keep going on the server overhead. Pods 18-20 (RunPod A100 SXM4 80 GB), Qwen3-32B Q4_K_M, PK5.
- Wave anatomy from the engine trace and the service's stability log, 32 clients: the requests reach the server
  over about 50 ms; the service forms the group; then **105-115 ms** pass before the prefill (pod 20 markers: the
  worker rendered the 32 prompts one by one in about 105 ms, the engine's gather window 25 ms); prefill 2.30-2.62 s by
  pod; 31 decode calls of about 31 ms with 1.06 ms between them; after the last decode call the last request ended
  75-90 ms later (33-40 ms on pod 19). The client's wall time equals the server-side wave within a few ms.
- A local model of the service's grouped streaming path (fake worker, 32 x 32 tokens, 31 ms steps,
  `lab/sv/evbench.py`) delivers a token in 1.7 ms (median) and ends 3.3 ms after the last frame, so the tail is
  outside that path; a lab thread profiler (yappi, `lab/sv/sitecustomize.py`) did not capture the serving threads.
  A shorter GIL switch interval (0.5 ms) changed nothing (pod 18).
- SV1 (`ironmule_product/worker.py`): a done frame carried a fresh engine summary and the gate record per member,
  written after a drain of the token writer; now one summary per engine step, sent through the writer. SV2
  (`ironmule/ggml_engine.py`): no gather wait once the burst fills every free sequence (pod 20: 0 ms instead of 25).
  SV3: no context-draft lookups in steps that have no row for a draft (the gaps between decode calls stayed
  1.06 ms, so this was not their cause). SV4: the group's prompts rendered with one chat-template call and one batched
  tokenization, used only after the first batch was checked equal to the one-by-one rendering (locally 3.7 against
  about 9 ms for 32 prompts). AM1: one argmax buffer per backend, and the engine recognises a fallback to full logits.
- SD1 with multi-row CUDA graphs (pod 20, after the AM1 fix): no abort and no invalid draft any more, but acceptance
  fell to 2-4 % (0.64-0.70 without graphs), so the draft model computes wrong tokens under those graphs; the answers
  stayed the target's (8/8 identical per wave). Not located; SD1 stays off.
- Same pod, three repeats, 18/18 facts in every run of both servers:

| pod | IronMule 8 clients | vLLM 8 | IronMule 32 clients | vLLM 32 | at 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 (SV1) | 166.7 / 168.9 / 167.7 | 197.5 / 198.7 / 196.9 | 284.6 / 286.7 / 278.5 | 293.4 / 293.4 / 292.2 | -3.3 % |
| 20 (SV1-SV3) | 160.5 / 162.6 / 165.7 | 188.0 / 189.1 / 188.9 | 264.4 / 267.3 / 265.3 | 271.6 / 272.3 / 273.6 | -2.5 % |

  Against pod 17 (level at 32 clients) the pods differ by up to 10 % for both servers. In every pod IronMule's last
  first token comes earlier than vLLM's (pod 20: 2.80 against 3.02-3.04 s); vLLM then needs about 0.72 s for the rest
  of the answers, IronMule about 1.05 s: the remaining gap is the decode phase (31 steps at about 31 ms against vLLM's
  about 23 ms) plus the server's dispatch and tail.
[pod 19](../lab/pk1/results/a100-ironmule-http-pod19-sv1.jsonl) / [vLLM](../lab/pk1/results/a100-vllm-awq-pod19.jsonl) ·
[pod 20](../lab/pk1/results/a100-ironmule-http-pod20-sv3.jsonl) / [vLLM](../lab/pk1/results/a100-vllm-awq-pod20.jsonl).
Balance after pod 20: 5.76 USD.

### 2026-11-23, A100 pod 21: SV1-SV4 against vLLM on the same pod
Same protocol as pods 19-20 (three repeats, 18/18 facts in every run of both servers): IronMule 157.7 / 159.2 / 163.1
against vLLM 182.0 / 184.4 / 183.3 tok/s at 8 clients (-12.7 %), 268.9 / 263.3 / 269.7 against 275.1 / 273.2 / 270.7
at 32 (-2.1 %), 53.8 / 91.5 / 97.7 against 51.5 / 52.1 / 51.8 at one client. The trace markers show SV4 did not
engage: rendering the 32 prompts still took about 103 ms, so the worker's tokenizer did not take the batched path (it
falls back to one-by-one rendering on any doubt; why is not checked). The tail after the last decode call was
153-166 ms on this pod. Over the four same-pod comparisons with PK5 (pods 17, 19, 20, 21) IronMule is level to 3.3 %
behind vLLM at 32 clients and 13-15 % behind at 8; the +10 % goal is not reached.
[IronMule](../lab/pk1/results/a100-ironmule-http-pod21-sv4.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod21.jsonl).
Balance after pod 21: 4.39 USD.

### 2026-11-23, local (4-core cloud container, CPU build): why the 32 prompts took 100 ms to render (TK1)
Maintainer change. The pod 21 trace left open why SV4 never engaged and why rendering the 32 benchmark prompts took
about 103 ms before every 32-client prefill. Measured locally with Qwen3-0.6B Q4_K_M's own tokenizer (`GgufTokenizer`,
the same Qwen2 vocabulary and template as Qwen3-32B): the chat template renders all 32 in 0.6 ms; llama.cpp's
tokenizer takes 44.6 ms for their 7,632 tokens. A call costs about 0.23 ms per text fragment between special tokens
plus about 0.8 us per token, because Qwen2's pre-tokenizer pattern has no hand-written splitter in llama.cpp and takes
the `std::regex` path, which compiled the pattern again for every fragment (five fragments per chat prompt).
- TK1 (`native/ggml_patches/tk1`, every build): each thread compiles a pattern once and keeps it. 32 prompts: 46 ->
  6.2 ms serially. The ids equal Hugging Face's Qwen3 tokenizer on 370 texts (the benchmark prompts and chat-wrapped
  ones, German, Japanese, Chinese, Korean, code, emoji, odd whitespace, written special tokens, 300 random Unicode
  strings).
- `GgufTokenizer` takes Hugging Face's batch interface (`tokenizer(texts)`, `apply_chat_template` over a list of
  conversations), tokenizing in up to four threads (llama.cpp releases the GIL; 4 threads: 46 -> 12.5 ms without
  TK1, 2.2 ms with it, the same ids). SV4's batch rendering in the worker now engages for GGUF serving and passed its
  equality check: 3.4 ms for the 32 prompts, against 46 ms before.
- Expected on the A100 pods: about 100 ms less per 32-client wave (about 2.5 % at 32 clients). Not yet measured there.
- The HC1 probe on Kaggle (T4) failed before measuring (no `PYTHONPATH`; it also did not keep positions consecutive);
  fixed and rerun, and its CUDA build is now kept as output so later Kaggle runs skip the 55-minute compile.
- Wave tail, local server (CPU build, 0.6B, 32 clients, `lab/sv/tail_probe.py`): the last request ended 1.2-3.8 ms
  after the last decode call, against 33-166 ms on the A100 pods. Locally a step takes about 200 ms; on the pods 31 ms,
  with 32 token frames per step. That the pod tail depends on the pod (33-40 ms on pod 19, 153-166 ms on pod 21) fits
  HC1 (the CPU threadpool spinning on every core while the server's Python threads stream), not yet shown.

### 2026-11-23, A100 pod 22: TK1 and EM1 on the GPU, and vLLM on the same pod
Maintainer session. RunPod A100 SXM4 80 GB (community cloud, 1.39 USD/h, 256 visible cores), Qwen3-32B Q4_K_M, build
with every patch including TK1 and EM1.
- EM1, engine A/B (`lab/em1/em1_probe.py`, waves of 8 and 32 benchmark prompts, 32-token answers): the answers with the
  embedding on the GPU equal those with it on the CPU (6/6 waves, 120/120 answers). Decode steps unchanged (32 rows
  32.46 against 32.44 ms, 8 rows 22.37 against 22.43); the 32-prompt prefill 10-15 ms shorter (2,575 / 2,418 against
  2,589 / 2,432 ms). The server's quality gate with EM1 then failed: its KV-setting tuning ran out of GPU memory
  (exit -6), since the table and the step's input rows now live on the GPU. EM1 is opt-in (`IRONMULE_GGML_EM1=1`).
- HC1 refuted on this pod: rendering the 32 benchmark prompts took 12.9 ms with the engine idle and 12.9 ms while it
  decoded (28 threads, busy-wait 50), with EM1 on or off; decode steps unchanged by the rendering. With TK1 the serial
  rendering took 12.9 ms here (103 ms on pod 21).
- Same pod, three repeats, IronMule `serve --agent on` (TK1, EM1 off) against vLLM 0.11.0 AWQ-Marlin:

| Clients | IronMule (TK1) | vLLM AWQ | difference of the means |
| ---: | ---: | ---: | ---: |
| 1 | 55.4 / 95.1 / 98.2 | 51.5 / 51.8 / 51.9 | +60 % (the first repeat is the cold one) |
| 8 | 161.5 / 163.3 / 165.7 | 190.4 / 186.5 / 189.5 | -13.4 % |
| 32 | 275.6 / 273.0 / 271.4 | 274.0 / 278.8 / 276.2 | -1.1 % |

  Facts 18/18 in every run except IronMule's third 32-client repeat (17/18).
- Wave anatomy from the engine trace, 32 clients: where the burst reached the worker as one group, 50 ms passed from
  the batch command to the packed prefill (106 ms on pod 21: the rendering is gone). In one wave the first request
  arrived alone and the engine started it at once: a 441-row prefill of 217 ms before the other 31 (7,167 rows,
  2.37 s), 240 ms from command to the big prefill. Prefill 2.37-2.50 s; 31 decode steps of 30.1-30.2 ms, the first two
  after a prefill 35 and 42-44 ms (a new CUDA graph).
- What this means for the +10 % goal at 32 clients: vLLM's wave takes about 3.71 s at 276 tok/s; +10 % needs about
  3.37 s. IronMule's GPU work alone is about 2.45 s of prefill plus 31 x 30.2 ms of decoding, 3.39 s, before any server
  time. The server's own time per wave is now small; the decode step (30.2 against about 23 ms in vLLM) is what
  remains, and within it the 4-bit matmuls at 32 rows (about 21 ms at 0.94 TB/s, against 1.37 TB/s at one row).
[IronMule](../lab/pk1/results/a100-ironmule-http-pod22-tk1.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod22.jsonl).
Pod time 0.87 h, about 1.20 USD; balance after pod 22: 3.17 USD.

### 2026-11-23, RTX 4090: why SD1's drafts failed under multi-row CUDA graphs (GR4), and GB1
Maintainer session. RunPod RTX 4090 (secure cloud, 0.89 USD/h, 0.47 h, about 0.42 USD; no Ampere card or A100 was
free), Qwen3-4B Q4_K_M as the served model and Qwen3-0.6B Q4_K_M as SD1's draft, engine waves of the benchmark prompts
(`lab/sd1/sd_wave.py`, 8 and 32 clients, 32-token answers, 1,024 context per sequence: 2,048 did not fit both models'
KV caches into 24 GB), W4R tuned on the card.
- The bug reproduced: drafts accepted 0.59-0.62 with one-row CUDA graphs, 0.03-0.05 with graphs up to 32 rows.
- SG1 (one W4R scratch per CUDA stream, grown buffers kept; commit c3f95f3) was not the cause: the same draft counts
  to the token (12 / 224, 16 / 456, 20 / 688 accepted) with the W4R library before and after it. SG1 stays as a
  robustness change (it touches no arithmetic).
- The cause was GR3's capture state ("changed", "capture next time", the pause counter): process-wide statics, so the
  draft model's context consumed or set the served model's flags and replayed a stored graph instance of another of
  its graphs. GR4 (`native/ggml_patches/gr4`) keeps that state per CUDA backend context: with 32-row graphs the drafts
  were accepted 0.62 / 0.61 / 0.59 again (the one-row numbers), with `IRONMULE_GR4=0` 0.05 / 0.04 / 0.03.
- On this card SD1 still lost at 8 clients (583 against 752 tok/s without drafts): next to a 4B model a 0.6B draft
  is expensive. Whether it pays next to Qwen3-32B is measured on the A100 (below).
- Side observation: the served model's answers with 32-row graphs differ from those with one-row graphs in 16-21 of
  120 answers (greedy divergence; not examined further here).
- GB1 (local, `tests/engine/test_ggml_engine.py`): after a group of several prompts, the next lone arrival waits with
  the burst's gather window once; on pods 20 and 22 a 32-client burst had started its first request alone (a 441-row
  prefill of 217 ms) with the other 31 behind it.

### 2026-11-23, A100 pod 24: the server with GB1, GR4 and SD1 (rejected)
RunPod A100 SXM4 80 GB (secure cloud, 2.19 USD/h; no community A100 was free), Qwen3-32B Q4_K_M,
`serve --agent on --draft-gguf Qwen3-0.6B-Q4_K_M` with `IRONMULE_GGML_SD=1 IRONMULE_GGML_SD_MAX_RUNNING=8` (SD1 drafts
while at most 8 requests run), GB1 and GR4 in the build. Startup 1,362 s (gate and the draft model's measurements).
Three repeats, facts 18/18 in every run:

| Clients | IronMule with SD1 (pod 24) | IronMule without SD1 (pod 22, another pod) |
| ---: | ---: | ---: |
| 1 | 52.1 / 52.4 / 52.9 | 55.4 / 95.1 / 98.2 |
| 8 | 148.1 / 153.6 / 152.2 | 161.5 / 163.3 / 165.7 |
| 32 | 269.2 / 269.0 / 267.6 | 275.6 / 273.0 / 271.4 |

- SD1 rejected for the server: at one client the batch drafter takes the place of the context drafts that gave
  95-98 tok/s, and at 8 clients a draft call plus a verification step of two rows per request cost more than the
  accepted drafts saved (pods differ by up to 10 %, but the one-client drop is far outside that). SD1 stays off by
  default; GR4 stays (it fixes a second CUDA context's graph state, whatever drafts).
- GB1 on the server: all three 32-client bursts were prefilled in one call with all 32 prompts (pods 20 and 22 had a
  lone first request in two of six bursts).
- No vLLM run on this pod (the remaining balance did not cover it, and this configuration is not the one to compare).
[IronMule](../lab/pk1/results/a100-ironmule-http-pod24-sd1.jsonl). Pod time 0.52 h, about 1.13 USD; balance after
pod 24: 1.64 USD.

### 2026-11-23, A100 pod 25: TK1, GB1 and GR4 (no SD1) against vLLM on the same pod
RunPod A100 SXM4 80 GB (community cloud, 1.39 USD/h), Qwen3-32B Q4_K_M, `serve --agent on`, build with every patch
(TK1, EM1 opt-in and off, GR4), SD1 off. Startup 1,221 s. Three repeats each:

| Clients | IronMule | vLLM 0.11.0 AWQ-Marlin | difference of the means |
| ---: | ---: | ---: | ---: |
| 1 | 55.3 / 95.7 / 101.4 | 51.8 / 52.7 / 52.1 | +62 % (the first IronMule repeat is the cold one) |
| 8 | 164.3 / 165.9 / 169.0 | 191.1 / 190.7 / 193.9 | -13.3 % |
| 32 | 283.3 / 280.3 / 280.0 | 287.4 / 287.4 / 288.0 | -2.2 % |

Facts 18/18 in every run except IronMule's third 32-client repeat (17/18). GB1 only partly held: prompts per prefill
call in the three 32-client bursts were 1 + 31, 2 + 31 and 32. Before the first burst a lone request (the end of the
8-client part) had waited in vain and cleared the burst memory; in the second the first two requests arrived more than
the 25 ms quiet gap before the rest. Over the six same-pod comparisons since PK5 (pods 17, 19, 20, 21, 22, 25)
IronMule is level with to 3.3 % behind vLLM at 32 clients and 13-15 % behind at 8; the +10 % goal is not reached.
[IronMule](../lab/pk1/results/a100-ironmule-http-pod25-gb1.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod25.jsonl).
Pod time 0.56 h, about 0.77 USD; balance after pod 25: 0.86 USD.

### 2026-11-24: width 24 in the W4R tuner, PS (precomputed Q4_K scales), PK2-M; the A100 measurement is pending
Maintainer session. Goal (owner): +10 % over vLLM at 32 clients on the A100 with Qwen3-32B (pod 25: 281.2 against
287.6 tok/s; about 405 ms per wave short) and a smaller gap at 8 (166.4 against 191.9). No A100 was reachable within
the approved GPU time (Kaggle has none; 0.77 USD left on RunPod against about 3.4 USD for the session below), so the
changes are built and checked locally and the session is prepared (`lab/w4bw/a100_20261124.sh`); no number below is
measured on a GPU yet.
- Width 24 (fix, on by default): `ironmule/w4_kernels.py` measured 1 / 8 / 16 / 32 tokens and serves every count by the
  measurement of its tile count, so 17-24 tokens (three tiles) had no table entry and fell back to llama.cpp's kernels;
  the engine's startup check then found no gain there. TOKENS now includes 24, and the v9 families (VWN, VW6N) are
  built for three tiles too. Test: `tests/engine/test_w4_kernels.py`. Pending: steps of 17-24 rows and HTTP at 24
  clients, against the old table (session step 2a and 3).
- PS / v12 (experimental, opt-in `IRONMULE_W4_PRE=1`): `repack_q4k_pre` stores each row's eight sub-block scales and
  negated minimums as fp16 in the tile-order copy (2,560 bytes a tile and super-block against 2,304, i.e. 160 against
  144 bytes a row and super-block), rounded as v9 rounds them in the kernel, so `w4dp` must equal `w4d` bit for bit.
  Per sub-block it replaces two byte permutes, a subtract and a multiply (scale and minimum each) by one permute.
  Compiled for sm_80: up to 255 registers per thread against 128-168 for v9 (the prefetched headers are twice as
  wide), little spilling; variants with a forced minimum occupancy (VWPB) are included. The benchmark
  (`lab/w4bw/bench.py --shapes Q4_K:5120:25600`) reports the best PS and the best v9 time per shape and token count and
  whether their outputs are identical. Pending: whether the saved work outweighs 11 % more weight bytes on the full
  5120 x 25600 matrix and in the 32-row decode step (session steps 1 and 2b).
- PK2-M (experimental, opt-in `IRONMULE_PK2_CALIBRATION=mix`): the startup calibration timed eight equal 444-token
  prompts; the 8-client mix (4 short, 4 long) runs as two passes on one KV stream per sequence, the short one of about
  130 rows below every threshold measured. "mix" times the thresholds (now with 128) on that mix and chooses by it.
  Test in `tests/engine/test_ggml_engine.py`. Pending: prefill and time to first token per threshold on the real mix
  (session step 2c).
- The failed fact in pod 25's third 32-client repeat: that repeat's seed (20263169) holds "What is 33 * 6?", the case
  pod 3 showed IronMule answering "200 - 2 = 198 ... **198**" instead of the number alone (a grader format miss, not a
  wrong fact; vLLM answers in the asked format). The pod 25 record kept only a hash of the answers, so this is an
  inference; `lab/pk1/http_bench.py` now records each missed case with its answer (`quality_misses`).

### 2026-11-24, A100 pod 26: width 24, PS and PK2-M measured; baseline, candidate and vLLM on the same pod
RunPod A100 SXM4 80 GB (community cloud, 1.39 USD/h, 1.28 h, about 1.77 USD), Qwen3-32B Q4_K_M, session
`lab/w4bw/a100_20261124.sh` (log: `lab/pk1/results/a100-pod26-session.log`). Baseline 255a58f, candidate b901f27.
Kernels (`lab/w4bw/bench.py`, random blocks in the model's shapes, every W4R variant, best per shape; the GPU read
1,733 GB/s with a plain streaming kernel):

| matrices of one decode step | 8 rows | 16 rows | 24 rows | 32 rows |
| --- | ---: | ---: | ---: | ---: |
| matmul time | 17.15 ms | 18.16 ms | 21.54 ms | 24.09 ms |
| share of the achievable bandwidth | 65 % | 61 % | 52 % | 46 % |

At 32 tokens the gate/up matrix (5120 x 25600) reached 56.5 %, down (Q4_K) 46.3 %, q and o 34-35 %, k and v
(5120 x 1024) 9 %: the small matrices take about 7.4 of the 24.1 ms for about a fifth of the bytes.
- PS (v12): bit for bit equal to v9 in every compared configuration; on the full 5120 x 25600 matrix 75.67 against
  75.32 us at 32 tokens, 59.80 against 57.19 at 16, 65.95 against 68.10 at 24 (the one shape and width where it won);
  slower on every other shape. In the decode step (engine, tables from one tuning run) 30.07 against 30.14 ms at
  32 rows and 26.38 against 27.15 at 24, with 167 of 168 answers equal (another configuration won one entry).
  Rejected: the saved instructions do not pay for 11 % more weight bytes and a second copy; it stays opt-in.
- Width 24 (engine, decode step by rows, base table without width 24 against the candidate's): 20 rows 49.04 ->
  25.90 ms, 24 rows 50.32 -> 27.15 ms; 8 and 32 rows unchanged (20.43 / 20.42, 30.22 / 30.14). Kept (on by default).
- PK2-M (engine, the 8-client mix, five seeds): llama.cpp's kernels 2.202 s a wave and 1,558 ms to the first token;
  threshold 1024 rows 1.458 s / 811 ms; "equal" (chose 256) 1.439 s / 792 ms; 128 rows 1.395 s / 748 ms; "mix" (chose
  128) 1.392 s / 746 ms. Made the default (below).
- HTTP, same pod, prompts, seeds and 32-token answers, three repeats each (output tok/s; first token median of the
  per-run medians):

| clients | baseline | candidate (width 24, PK2-M) | vLLM AWQ | candidate against vLLM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 48.5 / 95.5 / 100.2 | 56.0 / 100.9 / 102.8 | 51.9 / 52.0 / 52.1 | +66 % |
| 8 | 164.3 / 172.1 / 163.5 | 171.9 / 166.1 / 174.2 | 185.9 / 184.9 / 185.1 | -7.9 % (baseline -10.1 %) |
| 24 | 207.5 / 207.1 / 202.8 | 256.8 / 254.5 / 256.0 | 254.8 / 259.8 / 254.6 | -0.2 % (baseline -19.7 %) |
| 32 | 273.1 / 266.4 / 279.9 | 269.4 / 269.3 / 271.5 | 274.8 / 277.3 / 273.6 | -1.9 % (baseline -0.8 %) |

  First token at 8 clients: baseline 852 / 871 / 885 ms, candidate 831 / 843 / 797, vLLM 844 / 853 / 854. At 32
  clients the median first token is vLLM's (1.6-2.0 s against 2.7 s), the slowest IronMule's (2.73 against 3.0 s).
  Facts 18/18 in every run except the baseline's third 32-client repeat: 17/18, recorded now as
  `20263169:arithmetic-2` with the answer "200 - 2 = 198 / So, 33 * 6 = **198**" (right, not in the asked format).
- Decisions: width 24 kept (+24.3 % at 24 clients, 3 of 3 repeats above every baseline repeat); PK2-M made the
  default (first token earlier in 3 of 3 repeats, throughput +2.4 % with overlapping ranges); PS rejected. The +10 % at
  32 clients is not reached (-1.9 % here); the 8-client gap shrank from -10.1 % to -7.9 % on this pod.
- Next lever from the kernel table: the small matrices at 32 tokens (k and v at 9 %, q and o at 35 % of the bandwidth),
  i.e. one launch for q, k and v (and for gate and up) instead of three (two).
[baseline](../lab/pk1/results/a100-ironmule-http-pod26-base.jsonl) ·
[candidate](../lab/pk1/results/a100-ironmule-http-pod26-cand.jsonl) · [vLLM](../lab/pk1/results/a100-vllm-awq-pod26.jsonl) ·
[kernels](../lab/pk1/results/pod26/a100-w4bw-bench.json). Balance after pod 26: 49.03 USD.
- Kaggle T4 (T4W: the W4R kernels built for Turing, m16n8k8 and plain copies; Turing numbers, not an A100's): the
  T4 read 277 GB/s; a decode step's matmuls 75.8 / 78.3 / 83.3 / 92.4 / 99.0 ms at 1 / 8 / 16 / 24 / 32 tokens, 92 / 89 /
  84 / 76 / 70 % of that bandwidth; PS equal to v9 bit for bit and slower on every shape but k / v at 32 tokens.
  [T4 kernels](../lab/pk1/results/pod26/t4-w4bw-bench.json)

### 2026-11-24, A100 pod 27: Q/K/V in isolation, separate against one fused launch (v13)
RunPod A100 SXM4 80 GB (community, 0.18 h, about 0.25 USD; no model needed: random blocks of the real formats).
`lab/w4bw/qkv_bench.py`: a layer's q (5120 -> 8192), k and v (5120 -> 1024) as served, i.e. the best W4R variant of
each shape back to back on one stream with one prepared x shared (PK3), against v13 (`w4dq`, one launch over the
segments as one virtual matrix); weights in eight copies beyond the L2 cache, two alternating rounds, best of each.
Correction to pod 26's note: the 7.4 ms there included the o projection (after attention); Q/K/V alone is 4.8 ms at
32 rows.

| rows | layer (32 each) | separate | fused | per decode step: separate -> fused |
| ---: | --- | ---: | ---: | ---: |
| 8 | q + k + v (v Q4_K) | 45.50 us | 31.51 us | |
| 8 | q + k (v Q6_K apart, 15.62 us) | 36.63 us | 34.44 us | 3.128 -> 2.610 ms (-0.52) |
| 24 | q + k + v | 62.67 us | 41.40 us | |
| 24 | q + k (v Q6_K apart, 18.43 us) | 59.67 us | 48.74 us | 4.505 -> 3.474 ms (-1.03) |
| 32 | q + k + v | 66.73 us | 48.03 us | |
| 32 | q + k (v Q6_K apart, 19.71 us) | 63.97 us | 57.24 us | 4.813 -> 3.999 ms (-0.81) |

Every fused output within the error bound against float64 (relative 3.5-3.8e-4); not bit for bit equal to the separate
path (another configuration wins, so another summation order). At 32 rows that is about a quarter of the 3.3 ms the
small matrices were estimated to leave, about 2.7 % of a 30.1 ms step and about 25 ms of a 3.8 s wave. Next: v13 in
the served step behind a switch, the step at 8 / 24 / 32 rows with and without it. Not yet in llama.cpp's graph.
[measurements](../lab/pk1/results/pod27/a100-qkv-isolated.json)

### 2026-11-24, A100 pod 28: QK1 (q, k, v in one launch) in the served decode step -- no gain, HTTP skipped
RunPod A100 SXM4 80 GB (community, 0.42 h, about 0.58 USD), Qwen3-32B Q4_K_M, build with qk1, session
`lab/w4bw/a100_qkv.sh`. The isolated measurement repeated pod 27 (per step 3.32 -> 2.78 ms at 8 rows, 3.74 -> 3.10 at
16, 4.53 -> 3.49 at 24, 4.78 -> 4.00 at 32). The served decode step (engine, the pod's own W4R table, CUDA graphs up to
32 rows, rounds off / on / off / on, median step):

| rows | QKV off | QKV on |
| ---: | --- | --- |
| 8 | 20.615 / 20.626 ms | 20.681 / 20.682 ms |
| 24 | 26.588 / 26.582 ms | 26.475 / 26.435 ms |
| 32 | 30.500 / 30.522 ms | 30.595 / 30.533 ms |

Answers with QKV on: 189 of 192 equal to the first off round (off rounds 192 / 192). The 32-row gain stayed within the
spread, so by the rule fixed before the run no HTTP run. Kernel profile of 16 steps at 32 rows (CUPTI, per step):
q 2.11 ms (64 calls), k and the Q4_K v 1.18 ms (96), prep_x 1.19, reduce_split 0.50; with QKV on the fused q + k 2.72 ms
(63 calls), 33 separate 1024-row calls 0.41 ms, prep_x 1.40, reduce_split 0.28; all kernels 28.66 against 28.62 ms.
- The group took q and k in 63 of 64 layers but never the Q4_K v: in those 32 layers v's output shares memory with a
  tensor between q and v (QK1's overlap check), so v ran alone; and the fused path did not leave its prepared x for
  PK3, so those v calls prepared x again (+0.21 ms a step).
- The isolated benchmark overstated the separate path: under CUDA graphs q, k and v take 3.29 ms a step, not 4.78; a
  1024-row call takes 12.2 us, not 17-19 (launches on a stream outside a graph carry gaps the graph does not have).
  The small matrices' 9 % of the bandwidth was largely that.
- With v grouped and x shared the step would gain at most about 0.4 ms (1.3 %), about 13 ms of a 3.8 s wave: not
  worth more work. QK1 stays opt-in (`IRONMULE_KG_QKV=1`), off by default.
- Where a 32-row step's 28.7 ms of kernels go (QKV off): gate / up 9.35 ms (128 calls, already about 56 % of the
  bandwidth), down 5.74 (Q6_K 3.20, Q4_K 2.54), o 2.25, q 2.11, k / v 1.55 (Q4_K 1.18, Q6_K 0.37), prep_x 1.19,
  reduce_split 0.50, output 0.48, SwiGLU 0.42; matmuls about 21.5 ms. A gate / up group would not change gate / up's
  own efficiency (each already fills the GPU), so it is not the next experiment either; the matmul kernels themselves
  at 32 rows (gate / up, down) are.
[session](../lab/pk1/results/pod28/a100-pod28-session.log) · [profile](../lab/pk1/results/pod28/a100-pod28-profile.log)

### 2026-11-24, A100 pod 29: PF1, the prefill/decode timeline of one 32-client wave -- the decode step, not the schedule
RunPod A100 SXM4 80 GB (community, 0.685 h, about 0.95 USD), Qwen3-32B Q4_K_M (IronMule, build 96a4b6c with the
patches of pod 28, QK1 off) against Qwen3-32B AWQ (vLLM 0.11.0, AWQ-Marlin, its defaults: chunked prefill with a
2,048-token budget, prefix caching), session `lab/pf/a100_pf1.sh`. Workload: `lab/pk1/http_bench.py`'s 32-client mix
(16 short prompts of 33 tokens, 16 documents of about 444), 32-token answers, three waves with their own prompts, the
same seeds for both engines. Each arm is a fresh server process; vLLM's prefix-cache hits over the timed waves: 0 of
7,632 tokens a wave; IronMule reused 0-12 tokens of a prompt (its kept template prefix). Prompt tokens a wave: IronMule
7,413-7,423, vLLM 7,632 (chat templates differ). New instrumentation (trace only, no added synchronisation): the
engine trace's per-call prompt / decode / placeholder rows and per-request prefill start and end, the service's
submit / dispatch / first / end times, the client's arrival of every token, joined by the server's request id
(`lab/pf/analyze.py`).

Whole waves (single values, s; "last long first" is the last document's first token at the client, "after" is from
there to the wave's end, both on the same request's timeline):

| arm | tok/s | wave | last long first | after | short TTFT min / median / max | long TTFT median | short done median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IronMule, no trace | 267.3 / 267.4 / 278.3 | 3.823 / 3.824 / 3.676 | 2.771 / 2.790 / 2.650 | 1.05 / 1.03 / 1.03 | 0.48 / 2.74 / 2.75, 2.75 / 2.76 / 2.77, 2.62 / 2.62 / 2.63 | 2.75 / 2.77 / 2.63 | 3.78 / 3.78 / 3.65 |
| IronMule, trace | 275.0 / 270.0 / 275.0 | 3.720 / 3.788 / 3.720 | 2.674 / 2.773 / 2.716 | 1.05 / 1.02 / 1.00 | 0.38 / 2.65 / 2.65, 2.74 / 2.75 / 2.75, 2.68 / 2.69 / 2.70 | 2.66 / 2.76 / 2.70 | 3.68 / 3.75 / 3.68 |
| vLLM | 279.9 / 279.5 / 277.6 | 3.655 / 3.659 / 3.685 | 2.958 / 2.962 / 2.995 | 0.70 / 0.70 / 0.69 | 0.23 / 1.59 / 2.93, 0.22 / 1.60 / 2.94, 0.71 / 1.80 / 2.90 | 1.96 / 1.96 / 2.16 | 3.62 / 3.63 / 3.64 |

Facts: IronMule 18/18, vLLM 18/18 (one graded wave each, apart from the timed ones). The trace costs nothing visible
(270-275 against 267-278 tok/s).
- IronMule's schedule: the wave's prompts go into one native call (waves 2 and 3: 7,413 and 7,423 rows of 32
  sequences, 2.55 and 2.52 s; wave 1: 2 early arrivals alone, 474 rows in 0.29 s, then 7,155 rows of 30 in 2.25 s),
  then 31 decode calls of 32 rows (engine 30.26 ms median over 135 calls; 31.2-31.6 ms between tokens at the
  client). So a short request does not wait after its own prefill: its prefill ends in the same call as the longest
  prompt's, and its first token comes 0-30 ms after the documents' (every short request: 0 tokens when the last
  prefill ended, 1 at the last document's first token).
- vLLM's schedule (`/metrics` per wave): 37 steps, 4 of 1,025-2,048 tokens (the prompt chunks with the running
  requests' decode rows), 1 of 257-1,024, 32 of at most 64 (31 of at most 32). First tokens come in four groups about
  0.72 s apart, one per chunk. By the last document's first token the short requests had 1-6 tokens (median 3), the
  documents 1-6: mixing gained them three to four tokens, not a shorter path.
- Where the wave's difference sits: vLLM ends its last prefill 0.17-0.35 s later than IronMule (2.96-3.00 against
  2.65-2.79 s: IronMule's packed prompt call is the faster prefill, about 2,900 against 2,600 prompt tokens a second;
  vLLM's chunks also carry the running requests' decode rows), and needs 0.69-0.70 s instead of 1.00-1.05 s for the remaining 31 tokens:
  22.7-22.8 ms a step against 31.2-31.6 ms. vLLM's waves are 0.02-0.17 s shorter because of the decode step alone.
- vLLM mainly hands out first tokens earlier (short median 1.6-1.8 against 2.6-2.8 s; its slowest first token is
  later, 2.90-2.94 against 2.64-2.78 s); it does not finish the wave earlier by scheduling. With one burst of
  arrivals the wave cannot end before the last prompt's prefill plus its 31 decode steps; mixing decode rows into the
  prompt calls only moves that prefill later (and, with IronMule's KV streams, splits the call into passes). A mixed
  prefill/decode prototype would trade later documents for earlier short answers and is not the next experiment for
  throughput or wave time.
- Next: the 32-row decode step. At vLLM's 22.7 ms the same waves would take about 2.65-2.79 + 0.70 s = 3.35-3.49 s,
  5-9 % below vLLM's 3.65-3.69 s. The step's parts (pod 28): about 28.7 ms of kernels, of which gate / up 9.35 ms and
  down 5.74 ms (Q4_K and Q6_K), against a 30.26 ms native call and a 31.5 ms client gap: about 1.5 ms of launch and
  graph overhead and about 1.2 ms of host time (sampling, events) a step. The gate / up and down kernels at 32 rows
  come first, the host time per step second.
[session](../lab/pk1/results/pod29/a100-pod29-session.log) · [raw data](../lab/pk1/results/pod29/) (client timelines
`tl_*.jsonl`, engine trace `trace.txt`, service `stability-log.jsonl`, vLLM `metrics_vllm.jsonl`, analysis `pf_*.json`)

### 2026-11-24, PF1 numbers attributed, and A100 pod 30: DK1, what limits the 32-row decode matmuls -- v9 on gate / up is not HBM-bound; the ablations give no bounds
Owner's order for this round: first attribute PF1's numbers, then one dominant decode bottleneck, at most two variants.

PF1 (pod 29) on one time axis, from the raw data already saved:
- vLLM's 22.7 ms is a direct 32-row reference, not an average over shrinking batches: in every wave 26-27 consecutive
  token arrivals reached all 32 requests at once, 22.6-23.5 ms apart at the client (the last 4-5 steps had 31-1
  requests). It is vLLM's whole step period at the client; its GPU time per 32-row step (the 2026-11-20 study, another
  run) was about 20 ms, so about 2.8 ms of it are outside the kernels, not split further.
- IronMule, traced waves: 32-row native call 30.0-30.1 ms (median per wave), 1.2-1.3 ms between calls (median;
  0.5-1.7), start to start 31.2-31.4 ms = the client's spacing of full 32-token arrivals (31.2-31.5 ms); the first
  chunk arrives 0.6-0.8 ms after a call ends. Inside the call the kernels are not on this time axis; the 28.4 ms of
  kernels per 32-row step come from pod 13's profile (another run, CUPTI), so the 1.6-1.7 ms between them and the call
  stay unexplained here (pod 13 put about 0.8 ms of input setup and 0.1-0.7 ms of readback around its graph).
- What +10 % over vLLM needs, wave by wave (the same prompts; target = vLLM's wave / 1.10; IronMule's own time to the
  last document's first token and 31 steps after it):

| wave | vLLM wave | target | IronMule last first token (no trace / trace) | step period needed (no trace / trace) | now |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 3.655 s | 3.322 s | 2.771 / 2.674 s | 17.8 / 20.9 ms | 31.4 ms |
| 2 | 3.659 s | 3.327 s | 2.790 / 2.773 s | 17.3 / 17.9 ms | 31.4 ms |
| 3 | 3.685 s | 3.350 s | 2.650 / 2.716 s | 22.6 / 20.4 ms | 31.2 ms |

  So with the present prefill, a 17-23 ms step period (a native call of about 16-21 ms): a cut of 30-47 %, not the
  "20-22 ms" working figure; the same throughput could also come partly from the prefill (68-74 % of the wave).

DK1 on pod 30 (RunPod A100 SXM4 80 GB, community, 0.546 h, about 0.76 USD; Qwen3-32B Q4_K_M; build 38e4267, product
kernels unchanged; `lab/dk1`, raw data in `lab/pk1/results/pod30`):
- Nsight Compute is installed but RunPod's container refuses GPU counters (`ERR_NVGPUCTRPERM`, a host driver setting):
  no DRAM bytes, stall reasons or pipe utilisation. The limit was located by timing instead: lab-only compile switches
  in v9 (`IRONMULE_W4_ABL`, default off; the product build's SASS is unchanged, 2,072 instructions, 236 registers, no
  spills) that remove one part at a time, all builds loaded in one process and measured in turn after a warm-up, six
  rounds of 200 calls (spread below 1 %). Separate processes with short windows had scattered by up to 20 % and
  were discarded; a first pin of the blocks per SM through dynamic shared memory also shrank L1 to 28 KB and slowed
  the full kernel to 104-121 us, so it was replaced.
- The winners at 32 tokens (all variants measured again): gate / up v9 (variant 184: 2 warps, 2 tiles per warp, 236
  registers, 4 blocks = 8 warps per SM, 400 blocks for 432 slots) 75.6 us; down Q4_K v11 (VF(4,4,1), K-split 4, 122
  registers, 36 KB shared) 92.3 us; down Q6_K v11 111.9 us (times with prep_x and reduce_split; the pod's own tuning
  chose the same). Served steps without a profiler: 20.4 / 22.3 / 26.8 / 30.2 ms at 8 / 16 / 24 / 32 rows.
- Gate / up at 32 tokens, us per call (median of six rounds):

| arm (v9, variant 184) | us | what else the arm changed (SASS and ptxas of the same builds) |
| --- | ---: | --- |
| product build | 75.4 | 2,072 instructions per two super-blocks, 236 registers, 4 blocks per SM |
| largest L1 requested (carveout 0 %) | 75.4 | nothing: the driver already chose the largest L1 in this process |
| carveout 50 % / 100 % (L1 about 28 KB) | 79.1 / 103.3 | only the L1 size |
| weights streamed only (no x, no math) | 50.7 | 416 instructions, 122 registers, 8 blocks per SM |
| weights from L2 instead of DRAM | 72.2 | 2,192 instructions (address arithmetic), 236 registers, 4 blocks per SM |
| x from one sub-block (L1-resident) | 67.7 | x loads hoisted (40 LDG instead of 100), 194 registers, 5 blocks per SM |
| no mma | 63.4 | HMMA replaced by XORs (363 LOP3 instead of 186), 164 registers, 6 blocks per SM |
| no dequantisation (raw words into the mma) | 61.3 | 1,000 instructions, 206 registers, 4 blocks per SM |

  Down Q4_K with v9 (variant 184, K-split 4): 102.7 us full, 59.2 weights only, 95.3 weights from L2, 84.3 without
  dequantisation. This is v9 on the down shape, not the served down kernels (Q4_K: variant 256, v11 VF(4,4,1);
  Q6_K: variant 266, v11 for Q6_K; both K-split 4, 122-126 registers, 36 KB of shared memory, x staged in shared
  memory): nothing here is measured for them.
- What the measurements support (v9 on gate / up at 32 tokens): taking the DRAM stream away (weights from L2, the
  same registers and occupancy) saves 3.2 of 75.4 us, so the HBM bandwidth itself is not the dominant limit there.
  That does not show that other layouts would not help: a layout also changes the dequantisation, the registers and
  the accesses inside the SM, which this arm leaves as they are.
- What they do not support: the other arms are not cost blocks and not upper bounds. Each one also changed the
  instructions executed, the registers and with them the blocks per SM (5, 6 or 8 instead of 4), and the dependence
  chains (no mma: nothing waits for the dequantised registers; no dequantisation: the mma no longer waits for it), so
  their differences cannot be added or read as the most a change could gain. That HMMAs and dequantisation are
  interleaved in the SASS order says nothing about whether they overlap in execution; restructuring the data
  dependences (for example dequantising one sub-block ahead into a second register set) would be a different
  hypothesis from reordering the source, and is untested. A count of the SASS per super-block (about 1,036
  instructions, 128 HMMA, about 475 integer ops at half rate) estimates tensor pipe, integer pipe and issue at about
  25-27 us each of the 72 us; without counters this is an estimate.
- No variant was built in this round. The best-targeted mechanism found (nibble unpacking by LOP3 with a permuted
  tile layout, 7 -> 5 integer ops per 8 weights) was estimated at a few microseconds per gate / up call and needs a
  second copy of the weights in another layout; PS (pod 26), which removed the per-sub-block scale arithmetic, gained
  nothing at 32 tokens. This is an estimate, not a measured limit; the kernel line is open, not closed.
- For the whole step: all matmuls of a 32-row step at the copy bandwidth would save about 9.6 ms (pod 13: 21.1 ms at
  0.94 TB/s against 11.4 ms at 1.73), about the size of the needed cut. The other parts of a 32-row step known from
  earlier runs (not re-measured here): prep_x and reduce_split 1.95 ms (pod 13, an older build and run: a starting
  hypothesis, not an available saving), attention 2.5 ms against vLLM's 1.5, norms / RoPE / KV / SwiGLU about 2.7 ms
  against about 1.3, and 1.6-1.7 ms inside the native call outside the kernels plus 1.2 ms between calls (PF1). No
  single one of them closes the gap; the prefill is 68-74 % of the wave.
- Next: decode producers that write the matmul's prepared activations (PD1 below), then reduce_split separately.
[session](../lab/pk1/results/pod30/a100_dk1.log) · [ablations, separate processes](../lab/pk1/results/pod30/a100_dk1b.log) ·
[ablations, one process](../lab/pk1/results/pod30/a100_dk1c.log) · [down](../lab/pk1/results/pod30/a100_dk1d_down.jsonl)

### 2026-11-24, A100 pod 31: PD1 (decode producers write the matmul's fp16 activations) kept; RF1 (K-split reduced in the kernel) rejected
RunPod A100 SXM4 80 GB (community, 1.453 h, about 2.02 USD; balance afterwards 44.36 USD), Qwen3-32B Q4_K_M, build
with patch pd1, the pod's own W4R table (matmul variants and K-splits unchanged in every arm). Sessions
`lab/pd1/a100_pd1.sh` and `lab/rf1/a100_rf1.sh`; raw data in `lab/pk1/results/pod31`.

What prep_x did, from the code: per matmul input x [B][K] (float) it wrote the fp16 copy (saturated to +-65000,
rounded to nearest) into the W4R library's scratch with the token rows padded to the 8-token tile, plus the 16- and
32-value sums of the rounded copy. Only w4r and the 1-8-token kernels v7 / v8 read those sums; the 9-32-row
winners (v9, v11, w4d6) read the fp16 copy alone. PK3 already shared one copy between q / k / v and gate / up.
PD1: FU1's fused add + RMS norm + mul (the attention and FFN inputs of every layer but the first) and PK4's SWIGLU
write that copy themselves when the library offers a slot (`ironmule_kg_prepare`): this stream's scratch, recorded
under PK3's key (node, graph epoch, data, B, K, rows, stream, buffer) as prepared without sums. A matmul skips prep_x
only on that key and only with a variant that reads no sums; no slot is given where a table entry for that K and
batch reads sums, inside a capture without room, for other K or batch. The fp32 outputs are still written; the first
layer's norm, the output norm and the o projection (input from attention) keep prep_x.
- Correctness: one round with `IRONMULE_KG_PD1_CHECK=1` (prep_x run beside every producer-written copy, inside the
  CUDA graphs, compared bit for bit): 45,485,568,000 halves compared, 0 differed; answers of all 64 timed waves
  identical to the reference arm (8 / 16 / 24 / 32 requests, two prompt sets, four rounds).
- Decode step, reference and PD1 in one process (graphs captured anew after each switch, a discarded wave per
  width), medians of eight waves each, no profiler:

| rows | reference | PD1 | difference |
| ---: | ---: | ---: | ---: |
| 8 | 20.392 ms | 20.429 ms | +0.037 ms |
| 16 | 22.515 ms | 21.973 ms | -0.542 ms |
| 24 | 26.764 ms | 26.119 ms | -0.645 ms |
| 32 | 30.486 ms | 29.638 ms | -0.848 ms |

  The waves of the two arms did not overlap at 16-32 rows (32 rows: 30.47-30.56 against 29.58-29.78 ms). At 8 rows
  the table's variants read the sums, so prep_x still ran and the producer's copy was extra work; the slot is now
  refused there (8-row step with the corrected rule in the RF1 run: 20.372 ms).
- The chain, CUPTI kernel tables of 16 steps at 32 rows (diagnostic, apart): prep_x 19.49 ms (4,112 calls) -> 4.07 ms
  (1,040: the o projection and the output matrix), FU1 norm 15.46 -> 15.85 ms, SWIGLU 6.73 (llama.cpp's) -> 7.10 ms
  (PK4's with the copy), matmuls unchanged; kernels per step -0.92 ms.
- HTTP on the same pod (IronMule reference / PD1 / reference / PD1, two repeats each; vLLM 0.11.0 AWQ-Marlin), 32
  clients, tok/s, wave, the last document's first token and the rest of the wave (client timelines):

| arm | tok/s | wave s | last long first token s | after it s |
| --- | --- | --- | --- | --- |
| IronMule reference | 273.5 / 271.9 / (185.3) / 270.4 | 3.739 / 3.761 / (5.521) / 3.783 | 2.710 / 2.748 / (4.463) / 2.751 | 1.030 / 1.013 / (1.057) / 1.032 |
| IronMule PD1 | 283.4 / 277.6 / 270.3 / 278.8 | 3.609 / 3.684 / 3.781 / 3.668 | 2.577 / 2.665 / 2.775 / 2.663 | 1.031 / 1.019 / 1.006 / 1.006 |
| vLLM | 277.6 / 276.4 | 3.684 / 3.701 | 2.991 / 3.006 | 0.693 / 0.695 |

  At 8 clients: reference 174.0 / 176.4 / 172.5 / 169.3, PD1 172.7 / 177.0 / 172.8 / 175.9, vLLM 188.0 / 188.1 tok/s.
  Facts 18/18 in every run of every server. The bracketed reference wave was the first after a restart that reused
  the gate: its 32 requests reached the engine within 0.145 s, one was prefilled alone (first token 0.43 s), the
  packed prefill of the others gave first tokens only at 4.45 s instead of about 2.6 s; no engine trace ran, so its
  cause is open (not PD1's arm). Without it: reference 271.9, PD1 277.5, vLLM 277.0 tok/s mean. The decode part of
  the waves moved by the expected amount (31 steps x 0.85 ms = 26 ms: 1.013-1.032 against 1.006-1.031 s); the rest
  of the HTTP difference is the prefill's spread between server starts (last first token 2.58-2.78 s), which is
  larger than PD1's whole effect. So PD1 is worth about +0.7 % per 32-client wave on IronMule, and IronMule with PD1
  is level with vLLM on this pod (277.5 against 277.0), not ahead.
- PD1 is on by default (`IRONMULE_KG_PD1=0` turns it off). Its output is bit-identical to prep_x's, so the quality
  gate's cached results stay valid; the gate ran anew in the PD1 arm here and passed.

RF1 (`IRONMULE_KG_RF1=1`, off): the v11 kernels (down Q4_K 256/4, q 256/4, down Q6_K 267/8; 128 of the 160
reduce_split calls of a 32-row step) reduce their own K-split partials: every block fences and counts, the last block
of a row range adds the partials in reduce_split's order and resets the count; nothing waits. Outputs bit-identical
(isolated and all 64 waves), but slower, prep_x + matmul + whole reduction per call: down Q4_K 92.0 -> 114.4 us, q
40.3 -> 55.7, down Q6_K 117.2 -> 129.8; the w4d6 control 19.7 -> 19.7. Decode step with PD1 on in both arms:
+0.42 ms at 16 rows, +2.26 ms at 32, none at 8 and 24. The last block of each range sums 64 rows x 32 tokens x 4-8
partials with 128 threads after its own matmul, a serial tail of L2 loads at the end of the kernel that costs more
than the separate reduce kernel (0.51 ms per 32-row step in the profile). Rejected in this form; a reduction in the
consumer (the next residual add or norm reading the partials) would avoid that tail and is untested.

Where +10 % over vLLM stands (this pod's paired waves): vLLM 3.684 / 3.701 s gives a target of 3.349 / 3.365 s; with
IronMule's present time to the last first token (2.58-2.78 s) the decode part would have to fall from about 1.01 s
to 0.57-0.77 s. In IronMule's waves the prefill up to the last first token is 70-74 % of the wave and the decode
part 27-28 %, and the prefill alone varies by about 0.2 s between server starts. Next, by share of the wave: the
prefill, first its spread (one traced HTTP wave per start: gathering, packing and the packed call apart), then the
packed call itself.

### 2026-11-24, A100 pod 32: PF2, the prefill and its spread between server starts; PK1-C (gathering window) measured, opt-in
RunPod A100 SXM4 80 GB (community, 1.398 h, about 1.94 USD; balance afterwards 42.47 USD), Qwen3-32B Q4_K_M, the final
PD1 state (PD1 on with the 8-row rule, RF1 off), one state directory for every start (the first ran the quality gate
and the W4R tuning, the others reused both). Sessions `lab/pf2/a100_pf2.sh` and `lab/pf2/a100_pk1c.sh`; raw data in
`lab/pk1/results/pod32` (engine traces, client timelines, service logs, nvidia-smi every 200 ms, CUPTI summaries).
New trace fields (only with `IRONMULE_GGML_TRACE`, clock reads where calls return anyway): each native call's host
batch build, llama_decode and readback; when the engine made each request; the gathering window's start and end; a
start's stages and choices. `lab/pf2/balance.py` lays every wave out as one chain on one time axis.

Reference (PD1 final) over HTTP, the same prompts at the same position in every start and in vLLM, tok/s:

| | 32 clients, first wave | 32 clients, later waves | 8 clients (facts) |
| --- | --- | --- | --- |
| IronMule S1 (first start, 1,300 s) | 202.6 | 275.2 / 279.7 | 171.2 / 182.3 (18/18) |
| IronMule S2 (reused, 91 s) | 178.4 | 267.3 / 276.8 | 171.7 / 183.9 (18/18) |
| IronMule S3 (reused, 85 s) | 279.2 | 265.4 / 262.4 | 170.0 / 181.6 (18/18) |
| vLLM (141 s) | 210.7 | 278.8 / 277.5 | 186.2 / 187.3 (18/18) |

- Saving needed for +10 % over vLLM, wave by wave (IronMule wave - vLLM wave / 1.10, same prompts): 0.30-0.55 s in
  the later waves (0.380 / 0.304, 0.490 / 0.346, 0.519 / 0.549 s). PD1's 26 ms are inside every IronMule number above;
  the spread between starts and waves (later waves 3.65-3.90 s) is larger than that and is not read as a gain of
  anything.
- A start's stages, every engine of a start (the serving one and the gate's children): warm-up 0.3-2.4 s, argmax check
  0.15 s, W4R step measurement 17.5-18.9 s, PK2 calibration 21-24 s, Pilot 1.2-1.9 s. The choices were the same in
  every start: PK2 from 128 rows, W4R for 1-8 / 16 / 24 / 32 rows.
- The chain of a later wave (no split): burst 20-25 ms, through the service to the engine 36-147 ms, gathering 25-37
  ms, the packed prefill call 2.41-2.49 s for about 7,400 rows (host batch build 5-13 ms, readback 0.4-1.7 ms; the
  rest inside llama_decode), first token at the client 26-33 ms after it, the decode part 1.01-1.05 s.
- Inside the packed call (CUPTI, a separate engine process with the server's settings; 7,632 prompt tokens): 2.37-2.40
  s of kernels in a 2.46-2.48 s region: cuBLAS fp16 GEMMs 1.84-1.90 s, Q4_K / Q6_K dequantisation 0.115 s, SWIGLU 0.10,
  norms 0.09, attention 0.08, RoPE 0.03, activation conversion 0.026, KV writes 0.018; gaps 58-64 ms, host time before
  the first kernel 21-22 ms. First-time work in the first wave after the start: 2 cudaMalloc and 2 cudaFree (18 ms),
  nothing else. During server prefills the SM clock sat at 1,245-1,410 MHz with the software power cap (400 W) active.
- Split bursts: in 7 of the 9 32-client waves of S1-S3 the gathering window closed with 1-4 requests and the others
  waited for a second prefill call (a first call of 31-483 ms, then 16-108 ms until the second). Cause, from the
  service's log: the client sends 32 requests within 20-25 ms; the service admits the first eight 10-26 ms apart and
  the rest within 1 ms each (admits over about 100 ms), starts a group with the first one to three and forwards the
  others as joins 3 to over 60 ms after it; the engine's window (25 ms of quiet in a burst) closes in between. At 8
  clients the window held in 5 of 6 waves.
- Slow first waves: in S1 and S2 the first wave's large prefill call took 3.71 and 4.03 s instead of about 2.4 s; the
  GPU ran about 0.6 s, then idled 1.2-2.0 s in the middle of the call (utilisation 0-1 %, about 88 W), then finished:
  host-side waiting inside llama_decode, between the passes of a packed call. Not in S3's or S4's first wave, not in
  two separate engine processes (one with the server's split), no major page faults in S4's worker (the pod had 2 TB
  of memory). The same signature recurred twice in later waves of the A/B below (3.16 and 3.03 s calls). Cause open.
- vLLM's first wave was slow too (4.86 s against 3.67-3.69 s afterwards).

PK1-C (`IRONMULE_GGML_PK1C=1`, opt-in): within GB1's burst memory the gathering window expects as many requests as
the last run held at its peak; while fewer have come it waits for a longer quiet gap (inside the 150 ms burst limit)
and it ends at once when they are all there; a run that held several requests renews the burst memory. Three A/Bs on
the same pod, server starts alternating, the PF2 state reused, six (first: three) 32-client waves and two 8-client
waves per start, facts 18/18 in every one:
- 60 ms expected quiet, first form: c1's window lost its memory after a split first wave (GB1 cleared it on a lone
  arrival although the rest joined); fixed (the run's end renews it).
- 60 ms after the fix: the window held where the joins came within 60 ms; in other waves the rest came more than 60
  ms after the first request (`gathered 1 expect 32` after exactly 60 ms).
- 120 ms: later 32-client waves in one prefill call 10 of 10 with PK1-C against 6 of 10 without; mean wave 3.788 s
  (3.733 s without one wave with a 3.03 s call) against 3.765 s, time to the last first token 2.770 (2.715) against
  2.741 s; 8-client waves 1.480 / 1.361 and 1.531 / 1.376 against 1.522 / 1.404 and 1.455 / 1.400 s. The packed call
  of the same size itself took 2.37-2.64 s from wave to wave, more than the 33-320 ms a split costs, so no wave-time
  gain is resolved. A lone request within the memory (the benchmark's warm-up request) waited the full 150 ms.
- So PK1-C stays off by default (expected quiet 120 ms when on). The split is solved as a mechanism, the throughput
  gain is not shown, and +10 % over vLLM is not reached.

Next, by share of the wave: the packed prefill call (2.4-2.6 s of a 3.65-3.90 s wave; cuBLAS GEMMs 1.84-1.90 s of it),
first its spread of about 0.25 s between waves of the same size (the power cap and clocks above are the first
suspect; observed, not tested), then the host-side pause that turns some calls into 3.0-4.0 s.

### 2026-11-24, PF3 before the pod: PF2's conclusions made precise, the GEMM efficiency of the packed prefill, pass fragmentation (PX1) chosen
Local analysis of pod 32's raw data (`lab/pk1/results/pod32`), no GPU time. PD1 on, RF1 and PK1-C off.

Corrections to PF2:
- Split bursts, at equal prompt work: pairing every split 32-client wave with an unsplit one of the same repeat (same
  prompts, PF2 S1-S3 and the PK1-C A/Bs, pause calls excluded), the time to the last first token differs by a median
  of +24 ms (-49 to +79 ms) for the split waves. The first, small call computes rows the wave needs anyway; what a
  split adds is one more pass through the weights, and an unsplit wave spends 0.20-0.24 s in the service and the
  gathering window before its call where a split wave spends 0.05-0.09 s. So the split is not the largest avoidable
  block.
- PK1-C only waits: it holds the window open until the expected count is there. Ten of ten merged later waves show
  that the grouping works, not that the wave gets faster (it did not, within the call-to-call spread). The cause of
  the late joins is the admission path (below), not the window.
- Page faults: S4's worker had no major faults, but S4 had no pause. That does not exclude page faults (or anything
  else) in the paused calls of S1 and S2; the cause of those pauses stays open until a paused call is observed with
  the probe below.

GEMM efficiency of the packed prefill (CUPTI kernels of pod 32's separate engine process; Qwen3-32B, 64 layers):

| per layer (out x in) | q 8192x5120 | k, v 1024x5120 | o 5120x8192 | gate, up 25600x5120 | down 5120x25600 |
| --- | --- | --- | --- | --- | --- |
| share of the GEMM FLOPs | 8.6 % | 1.1 % each | 8.6 % | 26.9 % each | 26.9 % |

- Path (PK2, passes of 128 rows or more): Q4_K / Q6_K weights dequantised to fp16 per pass, then
  `cublasGemmEx(OP_T, OP_N)` with fp16 weights and activations, fp32 output, `CUBLAS_COMPUTE_32F`,
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (cuBLAS's heuristic; ggml sets no workspace of its own). Kernels:
  `ampere_s16816gemm_fp16_256x128_ldg8_stages_64x3_tn`, its 128x256 sibling and
  `cutlass_80_tensorop_s16816gemm_f16_128x256_32x6_tn_align8` (fp16 MMA, fp32 accumulation). 7 GEMMs a layer, 448 a
  pass; the tokens are the GEMM's N, unpadded (the tile edge wastes 1.2 % at 6,576 rows, 9 % at 1,056).
- The output projection runs only on the 32 rows that need logits (below the PK2 threshold, not cuBLAS).
- Wave 0 (passes of 1,056 and 6,576 rows, 7,632 tokens): 2 x 7,632 x 31.21e9 = 4.763e14 FLOP in 1,904 ms of GEMM
  kernels = 250 TFLOPS. Wave 3 (896 + 6,464 cuBLAS rows and five small W4R passes): about 250 TFLOPS as well.
- The A100's fp16 tensor peak is 108 SMs x 2,048 FLOP a clock: 312 TFLOPS at 1,410 MHz, 280 TFLOPS at the 1,268 MHz
  the SM clock held during these calls (power cap 400 W active). Achieved: 89 % of the clock-adjusted peak, 80 % of
  the nominal one. Lower bound for these FLOPs: 1.70 s at the observed clock, 1.53 s at 1,410 MHz.
- Practically supported target: large fp16 GEMMs with fp32 accumulation reach about 90-93 % of the clock's peak on
  A100s; fp16 accumulation has the same peak on the A100, so the type is no lever. From 89 % to 93 %: about 0.08 s a
  wave; a workspace or algorithm choice that recovers part of it: 0.03-0.06 s. The clock itself is a measurement
  condition here, not a shown reserve.
- So the GEMMs hold no reserve near the 0.30-0.55 s needed for +10 % over vLLM.

Pass fragmentation (`lab/pf3/passes.py`, the passes rebuilt from the trace's request lines with llama.cpp's
sequential split): with one KV stream per sequence a pass takes the run of consecutive sequence ids that all still
have tokens and advances each by the fewest any of them has left. Prefix reuse keeps 0-12 template tokens per stream,
different from stream to stream, so in later waves prompts of equal length become unequal and a 32-request prefill
runs as 6-14 passes (median 8; first waves: 2). Fitted over 60 large calls, robustly (pause calls excluded): 0.3347
ms a row plus 21.5 ms a pass (95 % bootstrap 18.6-23.9 ms), residual median 11.5 ms. Keeping the same prefix in every
stream of a packed group gives 2 passes in every call for a median of 67 more rows: predicted -110 ms a packed call
(median; 0 to -233 ms), in later waves only. That is 20-37 % of the needed saving, the largest supported candidate.

Admission path (local CPU server, py-spy): the client's 16-22 ms burst becomes 94-113 ms of submits, because every
request runs `ProductService.health()` (JSON reads under file locks, about 62 ms over 32 requests) and
`ServingLog.append` (about 23 ms) on the serialised Python path. Estimated 40-60 ms; the next candidate after PX1.

PX1 (`IRONMULE_GGML_PX1=1`, off until measured): a packed group keeps the smallest kept prefix in every stream when
that saves passes, for at most 32 recomputed rows per pass saved (`llama_memory_seq_rm` cuts the longer prefixes;
the recomputed tokens are an ordinary prefill). The trace notes the passes before and after either way.

Pause diagnostics prepared, for the server only with `IRONMULE_GGML_TRACE`: `IRONMULE_GGML_PAUSE` (per call: thread
CPU time, minor and major faults, run delay, cgroup throttling; for calls of 1 s or more, buffered samples every 5 ms
of the thread's state, system call with its file and wait channel; one `py-spy dump --native` past a set time) and
the llama.cpp patch TL1 (`IRONMULE_TL1`: per ubatch the host time for graph build, inputs, launch and output copy; no
added synchronisation; buffered). `lab/pf3/pauses.py` joins them and classifies a call's excess as compute, CPU
wait, driver call, other synchronisation, page faults or open. The CUDA API calls around a gap are not recorded (no
CUPTI in the timed server); a driver call shows as an ioctl on /dev/nvidia*.

### 2026-11-24, A100 pod 33: PF3 measured -- PX1 kept (on by default), no pause in 80 large prefill calls
RunPod A100 SXM4 80 GB (community, 0.818 h, 1.14 USD; balance afterwards 41.29 USD), Qwen3-32B Q4_K_M, PD1 on, RF1
and PK1-C off, at bff6984. Session `lab/pf3/a100_pf3.sh` with scope, stop rule and a 2.0 h watchdog fixed before
the pod (not reached, not changed); raw data in `lab/pk1/results/pod33/raw`. nvidia-smi every 200 ms: during the
HTTP arms the SM clock's median was 1,410 MHz (10th percentile 1,290-1,305 MHz) in every arm alike; during the dense
in-process prefills 1,305 MHz at the 400 W cap. Runtimes are not normalised.

Pause diagnostics (three server starts with the probe and TL1; the first ran the quality gate, 1,224 s): no pause.
The three first waves took 3.80, 3.70 and 3.58 s, their large calls 2.22-2.34 s for 7,122-7,599 rows in 2 passes.
In every large call the thread was running in every sample (`R`, no system call): llama_decode returns from the
launch after 0.36-0.43 s and then spins in the output copy's synchronisation for 1.07-1.13 s while the GPU computes;
CPU time 1.57-1.83 s user and 0.64-0.69 s system per call; no major faults, no run delay, no cgroup throttling. No
call in any trace of this pod (80 calls of 300 rows or more, HTTP and in-process) exceeded its rows' expectation by
more than 0.2 s. So the 1.2-2.0 s pauses of pod 32 did not recur, and their cause stays open; the probe is ready
(`IRONMULE_GGML_PAUSE`, `IRONMULE_TL1`, `lab/pf3/pauses.py`) for the next pod where one occurs.

PX1 in one engine process (`lab/pf3/px1_ab.py`; the benchmark's 32-request repeats 1 and 2, each after a primer of
the repeat before with the KV cleared first, so both arms find the same kept prefixes; 12 ABBA pairs, 32-token
answers, no profiler):

| | passes (reference -> PX1) | rows recomputed | prefill call | wave (prefill + decode) |
| --- | --- | --- | --- | --- |
| repeat 2, 6 pairs | 7 -> 2 | 99 | 2,405-2,413 -> 2,310-2,316 ms | 3.357-3.365 -> 3.262-3.271 s, -87 to -98 ms in every pair |
| repeat 1, 6 pairs | 5 (2 possible) | 99 not allowed | unchanged | -2 to +20 ms (noise) |

- Repeat 1 was not evened: 99 recomputed rows exceed the 32-rows-per-saved-pass limit (96 for 3 passes). By the
  pod-32 fit the break-even is about 64 rows a pass (21.5 / 0.335 ms), so a looser limit would likely gain about 30
  ms there too; noted post hoc, not changed in this run.
- Answers: 378 of 384 identical; the six differences are one request (repeat 2, request 19) diverging at its third
  token in every PX1 wave, deterministic (the recomputed prefix is computed in a different batch).
- Stop rule (median paired saving >= 40 ms, faster in >= 10 of 12 pairs): 53.7 ms, 11 of 12. HTTP followed.

HTTP, same pod, server starts alternating reference (PX1 off) / PX1 / reference / PX1, then vLLM 0.11.0 (AWQ-Marlin),
the same prompts at the same position everywhere:

| 32 clients | wave 0 | wave 1 | wave 2 | first token p50 |
| --- | --- | --- | --- | --- |
| reference R1 / R2 | 3.630 / 3.535 s | 3.701 / 3.670 s | 3.758 / 3.738 s | 2.571-2.672 s (median 2.617) |
| PX1 P1 / P2 | 3.593 / 3.585 s | 3.594 / 3.549 s | 3.691 / 3.509 s | 2.471-2.603 s (median 2.512) |
| vLLM | 3.669 s | 3.767 s | 3.661 s | 1.788-2.102 s (max 2.94-3.05 s) |

- Median 32-client wave over both starts: 3.589 s with PX1 against 3.686 s (-96 ms, -2.6 %); later waves only 3.572
  against 3.720 s (-148 ms); first waves unchanged (3.589 against 3.583 s mean). Throughput median 285.3 against
  277.9 tok/s (vLLM 279.1). First token p50 -105 ms. Facts 18/18 in all four starts and in vLLM.
- In the HTTP later waves PX1 evened 9-11 passes to 2 (extra rows 60-107) and the large prefill call took 2,221-2,318
  ms against 2,300-2,499 ms.
- 8 clients (two waves each): reference 1.479 / 1.394 and 1.494 / 1.390 s, PX1 1.451 / 1.402 and 1.479 / 1.397 s, vLLM
  1.382 / 1.356 s: no PX1 effect (the 8-request groups were not fragmented enough to qualify) and IronMule is 3-7 %
  behind vLLM here.
- Release rule (median 32-client wave >= 40 ms shorter, first token not later by more than 5 %, facts 18/18): met.
  PX1 is on by default (`IRONMULE_GGML_PX1=0` turns it off).

Against +10 % over vLLM (target wave = vLLM wave / 1.10, same prompts): PX1 leaves 0.25 s (wave 0), 0.15 s (wave 1)
and 0.27 s (wave 2) at 32 clients, and 0.17-0.21 s at 8 clients. Supported reserve for the next round: the admission
path (40-60 ms, both widths), the PX1 row limit (about 30 ms where 3 passes are at stake), the GEMM algorithm or
workspace (0.03-0.06 s); together still below the remaining gap, so a larger mechanism is needed (the decode part of
the wave, about 1.0 s, and the first waves' 2-pass prefill).

### 2026-11-25, before the pod: where the 32-client wave loses time outside the GPU -- ADM1, ADM2, BF1, PX2
Maintainer work towards +10 % over vLLM at 32 clients (reference 584ec37: IronMule 285.3, vLLM 279.1 tok/s on pod 33;
remaining gap 0.15-0.27 s a wave). Local analysis of pod 33's raw data and a local reproduction; no GPU time.

One wave on one time axis (`lab/adm1/anatomy.py`, `lab/adm1/tail.py`; pod 33, PX1 arms, 32 clients, ms from the first
send; the client sends all 32 requests within 17-25 ms):

| part | P1 / P2 waves | ideal |
| --- | --- | --- |
| first send to the large prefill call | 142-185 (unsplit waves) | about 30 |
| large prefill call | 2,221-2,368 | -- |
| prefill end to first decode call | 6-9 | -- |
| 31 decode steps, start to start | 31.0-31.8 a step (native call 29.8-30.5) | -- |
| last decode call's end to the last response's end | 47-85 | a few |

- Admission (`lab/adm1/adm.py` over the stability log): the service admitted the burst over 80-128 ms, its first
  requests 10-30 ms apart. Every chat request ran `ProductService.health()` (the optimizer status and the settings,
  read under file locks) and appended to the stability log with pathlib's exists, stat and open, on the Python path
  that 32 HTTP threads, the scheduler and the worker's reader share. Local reproduction (`lab/adm1/stub_server.py`:
  the real HTTP server, ProductService, scheduler and join path, a stub worker; the benchmark's own client): the burst
  reached the worker over 40-45 ms on 4 cores, 15-18 ms with the server pinned to one core (contention for the GIL,
  not work alone). The 4-core container is not the pod; the pod had 192-256 visible cores.
- The wave's end: the engine finished all 32 requests 10-13 ms after its last call (`# job` end times), but the
  service read their done frames 1.6 ... 52-85 ms after it (`member_done`), each request's last chunk 0.5 ms before.
  Every finished request costs about five log lines (member_done, http, end, the serving log, and the frames' JSON)
  and two thread hand-offs on the same contended path. Stub: removing the logs halved the tail (30 -> 15 ms).
- Host batch build of the large prefill call: 6-16 ms (`b=` in the trace), a Python loop over ctypes arrays.
- PX1's fixed limit against a cost decision, offline over pods 32/33's 48 large HTTP calls (`lab/adm1/px2_offline.py`,
  the fitted pass costs below): PX1 already evened nearly every fragmented call (6-14 passes to 2); a cost decision
  adds 1 ms a call on average (one call 42 ms). The 30 ms expected for PX2 holds for the in-process repeat 1 (5
  passes, 99 rows), not for the benchmark's HTTP waves.
- Pass costs fitted over 186 calls of pods 32/33 (pause calls out): a pass under PK2's 128-row threshold 16.5 ms +
  0.98 ms a row, a cuBLAS pass 91 ms + 0.287 ms a row (residual median 11.6 ms).

Changes, each with its own switch (all on by default until the pod decides; `IRONMULE_LAB_SWITCH` flips them inside
one server start for paired waves):
- ADM1 (`IRONMULE_ADM1`): a chat request asks `ProductService.ready()` (the same answer as health()["ready"], no
  file reads); log lines are appended with one open, fstat and write. Stub: last request at the worker 41 -> 20 ms.
- ADM2 (`IRONMULE_ADM2`): the stability and serving logs are written behind by one thread per log, every waiting
  line in one call; `read()`, `flush_logs()` and the service's close flush. Stub, with ADM1: tail 30 -> 19 ms.
- BF1 (`IRONMULE_GGML_BF1`): calls of 64 rows or more fill llama.cpp's batch with numpy copies (the rows point into
  one contiguous sequence-id array; llama.cpp's own arrays are given back before its free). Same tokens (CPU build,
  Qwen3-0.6B, 18 calls of 64+ rows); 500 rows 0.23 -> 0.07 ms on the container.
- PX2 (`IRONMULE_GGML_PX2`): a packed engine times one-sequence passes of 1, 8, 16, 32, 127, 128, 256 and 1,024 rows
  at its start (logits on the last row only) and prices a split as the sum of its passes' interpolated times; every
  kept-prefix level of the group is priced with the passes it would really make, the cheapest taken when it saves
  2 ms. Without the measurement PX1's fixed limit decides.

Larger lever, chosen by the remaining milliseconds (not more than three candidates):

| mechanism | evidenced share of a wave | expected net saving | effort | decisive test |
| --- | --- | --- | --- | --- |
| server path: admission and wave end (ADM1 + ADM2) | 110-150 ms before the prefill, 45-80 ms after the last call | 80-150 ms (stub 79 -> 39 ms) | low | paired waves in one server start |
| one large prefill pass instead of two (matmuls over all rows, attention per pass shape) | 91 ms a cuBLAS pass | about 90 ms | high (llama.cpp graph and KV views) | engine prefill of the benchmark mix |
| decode host time between native calls | 1.0-1.5 ms x 31 steps | 15-30 ms | medium | decode period at 32 rows |

Speculative decoding stays out: the verification path was measured already (pod 17: 64 rows 56.4 ms = 1.86 x a
32-row step; draft acceptance 0.66 at k = 1), which leaves no reserve at 32 clients. The server path is chosen: the
largest block with the smallest effort, and its pieces are measured per wave on the pod.

### 2026-11-25, A100 pod 34: ADM1 and ADM2 kept -- 312.1 against 293.4 tok/s at 32 clients (+6.4 %), +10 % not reached
RunPod A100 SXM4 80 GB (secure cloud: no community stock; 2.19 USD/h, 0.794 h, about 1.74 USD; balance afterwards
39.60 USD; 192 visible cores, power limit 500 W), Qwen3-32B Q4_K_M, builds d371f98 (code) and c4001f6 (session), phase E with DH1 at b1f33ea,
vLLM 0.11.0 AWQ-Marlin. Session `lab/adm2/a100_adm2.sh`, scope, rules and a 2.5 h watchdog fixed before the pod (not
reached, not changed); phase E (`lab/adm2/a100_e.sh`) was added inside the same limit. Raw data in
`lab/pk1/results/pod34/raw`. No profiler in any timed wave.

Paired waves in one server start (`lab/adm2/paired.py`, `IRONMULE_LAB_SWITCH`; rotated order, prompts of their own
per wave, one discarded wave per width), 32 clients, wave seconds and the paired difference to the round's reference:

| phase | arm | median wave | tok/s | paired vs reference | first token p50 | 8 clients (4 rounds) |
| --- | --- | ---: | ---: | --- | ---: | --- |
| A1 | R (584ec37: ADM1, ADM2, PX2, BF1 off) | 3.418 s (3.318-3.549) | 299.6 | -- | 2,406 ms | 1.334 s, 18/18 |
| A1 | A (ADM1) | 3.288 s (3.269-3.307) | 311.5 | -138.7 ms, 8 of 8 | 2,300 ms | 1.347 s, 18/18 |
| A1 | P (PX2) | 3.419 s (3.320-3.534) | 299.5 | -45.1 ms, 5 of 8 | 2,397 ms | 1.341 s, 18/18 |
| A1 | AP | 3.290 s (3.266-3.306) | 311.3 | -132.2 ms, 8 of 8 | 2,307 ms | 1.332 s, 18/18 |
| A2 | B = A | 3.312 s | 309.2 | -- | 2,311 ms | 1.413 s, 18/18 |
| A2 | C = A + ADM2 + BF1 | 3.276 s | 312.6 | -33.1 ms, 8 of 8 | 2,296 ms | 1.410 s (-3.8 ms, 4 of 4), 18/18 |
| E | F = C | 3.294 s | 310.9 | -- | 2,294 ms | -- |
| E | F + PX2 | 3.278 s | 312.4 | -9.2 ms, 11 of 12 | 2,286 ms | -- |

- By the rule A passed, P did not (5 of 8; its waves spread like R's), AP's saving was smaller than A's: B = A. C
  passed against B. Without ADM1 the 32-client waves spread over 0.23 s (split bursts); with it over 0.04 s.
- Admission and the wave's end (`lab/adm1/anatomy.py`, the fresh starts below): the large prefill call started
  27-61 ms after the first send with F against 58-116 ms with R (pod 33: 142-185 ms); the last response ended
  20-41 ms after the last native call (pod 33: 47-85 ms; R on this pod: 16-39 ms).
- PX2's pass costs on this pod (ms, one sequence): 1 / 8 / 16 / 32 rows 19.4 / 21.1-23.0 / 21.9-23.5 / 28.9-29.2,
  127 rows 125.0-125.7, 128 rows (cuBLAS) 121.4-121.7, 256 rows 146.8-147.7, 1,024 rows 382.7-383.3: about 68 ms
  fixed and 0.31 ms a row for a cuBLAS pass. PX2 stays on (rule: not 10 ms slower; phase E: -9.2 ms).
- Decode host time (DH1, `IRONMULE_GGML_STEPPROF`, medians of 31 full-width steps per wave, F): native call
  29.97-30.23 ms, accepting and streaming the 32 tokens 0.53-0.68 ms, pilot and admission 0.07, rows 0.045, the
  rest below 0.01; start to start 30.8-31.0 ms. At most about 20 ms a wave is left there.

HTTP, the existing benchmark, fresh server starts alternating reference / final, then vLLM, the same prompts at the
same positions (median over all 32-client waves, first waves included, as fixed before the pod):

| 32 clients | wave 0 | wave 1 | wave 2 | median tok/s | first token p50 |
| --- | --- | --- | --- | ---: | ---: |
| R1 / R2 | 280.96 / 297.87 | 301.61 / 302.92 | 309.33 / 303.29 | 302.3 | 2,363 ms |
| F1 / F2 (ADM1, ADM2, BF1) | 300.77 / 301.92 | 311.80 / 312.36 | 314.35 / 312.44 | 312.1 | 2,284 ms |
| vLLM | 288.69 | 293.50 | 293.41 | 293.4 | 1,843 ms (max 2.78-2.83 s) |

- F against vLLM: 1.064x (+6.4 %); F against R: +3.25 %. Remaining gap to vLLM's wave / 1.10: 108 ms (median wave
  3.281 against 3.173 s); by position 174 / 109 / 95 ms. IronMule's slowest first token is earlier than vLLM's
  (2.28-2.41 against 2.78-2.83 s), its median later (vLLM's chunked prefill hands out first tokens in groups).
- 8 clients: F 186.9 against R 183.1 tok/s (+2.0 %; waves 1.395 / 1.346 / 1.390 / 1.350 against 1.439 / 1.358 /
  1.456 / 1.359 s), vLLM 195.3 (F 4.3 % behind). Facts 18/18 in every start of every server.
- Not run: the diagnostic py-spy of the worker (ptrace refused in this container, `Permission denied`).

Where the 108 ms remain (F, later waves: about 30 ms to the prefill, 2,236 ms prefill, 8 ms, 31 x 31.0 ms decode, 22
ms to the last response): no server-side block of that size is left. The largest single item is the second large
prefill pass of every wave (short prompts in one pass, long ones in a second, as llama.cpp's split over one KV
stream per sequence requires): about 68 ms of fixed cost (dequantisation and per-pass work) measured here. The
missing solution is a prefill whose matmuls run once over all rows of both passes while attention keeps the two
per-stream shapes (a llama.cpp graph and KV-view change), worth about 68 ms; with the remaining decode host time
(about 20 ms) and wave end (about 15 ms) that still leaves the decode step itself, 30.0 ms at 32 rows against
vLLM's 22.7 ms step period, as the lever that closes the rest.

### 2026-11-25, before pod 35: the starting point, PS1 (one native pass for a packed prefill) and DH2 prepared
Starting point, two configurations (Qwen3-32B Q4_K_M, A100 SXM4 80 GB, pod 34):
- Measured over HTTP as "F" (+6.4 % against vLLM, 312.1 against 293.4 tok/s): `IRONMULE_ADM1=1 IRONMULE_ADM2=1
  IRONMULE_GGML_BF1=1 IRONMULE_GGML_PX2=0`, everything else at its 11f00e4 default.
- The 11f00e4 default: ADM1, ADM2, BF1 and PX2 on. PX2 was measured on top of F only in paired waves (-9.2 ms, 11 of
  12), not in the fresh-start HTTP benchmark. Gap of F to vLLM / 1.10, paired by position: 174 / 109 / 95 ms.
  Pod 35's reference arm R is this default; its final arm runs the default build with nothing set by environment.

PS1 (`native/ggml_patches/ps1`, every build; `IRONMULE_PS1`, read by the native library on every call, set by the engine
from its own default): llama.cpp's KV cache (unified-cache class, multi-stream mode) merges the rectangular ubatches
that one decode call's sequential split made into one ubatch, after `prepare()` placed every rectangle as before. The
graph then runs embedding, norms, projections, RoPE, FFN and the output once over all rows; attention runs per
rectangle on its rows of Q (a contiguous slice), its streams' K / V views and its own mask (filled after every
rectangle's cells were placed, so a stream split over two rectangles sees its earlier tokens and not its later ones).
With flash attention the rectangles write their outputs into one tensor of all rows (the FA results are views of it,
no copy); `IRONMULE_PS1_CONCAT=1` joins them with `ggml_concat` instead (diagnostics). Attention work is the same as
with separate passes (the same rectangles); the merged ubatch never exceeds n_ubatch; graphs are not reused across it.
Merging needs an untransposed V cache (flash attention) and `ggml_set_rows`; elsewhere nothing changes. With PS1 on,
PX1/PX2 no longer cut kept prefixes: the trace notes the prefix rows kept, the rows computed and the rectangles.
- CPU build, Qwen3-0.6B Q4_K_M, 8 KV streams with prompts of 25-140 tokens and kept prefixes of 0-12, one packed call
  of 616 rows (merged into 6 + 2 rectangles at n_ubatch 512) and 8 decode steps (`lab/ps1/check.py`): logits of every
  output row bit-identical to separate passes (max |diff| 0.0), with the view and the concat variant; tokens equal.
- The patch applies to a fresh 0.3.16 tree with every other patch (CPU and CUDA sets).
- Hypothesis for the GPU, to be confirmed: one cuBLAS pass of fixed cost (about 68 ms on pod 34, mostly the Q4_K ->
  fp16 dequantisation per pass) less a wave, and the 60-160 template rows PX1/PX2 recompute kept instead.

DH2 (`IRONMULE_GGML_DH2`, GPU engines): during a run the tokens' delivery (the worker's `_BatchStream.on_token`:
detokenizer, frame, queue; 0.53-0.68 ms a 32-row step on pod 34) goes to an emitter thread that runs while the next
native call holds no GIL, in order per request; a request's last token (its done frame reads engine state) is
delivered on the engine thread after the emitter drained; the run drains it before it returns. The worker's
detokenizer pool takes a lock. Upper bound about 0.6 ms x 31 steps = 19 ms a wave.

The decode path from logits to token, checked first (pod 33 TL1 timestamps, 205 calls of 32 rows): AM1 already picks
the argmax on the GPU and copies one index per output row; inside the native call 0.10 ms split, 0.21 ms inputs
(the KQ mask), 0.61 ms graph launch, 28.84 ms waiting for the GPU and the copy; after it 0.75 ms of Python (DH1). No
further logit transfer to remove; DH2 takes the largest part that can overlap without a new graph structure.

Session `lab/ps1/a100_ps1.sh` with its rules fixed in the file before the pod.

### 2026-11-25, A100 pod 35: PS1 measured (-90 ms a wave in one process, outliers over HTTP: rejected by the rule), DH2 no gain; +7.7 % against vLLM
RunPod A100 SXM4 80 GB (secure: no community stock; 2.19 USD/h, 1.263 h, 2.77 USD; balance afterwards 36.69 USD; 128
visible cores, 400 W), Qwen3-32B Q4_K_M, vLLM 0.11.0 AWQ-Marlin. Session `lab/ps1/a100_ps1.sh` (rules fixed in it
before the pod), watchdog at 2.5 h (not reached, not changed). Three runs inside the limit, each reason recorded; raw
data in `lab/pk1/results/pod35/raw` (`run1`, `run2`, `run3`, `h`).
- Run 1, C: the first GPU check ran out of graph objects (`GGML_ASSERT(obj_new)`: 8 rectangles x 64 layers above
  llama.cpp's 8 nodes a tensor). Fixed (24 with `IRONMULE_PS1=1`, at most 16 rectangles merged; 13f23c5), rebuilt on
  the pod. Then, 8 KV streams, 616 rows of unequal prompt and kept-prefix lengths: PS1 against the concat variant
  bit-identical (max |diff| 0.0: the in-place outputs are right on CUDA), against separate passes max |logit diff|
  0.47 (MMQ against cuBLAS for the small passes), top-1 8 of 8, tokens equal; the call 0.670 -> 0.215 s.
- Run 1, E (one process, the benchmark's prompts after a primer, 10 rotated rounds, no profiler): PS1 -89.9 ms a
  32-client wave in 10 of 10 rounds (wall 3.319 -> 3.227 s, prefill 2,380 -> 2,287 ms), prompt rows 7,472 -> 7,373,
  rows reused 160 -> 259, answers identical in 315 of 320.
- Run 1, A (paired HTTP waves, one start, lab switch): PS1 +50.6 ms (3 of 8); 5 of 13 merged prefills took 2.83-2.86 s
  instead of 2.29-2.32 s (the same row counts ran both ways). Suspected: the compute buffer regrown at every new
  maximum, since each rectangle's mask makes the need depend on the shape. Run 2 let an existing buffer grow to 1.5x
  its need (ggml-alloc.c): the worker crashed with SIGSEGV at its first request; withdrawn, not investigated further.
  Run 3 grows the buffer at the engine's start with one merged prefill of 32 different lengths (8,165 rows, 2.57 s).
- Run 3, A (8 rounds of 32 clients, 4 of 8; arms on top of the 11f00e4 default):

| arm | median wave | paired vs R | first token p50 | 8 clients: paired vs R | facts |
| --- | ---: | --- | ---: | --- | --- |
| R | 3.455 s | -- | 2,464 ms | -- | 18/18 |
| S (PS1) | 3.383 s | -82.5 ms, 5 of 8 | 2,398 ms | -49.9 ms, 4 of 4 (first token 697 against 770 ms) | 18/18 |
| D (DH2) | 3.462 s | +0.6 ms, 4 of 8 | 2,452 ms | +24.1 ms | 18/18 |
| SD | -- | -68.7 ms, 7 of 8 | -- | -- | 18/18 |

  3 of 16 merged 32-client prefills still took 0.45-0.55 s more (2,599 / 2,778 / 2,825 ms; 7,423 rows ran in 2,310 ms
  in another wave), not decreasing over the rounds; the cause is not shown. By the rule S failed (5 of 8), D failed,
  and SD may only be taken when both pass: F = R. PS1 and DH2 are off by default now.
- H (fresh starts, the existing benchmark, first waves included), replacing the script's H once F = R: R (PS1, DH2 off:
  the default after this pod) against P (PS1 and DH2 on), then vLLM:

| 32 clients | waves (tok/s) | median | vs vLLM | gap to vLLM / 1.10 (median wave; by position) |
| --- | --- | ---: | ---: | --- |
| R | 295.37 / 300.38 / 299.99, 293.11 / 301.93 / 301.78 | 300.2 | +7.7 % | 72 ms; 148 / 61 / 10 ms |
| P | 261.11 / 304.65 / 304.75, 262.91 / 300.16 / 305.47 | 302.4 | +8.5 % | 47 ms; 576 / 47 / -37 ms |
| vLLM | 279.34 / 278.81 / 274.40 | 278.8 | -- | -- |

  8 clients: R 180.5, P 189.2, vLLM 186.5 tok/s; first token p50 at 32: R 2,411, P 2,369, vLLM 1,607 ms (vLLM's slowest
  first token is later); facts 18/18 everywhere. P's first wave after a start is 0.5 s slower (the merged prefill's
  first shapes), its later waves are 1.6-1.7 % faster than R's and at position 2 above +10 % over vLLM.
- What blocks +10 % now: PS1 saves about 90 ms a wave where its prefill runs as expected, which is most of the
  remaining 72 ms; it fails on the merged prefills that take 0.45-0.55 s more, the first after a start and some
  later ones. The missing piece is a merged graph whose memory need does not depend on the rectangles (one mask
  tensor of fixed size for all of them, or the compute buffer reserved for the largest merged shape at the start),
  measured on the GPU first; then the same rule again.

### 2026-11-25, before pod 36: what the slow PS1 prefills have in common, TL2, and the start-up warm-up of the full micro-batch
From pod 35's raw data (local, no GPU time). Every merged 32-client prefill of runs 1 and 3 and of the H stage, in the
order the server ran them (rows = the merged call's prompt rows; "max before" = the largest merged call of the process
so far, the start's warm-up included):

| call | rows | max before | prefill |
| --- | ---: | ---: | --- |
| run 3 S0 (first PS1 wave) | 6,943 | about 6,100 (warm-up: 32 lengths merged as 16 + 16) | 2,599 ms, slow |
| run 3 S1 | 7,423 | 6,943 | 2,778 ms, slow |
| run 3 S2 | 7,382 | 7,423 | 2,308 ms |
| run 3 S7 | 7,423 | 7,423 | 2,310 ms |
| run 3 SD6 | 7,435 | 7,423 | 2,825 ms, slow |
| H P1 / P2, first wave | 7,629 | about 6,100 | 2,817 / 2,813 ms, slow |
| H P1 / P2, later waves | 7,344-7,373 | 7,629 | 2,266-2,278 ms |

Every slow call, the first waves after a start and the later ones alike, was the process's first merged call above all
earlier ones; no call below the maximum was slow (the same 7,423 rows ran in 2,778 and 2,310 ms). The reference never
exceeds 6,576 rows in a pass after its start and had no such call. So both cases have one trigger in these data; what
grows there, and where the 0.45-0.55 s go, is not shown by them (no TL1 ran on pod 35).

Prepared for the pod:
- TL2 (`native/ggml_patches/tl2`, `tl2c`; diagnostics, nothing without `IRONMULE_TL1` / `IRONMULE_TL2`): the TL1 line
  of every ubatch adds the graph's construction apart from the scheduler's allocation, the thread's CPU time and a
  merged ubatch's rectangles; every compute-buffer reallocation (ggml-alloc) and every CUDA pool growth (VMM:
  cuMemCreate / cuMemMap / cuMemSetAccess; legacy: cudaMalloc) is written with its sizes and host microseconds.
- `lab/ps1/growth.py`: one engine process, packed prefills of 8, 16, 16, 24, 24, 32, 32, 24, 32 fresh prompts (every
  rise a new maximum, every repeat a known shape), TL1/TL2 per wave, CUPTI for chosen waves (PF2's summary: kernels,
  busy time, gaps, first-time CUDA API calls).
- `IRONMULE_PS1_WARM` (with PS1 on): "2" (default) one merged call of the full micro-batch in 16 rectangles at the
  start, the largest merged call the engine can make (8,192 rows; CPU build: one merge of 16 rectangles, checked);
  "1" run 3's form; "0" none. PS1 stays off by default.
- Session `lab/ps1/a100_ps1b.sh`, stages d1 / d2 / d3 / h, rules in the file.

### 2026-11-25, A100 pod 36: PS1's slow prefills attributed and removed (PS1a), PS1 on by default, +9.2 % against vLLM
Maintainer session on one RunPod A100 SXM4 80 GB (secure cloud, US-WA-1, 32 vCPU; the community tier had no stock),
Qwen3-32B Q4_K_M GGUF, vLLM 0.11.0 with the official AWQ checkpoint (AWQ-Marlin). Build: cdd45ad's patch sets plus
PS1a below (the pod's tree was rebuilt in place after D1; 605bfef holds the same files). DH2 off in every arm.
Raw data: `lab/pk1/results/pod36/` (`ps1b_all.tgz`: every stage, TL1/TL2, traces, HTTP rows, vLLM; `d1.tgz`, `d2.tgz`,
`ps1b_d1.log`). Cost 1.45 h at 2.19 USD/h, 3.18 USD; pod deleted, watchdog (2.5 h) not reached.

**D1, the attribution** (`lab/ps1/growth.py`, one engine process, waves of 8, 16, 16, 24, 24, 32, 32, 24, 32 fresh
prompts, one packed prefill each, TL1/TL2 on). PS1 without the start's warm-up, the first call of each new size:

| call | rows | prefill | TL1: build (graph / allocation) | TL2 | GPU wait (out) |
| --- | ---: | ---: | --- | --- | ---: |
| first 32-request call | 7,632 | 2,761.5 ms | 386.8 ms (0.96 / 385.8) | CUDA0 compute buffer 14.02 -> 18.69 GB, 384.7 ms | 1,271 ms |
| the same shape again | 7,632 | 2,372.6 ms | 2.3 ms | none | 1,270 ms |
| first 24-request call | 5,724 | 1,797.7 ms | 11.2 ms | 9.35 -> 14.02 GB, 9.1 ms | 955 ms |

The whole extra time of the slow call is the scheduler's allocation: ggml-alloc freeing the 14 GB compute buffer and
allocating 18.7 GB (host time, 23 ms of it CPU; GPU execution unchanged, launch 1,088 against 1,085 ms). The buffer
grew by 2.2 MB for every merged row (5.15, 9.35, 14.02, 18.69 GB at 1,908 / 3,816 / 5,724 / 7,632 rows), 2.1 MB of it
exactly 64 layers x 8,192 floats: every layer's attention output at once. `CUDA_MODULE_LOADING=EAGER` changed nothing
(2,777 ms); CUPTI on the 24-request call showed one 851 ms stream synchronisation, the GPU's own work. The prepared
warm-up (one merged call of 8,192 rows at the start, 2.47 s) removed the slow calls in D1, but in the first server
start the start-up gate's W4R tuning then died with `out_of_memory` (SIGSEGV) next to the 18.7 GB buffer; that run
was discarded (`d2_failed/`).

The mechanism (ggml-alloc.c, `ggml_gallocr_alloc_graph_impl`): every graph leaf is allocated before the first node,
for the whole graph. PS1's per-layer tensor that the segments' flash attention writes into (`ggml_new_tensor_2d`, no
op) is a leaf, so all 64 were placed together. A first change (the segment outputs chained, so that their views
release it) left the growth byte for byte unchanged; with PS1's concat fallback (no such tensor) the buffer did not
grow at 3,816 rows (1,216 ms). **PS1a** (`native/ggml_patches/ps1a`, after tl2): a leaf named `ps1_all-<layer>` is
not pre-allocated; ggml-alloc places it at its first reader (the allocator's parent loop already does that) and, with
the chain, frees it after the layer's projection reads it. CPU build, Qwen3-0.6B, 16 streams with unequal prompts and
kept prefixes: logits and tokens identical to PS1 off (max |difference| 0.0). A100, the same D1 waves: no compute-buffer
growth after the start at all (TL2 lists only CUDA pool growths under 2 ms), first 32-request call 2,379 ms, its
repeats 2,380 / 2,382 ms; the reference on the same calls 2,445 ms. The warm-up is no longer needed (default "0").

**D2, the server** (PS1 on from the start, warm-up 0, trace, TL1, TL2; first start with the gate: 1,223 s, gate
passed, W4R ratio 0.994; 62.1 GB after ready): after ready one CUDA pool growth (0.25 ms) and no compute-buffer
growth; the allocation section of every merged call 0.7-3.3 ms. Every 32-client prefill after ready: S merged calls
of up to 23 rectangles with kept prefixes 2,281-2,322 ms, R's two-pass calls 2,387-2,402 ms, the benchmark's first
call (7,629 rows, no kept prefix yet) 2,359 ms. Paired R / S, 8 rounds of 32 clients: S -75.8 ms, 8 of 8.

**D3, decisive** (fresh start, PS1 on, no trace; paired waves through the lab switch, the rule fixed in
`lab/ps1/a100_ps1b.sh`): S -80.8 ms per paired 32-client wave, faster in 8 of 8 rounds; 8 clients -53.4 ms (wave
ratio 0.950, 4 of 4); 32-client first-token p50 ratio 0.969; facts 18/18 in both arms. S kept.

**H, fresh starts R1 S1 R2 S2 (trace on, as in every H), then vLLM.** Output tok/s, every wave, first waves included:

| arm | 32 clients (wave 1 / 2 / 3, two starts) | median | first-token p50 | 8 clients (median, facts) |
| --- | --- | ---: | ---: | --- |
| S (PS1 + PS1a) | 292.4 / 300.5 / 301.2, 291.5 / 301.5 / 299.6 | 300.0 | 2,360 ms | 194.3, 18/18 x 4 |
| R (PS1 off) | 284.3 / 289.2 / 291.4, 286.9 / 295.1 / 293.2 | 290.3 | 2,435 ms | 183.0, 18/18 x 4 |
| vLLM | 276.5 / 273.0 / 274.8 | 274.8 | 2,049 ms | 187.3, 18/18 x 2 |

- S against vLLM +9.2 % (300.0 against 274.8): +10 % not reached. Remaining gap: median S wave 3.413 s against vLLM's
  3.726 s / 1.10 = 3.387 s, 25.7 ms; by position 139.7 / -8.0 / 21.7 ms. R against vLLM +5.6 %. S against R +3.3 %
  at 32 clients, +6.1 % at 8 clients (8 clients: S ahead of vLLM by 3.7 %, R behind by 2.3 %).
- Startup (85 s) and memory after ready (61,579 against 61,569 MiB) the same for S and R.
- No slow call: every S prefill within 1.15 x its position's median (first 2,356 / 2,367, later 2,288-2,305 ms);
  R's 2,370-2,435 ms.
- Where S's first wave loses its 100 ms against its later waves (trace and client timeline): 35 ms from the first send
  to the prefill call (79-84 against 39-49 ms) and 65 ms in the prefill (7,629 rows without a kept prefix against
  about 7,390). Decode (31 calls, 1,016-1,032 ms) and the tail after the last call (28-48 ms) are the same in every
  position. vLLM's first wave is not slower than its others.
- The first start of a pod with PS1 now passes the gate; the 18.7 GB buffer was also what made the warm-up variant
  fail it.

Default since this session: `IRONMULE_PS1=1`, `IRONMULE_PS1_WARM=0`, DH2 off, patch set ps1a in every build: the
configuration measured as S. Next: the first wave's 35 ms before the prefill and its 65 ms of rows without a kept
prefix are the largest single remainder (140 ms of the first wave's gap); the later waves are within 22 ms.

### 2026-11-25, before pod 37: what sets the median wave, and two small levers (PK6, SV5)
Starting point 02b285b (PS1 + ps1a, warm-up 0, DH2 off). Pod 36: 300.0 against vLLM's 274.8 tok/s; +10 % needs
302.28 (+0.76 %), 25.7 ms a median wave. Local analysis of pod 36's raw data, no GPU time.

- Which waves decide: the six 32-client waves sorted are 291.5 / 292.4 (the two first waves) / 299.6 / 300.5 / 301.2
  / 301.5; the median is the mean of the third and fourth, both later waves. A faster first wave would not move it
  unless it overtook them; the later waves need about 26 ms.
- A later wave (3.39-3.42 s), from the client timeline, the engine trace and the stability log: first send to the
  prefill call 39-49 ms; prefill 2,288-2,305 ms; to the first decode call 8 ms; 31 decode calls of 31.3 ms (980 ms)
  and 30 gaps of 1.0-1.2 ms (28-37 ms); after the last call 28-48 ms (the 32 done frames handled about one a
  millisecond; locally 0.85 ms a frame, mostly thread hand-offs, frames of 7.8 KB; no small fix found).
- Before the prefill (later waves): the client's last request is sent at 14-17 ms, admitted at 25-30 ms; the worker
  receives 11-31 of the requests as one join at 30-36 ms and renders them one by one in 7-12 ms (the batch's first
  requests already go through SV4's batched rendering, joins did not), then 3-5 ms to the call. In the first wave the
  admission ends later (51-60 ms) and 16-24 ms more pass between the decision and the call (a first-time cost there).
- The prefill (pod 36 CUPTI, 5,724 rows): GEMM 1,431 of 1,757 ms of kernels (about 250 TFLOPS, 89 % of the
  clock-adjusted peak), SwiGLU 78, norms 66 (the fused add + RMS norm 42), attention 61, weight dequantisation 59,
  rope 26, activation conversion 20, KV writes 13 ms. Lever A as asked: a cuBLAS algorithm or workspace choice stays an
  unmeasured 30-60 ms (pod 32's estimate); a dead write in two of the bandwidth-bound kernels is a sure one.

Prepared (each with its own switch, read while running, so the lab switch can pair them in one server):
- PK6 (`native/ggml_patches/pk6`, `IRONMULE_PK6=1`): PK4 makes SwiGLU and FU1's add + RMS norm write an fp16 copy
  for the next matmul besides their f32 result; in a prefill the matmuls (Q4_K / Q6_K, more than 64 rows: KG1's
  libraries decline them, PK2 sends them to cuBLAS) read only the copy. The f32 write is skipped when every reader of
  the tensor in the graph is such a matmul and nothing in between (matmul, GLU, ADD) can replace PK3's copy; a matmul
  that still misses the copy of a skipped tensor aborts instead of reading unwritten memory. Values unchanged (the
  floats are not read); one GPU split holds the whole model (pod 36: 2 splits, the other the CPU input). Expected
  about -40 ms a later wave (SwiGLU 14 -> 10 bytes an element, -29 ms; the norm about 18 -> 14, -12 ms, at 7,390
  rows), if both kernels stay bandwidth-bound. Not compiled locally (no nvcc here); the CPU build is unaffected.
- SV5 (`IRONMULE_SV5=1`, worker): a join of several requests goes through SV4's batched rendering, checked equal to
  one-by-one rendering the first time. Locally 30 benchmark prompts 7.2-7.4 -> 3.3-3.6 ms; expected about -5 ms.
- Session `lab/pk6/a100_pk6.sh` (stages e, d, h; rules in the file): E checks PK6's logits on the GPU (must be equal)
  and its skips, and times it in one engine process (8 ABBA pairs); D pairs R / PK6 / SV5 / both in one server
  start; H alternates fresh starts R1 K1 K2 R2, then vLLM.
