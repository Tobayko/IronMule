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
attention is one; within 1.005) it passes. The product's gate applies the latter. A decision for the owner.

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
