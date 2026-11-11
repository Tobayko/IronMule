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
