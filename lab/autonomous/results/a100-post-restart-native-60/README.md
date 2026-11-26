# Native work observed below the trained PR 60 worker

Two independent restricted processes on the A100-SXM4-80GB ran the active startup head,
the merged source, 32 mixed completions cold, 32 warm and 18 independently graded facts.
The native callbacks enumerate 193 startup calls in each process. The second process
captured actual CUDA activity on the two cells chosen from the first. The verifier
checks the native binary for both C symbols, retained outputs, work geometry, source
lines, raw logical KV hashes and summed device-event accounting. Native ABI wall is a
host interval that may wait on the GPU. Summed CUPTI intervals may overlap.

During the first observation a separate tokenizer diagnostic attempted to initialize
CUDA. The system records both runs as confounded for timing, learning and authoring.
The common logical setup in measure_pass_costs and measure_padding consumed about
80 ms in the second inventory. The earlier _step_prices call with the same rows ran
inside the protected W4 receipt check. Retaining a logical input and output hash
cannot establish a matching accelerator graph/cache state.

The two samples resolved 3,993 actual CUDA device events. Large cuBLAS/CUTLASS GEMMs
occurred in the observed startup calibration and cold prefill. Emitted symbols also
match project-owned GGML patches for dequantization, SwiGLU and RMS norm. Changing
the GGML image changes the parent release binding and requires independent target
calibration and a freshly qualified release. No elapsed-time credit from this pair
was used for a training label or an author order. The 74.00 s of diagnostics cost
an estimated $0.0453 of the existing lease.

From this checkout, after creating a clean checkout of commit
63db7745998383509602601d46ae4bbff0076648 at /tmp/ironmule-pr60-replay, replay with

```sh
d=lab/autonomous/results/a100-post-restart-native-60
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
OPENBLAS_NUM_THREADS=1 python -m training.post_restart_native_plan \
  --observation "$d/97151.json" --observation "$d/97152.json" \
  --control "$r/control.json" --weights "$r/startup-head.npz" \
  --contract training/post_restart_native_contract.json --checkout /tmp/ironmule-pr60-replay \
  --incident "$d/product-nested-diagnostic.json" --hourly-usd 2.201904761904762 \
  --out /tmp/ironmule-native-diagnostic.json
cmp "$d/decision.json" /tmp/ironmule-native-diagnostic.json
```
