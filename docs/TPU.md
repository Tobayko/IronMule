# TPU decode-loop calibration

IronMule can measure a JAX decode loop on one local TPU chip. The probe uses random bf16 tensors with
Qwen3-1.7B's dimensions, a fixed KV cache and greedy decoding. It measures a compiler scheduling decision.
The HTTP server currently has GGUF and MLX execution paths. A TPU serving backend and real-checkpoint
quality and throughput results remain open.

From a checkout with JAX and the provider's TPU driver installed, run one workload at a time.

```bash
python tools/tpu_calibrate.py --profile-dir .local/tpu --rows 1 --steps 4 --budget-seconds 240
```

The time budget covers JAX initialization, compilation, warmup and evaluation. Reaching it terminates the worker.
Each invocation compares host-driven individual decode steps with the same steps inside a single XLA loop.
It checks every generated row and the final keys and values, then measures four alternating pairs.
The loop must save at least 5% wall time in every pair, and agree on unseen input tokens. The winner executes
on the held-out inputs. A rejected candidate leaves the individual steps selected.

A later invocation restores the measured choice for the matching device, JAX and driver versions,
compiler flags, code and shape. Positive and negative decisions are stored as local JSON. A restored
chained loop is checked again on the held-out inputs. Use `--recheck` to repeat the timings. The full
probe requires an actual TPU. For a bounded CPU test, add `--smoke`. Smoke results are stored separately.

`tools/tpu_step_probe.py` retains the original throughput-ceiling comparison at 1 and 8 rows.
This calibration qualifies only the loop shape it exercised. No result from synthetic weights establishes
a speed advantage over a model server or quality on real questions.
