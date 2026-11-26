# Xeon host time outside the two previously profiled files

Two fresh processes served the same four-slot CPU copy workload from frozen SmolLM2-135M
Q4_K_M. The seeds varied, and each process warmed once then served four measured
short/document waves of four clients. The 16/16 answers and 48 generated tokens passed
the serving trial's existing check in each process. Both used the same Xeon 8573C,
affinity 0–3, two native threads, Pilot checkpoint, enabled speculation, native libraries
and engine settings. Bindings, output hashes, per-request TTFT and memory appear in each
raw JSON. Additional paid service cost was USD 0.

| Seed | Wall time, four waves | All additional Python sites | Share |
| --- | ---: | ---: | ---: |
| 68421 | 5.689 s | 8.654 ms | 0.1521% |
| 68422 | 4.768 s | 3.654 ms | 0.0766% |

A new cProfile instance wrapped the unmodified `training.serving_trial.run_wave` call for
each wave. The warm-up was discarded. Each raw `measured_profiles[].sites` row contains
the implementation path, qualified function name, first line, call count and exclusive
seconds (`inlinetime`). Scope covered `ironmule/`, `ironmule_product/` and top-level
`ironmule_*.py`, excluding the engine and tokenizer already in the planner. Only
`ironmule/pilot.py` and `ironmule/speculation.py` were reached in that scope.
Exclusive accounting prevents nested time or native callees from being counted twice.
Any native work charged directly to a Python frame is included conservatively.
The largest individual site was `NgramIndex._index` with 3.127 and 1.011 ms respectively.

The two revision labels differ only in `training/restart_replan.py` and its test.
The complete runtime and measured evaluator sources are byte-identical.
The raw records preserve both source labels. Replaying the two outputs requires
no model or target hardware. From the repository root run:

```sh
PYTHONPATH=. python lab/autonomous/results/xeon-smollm2-broad-host/replay.py
git diff --name-only dceae359 16af87b
```

Even eliminating all measured time in the additional sites would fall short of the
1.9608% elapsed-time reduction required for 2% throughput. No source order was issued.
With two seeds this bounds these observed waves. A policy decision can affect native
compute time indirectly. HTTP product dispatch, other hardware and call shapes require
their own measurements.
