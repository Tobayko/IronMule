# B39a — B39 parent import-path correction

Experiment ID: B39a
Parent: B39
Registered: 2026-08-28, before any B39a model process
Status: infrastructure correction only; no B39 repetition

The first B39 pilot attempt used the direct script command
`python research/b39_combined_levers.py --experiment-id B39 --pilot`.
That invocation failed before parent initialization with return code `1`:
`ModuleNotFoundError: No module named 'research'` at `research/b39_combined_levers.py:22`.
No model was loaded, no child was started, no final or partial B39 output was
created, and the external crash-report count remained `30 -> 30` with no
residual model process. The failed import is retained as
`B39_pilot_import_failure_20260828.json` and is not a B39 measurement.

This amendment changes only the parent/child invocation path to the module form:
`python -m research.b39_combined_levers --experiment-id B39 --pilot`.
The B39 CLI experiment ID remains `B39`. All B39 arms, model, workload,
warmups, repeats, ordering, timing, correctness, memory, swap, crash, identity,
partial-evidence, pilot and no-activation rules remain unchanged. B39a is the
first authorized hardware attempt after the import-path correction; it does not
repeat or reinterpret B39.
