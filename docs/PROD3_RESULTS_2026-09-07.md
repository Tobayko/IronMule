# PROD3 results — 2026-09-07

The installed product now executes its bounded calibration protocol end to end,
records a verified local history, and independently declines an unqualified
optimization. This is real progress toward automatic optimization, not completion
of RL, online scheduling or deployment qualification.

## Installed execution

The wheel was installed into a separate Python environment outside the checkout.
Imports under `python -I` resolved from its site-packages, with no developer
worktree imports. All 27 product/shared Python files matched the inspected source
at the initial installation checkpoint. Model files remain explicitly registered
local cache data, not imported source code.

Environment: Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3, NumPy 2.5.2,
Transformers 5.15.1; Apple M1 Max, 32 GPU cores, 32 GiB RAM. The project runtime
environment hash did not change during isolated build/tool installations.

Final local regression: 855 tests passed, 16 explicit integration tests deselected;
native model evidence is reported separately below. Static product/evidence/export
checks, Xcode first-launch and diff checks passed.

| Model / attempt | Observed execution | Result |
| :-- | :-- | :-- |
| 1B installed attempt 1 | Stable real readiness; no model load | Failed on the corrected Metal-support string/boolean interface |
| 1B installed attempt 2 | 50 exact HTTP requests, two fresh workers | Rejected at CPU load ratio 0.972119 > 0.8 |
| 4B installed attempt 1 | All 90 exact HTTP requests, three fresh workers | Complete valid calibration; evaluator `inconclusive`, no activation |
| 12B installed attempt 1 | Model loaded; no generation | Rejected on swap growth before the first request |

All owned workers were closed/reaped. Failed attempts remain separate and are not
pooled with the complete 4B result. `tools/product_calibration_export.py` reconstructs
reports from the verified journal; independent reevaluation of the exported 4B
report reproduced its verdict and numbers.

## Complete 4B calibration

Model, hardware, runtime-code and environment identities matched before/after.
All output-token, text, finish/count and resource checks passed. Three distinct
worker PIDs completed and were reaped. Conservative inference-work accounting:
43.221072 s; maximum continuous block 0.657217 s; required breaks 372.568571 s;
additional cooldown sleeps 119.975648 s; measurement wall time 680.841029 s.

These are descriptive ratios, **not accepted speedup claims**:

| Output cap | Median candidate/reference ratio | Worker-cluster interval | Frozen A/A-derived effect floor | Verdict |
| --: | --: | :-- | --: | :-- |
| 1 | 0.993736 | 0.984620–1.008856 | 11.43% | Inconclusive |
| 8 | 0.960858 | 0.923480–0.970654 | 19.51% | Inconclusive |
| 32 | 1.007613 | 1.003072–1.034486 | 11.83% | Inconclusive; early EOS saves no forward |

The intervals use three worker-level ratios, not six independent workers. The
predeclared conservative noise rule is not loosened after seeing the result.
The mechanism saves one model-forward invocation when the cap is consumed; this
does not prove an equal reduction in GPU kernels, bandwidth, FLOPs or end-to-end
time. No broad deployment or performance authorization is issued.

## 12B memory result

Swap rose from 1,698,368,061 B to 4,155,831,746 B after loading the model: a
2,457,463,685 B increase, exceeding the fixed 256 MiB limit. Low CPU load and
a seemingly healthy memory-pressure percentage were not sufficient admission
evidence. The next work item measures loading peak/RSS/MLX allocation and improves
pre-load admission; it does not raise the swap ceiling to make this attempt pass.
Installed 12B generation is still unverified at this checkpoint.

## Sources and boundaries

- [1B initial failure](../research/raw/PROD3_1B_installed_20260907_attempt1.json)
- [1B partial run](../research/raw/PROD3_1B_installed_20260907_attempt2.json)
- [Complete 4B result](../research/raw/PROD3_4B_installed_20260907_attempt1.json)
- [12B memory rejection](../research/raw/PROD3_12B_installed_20260907_attempt1.json)
- [Frozen protocol](PROD3_AUTOCALIBRATION_SPEC.md)

The local dashboard snapshot contains verified metadata only. Its artifact schema
validated and its app-renderer handoff returned success. The portable HTML reader
failed its horizontal-overflow check; a passing standalone browser view is not
claimed. The JSON/CLI history is operational independently of that renderer.

Remaining: installed 12B generation under valid memory conditions, automatic
rescheduling and online inference priority, complete interaction-aware search,
independent deployment confirmation, RL, sustained server testing and multi-Mac
execution. The full product goal remains active.
