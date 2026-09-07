# Product checkpoint — 2026-09-07

The portable reference service is implemented and tested. Autonomous tuning/RL,
optimized serving qualification and sustained-production readiness are not done.
The new API does not inherit the older Python runtime's measured speedups.

## Verified delivery

- CLI: setup, cached model inventory/registration/removal, product status, serve.
- Isolated persistent GPU model worker; strict bounded IPC, deadlines and clean
  failure handling. Python `-I` prevents standard-library shadowing at startup.
- Exact-greedy text chat over real HTTP JSON/SSE; bounded queue and handlers,
  cancellation, custom stops, context admission, TLS/authentication boundaries.
- Private atomic settings/registry. No persisted user prompts/answers, implicit
  model downloads, automatic upload or pretending that an optimizer is running.
- [Start guide](PRODUCT_QUICKSTART.md).

| Check | Result | Scope |
| :-- | :-- | :-- |
| Gemma 3 1B / 4B / 12B, local 4-bit snapshots | All passed | Short exact comparison with separate stock MLX-LM processes; JSON/SSE/stops |
| Real cancellation / context overflow / recovery | All three passed | Same warm worker reused; short request only |
| Focused product regression | 118 passed | Filesystem, IPC, arithmetic, actual local HTTP/TLS |
| Existing CLI compatibility | 28 passed | Command/metadata compatibility |
| Clean GitHub CI, Python 3.11 | 764 passed, 16 integration tests deselected | Wheel build/install, static checks, engine suite; not target-Mac inference |
| Clean GitHub CI, Python 3.12 | 764 passed, 16 integration tests deselected | Same limits as above |
| Xcode / tiny MLX and PyTorch MPS operations | Passed | Device access/correctness, not performance |

CI: [run 34097092847](https://github.com/Tobayko/IronMule/actions/runs/34097092847),
commit `5fe3817`. Earlier CI failed on worker startup; the failure was reproduced
and fixed rather than skipped. Earlier Gemma attempt 1 failed on SSE body framing;
attempts 2 and 3 passed. Their source hashes belong to their exact historical
checkpoints, not silently to all future edits. Attempt 3's generator code was
unchanged by the later HTTP-only authentication correction.

## PROD2: proven mechanism, unqualified performance

The private candidate omits a final prefetch whose result cannot be consumed
at a finite output limit with a private discarded cache. On the real 1B model:

| Output limit | Stock model calls | Candidate model calls | Complete output comparison |
| --: | --: | --: | :-- |
| 1 | 3 | 2 | Exact |
| 8 | 10 | 9 | Exact |
| 32 | 34 | 33 | Exact |

These are **model-forward invocations**, not GPU kernel counts, FLOPs, bandwidth
or latency percentages. Calls have unequal cost. Early EOS before the limit
does not save that forward in this candidate. No automatic activation follows.

All attempts are retained separately:

| Attempt | Terminal result | Interpretation |
| :-- | :-- | :-- |
| 1B audit 1 | Failed: missing `os` import | No generation |
| 1B audit 2 | Failed: CPU readiness | No generation |
| 1B audit 3 | Passed: six traced HTTP calls | Mechanism/correctness only |
| 1B full pilot 1 | Failed: statistics import scope | Four calls retained; no complete AB/BA pair |
| 1B full pilot 2 | Failed: post-cell CPU readiness | Eight calls retained; full protocol incomplete and no speedup accepted |

Full pilot 2 recorded normalized load `0.805517578125` against the unchanged
maximum `0.8`; Low Power was off, thermal state nominal and source manifest
unchanged. The gate is not retroactively relaxed because the excess is small.
No partial result is promoted, pooled with retries or advertised as a gain.
All owned model workers were reaped. Paired confirmation on 1B/4B/12B is open.

Ruff 0.16.6 is now a pinned CI check for the new product/harness surface. Its
local isolated tool installation did not change the validated runtime environment
hash. Readiness samples, including rejected numeric observations, are persisted
before gate application; readiness is checked before loading weights as well.

## Remaining delivery

The backlog, not this result file, defines remaining work: installed-product
model checks, autonomous bounded calibration and complete local attempt history,
interaction-aware cost models and conservative RL behind an independent
evaluator, plus long desktop/server tests. Busy-host readiness needs controlled
deferral, not repeated speculative model loads. Multi-Mac execution and automatic
worker restart follow after the single-host runtime is qualified.
