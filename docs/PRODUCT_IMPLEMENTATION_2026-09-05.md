# IronMule product implementation — 2026-09-05

## Approved objective

The user approved implementation of the critically revised product plan, live
hardware tests for every locally available Gemma model, relevant primary-source
research, Luna subagents, and publication of completed work to GitHub. This
supersedes the earlier conversation's no-benchmark instruction. It does not turn
an unavailable GPU into a passing test or authorize fabricated results.

The product is an installable wrapper/runtime with CLI and API, explicit model
and backend contracts, desktop/server resource policies, and autonomous
configuration search. Exact is the default; Efficiency is opt-in with at most
0.5 percentage points of statistically qualified noninferiority margin on each
registered task metric. User prompts/answers are not persistently logged or
uploaded. RL remains a product objective, with training outside the token path
and a separate evaluator owning activation.

## Delivery sequence

1. Correct R2 population accounting in a separate derived report. Keep the
   original study, data, 320–399 holdout and failed/inconclusive verdict intact.
2. Inventory all local Gemma snapshots without requiring MLX to initialize;
   separate metadata completeness, device access and actual model qualification.
3. Deliver portable reference/optimized backends and a single installed CLI/API;
   remove dependence on a pinned private developer worktree.
4. Correct scheduler waits, cache identity/capacity checks, bounded compiled-body
   caching and byte-based memory accounting; validate real baseline/candidate
   pairs before making performance claims.
5. Implement autonomous collection, interaction-aware cost models and
   conservative multi-step RL with distinct train/validation/confirmation data.
6. Qualify every available local Gemma snapshot, then desktop/server sustained
   workloads. Connected Macs and automatic model-worker restart follow later.

Every candidate records its mechanism, validity scope, baseline, costs and kill
criterion in the backlog. Metadata count is not context diversity. Failed and
retried attempts must remain visible; omitted target-zero actions must not
change an IPS denominator silently. Historical research packages remain frozen.

## Initial execution environment (historical, 2026-09-05)

Base revision: `dc6d32c` (2026-09-05). Worktree was clean before this task.
Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3. AC power is present. Xcode's
first-launch check returned successfully. ProjectAtlas MCP was refused by the
session approval policy; its version-matched 0.4.5-rc1 CLI was used and the index
was refreshed after typed full-refresh guidance. No ProjectAtlas source changed.

Actual device access fails with `[metal::load_device] No Metal device available`.
`mx.metal.is_available()` nevertheless returns true. This reproduces a false
positive in the old CLI diagnostic: `_probe_metal()` returned
`(True, 'Metal available')` before the change. The corrected probe explicitly
opens the device and evaluates a tiny GPU operation in a child process; here it
correctly returns false. This verifies error reporting, not GPU execution.

The session has read-only `.git` permissions, and `git ls-remote origin` fails
with `Could not resolve host: github.com`. Commit/push and hardware qualification
cannot complete in this execution environment. No CPU surrogate result will be
reported as a GPU result, and no passing Gemma inference result is claimed.

## Local user-facing diagnostics

`ironmule doctor --json` provides an isolated, machine-readable readiness report.
`ironmule models list --family gemma --json` inventories local snapshots without
importing the inference runtime. `--cache-root` is repeatable and restricts the
inventory to explicit roots. Reports mark models as not loaded and not hardware
qualified. The original `ironmule models` interface remains compatible.

The MLX inventory follows the installed non-distributed loader's direct
`model*.safetensors` selection. Converted Gemma 4B/12B checkpoints retain upstream
indices that name different shards; these are warnings, not false missing-model
errors. Generic `loader="hf"` inventory continues to validate the index. Local
metadata shows Gemma 1B (732,577,304 weight bytes), 4B (3,400,569,562 bytes) and
12B (8,028,675,248 bytes). These are file sizes, not measured inference memory.

## Completed bounded verification

- PROD1-A: `experiments/r2_campaign/corrected_evaluation.py` reconstructs all 400
  original draws, 352 observations and 48 target-zero omissions. Holdout remains
  indices 320–399 (80 draws, 70 observations). Integrity and hash-chain checks
  use the existing read-only database API. The separate corrected JSON preserves
  `learning_claim=false`, `activation_allowed=false`, historical inconclusiveness,
  the undocumented retry caveat and unverified cross-commit timing comparability.
- 19 focused tests passed: population arithmetic/failure handling, cache inventory
  fixtures and real CLI child processes. No fake backend produced GPU evidence.
- Wheel built with the existing setuptools installation, then CLI and inventory
  loaded directly from that wheel with Python `-S` outside the source checkout.
  Metadata discovery worked without MLX, NumPy or Hugging Face imports. This is
  an isolated wheel-content smoke, not a full dependency install or model test.
- `git diff --check` passed. No source change to sealed research packages or to
  the original R2 evaluator/report. No dependency installation took place.

The above records the initial restricted session, not the current access state.

## Progress on 2026-09-07

The durable hardware authorization is retained. Reviewed escalations now provide
real Metal, loopback/TLS, Git writes and GitHub access. R2's four-file correction
was pushed as `c260c8c` on `Codex/ironmule-product`. Xcode first-launch validation
passes and the project-local MCP configuration points to ProjectAtlas 0.4.5-rc1.
Atlas's bounded refresh succeeded once, but subsequent queries still sometimes
return `dependency_closure_limit`; exact bounded source reads are used only when
that navigation state prevents a current answer. No Atlas source is modified.

The portable product package now has dependency-free setup/model metadata,
private atomic state, a persistent isolated GPU reference worker, bounded FIFO
admission, request deadlines, cancellation, custom stop filtering and real HTTP
JSON/SSE. Remote binding requires TLS plus authentication. Incomplete TLS clients
cannot block the accept loop; handler admission is bounded at 64. Desktop/server
currently differ in pending queue capacity, not yet in an optimized GPU policy.

The first complete Gemma screen passed for all locally cached 1B/4B/12B snapshots
in attempt 2. Attempt 1 remains failed and preserved: a finite SSE response had
waited for its idle timeout; explicit connection closure fixed the framing.
Attempt 3 extends the screen with real cancellation, context rejection and warm
worker recovery after subsequent source changes. Its raw report, not this plan,
is authoritative for its terminal status.

Attempt 3 is now terminal `passed` for all three models with unchanged source
digest during execution. A subsequent HTTP-only ASCII-authentication correction
has its own real-loopback regression; the model/generator code did not change.
118 focused product/IPC/filesystem/arithmetic/real-HTTP/TLS tests and 28 existing
CLI compatibility tests passed at the publication checkpoint.
An extracted built wheel also passed dependency-free setup, registry,
status, inventory and help smokes outside the checkout. That is packaging evidence,
not an independent clean dependency installation or a model performance claim.

See [the quick start](PRODUCT_QUICKSTART.md) for the usable service. Configuration
status intentionally says `configuration_only`: no optimizer/RL activation is
pretended. The next bounded candidate is PROD2's unused final prefetch; it is
private, version-bound, and needs an exact live mechanism audit plus paired API
timings before any benefit is stated. Autonomous optimization, learned cost
models, independent confirmation, installed-product model runs and long server
qualification remain open. Multi-Mac operation and automatic restart follow later.
