# Local MVP results — 2026-09-12

## Verdict

The runnable local MVP and its correctness/fault checks are complete. **Adaptive
added value is not demonstrated.** No candidate was activated. The strong stored
workflow is faster than the conservative adaptive runtime in this bounded experiment.
Direct-agent and actual Programmatic Tool Calling model benchmarks were not executed;
final agent-response time, tokens, billed cost and end-to-end amortization remain unknown.

## Final implementation measurement

Experiment: `MOLE1-LOCAL-2-20260912-attempt1`, plan version 2. Apple M1 Max, macOS arm64,
Node v25.5.0, official MCP SDK 1.29.0, negotiated inner protocol 2025-11-25. No inference
model, quantization or model identity applies to these local-only measurements.
Other agents were active on the shared host; background load and OS file cache were
not controlled. Separate subprocesses and an A/A control bound interpretation; they
do not establish a quiet-machine or cold-disk result.

Six independent process blocks per arm, four measured repetitions of four held-out
cases per process, two warmup sweeps and a separately recorded cold sweep. All **864
measured results passed** both the independent fixture oracle and cross-arm reference
comparison. There were **zero deoptimizations in the nominal benchmark**; injected
failures are tested separately and do not establish a zero real-world failure probability.

| Variant | Repeated median ms | p95 ms | Process start to first result, median ms | Median tool calls |
| --- | ---: | ---: | ---: | ---: |
| REBUILD | 27.330 | 49.036 | 387.953 | 25.0 |
| REFERENCE | 28.035 | 50.198 | 384.018 | 25.0 |
| PARALLEL | 18.838 | 31.978 | 369.592 | 25.0 |
| DEDUP | 28.309 | 50.240 | 383.586 | 25.0 |
| MERGE | 16.934 | 19.327 | 381.633 | 10.5 |
| PREPARATION | 28.062 | 48.924 | 380.067 | 25.0 |
| C_WORKFLOW | 12.537 | 14.416 | 374.789 | 10.5 |
| C_AA | 12.767 | 15.082 | 373.980 | 10.5 |
| D_ADAPTIVE | 27.627 | 49.079 | 382.715 | 25.0 |

The pooled timing distribution mixes four declared workloads; it is not one generic
repository latency. Per-case distributions, min/max/spread and individual samples
are in the artifacts. C's width **2** was chosen from 1/2/4 on the separate selection
repository. D retained the known sequential reference because qualifying agent-level
evidence is unavailable. The C/D gap includes different permitted execution plans,
not merely the router's overhead.

D/C block-median ratio is **2.2884**, 95% interval **[2.0240, 2.4228]**. The matched C_AA/C
control is **0.9961**, interval **[0.9508, 1.0603]**; it passes the declared control rule.
Intervals bootstrap the six independent block medians. The 96 measured samples per
arm are not represented as 96 independent processes. No interval establishes a
provider/model benefit.

## Separate contributions

The following ratios use the unchanged sequential REFERENCE in the same block,
repetition and case. They are not multiplied or added together.

| Variant / REFERENCE | Median paired ratio | Clustered 95% interval |
| --- | ---: | --- |
| REBUILD | 0.9637 | [0.9345, 1.0010] |
| PARALLEL | 0.6739 | [0.6542, 0.7249] |
| DEDUP | 0.9775 | [0.9614, 1.0173] |
| MERGE | 0.6857 | [0.6332, 0.7119] |
| PREPARATION | 0.9685 | [0.9419, 0.9910] |
| C_WORKFLOW | 0.4923 | [0.4637, 0.5104] |
| D_ADAPTIVE | 0.9939 | [0.9510, 1.0122] |

Parallel reads and interval coalescing lower local runtime on these fixtures. Exact
read deduplication is functionally verified on duplicate full-file ranges; this
benchmark's intervals contain no identical reads that need removing, and the DEDUP
arm makes no qualifying speed claim. Reusing the tiny stored plan does not demonstrate
a gain over rebuilding it. Indexed preparation is below the required 5% improvement
gate; no qualifying benefit is claimed. Canonical output is identical in all variants.
D/REFERENCE is indistinguishable from its reference within the measured uncertainty.

Median RETURN preparation cost is 0.178 ms for REFERENCE, 0.173 ms for PREPARATION
and 0.154 ms for C_WORKFLOW. Median runtime overhead is 11.957 ms for D and 3.157 ms
for C; these values include their different call patterns and must not be described
as pure routing overhead. Measured median route/compile phase is 0.579 ms for D and
0.328 ms for C. No inter-tool interval is called reasoning time.

## Learning, validation and missing cost terms

Training setup, three observed runs and mining: **416.321 ms**. Candidate selection
and preparation validation: **551.239 ms**. The 54 final child processes together
consumed **57,149.176 ms of observed child wall time**, including cold sweeps, warmups,
repeated runs, oracle checks, traces, process startup and shutdown. This excludes
controller/report overhead and human engineering time. The miner emitted three
same-task candidates; all remain `candidate`, with no automatic activation.

A. A well-configured model agent with direct tools: **not run**, no model adapter.
B. Actual Programmatic Tool Calling: **not run**, no supported integration configured.

A presence-only environment check found neither `OPENAI_API_KEY` nor
`ANTHROPIC_API_KEY`. Credential files were not inspected and no paid model execution
was started. The installed Codex CLI was used only for configuration checks.
C. Guarded stored workflow: measured locally, final model response absent.
D. Adaptive IronMole: measured locally, final model response absent.

Model calls/tokens/cost for an encompassing agent task stay null. The runtime itself
made zero model calls. There is no documented provider price basis because no model
usage was billed or observed. Handoff plus subsequent agent execution and lifetime
break-even remain unknown. Neither the local fixed workflow nor the manual seed
proves automatic learning or advantage over PTC.

## Earlier run retained separately

`MOLE1-LOCAL-1-20260912-attempt1` completed 768 correct measurements using plan version
1. C median/p95 was 11.469/13.134 ms; D was 25.954/44.634 ms. It instrumented RETURN
cost but lacked a separate preparation variant. LOCAL-2 prospectively added the
indexed preparation path and its arm without changing the success thresholds. The
runs are not pooled or treated as replications of identical code. Both conclude that
additional adaptive benefit is not demonstrated. Their sealed input hashes and raw
outcomes remain unchanged; private source archives preserve the measured revisions.

## Validation actually executed

- `npm ci --offline --ignore-scripts --cache <local-cache> --no-audit --no-fund`: clean
  lockfile installation, 96 packages, successful. No install scripts or downloads of models.
- `npm run build` followed by `node --test --test-concurrency=1 dist/tests/*.test.js`
  (the `npm test` script): **31 passed, 0 failed, 0 skipped** on the final implementation.
- Includes real inner and outer stdio MCP connections, schema drift checks, typed
  graph rejection, exact independent expected locations/excerpts, lifecycle/miner,
  650-hit pagination, empty results, byte limits and continuation.
- Includes traversal, symlink files/directories, prefix confusion, permission revocation,
  snapshot identity changes, invalid UTF-8, corrupt/foreign schema, malformed/dishonest
  replies, timeouts, cancellation, resource concurrency and partial parallel failure.
- Generated Codex TOML parsed and accepted by **codex-cli 0.153.4** using a temporary
  command-line override; its exact executable/arguments completed MCP task execution.
  No global Codex configuration or model conversation was changed.
- README registration/task/config-generation commands executed successfully on the
  package source snapshot: COMPLETE, two matches, REFERENCE_PLAN. The generated local
  Codex config is retained under ignored `.state/codex-mcp.toml`.
- Existing tracked Markdown link checks passed; sealed historical documents retain
  their established exemptions. All nine new Markdown documents passed a separate
  link check. Public JSON artifacts passed parsing and private-path checks.
- Python reporting uses `friday_evidence.statistics`; regeneration from public raw
  samples was byte-identical to the recorded summary.

IronMule inference/GPU tests and external model/PTC benchmarks were not run. This
package does not change that inference path. Node 24 is supported by the declared
APIs but only Node 25.5.0 was available for execution here. SDK 1.30/v2 compatibility
and arbitrary external MCP servers are not claimed. All benchmark child commands
exited normally and their owned SDK clients were closed. An independent OS-wide
process audit was unavailable because this session cannot list processes; no unrelated
processes were stopped.

## Inspect and reproduce

- [Final raw samples](../bench/results/MOLE1-LOCAL-2-20260912-attempt1.raw.json)
- [Final summary](../bench/results/MOLE1-LOCAL-2-20260912-attempt1.summary.json)
- [Sealed input hashes](../bench/results/MOLE1-LOCAL-2-20260912-attempt1.seal.json)
- [Validation effort and isolated contrasts](../bench/results/MOLE1-LOCAL-2-20260912-attempt1.effort.json)
- [Earlier raw samples](../bench/results/MOLE1-LOCAL-1-20260912-attempt1.raw.json)
- [Earlier summary](../bench/results/MOLE1-LOCAL-1-20260912-attempt1.summary.json)

Public raw artifacts replace synthetic excerpts with result digests while retaining
all recorded timing/outcome samples. Full synthetic outputs, local traces, snapshots
and the measured source archives remain under ignored `bench/runs/`. No private
repository contents, credentials or actual local paths are in the public artifacts.

```sh
npm ci --ignore-scripts
npm test
npm run benchmark
python3 bench/report.py \
  bench/results/MOLE1-LOCAL-2-20260912-attempt1.raw.json \
  /tmp/ironmole-recomputed-summary.json
```

Use a fresh report output path: evidence files are not overwritten. Current limits and
remaining extensions are documented in [contracts](CONTRACTS.md) and [next steps](NEXT.md).
