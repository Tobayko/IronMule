# IronMole MCP runtime

An experimental, provider-independent runtime for one recurring task:
`repository_context_bundle.v1`. It searches an immutable text snapshot, reads exact
line intervals and returns bounded, checksum-backed context. It runs through the
official MCP SDK over local stdio. It never runs code from the examined repository.

**The cost/latency hypothesis is unproven.** A manual seed workflow and an optimizer
are implemented. Trace mining proposes same-task candidates; no candidate is activated
by successful repetition alone. The current local benchmark cannot establish an
advantage over a model agent or actual Programmatic Tool Calling.

This package is independent of Python/MLX IronMule and the concurrently developed
Rust/PostgreSQL project in `../IronMole`. Neither is imported or changed. Local
benchmark reporting reuses `friday_evidence.statistics` from the parent repository.

## Run the complete demo

Prerequisites: Node 24 or 25, npm, Python 3.11+ for report/configuration checks. The
verified development runtime is recorded in the results. Node's built-in SQLite may
print an experimental-feature warning; no SQLite service is required.

```sh
cd ironmole_mcp
npm ci --ignore-scripts
npm test
npm run demo
```

The demo creates a private, synthetic repository snapshot under `.state/`, starts the
controlled MCP search/read child, executes the reference plan and prints a complete
bundle. An empty result is valid. Partial output has a continuation cursor.

Exact dependency versions and transitive resolutions are in `package-lock.json`.
This development session used the existing npm cache because shell networking was
unavailable. A normal `npm ci --ignore-scripts` needs registry access or a complete
cache; it does not download models. Package scripts run only IronMole's own code.

## Register a real text scope

Registration and snapshot creation are explicit host operations. Start with a narrow
text-only directory; binary files, symlinks, special files and oversized scopes fail
instead of silently disappearing. Use a new state directory for this example.

```sh
mkdir -p .state
chmod 700 .state
node dist/src/cli.js init --state .state/local --repository own-code \
  --root . --allow src > .state/setup.json
node --input-type=module <<'JS'
import { readFileSync, writeFileSync } from 'node:fs';
const setup = JSON.parse(readFileSync('.state/setup.json', 'utf8'));
writeFileSync('.state/task.json', JSON.stringify({
  taskType: 'repository_context_bundle.v1', repositoryId: setup.repositoryId,
  snapshotId: setup.snapshotId, query: 'repository_context_bundle',
  searchSemantics: 'literal_case_sensitive_line.v1', pathFilters: ['src'],
  contextLines: 2, maxMatches: 100, outputBudgetBytes: 32768,
  continuation: null
}, null, 2), { mode: 0o600 });
JS
node dist/src/cli.js run --state .state/local --host .state/local/host.json \
  --task .state/task.json
```

The host file sets the principal, expiry, permitted tools, repository root/scope,
resource concurrency and objective. It is private and never accepted as task input.
The default policy expires after 30 days. A host administrator can revoke it by
setting `enabled` to `false`. Already-open clients reread policy around each call.

After source changes, explicitly create another snapshot:

```sh
node dist/src/cli.js snapshot --state .state/local --host .state/local/host.json \
  --repository own-code
```

Use that returned snapshot ID in a new task. Old snapshots remain immutable. Plans
can apply to either snapshot, but no tool results are cached across executions.

## Codex and other MCP clients

```sh
node dist/src/cli.js codex-config --state .state/local \
  --host .state/local/host.json > .state/codex-mcp.toml
```

The generated TOML contains actual absolute paths for this installation. Add its
`[mcp_servers.ironmole]` table to your chosen Codex host configuration. This command
does not edit global configuration or grant additional permissions. The only outer
tool is `ironmole.execute`; pass the explicit task object above. STDIO works with
other clients implementing the negotiated SDK protocol as well.

`tests/mcp.test.ts` checks the generated TOML with Python's TOML parser, asks the
installed Codex CLI to read an ephemeral `-c` override, and starts the exact command
for initialize/list-tools/execute. It does not invoke a Codex model. The test reports
an explicit skip if Codex is absent; the independent MCP roundtrip still runs.

## Plans and observations

```sh
node dist/src/cli.js compile --state .state/local --plan examples/reference.moleir.json
node dist/src/cli.js mine --state .state/local
node dist/src/cli.js plans --state .state/local
```

The miner needs three successful, dependency-rich traces and two distinct validated
parameter bindings. It proposes only width 1/2/4 variants of this exact task, with
origins marked `mined`. Seed examples are marked `manual`. Host code can evaluate
and qualify candidates through `Registry`; no MCP tool promotes plans or accepts
performance claims from repository text. Runtime-only evidence is rejected for
activation. No default activation command fabricates missing agent measurements.

## Reproduce the local experiment

```sh
npm run benchmark
```

Each run gets a unique private directory under `bench/runs/`. The controller seals
inputs, trains/mines on one repository, selects C's width on another and evaluates
nine local variants on held-out repositories in separate subprocesses. Raw data,
selection results, seal and report are retained. Existing run IDs are never overwritten.
Failed attempts remain failed. See the [preregistration](bench/PREREGISTRATION.md)
and [results](docs/RESULTS.md) for thresholds, actual numbers and absent A/B arms.

The benchmark stops at the locally serialized task result. Final model response,
agent token use, billed cost and end-to-end amortization stay unknown. It makes no
MLX, inference-speed, reasoning-time or generic learning claim.

## Documents

- [Architecture and decisions](docs/ARCHITECTURE.md)
- [Task, trust and adapter contracts](docs/CONTRACTS.md)
- [MoleIR and deoptimization](docs/MOLEIR.md)
- [Related work and compatibility](docs/RELATED_WORK.md)
- [Milestone checklist](docs/PLAN.md)
- [Next bounded extensions](docs/NEXT.md)
