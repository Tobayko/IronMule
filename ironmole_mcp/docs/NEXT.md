# Next bounded extensions

These are hypotheses, not implemented capability or permission to broaden the MVP.
Each belongs in the owning backlog before implementation.

## Supported language server

Add one explicitly supported language/server version with immutable workspace binding,
validated document versions and an adapter mapping LSP definitions/references to exact
source evidence. Keep lexical hits and language-server symbol results distinct. Test
unresolved symbols, stale document versions and incomplete indexing. Kill criterion:
wrong identity, incomplete index called complete, or no improvement over direct LSP
usage under the exact task contract. Do not execute repository build scripts merely
to make indexing work without a separately approved contract.

## Second MCP adapter

Add one search/read adapter with materially different measured latency, pagination or
parallel-resource rules. Bind server identity, schemas, snapshot semantics, effect and
permission limits independently. Compare reference, best fixed workflow and adaptive
selection on held-out repositories. Kill criterion: incompatible evidence/permission
semantics or no net advantage over the best fixed workflow after routing and validation.
Do not infer network behavior from an injected sleep in the current server.

## One controlled browser application

Start with one local controlled application and a bounded read-only task. Bind origin,
session identity/expiry, current navigation and frame state, document generation,
element identity and expected postconditions. Verify redirects, stale DOM handles,
authentication changes, navigation races, duplicate elements and page-load failures.
Semantic selectors alone do not establish stable identity or robustness.

Any later form submission, payment, deletion or account mutation needs an explicit
side-effect contract: per-operation authorization, retry/idempotency semantics,
unknown-outcome handling and user/host confirmation where appropriate. A timeout or
cancellation is not evidence that submission did not happen. Kill criterion: a wrong
origin/session/element or unconfirmed side effect can be accepted as success. Do not
expand to arbitrary websites before this single-application contract is validated.
