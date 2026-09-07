# PROD10-S — one-hour 12B mixed-concurrency endurance protocol

Registered before this endurance invocation. This extends only the endurance
schedule of PROD10; the previously completed three model matrices, their
source hashes and their results remain historical and unchanged. The product
runtime remains the natively qualified code `47c416fb…edf7b7b`.

Use the frozen local 12B snapshot, the same three public greedy cases, and the
same independent fresh stock reference, product/JSON/SSE matrix and real
disconnect/recovery gate from `PROD10_OPEN_VALIDATION_SPEC.md`.

After those gates, maintain the same product worker for **at least 3600 s**.
Cycle long8, short32, long32 by absolute request index; alternate JSON/SSE by
even/odd index. At each nonzero index divisible by twelve, dispatch four
simultaneous clients for that index and the next three indices; otherwise
dispatch one. Record every response's logical index, batch index, concurrency,
hashes, counts and latency. Finish an already dispatched batch, but dispatch
no new batch after the one-hour boundary. There is no artificial delay or
hardware time, duty, load, RSS or swap rejection gate.

Every response must match its actual stock-reference case. One-second owned-
process/swap observations remain active. Interpret current physical footprint,
not a lifetime high-water counter, when examining memory trends. Missing
telemetry is an evidence error, never a resource-limit violation or a made-up
zero. This is a repeated fixed-workload stability test, not universal leak
freedom, arbitrary context coverage, semantic answer quality or peak throughput.

At the end `/health` must show ready, no active/queued or failed requests,
exactly one cancellation (the deliberate pre-soak disconnect), and exactly
`11 + soak_request_count` completed HTTP requests. The same owned product PID
must survive through the workload. Both stock and product workers must close
with normal exit code zero; identities/source/provider/model metadata must
remain unchanged. No auto-restart or retry hides a fault.

The updated harness is separately source-bound for this invocation and adds
this specification to its source manifest. No new model/math/kernel code is
introduced by the endurance schedule.
