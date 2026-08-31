# Q3f terminal result — 2026-09-01

Q3f consumed its single permitted execution path and ended `FAILED`. This is a
terminal safety result, not a performance result. No candidate was promoted,
activated or routed; the fallback remains `BASE/current incumbent`.

## Frozen evidence

- Raw result: `research/raw/Q3f_q3c_final_20260901.json`
- Raw size: `2,487,533` bytes
- Raw SHA-256: `e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`
- Schema: `ironmule.q3c_result.v1`
- Q3c preregistration SHA-256: `3bf63ff0dcf442855b6d7b97278fb1d43583a9f18e3f5b6c3caa507582a9ffc5`
- Q3f preregistration SHA-256: `345c63cba5f019ab0314761404f7de398ceee876ffcee82d80c3578f9db8e31b`
- Code commit: `98258e39e21739644119e63ad9a50f5b60d9d84c`
- Runtime-code SHA-256: `55b134c67337daaced0c14ce198334fb579b6df9994280a19c0c117cee3f488e`
- Model: `mlx-community/gemma-3-4b-it-4bit`
- Model revision: `93724907d4ed1745d2fe50baadf3b0b01a65abf2`
- Model manifest SHA-256: `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`

## Preconditions and execution

All 14 preflight checks passed. The Mac was on AC power, low-power mode was
off and thermal state was nominal. Free memory was `67%`; installed memory was
`34,359,738,368` bytes. Swap was known at `2,609,643,520` bytes and remained
constant across the 25 recorded samples (`delta=0`, no sampler errors).

The phase worker exited with status `2` before producing a child-start marker,
child ledger entry, model timing, token, phase-gate or resource result. The
bounded failure was:

`ABRunError: child 0 start callback failed`

The raw record therefore contains no accepted Phase R or Phase N measurement.
The exact lower-level cause of the callback failure is not preserved in the
bounded evidence and must not be stated as fact. An immediate process/command
visibility race is only a possible hypothesis for a future, separately
authorized investigation.

## Cleanup and safety decision

Cleanup used `ironmule.cleanup.v2`. The worker was reaped and two independent
verification snapshots were valid. No worker-group member or descendant
remained, no descendant kill was attempted, and the only signal record was
`SIGTERM: not_needed_group_already_gone`.

The cleanup evidence was nevertheless rejected because the guard and direct
child-start ledger were unavailable. A stable same-UID Spotify Helper process
(`PID 52017`) appeared after the worker baseline and was outside the worker
group; it was not killed. The raw result also records a new same-UID process
attribution condition. This is the intended fail-closed outcome: the process
was not proven to belong to the worker, but the missing guard/ledger proof also
prevented accepting it as unrelated.

At the time of recording, no orphan from the Q3f worker remained. The result is
`FAILED`, `promotion_allowed=false`, with the BASE/current-incumbent fallback.
There is no retry, no pooling with Q3e/Q3d/Q3c, no 27B run, no download or
installation, and no automatic activation.

## Consequence

Q3f did not establish that the historical Q2 speed values were reproduced, and
it did not establish any new speed improvement. The next possible lesson is
limited to preserving the bounded child-start error while making child identity
visibility robust against a documented race; that would require a new explicit
authorization and a new preregistration. It is not a Q3f rerun.
