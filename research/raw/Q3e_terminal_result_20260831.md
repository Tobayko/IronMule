# Q3e terminal result — one Phase-R result rejected by cleanup attribution

Captured 2026-08-31 from the retained raw record
`research/raw/Q3e_q3c_final_20260831.json`. This is a redacted, human-readable
result note; the JSON remains the authoritative evidence.

## Terminal state

- Status: `FAILED`.
- Fallback: `BASE/current incumbent`.
- Promotion: `false`.
- Raw size: `2,205,857` bytes.
- Raw SHA-256: `1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`.
- Q3c preregistration SHA in the raw result:
  `3bf63ff0dcf442855b6d7b97278fb1d43583a9f18e3f5b6c3caa507582a9ffc5`.
- Q3e preregistration SHA:
  `71901b0d2220d7e9559bad536afaf15d04fac5ea1714f7ceade5bc37811dfd47`.

Q3e's preflight passed all 14 checks. It recorded AC power, low-power mode
off, nominal thermal state, 64% free memory, known 32-GiB installed memory,
known swap `2,643,334,266` bytes, load maximum `5.15771484375` and spread
`0.42138671875`, no competing model process, clean/bound Git, and the exact
local Gemma identity:

- model: `mlx-community/gemma-3-4b-it-4bit`;
- revision: `93724907d4ed1745d2fe50baadf3b0b01a65abf2`;
- manifest SHA-256: `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`;
- commit: `b7884b0ddd1bf6196ed30197a6cd20e2170efc25`;
- runtime-code SHA-256:
  `dcf31430579934c9675bb4577bbb9a4fe1612e58c0dc19b8d60264b8bf02f70b`.

## Phase R evidence

Phase R completed all six fresh processes, alternating `AB, BA, AB, BA, AB,
BA`, with two warmups and seven measured repeats per arm. Every measured
repeat preserved the exact 23 logical and physical output tokens, counts,
`eos` stop reason, capacity `384`, prompt count `322`, decode-step count `22`
and determinism. Swap stayed at `2,643,334,266` bytes with zero delta; all
sampled resource values were within the frozen limits.

The descriptive incumbent/BASE total-time ratio was
`0.857466859207542`, with bootstrap 95% CI
`[0.8551668079699586, 0.8611021999710893]`. This meets the frozen Phase-R
historical checks: distance from `0.8568` is below `0.03`, the CI contains
`0.8568`, and its high endpoint is below `1.0`. Descriptively this is
`14.2533140792%` faster total time, with prefill `16.0213211%` and decode
`10.0321650%` faster; physical output rate is `16.6225872%` higher and decode
steps/s `11.1510142%` higher. These numbers are not an accepted performance
result because the phase failed its cleanup gate.

Phase N did not run. No profile was promoted or activated.

## Why the result is rejected

The worker was reaped and both independent final process snapshots showed no
worker-group member, leader or known descendant. However, four stable new
same-UID processes appeared after the worker baseline. Their bounded records
were retained in both snapshots:

| PID | Process evidence | PPID | PGID | SID | State |
|---:|---|---:|---:|---:|---|
| 28095 | `.../ExtensionFoundation.framework/.../extensionkitservice` | 1 | 28095 | 28095 | `Ss` |
| 28209 | `/Applications/STARFACE.app/.../HeadsetXPCService.xpc/...` | 1 | 28209 | 28209 | `Ss` |
| 28636 | `.../Metadata.framework/.../mdworker_shared -s mdworker -c MDSImporterWorker -m com.apple.mdworker.shared` | 1 | 28636 | 1 | `S` |
| 28964 | `.../UniversalAccess.framework/.../AXVisualSupportAgent.app/... launchd -s` | 1 | 28964 | 1 | `S` |

They were outside the worker and known-child PID sets, outside the worker
process group/session, not descendants of the worker, and had known non-zombie
states. Nevertheless, the Q3e policy treated every new same-UID process as
unresolved. Cleanup therefore recorded
`new same-UID process appeared after worker spawn baseline`, set
`group_gone=false`, and rejected the otherwise complete phase. This is a
conservative attribution failure, not evidence that those unrelated services
were unsafe or that the model changed tokens.

Q3e is terminal. Its timing values remain descriptive retained evidence only;
they must not be pooled with Q2, Q3b, Q3c attempts, Q3d or another run.
