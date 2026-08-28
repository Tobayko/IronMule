# B36a — B36 artifact-scope clarification

Experiment ID: B36a
Parent: B36
Registered: 2026-08-28, before any model process
Authorization: same 2026-08-28 user authorization as B36

This clarification supersedes only the earlier metadata-only wording. B36
children hash every file in the parent manifest before model load and again
after model load; the full-weight hashing intentionally applies the same
prefaulting work to both arms. All other B36 gates, constants, arm order,
workload, no-retry policy, correctness rules, and no-activation rule remain
unchanged.
