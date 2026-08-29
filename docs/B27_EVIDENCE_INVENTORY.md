# B27 evidence inventory

This is a read-only corpus audit. It is not a performance result, a training
dataset, or permission to route a strategy. Absolute local paths are omitted.

Dataset SHA-256: `ee414c9ee51c6e583ada094444ce66d5e22dca6c15c197dda1d7cd004e30bf32`

## Sources

| Alias | Git head | Artifacts | Tracked | Local-only |
| --- | --- | ---: | ---: | ---: |
| `branch` | `d422fdb00fced3238dfaa6b5e9e993294adb72cd` | 42 | 42 | 0 |
| `local-unpublished` | `897b789a28635328c9ccbd5e9b2d06de3f86eae3` | 92 | 41 | 51 |

## Artifact classes

| Kind | Count |
| --- | ---: |
| `legacy_summary` | 14 |
| `partial_result` | 5 |
| `preregistration` | 40 |
| `preregistration_checksum` | 16 |
| `public_summary` | 5 |
| `raw_result` | 48 |
| `review` | 6 |

Total rows: **134**; unique contents: **92**; duplicate-content groups: **42**.

## Structural JSON coverage

Presence below is not a quality or trust judgment.

| Group | Present | Eligible valid JSON |
| --- | ---: | ---: |
| `environment` | 43 | 72 |
| `workload` | 52 | 72 |
| `baseline` | 25 | 72 |
| `candidate` | 25 | 72 |
| `measurements` | 43 | 72 |
| `correctness` | 37 | 72 |
| `resources` | 64 | 72 |
| `provenance` | 53 | 72 |

## Gate

Raw samples, formal preregistrations, reviews, partials, legacy summaries and
public summaries remain separate quality classes. No learned or generalized
claim is permitted until provenance, replayability, missingness, censoring and
leakage are validated per experiment. Local-only artifacts must not be copied
into the public repository merely to make the counts look complete.
