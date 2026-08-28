# B39/B39a/B39b review record — safety-only status

This is a documentation-only review. It contains no B39 performance result and
does not authorize a hardware run.

## B39 direct-script import failure

The first pilot used the direct script invocation and failed before parent
initialization with return code `1` and
`ModuleNotFoundError: No module named 'research'` at
`research/b39_combined_levers.py:22`. No model or child ran, no output or
partial was created, crash reports stayed `30 -> 30`, and no residual model
process remained. Raw: `B39_pilot_import_failure_20260828.json`.

## B39a module pilot failure

B39a used the corrected module invocation and attempted only arm A. The child
loaded the model but stopped at `after_model_load` with return code `3` and
`RuntimeError: B36 checkpoint gate failed: after_model_load`. No warmups or
timed repeats ran; there is no timing evidence and no performance claim.

The partial retains parent system Swap of `1,704,921,661 B` before the child
and `8,568,438,784 B` afterwards, delta `6,863,517,123 B` (approximately
`6.39 GiB`), far above the `268,435,456 B` allowance. This strongly suggests
a resource/swap failure, but the exact checkpoint subtype is unobservable
because the child event stream was discarded. Crash delta was `0`; no residual
model process remained. The parent then raised
`StatisticsError: no median for empty data`, produced no final JSON, and left
the partial sidecar. Raw: `B39a_pilot_failure_20260828.json` plus
`.B39a_pilot_gemma12b_combined_20260828.json.partial`.

No retry and no B39 main run occurred. The event loss and empty-summary crash
are evidence/harness defects, not model or performance evidence.

## B39b safety correction

B39b is safety/evidence-only and authorizes no hardware run. SHA-256:
`403eb1b098d49bff891a52ac16b974857b4fad3e0ed2984f554436acf0e9e7cb`.
Its sole corrections are preserving parent/child checkpoint events in failure
raw data, returning structured `INCONCLUSIVE` for an empty summary while
retaining partial evidence, and enforcing an absolute pre-spawn system-Swap
ceiling of `268,435,456 B` (`256 MiB`) in addition to the unchanged
process-start-to-end delta gate. B39 arms, workload, statistics, thresholds,
and no-activation rules remain unchanged.

## Separate xdist incident

An accidental non-`-n0` test invocation used the `pytest.ini` `xdist -n auto`
path and produced `23` new Python `SIGABRT` reports between `11:38:38` and
`11:38:49`, parent PID `80772`, through MLX/`libmlx`. Representative reports
are `Python-2026-08-28-113838.ips` and `Python-2026-08-28-113849.ips` in
`~/Library/Logs/DiagnosticReports/`. The cause was parallel xdist workers,
not a B39 model result; all B39 CPU tests must run serially with `-n0`.
The later serial verification recorded `46` passing CPU tests and crashreport
count `30 -> 30` with diff-check green. No B39 performance claim follows.

## Scope and UI

The existing X1/B39 evidence remains historical or safety-only; no profile,
routing, or activation change is permitted. The local history UI was not
changed: there is no approved append-only JSON source, and the existing
SQLite-backed runtime history is an architectural surface outside this record.

## B39b pilot — diagnostic only

After a clean preflight (system Swap `0 B`, memory pressure `93%` free, no
Gemma process), the B39b one-block pilot ran four fresh serial children in
order A/B/D/C. All four returned `0` and passed correctness, environment,
workload, crash, and canonical-output gates; every arm recorded two warmups and
one measured repeat. All six requests produced `48` physical, logical, and
visible tokens with stop reason `length`; the canonical output digest was
identical across arms. Swap was `0 B`; no relevant crashreport or residual
process appeared.

The pilot was nevertheless `INCONCLUSIVE` because the block peak gate failed
solely on RSS C/A `3.6523564` (D/B `1.0001511`). MLX peak ratios were C/A
`1.0064033` and D/B `1.0257863`; MLX peaks A/B/D/C were
`7,796,516,616`/`7,801,367,483`/`8,002,535,534`/`7,846,439,900 B`, and
RSS peaks were `2,166,931,456`/`7,916,470,272`/`7,917,666,304`/
`7,914,405,888 B`. Diagnostic outer-wall ms / physical=visible tok/s were
A `10308.915125`/`27.936984300`, B `9072.028833`/`31.745930850`, C
`9805.518458`/`29.371215937`, D `8524.246458`/`33.785977613`.
Wall ratios B/A, C/A, D/A, D/B, D/C were `0.880017802`, `0.951168803`,
`0.826881040`, `0.939618537`, `0.869331540`; corresponding rate ratios were
`1.136340648`, `1.051338098`, `1.209363804`, `1.064261677`, `1.150309122`.
Interaction D*A/(B*C) was `0.987856765`. These single-repeat values are
diagnostic only and no performance claim is made.

The RSS pattern (A `2.17 -> 1.26 GB` during checkpoints versus B/D/C near
`7.9 GB`, with MLX active memory identical near `7.188 GB`) makes process-order
or page-residency confounding plausible. No arm attribution is permitted.
Raw: `B39b_pilot_gemma12b_combined_20260828.json`. Final status remains
`INCONCLUSIVE`, `activation_allowed=false`; no main run, retry, routing, or
activation occurred. B39c's two-new-block crossover is pending after a clean
state; pilot data must not be reused or pooled.

## B39c Memory-Order Diagnostic

After clean preflight (system Swap `0 B`, no residual model process), B39c ran
two fresh serial blocks, `ABDC` then `CDBA`, with all eight children returning
`0`. Correctness, identity, workload, crash, post-state, absolute memory and
Swap gates passed; Swap was `0 B`, final H2 was true, and no relevant crashreport
or residual process appeared.

Block peak ratios were MLX C/A `1.0064022925` and `1.0064018108`, MLX D/B
`1.0257847094` and `1.0257859921`; RSS C/A `0.9999502092` and `1.0007563638`,
RSS D/B `0.9997158295` and `0.9998923418`. Position peak ratios were
`A@0/C@3 = 1.0000497933` and `C@0/A@3 = 1.0007563638`. Absolute MLX peaks
were approximately `7.80–8.00 GB`; RSS peaks were approximately
`7.897–7.914 GB`.

Classification and top status were both `INCONCLUSIVE`: neither the
preregistered RSS order flip nor the reproduced-core condition occurred.
Crucially, historical B39b RSS C/A `3.6524` did not reproduce; block 0 was
`0.9999502` and all RSS values were near `7.897–7.914 GB`. No arm attribution
is permitted. B39c has `valid_for_performance=false` and
`activation_allowed=false`; it contains no timing summary and causes no B39
main run, retry, routing, or activation. B39d's position-balanced performance
main with two new crossover blocks remains pending after clean-state
preflight; B39c data are not reused or pooled.

## B39d — Final performance-main result review (2026-08-28)

The exact B39d module invocation completed with return code `0` and produced
[`B39d_gemma12b_combined_20260828.json`](B39d_gemma12b_combined_20260828.json).
The result has 8/8 blocks and 32/32 fresh serial children,
`status=QUALIFIED`, `valid_for_performance=true`, and
`activation_allowed=false`. All child return codes were `0`; complete
correctness, canonical identity, workload, environment, resource, crash and
final-H2 gates were true. Across the 192 measured requests, every physical,
logical and visible output had 48 tokens and `stop_reason=length`, with one
canonical token digest. Swap delta was `0 B`; no new relevant crashreport or
residual model process was present. Maximum MLX peak was `8,002,539,246 B` and
maximum RSS peak `7,916,519,424 B`; the positions-balanced RSS gate passed with
global `C/A=1.000449911553665` and `D/B=1.0002091397755728`.

Absolute endpoint medians and 97.5% CIs (wall ns; physical/visible tok/s) were:

| arm | wall | rate |
| --- | ---: | ---: |
| A | `11,238,261,187.5 [11,160,058,417; 11,407,090,125]` | `25.6268096092 [25.2474554723; 25.8063165298]` |
| B | `9,804,256,146 [9,746,705,041; 9,953,182,750]` | `29.3751295028 [28.9354679035; 29.5484472741]` |
| C | `10,647,817,688 [10,494,913,166; 10,722,052,334]` | `27.0483488952 [26.8605292185; 27.4418659254]` |
| D | `9,206,717,688 [9,178,958,958; 9,380,620,959]` | `31.2815138465 [30.7015922782; 31.3761071727]` |

All ratio medians and 97.5% CIs (wall; rate) were:

| ratio | wall | rate |
| --- | ---: | ---: |
| B/A | `0.8758819112 [0.8513996079; 0.8899192300]` | `1.1417105861 [1.1236974843; 1.1745365992]` |
| C/A | `0.9430849603 [0.9376590283; 0.9482680892]` | `1.0603530881 [1.0545540985; 1.0664857585]` |
| D/A | `0.8194867050 [0.8067160263; 0.8394204565]` | `1.2202787058 [1.1912981061; 1.2395935713]` |
| D/B | `0.9383079941 [0.9222134455; 0.9588925258]` | `1.0657544693 [1.0428697410; 1.0843476690]` |
| D/C | `0.8694078240 [0.8560822753; 0.8852142827]` | `1.1502701579 [1.1296699788; 1.1681120248]` |

The headline must distinguish scales: D/A and D/B wall reductions are `18.05%`
and `6.17%`, while rate gains are `22.03%` and `6.58%`. Interaction
`D*A/(B*C)` was median `1.0027137194`, 97.5% CI
`[0.9619774991; 1.0185403335]`. Epoch/order drift was not material; the
small-n diagnostics explicitly retain uncertainty for `order:D/B` and
`epoch:contrasts:D/B`.

Identity was bound to model digest
`e08dd84591588722a11c43d9ff7ee4b3f50d01f15371c8a4429c4f9857d37fb6`, code
digest `3adaa1bf467b0efd9fa7c06b3da628de5bbadcd3d8d1e3250c462c3c9ff49ce4`,
and B39d preregistration SHA
`f6fcfccc14afb0535cd0d360d0b956cb6e2bb86873e6e5cfdc827784a7d0bd49` on
Python `3.12.13`, MLX `0.32.0`, mlx-lm `0.31.3`, Apple M1 Max, 32 GiB,
macOS `26.5.2`.

X1's historical `+15.42%` is correctly a rate ratio `1.1542`, equivalent to
wall ratio `0.86640097`; raw X1 `.8458` flags used the semantically invalid
`1-.1542` construction and are non-gating descriptive flags only. B39d exceeds
X1 on both correctly defined scales. This is scoped B39d evidence only:
no activation, routing or general claim. B39 is finished; B40's Gemma-12B
`max_width` 2/3/4 test is pending.

## B40 — Final width-sweep result review (2026-08-28)

B40 completed all six mirrored blocks and 18 fresh serial children, then
returned `rc=2` because the preregistered drift gate classified the result as
inconclusive. Final raw output is
[`B40_gemma12b_width_sweep_20260828.json`](B40_gemma12b_width_sweep_20260828.json);
the atomic partial remains retained. All 18 children returned `0`, were
complete and canonical-correct, passed environment/workload identity and
`no_crash`, and produced 48 tokens per request with `stop_reason=length`.
Swap delta was `0 B`; maximum MLX peak was `8,002,539,246 B`, maximum RSS
peak `7,921,287,168 B`, and final H2 was true. The post-run RSS summary was
`PASS` with global W2/W4 `0.9994507845985368` and W3/W4
`0.9995676763827266`; no relevant crashreport or residual process was found.

Realized widths matched the configured ceilings: W2 mean/max `2/2`, W3 `3/3`,
W4 `3.971830985915493/4`. Candidate/W4 ratios (six block medians, 10,000
bootstrap resamples, 97.5% CI) were:

| comparison | wall median [97.5% CI] | physical/visible rate median [97.5% CI] |
| --- | ---: | ---: |
| W2/W4 | `1.1033961300051331 [1.0849631490673945; 1.1335189058508615]` | `0.9062977565937032 [0.8823856803711523; 0.9218292054497783]` |
| W3/W4 | `1.040445749841422 [1.022514206934345; 1.0723726405010425]` | `0.9611730621034691 [0.9325739636675945; 0.9779881471327478]` |

Every W2 and W3 block pointed in the same descriptive direction (Wall ratio
above 1 and rate ratio below 1 versus W4), but that direction is not promoted
to a performance claim. Both candidates are robust misses; material epoch drift
blocked retention/selection: W3 `0->5 = 1.0313798935311982`, W4
`1->4 = 0.9721113375197978`, and W4 `2->3 = 1.0226246692862697`. Position
residuals stayed near 1 (W2 `1.0038336708/0.9883530874/1.0131131257`, W3
`1.0097000444/0.9991497382/0.9966448181`, W4
`1.0078396313/0.9986091166/1.0000000000`).

The formal result is `status=INCONCLUSIVE`, `classification=INCONCLUSIVE`,
`selected_width=null`, `valid_for_performance=false`, and
`activation_allowed=false`. W4 remains the unchanged operational baseline;
there is no B40-selected width, no retry and no cherry-picking of the timing
direction. Model digest `e08dd84591588722a11c43d9ff7ee4b3f50d01f15371c8a4429c4f9857d37fb6`,
code digest `473980a41d7f5d46f0bc1e76452edcc89238a31edc2bb1943313c243b3a27120`,
and B40 preregistration SHA
`23d0c59d9903875a68131d1f7ac6dc902f671a48b30fa25238ff7dfda34ca0a6` are
recorded. The next architectural route is existing B3 and requires separate
approval; no activation follows B40.

## Public evidence boundary

The complete local B39d/B40 raw JSON files and retained partial sidecars are
intentionally excluded from the public repository because they contain local
process and system evidence. `B39d_public_summary_20260828.json` and
`B40_public_summary_20260828.json` are publication artifacts with privacy
redaction; they are not replacements for the local immutable raw evidence.
