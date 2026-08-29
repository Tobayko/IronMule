# `Q2` preregistration — the self-tuning loop, run once for real

Written **before** the run. Nothing below may be edited after the first measurement;
results go in a separate section appended at the end.

- **Written:** 2026-08-29, before any `tune` invocation on this machine
- **Author:** session `tobiasburandt-5f`, on the user's standing instruction
- **Authority:** user decision of 2026-08-29 — the run happens *after* the onboarding
  push, as its own preregistered run, not inside the onboarding PR.

## Why this run exists

`tune()` is the mechanism behind IronMule's central claim: an unseen machine tunes
itself once and every later start reuses the result. That claim has never been
executed end to end. `tests/test_r6_r7.py` covers the control flow with `Engine`
replaced by `FakeEngine`, `probe` stubbed and `gpu_busy` patched to `None`
(lines 352-358), and `~/.ironmule` does not exist on this machine. So the search runs,
but nobody has watched it run against a real model.

## Hypothesis

Coordinate descent over the nine knobs in `tune.SEARCH` finds at least one setting that
beats the untuned `BASELINE` end to end by more than the `KEEP_IF_RATIO_BELOW = 0.995`
threshold, survives the 6-process / 7-repeat paired confirmation, and emits a token
sequence identical to the baseline's at every step.

## Exact conditions

| | |
| :-- | :-- |
| Command | `ironmule tune --model mlx-community/gemma-3-4b-it-4bit` |
| Store | `IRONMULE_HOME` pointed at this scratchpad, **never** `~/.ironmule` |
| Model | `mlx-community/gemma-3-4b-it-4bit`, from the local HF cache |
| Framework | MLX `0.32.0`, from `/Users/tobiasburandt/Project_Friday/.venv` |
| Machine | Apple M1 Max, 32 GB unified memory |
| Power | AC, mains connected |
| `max_tokens` | 32 (default) |
| `repeats` | 5 (default) |
| Confirmation | `CONFIRM_PROCESSES = 6`, `CONFIRM_REPEATS = 7`, unmodified |
| Other load | none; run only once `tobiasburandt-ec` reports its suite finished |

`IRONMULE_HOME` is redirected for one reason: a tuned profile in the default store
changes runtime behaviour for every other process on this machine, including the peer
session's test runs. An isolated store keeps this run from becoming everyone's silent
new baseline.

## Search space, as it stands in the code

Nine knobs, twelve candidate values, searched in this order — `fuse_projections` last
because it forces a model reload:

`compiled_fixed_cache[True]`, `fused_argmax[True]`, `head_skip_prefill[True]`,
`prefill_into_fixed[True]`, `readback_every[2,4,8]`, `speculate_k[4]`,
`capacity_slack[128]`, `wired_fraction[0.6]`, `fuse_projections[True]`

Two of these are recorded as dead in `docs/BACKLOG.md` Tier 0 and are expected to lose,
which is itself a check that the search reproduces known results rather than inventing
new ones:

- `prefill_into_fixed` — `E1` killed it: the phase it optimises costs `1.47 ms` of
  `537 ms`, so the best imaginable ratio is `0.9973`, above this search's `0.995` gate.
  It cannot legitimately win here.
- `speculate_k` — the code comment records that it loses on MLX 0.32, so its ratio
  should exceed `1.0`.

**A search that "wins" on either of those two is evidence of a broken harness, not a
new result.**

`readback_every` is deliberately **not** on that list, and an earlier draft of this
document had it there in error. The predecessor's cycle 17 measured `0.9581` for
readback 8 — faster in every pair — and retired it only because it missed *that
experiment's* preregistered 5% threshold. This search gates at
`KEEP_IF_RATIO_BELOW = 0.995`, which `0.9581` clears comfortably. So `readback_every`
winning here is the **expected** outcome and contradicts nothing: the two runs ask the
same question against different bars. Treating that win as a bug would have thrown away
the most likely real result of the run.

## What gets recorded

For every candidate: knob name, value, measured ns, ratio against the running best,
token match yes/no, and the accept/reject decision. Then the winning profile as written
to the store, the reported `gain`, the paired confirmation output, total wall time, and
`environment()` from `bench.py` (power source, thermal, loadavg) before and after.

Then a second, separate invocation to confirm the stored profile is loaded rather than
re-tuned.

## Kill criteria

The entry closes as a **failure of the shipped claim** — and the README's
self-optimisation wording must be withdrawn before publication — if any of these hold:

1. The run does not complete, or completes only with `--force`.
2. No candidate beats `0.995`, so the tuned profile equals the baseline. The
   self-tuning feature would then be real code with nothing to find on its own
   reference hardware.
3. Any accepted candidate changes the token sequence. This is the safety property the
   whole design rests on; a violation is a release blocker, not a tuning result.
4. `prefill_into_fixed` or `speculate_k` wins — either indicates a broken harness.
   (`readback_every` winning does **not** trigger this; see the search-space note.)
5. The second invocation re-tunes instead of loading the stored profile — the "tunes
   itself once" half of the claim would be false.

The entry closes as a **success** only if the run completes unforced, the winner clears
`0.995` under paired confirmation, every accepted candidate is token-identical,
`prefill_into_fixed` and `speculate_k` stay dead, and the second start loads the profile.

**A `gain` of any size is a valid success.** The number is not the point of this run;
whether the loop closes at all is.

## Explicitly not claimed by this run

One machine, one model, one MLX build. This measures **that the procedure works**, not
what any other Mac will get. Per `docs/LIMITS.md`, no result here extends the validity
domain by a single cell.

---

## Results

Run completed 2026-08-29, exit code 0, no `--force`. Code under test: commit `a65563f`,
clean working tree. Environment at start: AC power, low-power mode off, loadavg 2.28,
MLX 0.32.0, macOS 26.5.2. Hardware fingerprint `dc652d66f24ac207`.

**Outcome: success. All five kill criteria passed.**

Baseline: `936.89 ms` (prefill `644.63`, decode `292.64`), 23 tokens, capacity 384.

| Knob | Value | Ratio | Decision |
| :-- | --: | --: | :-- |
| `compiled_fixed_cache` | True | 0.9679 | **kept** |
| `fused_argmax` | True | 0.9707 | rejected |
| `head_skip_prefill` | True | 0.8606 | **kept** |
| `prefill_into_fixed` | True | 0.8699 | rejected |
| `readback_every` | 2 | 0.8543 | **kept** |
| `readback_every` | 4 | 0.8737 | rejected |
| `readback_every` | 8 | 0.8688 | rejected |
| `speculate_k` | 4 | 1.3829 | rejected |
| `capacity_slack` | 128 | 0.8538 | rejected |
| `wired_fraction` | 0.6 | 0.8552 | rejected |
| `fuse_projections` | True | 0.8526 | rejected |

Paired confirmation, 6 processes x 7 repeats: ratio `0.8568`, CI `[0.8549; 0.9402]`,
tokens identical, accepted. Stored gain `0.1457`. Second start loaded the profile:
`knobs_for()` returned the tuned knobs, not `BASELINE`. Profile key
`dc652d66f24ac207/2730e8b1…`.

### Against the criteria

1. Completed unforced — yes.
2. Winner beat `0.995` — yes, `0.8568` confirmed.
3. Token identity — held on every accepted candidate.
4. `prefill_into_fixed` (`0.8699`) and `speculate_k` (`1.3829`) both stayed dead, as
   predicted. `speculate_k` is 38% *slower*, matching the code comment exactly.
5. Second start loaded rather than re-tuned — yes.

`readback_every=2` is one of the three winning knobs. The pre-run correction to this
document was therefore load-bearing: the earlier draft would have classified this run's
own result as a harness bug and thrown it away.

### Two honest caveats on the headline number

**The published gain is the screening measurement, not the confirmed one.**
`tune.py:430` computes `gain = 1.0 - best_result["total_ns"] / base["total_ns"]`, where
`best_result` is the single-process screening run. The 6-process paired A/B at line 424
only gates accept/reject; it never reaches the stored profile. So `0.1457` is `1 −
0.8543` (screening), while the more trustworthy paired measurement was `0.8568`, i.e.
`14.32%`. The difference is small here, but the direction is systematic: the number
shown to the user is always the one that survived less scrutiny.

**The confidence interval is wide and asymmetric.** `[0.8549; 0.9402]` — the upper bound
corresponds to roughly `6%`, not `14%`. The point estimate is the honest headline, but
`at least 6% on this machine` is the honest floor, and the interval never reaches the
user at all.

Neither caveat changes the outcome: the loop closes, the safety property holds, the
feature is real. Both are reported to the peer session as review findings.

### Still not claimed

One machine, one model, one MLX build, greedy decoding, 32 output tokens. This run
extends the validity domain in `docs/LIMITS.md` by exactly zero cells. It shows the
procedure works here, not what any other Mac will get.
