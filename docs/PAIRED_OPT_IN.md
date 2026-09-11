# Paired decode with a shared weight load — opt-in, off by default

Two ready requests share one sweep of the `K=3840` weights instead of reading them
twice. Measured against IronMule's shipped `ThroughputMode`, a pair of requests finishes
**12.7 to 12.8 per cent sooner** (`B46`, two preregistered sessions). It is not a faster
single chat and not a general model gain.

## Turning it on

The option is a service mode, chosen the same way `ThroughputMode` is. There is no CLI
flag; modes are a library-level choice in this runtime.

```python
from ironmule.service import PairedThroughputMode, Request, Runtime, paired_status
from ironmule.plans import StrictOneShotPlan

mode = PairedThroughputMode()                 # explicit; nothing enables this for you
runtime = Runtime.load("mlx-community/gemma-3-12b-it-4bit", mode=mode)

requests = [
    Request(prompt_ids=runtime.encode(first), max_tokens=24, plan=StrictOneShotPlan()),
    Request(prompt_ids=runtime.encode(second), max_tokens=24, plan=StrictOneShotPlan()),
]
results = runtime.serve(requests)
```

## Checking what it did

`paired_status` answers for any mode, so "disabled" is a real answer rather than a
missing one.

```python
paired_status(runtime.mode)
# {'mode': 'paired_throughput', 'enabled': True, 'sharing': True, 'admitted': True,
#  'admitted_projections': 240, 'paired_steps': 21, 'solo_steps_no_partner': 3}
```

* `enabled` — the mode was chosen at all.
* `admitted` — this machine, library build, model revision and projection shapes passed.
* `paired_steps` — steps that actually shared a weight load.
* `solo_steps_no_partner` — steps taken alone because no partner was ready.

## Turning it off

Assign any other mode, or load without one. The default is off:

```python
runtime.mode = ThroughputMode()               # or InteractiveMode()
runtime = Runtime.load(model_id)              # default mode, paired path never engaged
```

## Letting the profile choose, also opt-in

Naming the mode is one way. The other is to let the tuned profile choose between the
established mode and this one, per request group. That is off too, and it needs a second
opt-in:

```python
runtime = Runtime.load(model_id, automatic_service_mode=True)
runtime.mode.status()
# {'mode': 'automatic', 'strategy': 'throughput', 'reason': 'nothing served yet', ...}
```

Three gates must all open before the paired path is chosen:

1. the caller passed `automatic_service_mode=True`;
2. the profile carries a complete `service_strategy` record;
3. this machine, model, library build and the number of ready requests are inside the
   range that record admits.

A profile without the record, a record missing a field, and an older profile all mean the
same thing: the established mode. The decision uses only what is known when it is made —
how many requests are ready now, and the identity of machine, model and libraries. The
length a request will eventually produce is not used and not predicted.

**Ready means ready.** A request whose `arrival_ms` still lies ahead, or that already
finished during prefill, is not a partner. Two submitted requests with one of them still to
arrive count as one ready request and keep the established mode. The status reports both
numbers, so `ready_requests` and `group_requests` can disagree and say why:

```python
runtime.mode.status()["ready_requests"], runtime.mode.status()["group_requests"]
# (1, 2)
```

Pairing itself uses the same rule one level down: `PairedGroupedExecutor` only ever pairs
requests the scheduler has admitted at a step boundary, and it never waits for a partner.

After serving, the status says what was chosen and why:

```python
runtime.mode.status()
# {'mode': 'automatic', 'strategy': 'paired_throughput', 'paired_steps': 23,
#  'reason': '2 ready requests inside the admitted range 2 to 4, on the qualified '
#            'machine, model and libraries',
#  'evidence_run_ids': ['B45_shared_weight_verdict_20260909', ...], ...}
```

The decision is made once per `serve`, before any token is produced; nothing is added to
the per-token path. Choosing automatically costs nothing measurable against naming the
same strategy by hand: `0.9973`, CI `0.9937 – 1.0018`, inside a ±2 per cent margin fixed
before the run (`B52`, release run `B52R_automatic_selection_release_20260909`).

### Writing the record

The record lives beside the knobs, not among them, because a service mode changes how
requests are grouped rather than how a kernel computes. Build it and write it through the
ordinary profile write path:

```python
from ironmule import service_strategy as ss
from ironmule.tune import load_profile, save_profile

profile = load_profile(model_id)
record = ss.build_record(
    strategy=ss.STRATEGY_PAIRED, min_ready=2, max_ready=4,
    identity_sha256=..., fingerprint=..., mlx=..., mlx_lm=...,
    correctness_contract="...", evidence_run_ids=["B51_identity_gap_20260909"],
)
save_profile(ss.attach(profile, record))
```

`attach` returns a copy and refuses a record it would not read back; nothing is written
until `save_profile` is called, so reading a profile never migrates it. Test the migration
on a working copy first by pointing `IRONMULE_HOME` at one.

## The aligned extension, off and staying off

`PairedThroughputMode(share_aligned=True)` also shares `o_proj` (`K=4096`) and
`down_proj` (`K=15360`). Both kernels are bit-identical and faster in isolation, but at
product level the extension bought about **1 per cent** against the default paired path,
against a 5 per cent goal (`B48`). It is available, off by default, and not recommended.

## When it helps

Eight load cases against the unchanged `ThroughputMode`, each decided on its own, run
twice (`B50`). Both runs agree on every case, and no case regressed. Group completion
ratio, lower is better:

| load case | verdict | selection | confirmation |
| :-- | :-- | --: | --: |
| single request, short output | no advantage | `0.9965` | `1.0045` |
| single request, long output | no advantage | `0.9983` | `1.0028` |
| two requests, short prompt, 8 tokens | advantage | `0.9207` | `0.9225` |
| two requests, short prompt, 48 tokens | advantage | `0.8537` | `0.8513` |
| two requests, long prompt, 8 tokens | advantage | `0.9604` | `0.9651` |
| two requests, long prompt, 48 tokens | advantage | `0.8876` | `0.8836` |
| two requests, staggered arrival | advantage | `0.8882` | `0.8917` |
| four requests, served as two pairs | advantage | `0.9411` | `0.9481` |

Read it as three rules, all using facts known when the decision is made:

* **Two requests must be ready at the same step.** A single request shows zero paired
  steps and no effect; the path never waits for a partner.
* **More requested tokens, more gain.** `max_tokens` 48 gave `0.85` to `0.89`; 8 gave
  `0.92` to `0.97`. Every case here stopped on the token limit, so a request that ends
  early on a real end token spends fewer shared steps and would gain less.
* **Longer prompts, less gain.** Prefill is per request and never shared.

No average is given: it would hide a case that regressed, and the point of the table is
that none did. Loading and compiling cost `13.1 s` once, outside these numbers and equal
for both paths.

## Requirements

Admission runs once at load and refuses loudly outside the qualified box:

| | required |
| :-- | :-- |
| hardware fingerprint | `dc652d66f24ac207` (this M1 Max) |
| MLX / mlx_lm | `0.32.0` / `0.31.3` |
| model identity | `2b5b13a3…1474`, `mlx-community/gemma-3-12b-it-4bit` |
| architecture | `gemma3` |
| projections | reduction length `3840`, 4 bit, group size 64, bfloat16 scales, output width a multiple of 8 |

`K == 3840` on its own admits nothing. A projection outside the set stays on the library
path, and so do prefill, multi-token inputs, sampling and grouped execution.

## What is verified

Logit bit patterns, tokens, stop reasons and the full KV state match independent library
runs, per request. All eight load cases above were re-checked at bit level (`B51`): the
full logit bit pattern of every step and the whole KV state of every request, plus tokens,
stop reason and step count, identical in all ten cases and 22 requests. Two of those cases
end on a genuine end token, one where a request stops after 19 steps while the other runs
to 64, one where both stop, at 19 and 17. Operational cases checked on real model
computation (`B47`): a genuine end token on one request while the other continues, unequal
lengths returning to the single path, a late partner joining only at a step boundary, and
a lone request that never waits. An injected fault in the shared step falls back through the existing mechanism with
no duplicate tokens; that one is a control-flow test, not a naturally occurring error.

## What is not verified

**Cancellation mid-flight.** `ironmule.service.Request` carries `prompt_ids`,
`max_tokens`, `plan` and `arrival_ms`, and `serve()` runs to completion. There is no
cancel handle on the shipped surface, so a mid-flight cancellation could not be exercised.
This is a gap, not a pass.

Everything outside the table above: other models, other reduction lengths, width four,
sampling, and any machine that is not this one.

## Fallback

`ironmule.qmv_k3840.disable(model)` restores the original module objects. Within the
paired mode, an error in the shared step routes through the executor's existing
sequential fallback; a caller must still discard a KV cache built during a failed step.
