### `R13` — E12 cannot take R12's fork without a granularity decision first

**Mechanism.** `R12` gave `E14b` and `E15` one OS process per block, which held their
per-block peaks flat and cost nothing — measured at 12B, a forked block ran `147 s`
against `157 s` in-interpreter, because allocator pressure costs more than a process
start plus a model load. `E12` looks like the same shape and is not, for one reason that
makes the whole difference:

| Harness | Signature | Where the model is loaded |
| :-- | :-- | :-- |
| `e14b_arms.py:163` | `run_process(model_id, index, pilot)` | inside, `Harness(model_id)` |
| `e15_service.py:302` | `run_process(model_id, index, pilot)` | inside, `load_engine(model_id, KNOBS)` |
| `e12_window_falsification.py:252` | `run_case(h: Harness, kind, length, preamble, corpus, model_id)` | **outside, once, shared** |

`E14b` and `E15` already load a model per block, so forking *moves* a load that happens
anyway — the fork is free, and measurably slightly cheaper. `E12` builds one `Harness`
before its loop (`:439`), builds every preamble once, and runs its warmup once outside.
`PREFIX_LENGTHS` has 13 entries and `--types` defaults to `natural,synthetic`, so a fork
per case turns **1 model load into 26** — genuinely added work, not relocated work.

That collides with a limit `E12` already has: `STAGE_WALL_LIMIT_S = 45 * 60` (`:42`,
used at `:456`). `E15` measured `26.2 min` for four blocks with 18.8 min of headroom, and
that headroom survives only because its fork adds nothing. A 26-load version of `E12`
would spend its headroom on model loading and could trip the limit it is meant to be
protected by.

So the question `E12` raises is not "port or not" but **at what granularity**, and that
is a design decision with no obviously right answer:

- **Per case** — 26 loads. Cleanest isolation, almost certainly unaffordable.
- **Per type** — 2 forks of 13 cases each. Carry-over persists within a type but cannot
  cross between them, which is where the two arms are compared.
- **Not at all** — `E12`'s cases share one `Harness` deliberately; whether allocator
  carry-over even distorts them the way it distorted `E14b` is unmeasured.

**Test.** Before choosing, measure what is actually wrong: run `E12` as it stands and
record the per-case peak across all 26 cases. If it climbs the way `E14b`'s did
(`17.51 -> 23.14 GB`) the problem transfers and the granularity question is worth
answering. If it stays flat, there is nothing to fix and this entry closes without any
code. Then, only if it climbs, time one `E12` case to price the per-type option against
the wall limit.

**Kill.** Per-case peaks stay flat across the 26 cases — then `E12`'s shared harness does
not accumulate the way a per-block model load does, `R12` does not transfer, and `E12`
keeps its structure. This is the likely outcome and the cheap one to check.

It also closes as *unaffordable* if peaks do climb but per-type forking leaves less than
the current 18.8 min of wall headroom: then the fix costs more than the defect, and the
honest move is to record the carry-over as a limitation the way `M2` did, rather than
restructure an experiment that has already run.

**Not claimed.** `E14b`'s carry-over was measured on a 12B model at `17.51 GB` per block,
where the machine had 32 GB. `E12` runs at 4B and peaked at `9.62 GB` in the comparable
`E15` run. Nothing says an effect that mattered at 55% of memory matters at 30%, and this
entry must not assume it does.
