### `R11` — Gate measurement on swap, not on a hard-coded byte count

**Mechanism.** Three harnesses abort when a block's MLX peak exceeds a literal
`12 * 1024**3` (`e14_dispatch.py`, `e14b_arms.py`, `e15_service.py`). The number is
doing a job it cannot do. What actually invalidates a timing run is the machine
swapping; peak allocation is only a proxy for it, and a badly calibrated one. On this
32 GB machine the proxy is wrong in both directions at once:

- **Too strict.** `gemma-3-12b-it-4bit` peaks at `17.51 GB` per block and is refused,
  although `B7` measured it with swap steady at `0.06 MB` — the machine was never under
  pressure. 12B is therefore capped at one block, which is not a sample size, and every
  scaling question beyond 4B is blocked. Gemma 3 27B (`~16.78 GB` in `SCALING.md`) and
  Qwen 27B (`14.98 GiB` of weights) are refused for the same non-reason.
- **Too lax.** `B7`'s confirmation run stayed *under* the guard at every block and was
  still invalid: macOS grew the swap file from 1 GB to 4 GB and reached `2816 MB` in
  use, and every cell slowed by a uniform `1.10x`–`1.15x`. The guard passed a run that
  had to be discarded, because it was watching the wrong thing.

Raising the constant fixes neither half. It moves the too-strict edge to the next model
and leaves the too-lax edge exactly where it is.

**Test.** Replace the literal with a measured condition. Sample `vm.swapusage` before
the first block and after every block; abort when swap *in use* rises by more than a
preregistered delta above the run's own starting value, and record that delta with the
result. Keep a byte ceiling only as a coarse backstop against a genuinely unbounded
allocation, set from the machine's installed memory rather than typed in, and make it a
parameter with a documented default rather than a literal in three files.

Verify against both known cases: `B7`'s 12B run (peak `17.51 GB`, swap flat) must
complete four blocks, and `B7`'s discarded confirmation run (peak under the guard, swap
`+2816 MB`) must abort. A synthetic case that allocates without swapping must not abort.

**Kill.** Swap is a machine-wide signal, so an unrelated process can trip it and abort a
valid run — the condition must therefore record what it saw rather than only failing,
or it trades a false pass for a false abort and nothing is gained. It also closes as a
failure if swap proves to be a lagging indicator on this platform: if timings degrade
measurably before `vm.swapusage` moves, the gate fires after the damage and a memory
*pressure* signal is needed instead. Both are measurable on the two runs above before
any code changes.

**Related.** `R10` records the other half of this: when a guard does fire, the abort must
be visible in the result file rather than only on stdout. The two are worth doing
together — a gate that fires correctly and silently is still a gate that produces a
truncated file looking like a finished one.
