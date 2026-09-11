# IronMule — Hardware-Aware AI Runtime for Apple Silicon

**IronMule** (Project Friday) is a tightly measured local LLM inference runtime for
Apple Silicon, built on MLX around Gemma 3 (4-bit). It is a research vehicle, not a
product: every performance claim is bound to a preregistration, a sealed code hash,
paired AB/BA sampling, an A/A noise gate, and exact token identity. A number that has
not passed that pipeline is labelled *exploratory*, not shipped.

> **Status 2026-09-11.** The research tree now carries a complete local learning
> lifecycle, measured end to end on one Apple M1 Max: cold start (`B75`), fourteen
> sessions of temporal evidence (`B76`), a re-scoring that found the old verdict rule
> defective and kept the verdict it produced (`B77`), a persistent controller (`B78`),
> opt-in learned dispatch on real requests (`B79`), passive drift monitoring (`B80`),
> explicit requalification with every refusal path exercised (`B81`), and one live
> requalification that passed (`B82`). Learned dispatch is off by default. Nothing here
> generalises to a second machine; `B73` still needs one.
>
> **Earlier status, kept.** A 2026-09-03 review found that several benchmark harnesses
> added during the 2026-09-02/03 "serving" work measured incorrectly (single-shot
> instead of paired, lazy MLX graphs evaluated once, baseline truncated to candidate
> length) and that a former auto-tuner wrote fabricated statistics into the sealed
> device profile. The harnesses were repaired and the affected numbers re-measured the
> same day on real hardware. The three headline "breakthroughs" of the Gemini phase —
> sub-4-bit `1.24x`, double-buffering `+2.9%`, prompt-lookup `+29%` bit-exact — were all
> measurement artifacts and remain **refuted**. Those numbers are exploratory (session
> noise floor MDE 2.86%). See `docs/GEMINI_SELF_LEARNING_SYSTEM.md`.

---

## What it does

- Loads one Gemma 3 model into Apple Silicon Unified Memory and serves it behind an
  OpenAI-compatible HTTP/SSE endpoint (`/v1/chat/completions`, `/v1/models`).
- Applies an engine knob only if *this device's* calibration profile verified it as
  token-identical (`head_skip_prefill`, `compiled_fixed_cache`, `readback_every = 8`).
  An unverified knob is off; a profile that verified nothing serves the baseline.
- Derives request scope from the actual tokens and the loaded model, never from a
  caller's label; a request outside the calibrated scope (non-greedy, batched, wrong
  model revision) runs the baseline.
- Latches a persistent circuit breaker on any failure of an optimised path, so a knob
  that failed once stays off across restarts.
- Runs the LinUCB `AdaptiveRLController` in **shadow mode only**: it logs the knob set
  it *would* pick (`.friday-data/rl-shadow-decisions.jsonl`) but never applies it and
  never learns from a serving request. RL stays NO-GO until R2.

---

## Measurement tools (`python tools/friday.py <tool>`)

Every measuring tool refuses to run without an explicit `--execute` and enforces the
shared gate: AC power, `BudgetGuard` duty cycle, offline environment, paired sampling.

| Command | Script | What it measures |
| --- | --- | --- |
| `loop` | `optimization_loop.py` | Self-optimisation loop over execution plans, confirms its own winner |
| `dispatch` | `measure_dispatch_plan.py` | One execution plan vs. baseline, paired, frozen threshold |
| `cooldown` | `measure_cooldown_effect.py` | How an idle pause slows the next operation |
| `aa` | `run_h0_aa.py` | The preregistered H0 A/A null control (calibration, no optimisation) |
| `model-loop` | `model_loop.py` | H2: a local model proposes execution plans, the harness judges them |
| `codegen` | `codegen_loop.py` | H2 full: a local model writes execution plans, sandboxed and judged |
| `roofline` | `measure_roofline.py` | Whether inference is memory- or compute-bound per phase |
| `fusion` | `measure_fusion_layer.py` | Fuse a model's forward pass, measure the gain |
| `guard` | `run_h01_guard.py` | Verify the H0.1 analysis core stays stdlib-only (no GPU) |
| `evidence` | `evidence.py` | Verify or display the append-only H1/H2 evidence history (no GPU) |
| `serve` | `run_serve.py` | Start the OpenAI-compatible HTTP/SSE server + terminal cockpit |
| `monitor` | `monitor.py` | Remote in-place terminal telemetry cockpit |

Device-profile calibration is separate: `python tools/run_calibration.py run --execute`
(paired AB/BA, bootstrap CI, A/A noise gate) is the **only** writer of the sealed
`.friday-data/device-profile.sqlite3`.

---

## Quickstart

```bash
# Apple Silicon Mac, macOS 14+, Python 3.12+
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-apple-silicon.txt

python tools/friday.py doctor          # verify Metal GPU, Python, AC power
python tools/friday.py status          # device profile, verified knobs, runtime state
python tools/run_calibration.py run --execute --pairs 6   # calibrate this machine
python tools/friday.py serve --port 8080
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4b", "stream": true,
       "messages": [{"role": "user", "content": "Why is Apple Silicon fast?"}]}'
```

---

## Testing

```bash
pytest tests/          # -n auto is set in pytest.ini
```

Hardware execution paths use no mocks or simulations; a test that claims GPU, MLX or
model behaviour must have run it on the target device. Synthetic data is allowed only
for edge and error cases and grounds no performance claim.

---

## Empirical findings (bound to evidence)

The load-bearing results and their retractions live in
[`docs/ERGEBNISSE.md`](ERGEBNISSE.md); the full history is in the append-only
[`research/LEDGER.md`](../research/LEDGER.md).

1. **Unpaired variance dwarfs the effects.** Run-to-run variance on the M1 Max is far
   larger than any optimisation gain measured; every calibration therefore uses paired
   block sampling with a bounded confidence interval. This is the project's central
   result — an unpaired number here is meaningless.
2. **Prefill dominates the short-answer regime.** For the sealed 897-token / 32-token
   workload prefill is ~80 % of the request; a decode-only knob cannot reach the F1
   end-to-end threshold in that regime.
3. **Prompt-lookup speculation loses on the delivery workload.** Acceptance is 0.0 on
   the sealed prompt; `speculate_k` stays 0 in the delivery path and speculation is
   never treated as token-identical (bf16 1-ULP breaks). See
   `docs/GEMINI_SELF_LEARNING_SYSTEM.md` E01–E03.
4. **The device profile replaces frozen host constants.** A macOS update once broke a
   sealed hardware hash on the origin machine itself; a profile asks instead whether a
   knob was verified token-identical on *this* device against *this* model snapshot.

---

## The local learning lifecycle (`B75` to `B82`)

Measured on one Apple M1 Max, `gemma-3-12b-it-4bit`, single short requests on the
sequential path. Every study's preregistration and verdict is in
[`research/LEDGER.md`](../research/LEDGER.md), including the ones that failed.

| study | what it settled |
| :-- | :-- |
| `B75` | a fresh install measured its way to a defensible local decision in `286 s` |
| `B76` | fourteen independent sessions: the sign is stable, the magnitude is not |
| `B77` | the verdict rule that produced `B76`'s `FAIL` conflated three quantities; the verdict stands and the defect is documented |
| `B78` | a persistent controller that qualifies nothing without enough evidence |
| `B79` | the first dispatch a learned preference was allowed to change, opt-in, with an immediate reference fallback |
| `B80` | passive drift monitoring: zero false alarms, a defined shift caught in eight observations |
| `B81` | explicit requalification; two live attempts refused by their own gates, the reference kept serving |
| `B82` | a readiness probe, then one live requalification that passed and handed the action back |

**What is not shown.** Anything about a second machine, any other Apple Silicon
generation, reinforcement learning, a universal gain, or activation by default. See
[`docs/LIMITS.md`](LIMITS.md).

## License

MIT License. Developed as part of Project Friday research.
