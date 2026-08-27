# Contributing

Issues and pull requests are welcome.

For a first contribution, start with the [README](README.md), install the published
package with `python -m pip install ironmule`, and use the checkout installation below
when you need the test suite or source tree. Performance and correctness changes need
the evidence described in the research ledger.

## Start at the backlog

**[`docs/BACKLOG.md`](docs/BACKLOG.md) is the work list for anything meant to make the
runtime faster.** Every open hypothesis is there with its mechanism, the evidence for
and against it, a test, and the result that would close it.

Read **Tier 0** before proposing an optimisation. It lists what has already been
measured and rejected, in this project and in the one it grew out of — a draft model
for speculation, compiling decode subgraphs on a growing cache, projection fusion as a
decode win, a dedicated `M=1` fast path, and more. Each cost real GPU time to learn.

Three conventions keep that file usable, and a pull request that touches it is expected
to follow them:

1. **Work from it.** If the work is not an entry, add it as one first. An entry needs a
   mechanism and a kill criterion — what result would close it for good. Without a kill
   criterion it is a wish, not a hypothesis.
2. **Delete what is finished.** An answered entry leaves its tier in the same pull
   request that answers it. The result moves to `research/LEDGER.md` if it shipped, or
   to Tier 0 as one line with its number and experiment ID if it was rejected. A backlog
   that keeps its corpses stops describing what is left to do.
3. **Put back what you learn.** A new idea, a new dead end, a number that surprised you
   — the same day, even half-formed, even if it is probably wrong.

## The one rule that matters here

**A performance claim needs a measurement.**

If a change is meant to be faster, say how you measured it and against what
baseline. `research/LEDGER.md` shows the format that is used throughout this
project: a question, competing explanations, criteria fixed before the run, the
result, and what it does not show.

Four of the sixteen experiments in that ledger exist because a promising idea turned
out not to work. Negative results are kept, not deleted — they are the reason the
positive ones are worth anything.

## Practical

```bash
pip install -e ".[dev]"
pytest tests/test_ironmule_runtime.py -q              # fast, no model needed
pytest tests/test_ironmule_runtime_integration.py -q  # needs a local model snapshot
python -m ironmule.benchmark --requests 6 --max-tokens 48
```

Before opening an environment or installation issue, run `ironmule doctor` and include
its output. To share a result from another Mac, use the [benchmark issue template](.github/ISSUE_TEMPLATE/benchmark_submission.md)
and paste the complete benchmark output.

A change that touches the executor, the plans or the cache must keep
`tests/test_ironmule_runtime.py` green. Those tests cover token identity, stop
reasons, ragged lengths, early finishers, reversed order, staggered arrival, group
widths one to four, fallback, and the absence of state aliasing.

## Contributions and licence

By contributing you grant the licensor the rights described in §10.6 of
[`LICENSE.md`](LICENSE.md). No copyright assignment is required and you keep all
other rights in your contribution.
