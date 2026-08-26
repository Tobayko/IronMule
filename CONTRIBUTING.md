# Contributing

Issues and pull requests are welcome.

For a first contribution, start with the [README](README.md), install the published
package with `python -m pip install ironmule`, and use the checkout installation below
when you need the test suite or source tree. Performance and correctness changes need
the evidence described in the research ledger.

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
