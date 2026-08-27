# Agent instructions

**Start at [`docs/BACKLOG.md`](docs/BACKLOG.md).** It is the work list for anything meant
to make the runtime faster: every open hypothesis with its mechanism, the evidence for
and against it, a test, and the result that would close it.

Read **Tier 0** before proposing an optimisation. It records what has already been
measured and rejected, and each of those entries cost real GPU time to learn.

Three conventions:

1. **Work from it.** Work that is not an entry gets added as one first. An entry needs a
   mechanism and a kill criterion; without a kill criterion it is a wish.
2. **Delete what is finished.** An answered entry leaves its tier in the same change that
   answers it. The result moves to [`research/LEDGER.md`](research/LEDGER.md) if it
   shipped, or to Tier 0 as one line with its number if it was rejected.
3. **Put back what you learn.** New idea, new dead end, a number that surprised you — the
   same day, even half-formed.

The rest — evidence standards, correctness gates, the validity domain — is in
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/LIMITS.md`](docs/LIMITS.md) and
[`research/LEDGER.md`](research/LEDGER.md). No performance claim ships without a
measurement and a stated baseline.
