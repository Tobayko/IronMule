# `B7fix_*_INVALIDATED_swap.json` — discarded, do not use as evidence

These two files are a confirmation run of `B7` that was **discarded under its own
preregistered kill criterion 2**: "swap delta is nonzero at any model size".

Swap during the valid `B7` runs was `0.06 MB` throughout. During this run macOS grew the
swap file from 1 GB to 4 GB and reached `2816 MB` in use, and every cell slowed by a
uniform `1.10x`–`1.15x`.

They are kept, renamed and documented rather than deleted because the uniformity is
itself informative: because the slowdown was flat across both arms, both model sizes and
all four batch sizes, it cancels in the ratios `B7` uses. Read only as a robustness
check, they reproduce the valid run's `submission` ratios to within `0.02`
(3.66/3.75/3.71/3.66 against 3.68/3.77/3.72/3.68).

That is a remark, not a result. `B7`'s tables use the swap-free numbers.
