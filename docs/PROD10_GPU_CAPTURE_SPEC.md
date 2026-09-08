# PROD10 — private Metal GPU capture, diagnostic only

This is the bounded next diagnostic for PROD10’s open GPU/host bottleneck
question. It does not benchmark an optimisation and does not establish a GPU
duration or speedup claim.

`tools/product_gpu_capture.py --execute` starts one owned isolated worker with
`MTL_CAPTURE_ENABLED=1`. Before MLX is imported, the worker resolves the
registered immutable Gemma 3 12B snapshot and validates it with the installed
model policy. It then loads unmodified MLX-LM stock generation and performs
exactly four `long_8` requests: three warmups with capture availability enabled
but no active capture, followed by one request enclosed solely by
`mx.metal.start_capture(path)` / `stop_capture()` in a `finally` block.
The isolated `-I` worker loads the shared stock protocol helper by an exact
`importlib` file specification pointing at the manifest-bound
`tools/product_open_validation.py`; it never adds the repository root to
`sys.path`, so project source packages cannot shadow the installed runtime.

Each request must exactly match the completed stock `long_8` row in
`research/raw/PROD10_12B_open_20260907_attempt2.json` for output, token and
text hashes, prompt/completion counts, and finish reason. The controller emits
only bounded hash/count metadata, source/package/provider/model identities,
worker PID/real exit, MLX load/peak memory, and trace existence/byte count after
the worker has closed. It records only the trace basename publicly.

The raw reference itself is source-manifest-bound. It is accepted only when its
top-level status, model, revision, four stock `long_8` rows, warmup schedule and
before/after model/environment/hardware/code identities are complete and stable.
The new run must match those identities. Provider, snapshot-metadata, installed
module, source, hardware and host observations are captured before and after;
resource observations are appended to the private event journal without acting
as admission or termination gates. An exclusive product model lease covers the
worker's complete lifetime.

The trace lives at `.friday-data/profiles/<fresh-run>/capture.gputrace` and can contain model tensors;
it must never be published or copied into a public report. A missing, empty, or
unsupported capture is an explicit failed result—there is no fallback that
pretends a capture succeeded. The parent has no request timeout, no budget/duty
cycle/readiness/RSS/swap kill gate; Ctrl-C still triggers owned-worker cleanup.
The child uses a private umask. Cleanup uses the already-proven bounded protocol
and terminate/wait/kill fallback, and completes before trace inspection,
after-identities, terminal journaling, or final report hashing. Failures retain
completed samples, bounded partial-frame metadata and the primary error.

Trace metadata uses `lstat`/`scandir` without following aliases. Future `size_bytes` is the
logical byte total of unique regular-file inodes (hardlinks counted once); `allocated_bytes`
is the corresponding unique-inode filesystem allocation, and metadata also records regular-file
and symlink counts. The historical attempt2 `size_bytes` field was produced by a recursive
`Path.stat()` sum that followed 631,458 symlink aliases and is therefore alias-inflated and
invalid as a payload-size measurement; it must not be compared with the corrected field.

Capture/replay wall timing is diagnostic correlation for later Xcode analysis,
not untraced GPU time. No model run occurred while this harness/spec was added.

## Navigation note

ProjectAtlas runtime `0.4.5-rc1` was available, but its requested focused
navigation returned `refresh_required: dependency_closure_limit`, exactly
10,001 modified paths, and prescribed a full scan. The task explicitly forbade
that scan, so implementation used the bounded CLI fallback limited to this
specification, `product_open_validation.py`, the two cited helper modules, and
the one named raw reference export. This is not a claim that the index is fresh.
