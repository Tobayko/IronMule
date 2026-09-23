# PERF1 run 7: the `native` plan through the product, not the harness. Private notebook,
# internet on. Free quota: one run, <= 50 min. Backlog: PERF1-H.
#
# Runs 1-6 measured the kernel through `perf1.py`'s global patch. This runs IronMule itself
# with the working tree's `ironmule/cuda_native.py` (`compute_dtype="native"`: installed per
# model after fusion, probe-gated, pinned arithmetic), carried as a patch against the pinned
# commit. Rules fixed here:
#   * `cross.py`, the harness every numeric-plan cell comes from, 2 interleaved repetitions:
#     stock against native with no knobs, with `compiled_fixed_cache`, and with PORT2's tuned
#     knobs in throughput mode. The knobbed arms must reproduce the knob-free native tokens
#     (6/6), else that combination is reported as not exact under the plan.
#   * product smoke: `ironmule doctor`, `ironmule benchmark --compute-dtype native`, and
#     `ironmule serve --compute-dtype native` answering JSON, SSE and four concurrent requests,
#     with `health` reporting the plan.
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
CROSS_SOURCE = '"""PORT1: stock against IronMule in separate processes, the like-for-like CUDA comparison.\n\nUsage: python cross.py MODEL_ID REVISION CONFIGS_JSON OUT.json [REPS] [MEASURE]\n\n`abcd.py` runs every arm in one process, so an environment setting such as IronMule\'s CUDA\ngraph default reaches its baseline arm too. Here each configuration runs in fresh\nprocesses interleaved by repetition: "stock" pins MLX\'s own graph limits, baseline knobs\nand interactive mode; IronMule configurations use whatever IronMule applies by itself.\nThe first configuration is the reference. A configuration with `"dtype": "float32"` changes\nnumerics, so it reports token agreement instead of claiming identity.\nCONFIGS_JSON: [{"name", "env": {...}, "knobs": {...}, "mode": "interactive"|"throughput",\n"dtype": null|"float32"}, ...]\n"""\nimport hashlib\nimport json\nimport os\nimport statistics as st\nimport subprocess\nimport sys\nimport time\n\n\ndef child(model_id, revision, config_json, measure):\n    import mlx.core as mx\n\n    import ironmule\n    from ironmule import benchmark as bench\n    from ironmule.tune import load_engine\n\n    config = json.loads(config_json)\n    # The product path: IronMule\'s own opt-in numeric plan, not a post-load cast.\n    engine, tokenizer = load_engine(model_id, ironmule.Knobs(**config["knobs"]), revision=revision,\n                                    compute_dtype=config.get("dtype"))\n    mode = ironmule.ThroughputMode if config["mode"] == "throughput" else ironmule.InteractiveMode\n    with ironmule.Runtime(engine, tokenizer, model_id=model_id) as rt:\n        walls, outputs = [], None\n        for index in range(1 + measure):\n            requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", 6, 48)\n            results, snapshot = bench._run(rt, ironmule, mode(), requests)\n            outputs = [[int(t) for t in r.tokens] for r in results]\n            if index:\n                walls.append(snapshot["outer_wall_ms"])\n    print(json.dumps({"walls_ms": walls, "median_ms": st.median(walls), "tokens": outputs,\n                      "outputs_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),\n                      "peak_memory_bytes": int(mx.get_peak_memory()),\n                      "graph_env": {k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")}}))\n\n\ndef main(model_id, revision, configs_json, out, reps, measure):\n    configs = json.loads(configs_json)\n    runs = {c["name"]: [] for c in configs}\n    started = time.time()\n    for rep in range(reps):\n        order = configs[rep % len(configs):] + configs[:rep % len(configs)]\n        for config in order:\n            env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_MAX_")}\n            env.update(config.get("env", {}))\n            proc = subprocess.run([sys.executable, __file__, "child", model_id, revision, json.dumps(config),\n                                   str(measure)], env=env, capture_output=True, text=True, timeout=1800)\n            row = {"rep": rep, "exit": proc.returncode, "stderr_tail": proc.stderr[-1500:]}\n            if proc.returncode == 0:\n                row.update(json.loads(proc.stdout.strip().splitlines()[-1]))\n            runs[config["name"]].append(row)\n            print(config["name"], rep, row.get("median_ms"), row["exit"], flush=True)\n    base = configs[0]["name"]\n    reference = next((r["tokens"] for r in runs[base] if r.get("tokens")), None)\n    summary = {}\n    for config in configs:\n        name = config["name"]\n        pairs = [r["median_ms"] / b["median_ms"] for r, b in zip(runs[name], runs[base])\n                 if r.get("median_ms") and b.get("median_ms")]\n        tokens = next((r["tokens"] for r in runs[name] if r.get("tokens")), None)\n        summary[name] = {\n            "ratio_vs_reference_per_rep": pairs,\n            "median_ratio": st.median(pairs) if pairs else None,\n            "max_ratio": max(pairs) if pairs else None,\n            "identical_requests": (sum(a == b for a, b in zip(tokens, reference))\n                                   if tokens and reference else None),\n            "deterministic_across_processes": len({r.get("outputs_sha256") for r in runs[name]}) == 1,\n            "failed_processes": sum(r["exit"] != 0 for r in runs[name]),\n        }\n    report = {"schema": "ironmule.port1-cross.v1", "model_id": model_id, "revision": revision,\n              "configs": configs, "reps": reps, "measure": measure, "runs": runs, "summary": summary,\n              "seconds": time.time() - started, "performance_claim": False}\n    with open(out, "w") as stream:\n        json.dump(report, stream, indent=1)\n    print(json.dumps(summary, indent=1))\n\n\nif __name__ == "__main__":\n    if sys.argv[1] == "child":\n        child(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))\n    else:\n        main(*sys.argv[1:5], int(sys.argv[5]) if len(sys.argv) > 5 else 3,\n             int(sys.argv[6]) if len(sys.argv) > 6 else 3)\n'  # experiments/kaggle_compat/cross.py
PATCH_SOURCE = 'diff --git a/ironmule/benchmark.py b/ironmule/benchmark.py\nindex d16d347..a06de58 100644\n--- a/ironmule/benchmark.py\n+++ b/ironmule/benchmark.py\n@@ -669,7 +669,7 @@ def main(argv: list[str] | None = None) -> int:\n     parser.add_argument("--plan", choices=["strict", "reusable"], default="strict")\n     parser.add_argument("--model", default=None)\n     parser.add_argument("--json", type=Path, default=None)\n-    parser.add_argument("--compute-dtype", choices=("float32",), default=None,\n+    parser.add_argument("--compute-dtype", choices=("float32", "native"), default=None,\n                         help="opt-in numeric plan for GPUs that emulate bf16; changes output")\n     args = parser.parse_args(argv)\n     if args.requests < 1 or args.max_tokens < 1:\ndiff --git a/ironmule/cuda_native.py b/ironmule/cuda_native.py\nnew file mode 100644\nindex 0000000..7488ae5\n--- /dev/null\n+++ b/ironmule/cuda_native.py\n@@ -0,0 +1,203 @@\n+"""Native 4-bit matmuls for CUDA GPUs that emulate bfloat16 — the opt-in `native` plan.\n+\n+Below compute capability 8 (Volta, Turing) CUDA has no bfloat16 arithmetic. MLX\'s CUDA\n+`quantized_matmul` accumulates in the activation type, so a bfloat16 checkpoint spends its\n+decode step in emulation: Qwen 3 8B decoded at ~6.5 tok/s on a T4 whose bandwidth allows\n+~70 (PORT1, PERF1). This plan keeps the checkpoint and its bfloat16 activations and changes\n+only how the 4-bit affine matmuls are computed:\n+\n+* up to `MAX_ROWS` activation rows (decode, and the small batches of a server) go through one\n+  kernel that reads activations, scales and biases as raw 16-bit patterns — a bfloat16 is the\n+  top half of a float32, so widening is a shift — accumulates in float32, and rounds once;\n+* more rows (prefill) dequantise the weight to float16 and run one cuBLAS GEMM.\n+\n+It changes output, so it is opt-in like every numeric plan (`compute_dtype="native"`); its\n+measurements and quality gate are in `research/LEDGER.md`, PERF1. The arithmetic is pinned\n+(`__fmaf_rn`, `__fadd_rn`): each output row is computed with the same roundings whatever\n+else shares the launch, so a request\'s answer does not depend on how many others the server\n+batches with it. Installation is scoped to one model: eligible modules have their class\n+swapped, nothing global is patched, and nothing is installed unless a probe on the model\'s\n+own first eligible weight agrees with a float32 reference.\n+"""\n+\n+from __future__ import annotations\n+\n+from typing import Any\n+\n+import mlx.core as mx\n+import mlx.nn as nn\n+\n+GROUP_SIZE = 64\n+BITS = 4\n+MAX_ROWS = 8\n+PROBE_TOLERANCE = 1e-2\n+\n+HEADER = r"""\n+__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }\n+__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }\n+__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }\n+__device__ __forceinline__ unsigned short to_bf(float f) {\n+  unsigned int u = __float_as_uint(f);\n+  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;\n+  u += 0x7fffu + ((u >> 16) & 1u);\n+  return (unsigned short)(u >> 16);\n+}\n+"""\n+\n+# One warp per R output rows and M <= 8 activation rows. A uint4 of packed weights is 32\n+# nibbles inside one group of 64 (MLX packing: low nibble first). Each chunk\'s nibbles are\n+# converted once and reused for all M rows.\n+SOURCE = r"""\n+  const unsigned int gid = cooperative_groups::this_grid().thread_rank();\n+  const unsigned int row0 = (gid >> 5) * R;\n+  const unsigned int lane = threadIdx.x & 31;\n+  if (row0 >= N) return;\n+  const uint4* w4 = reinterpret_cast<const uint4*>(w);\n+  const uint4* xv = reinterpret_cast<const uint4*>(x);\n+  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);\n+  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);\n+  float acc[R * M];\n+#pragma unroll\n+  for (int i = 0; i < R * M; ++i) acc[i] = 0.f;\n+  for (unsigned int c = lane; c < K / 32; c += 32) {\n+    const unsigned int g = (c * 32) / GS;\n+    float xf[32];\n+    float sxm[M];\n+#pragma unroll\n+    for (int mo = 0; mo < M; ++mo) {\n+      LOAD_X\n+      float sx = 0.f;\n+#pragma unroll\n+      for (int i = 0; i < 32; ++i) sx = __fadd_rn(sx, xf[i]);\n+      sxm[mo] = sx;\n+    }\n+#pragma unroll\n+    for (int r = 0; r < R; ++r) {\n+      const unsigned int row = row0 + r;\n+      if (row < N) {\n+        const uint4 q = w4[(size_t)row * (K / 32) + c];\n+        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};\n+        float qf[32];\n+#pragma unroll\n+        for (int j = 0; j < 4; ++j) {\n+#pragma unroll\n+          for (int e = 0; e < 8; ++e) qf[j * 8 + e] = (float)((qw[j] >> (4 * e)) & 0xFu);\n+        }\n+        const size_t sg = (size_t)row * (K / GS) + g;\n+        const float s = one_bf(ss[sg]);\n+        const float b = one_bf(bs[sg]);\n+#pragma unroll\n+        for (int mo = 0; mo < M; ++mo) {\n+          if (M > 1) {\n+            LOAD_X\n+          }\n+          float qx = 0.f;\n+#pragma unroll\n+          for (int i = 0; i < 32; ++i) qx = __fmaf_rn(qf[i], xf[i], qx);\n+          acc[r * M + mo] = __fadd_rn(acc[r * M + mo], __fmaf_rn(s, qx, __fmul_rn(b, sxm[mo])));\n+        }\n+      }\n+    }\n+  }\n+#pragma unroll\n+  for (int i = 0; i < R * M; ++i) {\n+    float a = acc[i];\n+#pragma unroll\n+    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_down_sync(0xffffffffu, a, o));\n+    const unsigned int row = row0 + i / M;\n+    if (lane == 0 && row < N) out[(i % M) * N + row] = to_bf(a);\n+  }\n+""".replace("LOAD_X", """\n+#pragma unroll\n+    for (int j = 0; j < 4; ++j) {\n+      const uint4 xa = xv[mo * (K / 8) + c * 4 + j];\n+      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};\n+#pragma unroll\n+      for (int e = 0; e < 4; ++e) {\n+        xf[j * 8 + 2 * e] = lo_bf(xw[e]);\n+        xf[j * 8 + 2 * e + 1] = hi_bf(xw[e]);\n+      }\n+    }\n+""")\n+\n+_kernel: Any = None\n+\n+\n+def _rows_per_warp(n: int) -> int:\n+    # PERF1 run 1: two rows per warp won on the small Qwen 3 shapes, four from N >= 8192.\n+    return 4 if n >= 8192 else 2\n+\n+\n+def _qmv(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:\n+    global _kernel\n+    if _kernel is None:\n+        _kernel = mx.fast.cuda_kernel(name="ironmule_native_qmv4", input_names=["x", "w", "scales", "biases"],\n+                                      output_names=["out"], source=SOURCE, header=HEADER)\n+    n, k = w.shape[0], w.shape[1] * 8\n+    m = x.size // k\n+    rows = _rows_per_warp(n)\n+    threads = -(-(-(-n // rows) * 32) // 256) * 256  # whole blocks; surplus warps return at once\n+    out = _kernel(inputs=[mx.contiguous(x.reshape(m, k)), w, scales, biases],\n+                  template=[("N", n), ("K", k), ("GS", GROUP_SIZE), ("R", rows), ("M", m)],\n+                  grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(m, n)],\n+                  output_dtypes=[mx.uint16])[0]\n+    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)\n+\n+\n+def matmul(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:\n+    """`x @ dequantize(w).T` for a 4-bit, group-64 affine weight with bfloat16 scales."""\n+    if x.dtype != mx.bfloat16:\n+        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS)\n+    if x.size <= MAX_ROWS * x.shape[-1]:\n+        return _qmv(x, w, scales, biases)\n+    dense = mx.dequantize(w, scales.astype(mx.float16), biases.astype(mx.float16),\n+                          group_size=GROUP_SIZE, bits=BITS)\n+    return mx.matmul(x.astype(mx.float16), dense.T).astype(x.dtype)\n+\n+\n+class NativeQuantizedLinear(nn.QuantizedLinear):\n+    def __call__(self, x):\n+        y = matmul(x, self["weight"], self["scales"], self["biases"])\n+        return y + self["bias"] if "bias" in self else y\n+\n+\n+class NativeQuantizedEmbedding(nn.QuantizedEmbedding):\n+    def as_linear(self, x):\n+        return matmul(x, self["weight"], self["scales"], self["biases"])\n+\n+\n+def _eligible(module: nn.Module) -> bool:\n+    if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):\n+        return False\n+    if (getattr(module, "mode", "affine") != "affine" or module.bits != BITS\n+            or module.group_size != GROUP_SIZE or module.get("biases") is None):\n+        return False\n+    return module["scales"].dtype == mx.bfloat16 and (module["weight"].shape[1] * 8) % 32 == 0\n+\n+\n+def _probe(module: nn.Module) -> float:\n+    """Relative error of the kernel on the module\'s own weight against a float32 reference."""\n+    w, scales, biases = module["weight"], module["scales"], module["biases"]\n+    x = mx.random.normal((1, w.shape[1] * 8), key=mx.random.key(0)).astype(mx.bfloat16)\n+    reference = x.astype(mx.float32) @ mx.dequantize(\n+        w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=GROUP_SIZE, bits=BITS).T\n+    error = mx.max(mx.abs(_qmv(x, w, scales, biases).astype(mx.float32) - reference))\n+    return float(error / mx.maximum(mx.max(mx.abs(reference)), 1e-6))\n+\n+\n+def install(model: nn.Module, device_info: dict[str, Any] | None) -> dict[str, Any]:\n+    """Swap every eligible quantised module of `model` to the native kernels, or refuse."""\n+    from .numeric_plans import CUDA_PRE_AMPERE, device_class\n+\n+    if device_class(device_info) != CUDA_PRE_AMPERE:\n+        raise ValueError("the native plan exists for CUDA GPUs below compute capability 8 only")\n+    modules = [m for _, m in model.named_modules() if _eligible(m)]\n+    if not modules:\n+        raise ValueError("the native plan found no 4-bit group-64 affine bfloat16 weights")\n+    error = _probe(modules[0])\n+    if not error <= PROBE_TOLERANCE:\n+        raise ValueError(f"the native kernel disagrees with the float32 reference ({error:.3g})")\n+    for module in modules:\n+        module.__class__ = (NativeQuantizedEmbedding if isinstance(module, nn.QuantizedEmbedding)\n+                            else NativeQuantizedLinear)\n+    return {"modules": len(modules), "probe_rel_error": error}\ndiff --git a/ironmule/model_identity.py b/ironmule/model_identity.py\nindex efbe823..fa47dd6 100644\n--- a/ironmule/model_identity.py\n+++ b/ironmule/model_identity.py\n@@ -87,7 +87,12 @@ def _sha256_file(path: Path) -> str:\n \n def _files(root: Path) -> tuple[dict[str, Any], ...]:\n     rows = []\n-    allowed_root = root.parent.parent if root.parent.name == "snapshots" else root\n+    if root.parent.name == "snapshots":\n+        repo = root.parent.parent\n+        # huggingface_hub 1.32 moved blobs from `models--*/blobs` to one hub-wide `blobs/`.\n+        allowed_roots = (repo.resolve(), (repo.parent / "blobs").resolve())\n+    else:\n+        allowed_roots = (root.resolve(),)\n     for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):\n         if path.is_symlink() and not path.is_file():\n             raise ModelIdentityError(f"broken model symlink: {path.relative_to(root).as_posix()}")\n@@ -95,11 +100,14 @@ def _files(root: Path) -> tuple[dict[str, Any], ...]:\n             continue\n         if path.is_symlink():\n             try:\n-                path.resolve(strict=True).relative_to(allowed_root)\n-            except (OSError, ValueError) as exc:\n+                target = path.resolve(strict=True)\n+            except OSError as exc:\n                 raise ModelIdentityError(\n                     f"model symlink escapes its allowed root: {path.relative_to(root).as_posix()}"\n                 ) from exc\n+            if not any(target.is_relative_to(allowed) for allowed in allowed_roots):\n+                raise ModelIdentityError(\n+                    f"model symlink escapes its allowed root: {path.relative_to(root).as_posix()}")\n         relative = path.relative_to(root).as_posix()\n         rows.append({"path": relative, "bytes": path.stat().st_size,\n                      "sha256": _sha256_file(path)})\ndiff --git a/ironmule/q3f_child_guard.py b/ironmule/q3f_child_guard.py\nindex 2483675..1f9a9e3 100644\n--- a/ironmule/q3f_child_guard.py\n+++ b/ironmule/q3f_child_guard.py\n@@ -90,6 +90,10 @@ REVIEWED_SOURCE_MODULES = frozenset({\n     # strings plus predicates over it, importing only dataclasses, typing and\n     # ironmule.runtime, with no operation from OPERATION_SET and no dynamic call path.\n     "ironmule.numeric_plans",\n+    # Reached from load_engine when a caller asks for the `native` plan. Reviewed: mlx,\n+    # mlx.nn, typing and ironmule.numeric_plans only; a kernel source string and module\n+    # class swaps, no operation from OPERATION_SET and no dynamic call path.\n+    "ironmule.cuda_native",\n })\n #: The bare names a relative import inside the package can use. Derived from the allowlist\n #: rather than repeated, because the two must agree: a name missing here resolves to a\ndiff --git a/ironmule/tune.py b/ironmule/tune.py\nindex 11fe9e6..d9a330f 100644\n--- a/ironmule/tune.py\n+++ b/ironmule/tune.py\n@@ -315,7 +315,10 @@ def gpu_busy() -> str | None:\n # of bf16 against float32\'s 0.58, so it could be worth about another 1.75x — but float16\'s\n # exponent range is not bf16\'s, and no quality gate has been run on it. It is selectable so\n # it can be measured; nothing recommends it until that measurement exists.\n-COMPUTE_DTYPES = ("float32", "float16")\n+# `native` keeps the bf16 checkpoint and computes its 4-bit matmuls with IronMule\'s own CUDA\n+# kernels instead of emulated bf16 (`ironmule/cuda_native.py`, PERF1); CUDA below compute\n+# capability 8 only, refused everywhere else.\n+COMPUTE_DTYPES = ("float32", "float16", "native")\n \n \n def _check_compute_dtype(compute_dtype: str | None) -> str | None:\n@@ -377,9 +380,15 @@ def load_engine(model_id: str, knobs: Knobs, *, offline: bool | None = True,\n         # Spelled out rather than `getattr(mx, compute_dtype)`: the Q3f child guard scans\n         # this surface statically and refuses a dynamic attribute lookup, which is the right\n         # call — a plan name coming from a CLI flag must not become a module attribute path.\n-        model.set_dtype(mx.float32 if compute_dtype == "float32" else mx.float16)\n+        if compute_dtype != "native":\n+            model.set_dtype(mx.float32 if compute_dtype == "float32" else mx.float16)\n     engine = Engine(model, tokenizer, knobs)\n     engine.compute_dtype = compute_dtype\n+    engine.native_admission = None\n+    if compute_dtype == "native":\n+        from .cuda_native import install\n+        # After the Engine, because projection fusion builds fresh quantised modules.\n+        engine.native_admission = install(engine.model, info)\n     engine.model_identity = resolved.identity if resolved is not None else None\n     # Admission runs here because it needs the identity, and only here: nothing in the\n     # per-token path may hash a model or re-check a version.\ndiff --git a/ironmule_product/backend.py b/ironmule_product/backend.py\nindex b2ab81f..1268632 100644\n--- a/ironmule_product/backend.py\n+++ b/ironmule_product/backend.py\n@@ -63,8 +63,8 @@ class MLXWorkerClient:\n                  compute_dtype: str | None = None) -> None:\n         if not isinstance(spec, ModelSpec):\n             raise TypeError("spec must be ModelSpec")\n-        if compute_dtype not in (None, "float32"):\n-            raise ValueError("compute_dtype must be None or float32")\n+        if compute_dtype not in (None, "float32", "native"):\n+            raise ValueError("compute_dtype must be None, float32 or native")\n         if compute_dtype is not None and execution_variant != "reference":\n             raise ValueError("compute_dtype is available on the reference worker only")\n         # Opt-in numeric plan (PORT1): changes output, reported in health.\ndiff --git a/ironmule_product/cli.py b/ironmule_product/cli.py\nindex 236698f..b6ec25e 100644\n--- a/ironmule_product/cli.py\n+++ b/ironmule_product/cli.py\n@@ -120,7 +120,7 @@ def serve(argv: list[str]) -> int:\n     parser.add_argument("--api-key-env", default="IRONMULE_API_KEY", help="environment variable holding the API token")\n     parser.add_argument("--tls-cert", type=str)\n     parser.add_argument("--tls-key", type=str)\n-    parser.add_argument("--compute-dtype", choices=("float32",), default=None,\n+    parser.add_argument("--compute-dtype", choices=("float32", "native"), default=None,\n                         help="opt-in numeric plan for GPUs that emulate bf16 (NVIDIA below Ampere); "\n                              "about 2x faster there, changes output")\n     args = parser.parse_args(argv)\ndiff --git a/ironmule_product/worker.py b/ironmule_product/worker.py\nindex 4304d14..30d0362 100644\n--- a/ironmule_product/worker.py\n+++ b/ironmule_product/worker.py\n@@ -737,7 +737,7 @@ def main(argv: list[str] | None = None) -> int:\n     parser.add_argument("--prefix-cache-max-entries", type=int, default=4)\n     parser.add_argument("--prefix-cache-max-bytes", type=int, default=1024**3)\n     parser.add_argument("--selection-evidence", default="[]")\n-    parser.add_argument("--compute-dtype", choices=("float32",), default=None)\n+    parser.add_argument("--compute-dtype", choices=("float32", "native"), default=None)\n     args = parser.parse_args(argv)\n     startup_started = time.monotonic()\n     prefix_session = engine_bridge = automatic_runtime = None\n@@ -757,11 +757,15 @@ def main(argv: list[str] | None = None) -> int:\n             from friday_evidence.identity import assert_model_unchanged, runtime_identity\n             identity = runtime_identity(spec)\n         model, tokenizer, stream_generate, device = _load(spec)\n-        if args.compute_dtype == "float32":\n+        if args.compute_dtype is not None:\n             if args.execution_variant != "reference":\n                 raise ValueError("compute_dtype is available on the reference worker only")\n             import mlx.core as mx\n-            model.set_dtype(mx.float32)  # floating parameters only; opt-in, changes output\n+            if args.compute_dtype == "float32":\n+                model.set_dtype(mx.float32)  # floating parameters only; opt-in, changes output\n+            else:\n+                from ironmule.cuda_native import install\n+                install(model, mx.device_info() if mx.cuda.is_available() else None)\n         if identity is not None:\n             assert_model_unchanged(spec, identity)\n         if args.execution_variant == "prefix_reuse":\n'  # the working tree's IronMule changes (identity fix, native plan)
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
CONFIGS = [STOCK,
           {"name": "ironmule_native", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "native"},
           {"name": "ironmule_native_compiled", "env": {}, "knobs": {"compiled_fixed_cache": True},
            "mode": "interactive", "dtype": "native"},
           {"name": "ironmule_native_tuned", "env": {}, "knobs": TUNED, "mode": "throughput", "dtype": "native"}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 50 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v7", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           IRONMULE_API_KEY="perf1-local-only")


def save():
    with open(f"{WORK}/perf1-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd="/tmp"):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=min(timeout, left))
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        code = "timeout"
    with open(f"{WORK}/logs/{name}.log", "w") as stream:
        stream.write(out)
    report["stages"][name] = {"exit": code, "seconds": round(time.time() - started, 1), "tail": out[-2500:]}
    save()
    print(f"== {name}: exit={code} {report['stages'][name]['seconds']}s", flush=True)
    return code, out


def post(path, payload, auth, timeout=900):
    request = urllib.request.Request(f"http://127.0.0.1:8080{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **auth})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("cli_doctor", "ironmule doctor")

(eight, rev8), (fourteen, rev14) = [(m, r) for _, m, r in MODELS]
for key, model_id, revision in MODELS:
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

sh("benchmark_8b_native", f"ironmule benchmark --model {eight} --compute-dtype native "
                          f"--json {WORK}/benchmark-8b-native.json", timeout=1200)

state = "--state-dir /tmp/ironmule-product"
sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {eight}")
server = subprocess.Popen(f"exec ironmule serve {state} --model {eight} --port 8080 --compute-dtype native",
                          shell=True, env=env, cwd="/tmp", stdout=open(f"{WORK}/logs/serve.log", "w"),
                          stderr=subprocess.STDOUT)
auth = {"Authorization": "Bearer perf1-local-only"}
http = {"health": None}
for _ in range(300):
    try:
        with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/health", headers=auth),
                                    timeout=2) as response:
            http["health"] = json.loads(response.read())
            break
    except Exception as exc:  # noqa: BLE001
        http["health_error"] = repr(exc)
        if server.poll() is not None:
            break
        time.sleep(2)
if http["health"] is not None:
    body = {"model": eight, "max_tokens": 32, "messages": [{"role": "user", "content": "Say hello in five words."}]}
    try:
        http["completion"] = json.loads(post("/v1/chat/completions", body, auth))
        http["stream"] = post("/v1/chat/completions", {**body, "stream": True}, auth)[-1200:]
        questions = ["Name three rivers in Europe.", "What is 17 times 23?", "Define entropy in one sentence.",
                     "Give a synonym for happy."]
        answers, began = {}, time.time()

        def ask(i, q):
            answers[i] = json.loads(post("/v1/chat/completions", {**body, "max_tokens": 64,
                                         "messages": [{"role": "user", "content": q}]}, auth))
        threads = [threading.Thread(target=ask, args=(i, q)) for i, q in enumerate(questions)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        http["concurrent"] = {"requests": len(questions), "wall_s": round(time.time() - began, 2),
                              "answers": [answers[i]["choices"][0]["message"]["content"] for i in sorted(answers)],
                              "usage": [answers[i].get("usage") for i in sorted(answers)]}
    except Exception as exc:  # noqa: BLE001
        http["request_error"] = repr(exc)
server.terminate()
try:
    server.wait(timeout=20)
except subprocess.TimeoutExpired:
    server.kill()
http["server_exit"] = server.poll()
report["serve_http"] = http
save()

for key, model_id, revision in MODELS:
    sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(CONFIGS)}' "
                       f"{WORK}/cross-native-{key}.json 2 1", timeout=2400)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
