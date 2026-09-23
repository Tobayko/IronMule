# PERF1 run 2: the native kernel in both numeric plans, tensor-core prefill, and a quality check.
# Private notebook, internet on. Free quota: one run, <= 1.5 h; PERF1 budget 3 h, run 1 used 1.05 h.
# Backlog: docs/PROJECT_FRIDAY_BACKLOG.md, PERF1-C. Screening, no performance claim.
#
# Run 1 (`69dbc7af`) settled tensor parallelism (0.34x, killed) and showed the bf16 kernel at
# 5.5x decode with prefill untouched. This run asks three things, with rules fixed here:
#   * does the kernel also carry the qualified float32 plan ("k32"), and does float16
#     tensor-core prefill ("p16") cut TTFT, each in both plans;
#   * a part runs end to end only if the probe finds it correct (rel. error <= 1e-2 on every
#     shape) and faster than what it replaces (k32/kernel: float32 `quantized_matmul`;
#     p16: bf16 `quantized_matmul` at 512 rows);
#   * quality: mean next-token NLL on WikiText-2, 4 x 256 tokens, through the path each part
#     changes — decode (teacher-forced) for kernel/k32, prefill for p16 — against the same path
#     without it. Diagnostic only: 4 chunks cannot qualify a plan.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = '"""PERF1: native 4-bit matvec and tensor-core prefill against emulated bfloat16 on a T4.\n\nUsage: python perf1.py kernel OUT.json\n       python perf1.py e2e MODEL_PATH ARM OUT.json\n       python perf1.py nll MODEL_PATH ARM PATH_MODE TEXT OUT.json    (PATH_MODE: prefill|decode)\n\nARM is "+"-joined parts: "stock" (bf16 checkpoint as loaded), "fp32" (IronMule\'s float32\nplan, `set_dtype`), "kernel" (single-row bf16 4-bit matmuls through the native kernel),\n"k32" (the same kernel reading float32 activations and scales, for the float32 plan),\n"p16" (multi-row 4-bit matmuls, i.e. prefill: dequantise to float16, one tensor-core GEMM,\ncast back). E.g. "kernel+p16", "fp32+k32+p16".\n\nWhy: run 1 (`69dbc7af`) decoded Qwen 3 8B at 6.7 tok/s stock and 36.9 with the kernel; the\nT4\'s bandwidth allows ~70. MLX\'s CUDA `qmv` accumulates in the activation type and bfloat16\nis emulated below compute capability 8. A bfloat16 is the top half of a float32: widening\nit is one shift. The kernel reads activations, scales and biases as raw bits, accumulates\nin float32 and rounds once at the end. Prefill stayed at 26 s for 512 tokens, because the\nmulti-row path is the same emulated bfloat16; "p16" hands it to cuBLAS in float16 instead.\nBoth are numeric plans, not token-identical to stock; the "nll" mode is their quality check.\nWeights and packing are MLX\'s own: 4-bit, low nibble first, group size 64.\n\nTiming: every arm runs a full warmup generation first, then REPS measured generations of a\nfixed ~512-token prompt; TTFT is prefill plus the first token, decode rate is the remaining\ntokens over their wall time, mlx-lm\'s own async-eval pattern. No performance claim.\n"""\nimport json\nimport math\nimport os\nimport statistics as st\nimport sys\nimport time\n\nimport mlx.core as mx\n\nPROMPT_TOKENS = 512\nNEW_TOKENS = 128\nREPS = 3\nGS = 64\nNLL_CHUNKS = 4\nNLL_TOKENS = 256\n\nHEADER = r"""\n__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }\n__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }\n__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }\n__device__ __forceinline__ unsigned short to_bf(float f) {\n  unsigned int u = __float_as_uint(f);\n  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;\n  u += 0x7fffu + ((u >> 16) & 1u);\n  return (unsigned short)(u >> 16);\n}\n"""\n\n# One warp per R output rows. A uint4 of packed weights is 32 4-bit values, which sits inside\n# one group of 64; the matching 32 activations are widened once per chunk and reused for all\n# R rows (one row per warp would read the activations more often than the weights).\nSOURCE = r"""\n  const unsigned int gid = cooperative_groups::this_grid().thread_rank();\n  const unsigned int row0 = (gid >> 5) * R;\n  const unsigned int lane = threadIdx.x & 31;\n  if (row0 >= N) return;\n  const uint4* w4 = reinterpret_cast<const uint4*>(w);\n  @PTRS@\n  float acc[R];\n#pragma unroll\n  for (int r = 0; r < R; ++r) acc[r] = 0.f;\n  for (unsigned int c = lane; c < K / 32; c += 32) {\n    float xf[32];\n    @LOAD@\n    float sx = 0.f;\n#pragma unroll\n    for (int i = 0; i < 32; ++i) sx += xf[i];\n    const unsigned int g = (c * 32) / GS;\n#pragma unroll\n    for (int r = 0; r < R; ++r) {\n      const unsigned int row = row0 + r;\n      if (row < N) {\n        const uint4 q = w4[(size_t)row * (K / 32) + c];\n        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};\n        float qx = 0.f;\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int e = 0; e < 8; ++e) qx += (float)((qw[j] >> (4 * e)) & 0xFu) * xf[j * 8 + e];\n        }\n        const size_t sg = (size_t)row * (K / GS) + g;\n        acc[r] += @SCALE@ * qx + @BIAS@ * sx;\n      }\n    }\n  }\n#pragma unroll\n  for (int r = 0; r < R; ++r) {\n    float a = acc[r];\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);\n    if (lane == 0 && row0 + r < N) out[row0 + r] = @STORE@;\n  }\n"""\nVARIANTS = {\n    "bf16": {"@PTRS@": "const uint4* xv = reinterpret_cast<const uint4*>(x);\\n"\n                       "  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);\\n"\n                       "  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);",\n             "@LOAD@": "#pragma unroll\\n    for (int j = 0; j < 4; ++j) {\\n"\n                       "      const uint4 xa = xv[c * 4 + j];\\n"\n                       "      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};\\n"\n                       "#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n"\n                       "        xf[j * 8 + 2 * e] = lo_bf(xw[e]);\\n"\n                       "        xf[j * 8 + 2 * e + 1] = hi_bf(xw[e]);\\n      }\\n    }",\n             "@SCALE@": "one_bf(ss[sg])", "@BIAS@": "one_bf(bs[sg])", "@STORE@": "to_bf(a)"},\n    "f32": {"@PTRS@": "const float4* xv = reinterpret_cast<const float4*>(x);\\n"\n                      "  const float* ss = reinterpret_cast<const float*>(scales);\\n"\n                      "  const float* bs = reinterpret_cast<const float*>(biases);",\n            "@LOAD@": "#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n"\n                      "      const float4 xa = xv[c * 8 + j];\\n"\n                      "      xf[4 * j] = xa.x; xf[4 * j + 1] = xa.y; xf[4 * j + 2] = xa.z; xf[4 * j + 3] = xa.w;\\n"\n                      "    }",\n            "@SCALE@": "ss[sg]", "@BIAS@": "bs[sg]", "@STORE@": "a"},\n}\n_kernels = {}\n\n\ndef _source(variant):\n    source = SOURCE\n    for key, value in VARIANTS[variant].items():\n        source = source.replace(key, value)\n    return source\n\n\ndef rows_for(n):\n    # Fixed from run 1\'s probe: two rows won on the small shapes, four on gate/up, down and\n    # lm_head. PERF1_ROWS overrides it for the probe.\n    forced = os.environ.get("PERF1_ROWS")\n    return int(forced) if forced else (4 if n >= 8192 else 2)\n\n\ndef qmv(x, w, scales, biases, rows=None):\n    """x (..., K) with one row; w (N, K/8) uint32; scales/biases (N, K/64), all bf16 or all f32."""\n    variant = "f32" if x.dtype == mx.float32 else "bf16"\n    if variant not in _kernels:\n        _kernels[variant] = mx.fast.cuda_kernel(\n            name=f"perf1_qmv4_{variant}", input_names=["x", "w", "scales", "biases"],\n            output_names=["out"], source=_source(variant), header=HEADER)\n    n, k = w.shape[0], w.shape[1] * 8\n    rows = rows or rows_for(n)\n    threads = -(-n // rows) * 32\n    threads = -(-threads // 256) * 256  # whole blocks; surplus warps return before any work\n    out_dtype = mx.float32 if variant == "f32" else mx.uint16\n    out = _kernels[variant](\n        inputs=[mx.contiguous(x.reshape(k)), w, scales, biases],\n        template=[("N", n), ("K", k), ("GS", GS), ("R", rows)], grid=(threads, 1, 1),\n        threadgroup=(256, 1, 1), output_shapes=[(n,)], output_dtypes=[out_dtype])[0]\n    if variant == "bf16":\n        out = out.view(mx.bfloat16)\n    return out.reshape(*x.shape[:-1], n)\n\n\n_original = mx.quantized_matmul\nroutes = set()\nrouted = {"kernel": 0, "k32": 0, "p16": 0, "fallback": 0}\n\n\ndef _patched(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", **kw):\n    k = x.shape[-1]\n    if (biases is not None and transpose and (group_size or 64) == GS and (bits or 4) == 4\n            and mode == "affine" and not kw and k % 32 == 0 and w.ndim == 2 and w.shape[1] * 8 == k\n            and scales.dtype == x.dtype):\n        if x.size == k:\n            if "kernel" in routes and x.dtype == mx.bfloat16:\n                routed["kernel"] += 1\n                return qmv(x, w, scales, biases)\n            if "k32" in routes and x.dtype == mx.float32:\n                routed["k32"] += 1\n                return qmv(x, w, scales, biases)\n        elif "p16" in routes:\n            routed["p16"] += 1\n            wd = mx.dequantize(w, scales.astype(mx.float16), biases.astype(mx.float16),\n                               group_size=GS, bits=4)\n            return mx.matmul(x.astype(mx.float16), wd.T).astype(x.dtype)\n    routed["fallback"] += 1\n    return _original(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits,\n                     mode=mode, **kw)\n\n\ndef apply_arm(model, arm):\n    parts = set(arm.split("+"))\n    if not parts <= {"stock", "fp32", "kernel", "k32", "p16"}:\n        raise ValueError(arm)\n    if "fp32" in parts:\n        model.set_dtype(mx.float32)\n    routes.update(parts & {"kernel", "k32", "p16"})\n    if routes:\n        mx.quantized_matmul = _patched  # nn.QuantizedLinear looks it up per call\n\n\ndef prompt_ids(tokenizer):\n    text = ("The history of computing is a history of moving work closer to the hardware that does "\n            "it: from interpreted loops to compiled kernels, from general processors to units built "\n            "for one operation. Summarise the argument and give three examples. ")\n    ids = list(tokenizer.encode(text * 40))\n    return ids[:PROMPT_TOKENS]\n\n\ndef generate(model, ids):\n    from mlx_lm.models.cache import make_prompt_cache\n\n    cache = make_prompt_cache(model)\n    began = time.perf_counter()\n    y = mx.argmax(model(mx.array(ids)[None, :], cache=cache)[:, -1, :], axis=-1)\n    mx.eval(y)\n    first = time.perf_counter()\n    tokens = []\n    for _ in range(NEW_TOKENS - 1):\n        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[:, -1, :], axis=-1)\n        mx.async_eval(nxt)\n        tokens.append(int(y.item()))\n        y = nxt\n    tokens.append(int(y.item()))\n    done = time.perf_counter()\n    return tokens, (first - began) * 1000, (NEW_TOKENS - 1) / (done - first)\n\n\ndef measure(model, tokenizer, arm):\n    apply_arm(model, arm)\n    ids = prompt_ids(tokenizer)\n    generate(model, ids)  # warmup: JIT, graph capture, allocator\n    ttft, tps, tokens = [], [], None\n    for _ in range(REPS):\n        out, t, r = generate(model, ids)\n        if tokens is not None and out != tokens:\n            raise RuntimeError("tokens differ between repetitions of one arm")\n        tokens = out\n        ttft.append(round(t, 2))\n        tps.append(round(r, 3))\n    return {"arm": arm, "prompt_tokens": len(ids), "new_tokens": NEW_TOKENS, "ttft_ms": ttft,\n            "ttft_ms_median": st.median(ttft), "decode_tps": tps, "decode_tps_median": st.median(tps),\n            "tokens": tokens, "routed": dict(routed), "peak_memory_bytes": int(mx.get_peak_memory()),\n            "device": str(mx.default_device()), "graph_env": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),\n            "performance_claim": False}\n\n\ndef nll(model, tokenizer, arm, path_mode, text_path):\n    """Per-chunk mean next-token NLL, through the prefill path (one forward over the chunk) or\n    the decode path (teacher-forced, one token at a time through the cache) — the latter is the\n    only way the single-row kernel is exercised."""\n    from mlx_lm.models.cache import make_prompt_cache\n\n    apply_arm(model, arm)\n    with open(text_path) as stream:\n        ids = tokenizer.encode(stream.read())\n    stride = (len(ids) - NLL_TOKENS - 1) // NLL_CHUNKS\n    chunks, nonfinite = [], 0\n    started = time.time()\n    for i in range(NLL_CHUNKS):\n        seq = ids[i * stride:i * stride + NLL_TOKENS + 1]\n        if path_mode == "prefill":\n            logits = model(mx.array(seq[:-1])[None, :])[0].astype(mx.float32)\n            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)\n            per = -mx.take_along_axis(lp, mx.array(seq[1:])[:, None], axis=-1)[:, 0]\n            values = [float(v) for v in per.tolist()]\n        elif path_mode == "decode":\n            cache = make_prompt_cache(model)\n            values = []\n            for t in range(NLL_TOKENS):\n                logits = model(mx.array([[seq[t]]]), cache=cache)[0, -1].astype(mx.float32)\n                value = mx.logsumexp(logits) - logits[seq[t + 1]]\n                mx.eval(value)\n                values.append(float(value.item()))\n        else:\n            raise ValueError(path_mode)\n        nonfinite += sum(not math.isfinite(v) for v in values)\n        chunks.append(sum(values) / len(values))\n        print(i, chunks[-1], flush=True)\n    return {"arm": arm, "path_mode": path_mode, "chunks": NLL_CHUNKS, "chunk_tokens": NLL_TOKENS,\n            "chunk_nll": chunks, "mean_nll": sum(chunks) / len(chunks), "nonfinite": nonfinite,\n            "routed": dict(routed), "seconds": round(time.time() - started, 1),\n            "performance_claim": False}\n\n\ndef bench(fn, n=100):\n    for _ in range(10):\n        mx.eval(fn())\n    began = time.perf_counter()\n    for _ in range(n):\n        mx.eval(fn())\n    return (time.perf_counter() - began) / n * 1000\n\n\ndef kernel_probe(out):\n    # Qwen 3 8B decode shapes: q/o, k/v, gate/up, down, lm_head. (N, K)\n    shapes = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (12288, 4096),\n              "down": (4096, 12288), "lm_head": (151936, 4096)}\n    per_layer = {"q_o": 2, "k_v": 2, "gate_up": 2, "down": 1}  # 36 layers\n    report = {"schema": "ironmule.perf1-kernel.v2", "device": str(mx.default_device()), "shapes": {},\n              "performance_claim": False}\n    mx.random.seed(0)\n    for name, (n, k) in shapes.items():\n        wf = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)\n        w, s, b = mx.quantize(wf, group_size=GS, bits=4)\n        s32, b32 = s.astype(mx.float32), b.astype(mx.float32)\n        x = mx.random.normal((1, k)).astype(mx.bfloat16)\n        x32 = x.astype(mx.float32)\n        ref = x32 @ mx.dequantize(w, s32, b32, group_size=GS, bits=4).T\n        scale = mx.max(mx.abs(ref))\n        row = {"N": n, "K": k}\n        for label, xin, sin, bin_ in (("bf16", x, s, b), ("f32", x32, s32, b32)):\n            try:\n                y = qmv(xin, w, sin, bin_).astype(mx.float32)\n                row[f"kernel_{label}_rel_err"] = float(mx.max(mx.abs(y - ref)) / scale)\n                row[f"kernel_{label}_ms"] = bench(lambda: qmv(xin, w, sin, bin_))\n            except Exception as exc:  # noqa: BLE001 — a kernel that does not compile is the result\n                row[f"kernel_{label}_error"] = f"{type(exc).__name__}: {exc}"\n        row["qmm_bf16_ms"] = bench(lambda: _original(x, w, s, b, transpose=True, group_size=GS, bits=4))\n        row["qmm_fp32_ms"] = bench(lambda: _original(x32, w, s32, b32, transpose=True, group_size=GS, bits=4))\n        # The prefill path at 512 rows: emulated bf16 against a float16 dequantise + GEMM.\n        x512 = mx.random.normal((512, k)).astype(mx.bfloat16)\n        ref512 = x512.astype(mx.float32) @ mx.dequantize(w, s32, b32, group_size=GS, bits=4).T\n        routes.add("p16")\n        y512 = _patched(x512, w, s, b, transpose=True, group_size=GS, bits=4).astype(mx.float32)\n        routes.discard("p16")\n        row["p16_rel_err"] = float(mx.max(mx.abs(y512 - ref512)) / mx.max(mx.abs(ref512)))\n        row["p16_512_ms"] = bench(lambda: mx.matmul(x512.astype(mx.float16), mx.dequantize(\n            w, s.astype(mx.float16), b.astype(mx.float16), group_size=GS, bits=4).T).astype(mx.bfloat16), 20)\n        row["qmm_bf16_512_ms"] = bench(lambda: _original(x512, w, s, b, transpose=True, group_size=GS, bits=4), 5)\n        report["shapes"][name] = row\n        print(name, row, flush=True)\n    for key in ("kernel_bf16_ms", "kernel_f32_ms", "qmm_bf16_ms", "qmm_fp32_ms", "p16_512_ms", "qmm_bf16_512_ms"):\n        rows = report["shapes"]\n        if all(key in rows[name] for name in shapes):\n            report[f"step_{key}"] = round(sum(rows[n][key] * c for n, c in per_layer.items()) * 36\n                                          + rows["lm_head"][key], 3)\n    with open(out, "w") as stream:\n        json.dump(report, stream, indent=1)\n    print(json.dumps({k: v for k, v in report.items() if k.startswith("step")}), flush=True)\n\n\ndef main():\n    mode = sys.argv[1]\n    if mode == "kernel":\n        return kernel_probe(sys.argv[2])\n    from mlx_lm import load\n\n    path, arm = sys.argv[2:4]\n    model, tokenizer = load(path)\n    if mode == "e2e":\n        out = sys.argv[4]\n        report = measure(model, tokenizer, arm)\n    elif mode == "nll":\n        path_mode, text_path, out = sys.argv[4:7]\n        report = nll(model, tokenizer, arm, path_mode, text_path)\n    else:\n        raise SystemExit(__doc__)\n    report["mode"], report["model_path"] = mode, path\n    with open(out, "w") as stream:\n        json.dump(report, stream, indent=1)\n    print(json.dumps({k: v for k, v in report.items() if k != "tokens"}), flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'  # experiments/kaggle_compat/perf1.py
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v2", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400")


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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

sh("kernel_probe", f"{PY} /tmp/perf1.py kernel {WORK}/kernel-probe.json", timeout=900)
ok = {"kernel": False, "k32": False, "p16": False}
try:
    with open(f"{WORK}/kernel-probe.json") as stream:
        probe = json.load(stream)
    shapes = probe["shapes"].values()
    correct = lambda key: all(s.get(key, 1.0) <= 1e-2 for s in shapes)  # noqa: E731
    ok["kernel"] = correct("kernel_bf16_rel_err") and probe["step_kernel_bf16_ms"] < probe["step_qmm_fp32_ms"]
    ok["k32"] = correct("kernel_f32_rel_err") and probe["step_kernel_f32_ms"] < probe["step_qmm_fp32_ms"]
    ok["p16"] = correct("p16_rel_err") and probe["step_p16_512_ms"] < probe["step_qmm_bf16_512_ms"]
except (OSError, ValueError, KeyError):
    pass
report["parts_ok"] = ok
save()


def allowed(arm):
    return all(ok.get(part, True) for part in arm.split("+"))


E2E = [a for a in ("stock", "fp32", "kernel", "kernel+p16", "fp32+k32", "fp32+k32+p16") if allowed(a)]
NLL = [(a, m) for a, m in (("stock", "decode"), ("kernel", "decode"), ("fp32", "decode"), ("fp32+k32", "decode"),
                           ("stock", "prefill"), ("p16", "prefill"), ("fp32", "prefill"), ("fp32+p16", "prefill"))
       if allowed(a)]
report["e2e_arms"], report["nll_arms"] = E2E, NLL
save()

sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

for index, (key, model_id, revision) in enumerate(MODELS):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    for arm in E2E:
        sh(f"e2e_{key}_{arm}", f"{PY} /tmp/perf1.py e2e {path} '{arm}' {WORK}/e2e-{key}-{arm}.json", timeout=900)
    if index == 0:  # quality on the 8B only; the 14B is speed
        for arm, path_mode in NLL:
            sh(f"nll_{key}_{arm}_{path_mode}", f"{PY} /tmp/perf1.py nll {path} '{arm}' {path_mode} /tmp/wikitext.txt "
                                               f"{WORK}/nll-{key}-{arm}-{path_mode}.json", timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
