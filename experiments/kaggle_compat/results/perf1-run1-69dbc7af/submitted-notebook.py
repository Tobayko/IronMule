# PERF1 run 1: two T4s against one, and a native 4-bit matvec against emulated bfloat16.
# Private notebook, internet on. Free quota: one run, <= 1.5 h of the 3 h PERF1 budget.
# Backlog: docs/PROJECT_FRIDAY_BACKLOG.md, PERF1. Screening, no performance claim.
#
# Every arm is a fresh process with MLX_MAX_OPS_PER_BUFFER=400 (IronMule's own setting on
# compute capability < 8), so the arms differ only in what they name. The kernel arm only
# runs end to end when the microbenchmark says the kernel is correct.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = '"""PERF1: two T4s, and a native 4-bit matvec that never touches emulated bfloat16.\n\nUsage: python perf1.py kernel OUT.json\n       python perf1.py e2e MODEL_PATH ARM OUT.json\n       mlx.launch --hosts 127.0.0.1 -n 2 --backend ring perf1.py tp MODEL_PATH ARM OUT_PREFIX\n\nARM: "stock" (bf16 checkpoint as loaded), "fp32" (IronMule\'s float32 plan, `set_dtype`),\n"kernel" (bf16 checkpoint, every single-row 4-bit affine `quantized_matmul` routed through\nthe kernel below; everything else unchanged).\n\nWhy the kernel: PORT2 decodes Qwen 3 8B at 5.8 tok/s on a T4 whose bandwidth allows ~70.\nMLX\'s CUDA `qmv` accumulates in the activation type and bfloat16 is emulated below compute\ncapability 8, so the card is compute-bound on arithmetic it does not have. A bfloat16 is\nthe top half of a float32: widening it is one shift. This kernel reads activations, scales\nand biases as raw 16-bit patterns, widens them, accumulates in float32 and rounds once to\nbfloat16 at the end. It is a new numeric plan — not token-identical to stock — and its\nquality gate is a separate entry. Weights and packing are MLX\'s own: 4-bit, low nibble\nfirst, group size 64 (checked against `mx.dequantize` on Apple Silicon before submission).\n\nTiming: every arm runs a full warmup generation first, then REPS measured generations of a\nfixed ~512-token prompt; TTFT is prefill plus the first token, decode rate is the remaining\ntokens over their wall time, mlx-lm\'s own async-eval pattern. No performance claim.\n"""\nimport json\nimport os\nimport statistics as st\nimport sys\nimport time\n\nimport mlx.core as mx\n\nPROMPT_TOKENS = 512\nNEW_TOKENS = 128\nREPS = 3\nGS = 64\n\nHEADER = r"""\n__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }\n__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }\n__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }\n__device__ __forceinline__ unsigned short to_bf(float f) {\n  unsigned int u = __float_as_uint(f);\n  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;\n  u += 0x7fffu + ((u >> 16) & 1u);\n  return (unsigned short)(u >> 16);\n}\n"""\n\n# One warp per R output rows. A uint4 of packed weights is 32 4-bit values, which sits inside\n# one group of 64; the matching 32 activations are four uint4 of bfloat16 pairs. They are\n# widened once per chunk and reused for all R rows: one row per warp would read the\n# activation vector (2 bytes per element) four times as often as the weights (half a byte).\nSOURCE = r"""\n  const unsigned int gid = cooperative_groups::this_grid().thread_rank();\n  const unsigned int row0 = (gid >> 5) * R;\n  const unsigned int lane = threadIdx.x & 31;\n  if (row0 >= N) return;\n  const uint4* xv = reinterpret_cast<const uint4*>(x);\n  const uint4* w4 = reinterpret_cast<const uint4*>(w);\n  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);\n  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);\n  float acc[R];\n#pragma unroll\n  for (int r = 0; r < R; ++r) acc[r] = 0.f;\n  for (unsigned int c = lane; c < K / 32; c += 32) {\n    float xf[32];\n    float sx = 0.f;\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n      const uint4 xa = xv[c * 4 + j];\n      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        xf[j * 8 + 2 * e] = lo_bf(xw[e]);\n        xf[j * 8 + 2 * e + 1] = hi_bf(xw[e]);\n        sx += xf[j * 8 + 2 * e] + xf[j * 8 + 2 * e + 1];\n      }\n    }\n    const unsigned int g = (c * 32) / GS;\n#pragma unroll\n    for (int r = 0; r < R; ++r) {\n      const unsigned int row = row0 + r;\n      if (row < N) {\n        const uint4 q = w4[(size_t)row * (K / 32) + c];\n        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};\n        float qx = 0.f;\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int e = 0; e < 8; ++e) qx += (float)((qw[j] >> (4 * e)) & 0xFu) * xf[j * 8 + e];\n        }\n        const size_t sg = (size_t)row * (K / GS) + g;\n        acc[r] += one_bf(ss[sg]) * qx + one_bf(bs[sg]) * sx;\n      }\n    }\n  }\n#pragma unroll\n  for (int r = 0; r < R; ++r) {\n    float a = acc[r];\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);\n    if (lane == 0 && row0 + r < N) out[row0 + r] = to_bf(a);\n  }\n"""\n\nROW_VARIANTS = (1, 2, 4, 8)\nROWS = int(os.environ.get("PERF1_ROWS", "4"))  # rows per warp; the run picks it from the probe\n_kernel = None\n\n\ndef qmv(x, w, scales, biases, rows=None):\n    """x (..., K) bf16 with one row; w (N, K/8) uint32; scales/biases (N, K/64) bf16."""\n    global _kernel\n    if _kernel is None:\n        _kernel = mx.fast.cuda_kernel(name="perf1_qmv4", input_names=["x", "w", "scales", "biases"],\n                                      output_names=["out"], source=SOURCE, header=HEADER)\n    rows = rows or ROWS\n    n, k = w.shape[0], w.shape[1] * 8\n    threads = -(-n // rows) * 32\n    threads = -(-threads // 256) * 256  # whole blocks; surplus warps return before any work\n    out = _kernel(inputs=[mx.contiguous(x.reshape(k)), w, scales, biases],\n                  template=[("N", n), ("K", k), ("GS", GS), ("R", rows)], grid=(threads, 1, 1),\n                  threadgroup=(256, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.uint16])[0]\n    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)\n\n\n_original = mx.quantized_matmul\nrouted = {"kernel": 0, "fallback": 0}\n\n\ndef _patched(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", **kw):\n    k = x.shape[-1]\n    if (biases is not None and transpose and (group_size or 64) == GS and (bits or 4) == 4\n            and mode == "affine" and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16\n            and x.size == k and k % 32 == 0 and w.ndim == 2 and w.shape[1] * 8 == k and not kw):\n        routed["kernel"] += 1\n        return qmv(x, w, scales, biases)\n    routed["fallback"] += 1\n    return _original(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits,\n                     mode=mode, **kw)\n\n\ndef apply_arm(model, arm):\n    if arm == "fp32":\n        model.set_dtype(mx.float32)\n    elif arm == "kernel":\n        mx.quantized_matmul = _patched  # nn.QuantizedLinear and the sharded layers look it up per call\n    elif arm != "stock":\n        raise ValueError(arm)\n\n\ndef prompt_ids(tokenizer):\n    text = ("The history of computing is a history of moving work closer to the hardware that does "\n            "it: from interpreted loops to compiled kernels, from general processors to units built "\n            "for one operation. Summarise the argument and give three examples. ")\n    ids = list(tokenizer.encode(text * 40))\n    return ids[:PROMPT_TOKENS]\n\n\ndef generate(model, ids):\n    from mlx_lm.models.cache import make_prompt_cache\n\n    cache = make_prompt_cache(model)\n    began = time.perf_counter()\n    y = mx.argmax(model(mx.array(ids)[None, :], cache=cache)[:, -1, :], axis=-1)\n    mx.eval(y)\n    first = time.perf_counter()\n    tokens = []\n    for _ in range(NEW_TOKENS - 1):\n        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[:, -1, :], axis=-1)\n        mx.async_eval(nxt)\n        tokens.append(int(y.item()))\n        y = nxt\n    tokens.append(int(y.item()))\n    done = time.perf_counter()\n    return tokens, (first - began) * 1000, (NEW_TOKENS - 1) / (done - first)\n\n\ndef measure(model, tokenizer, arm):\n    apply_arm(model, arm)\n    ids = prompt_ids(tokenizer)\n    generate(model, ids)  # warmup: JIT, graph capture, allocator\n    ttft, tps, tokens = [], [], None\n    for _ in range(REPS):\n        out, t, r = generate(model, ids)\n        if tokens is not None and out != tokens:\n            raise RuntimeError("tokens differ between repetitions of one arm")\n        tokens = out\n        ttft.append(round(t, 2))\n        tps.append(round(r, 3))\n    return {"arm": arm, "prompt_tokens": len(ids), "new_tokens": NEW_TOKENS, "ttft_ms": ttft,\n            "ttft_ms_median": st.median(ttft), "decode_tps": tps, "decode_tps_median": st.median(tps),\n            "tokens": tokens, "routed": dict(routed), "peak_memory_bytes": int(mx.get_peak_memory()),\n            "device": str(mx.default_device()), "graph_env": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),\n            "performance_claim": False}\n\n\ndef bench(fn, n=100):\n    for _ in range(10):\n        mx.eval(fn())\n    began = time.perf_counter()\n    for _ in range(n):\n        mx.eval(fn())\n    return (time.perf_counter() - began) / n * 1000\n\n\ndef kernel_probe(out):\n    # Qwen 3 8B decode shapes: q/o, k/v, gate/up, down, lm_head. (N, K)\n    shapes = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (12288, 4096),\n              "down": (4096, 12288), "lm_head": (151936, 4096)}\n    per_layer = {"q_o": 2, "k_v": 2, "gate_up": 2, "down": 1}  # 36 layers\n    report = {"schema": "ironmule.perf1-kernel.v1", "device": str(mx.default_device()), "shapes": {},\n              "performance_claim": False}\n    mx.random.seed(0)\n    for name, (n, k) in shapes.items():\n        wf = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)\n        w, s, b = mx.quantize(wf, group_size=GS, bits=4)\n        x = mx.random.normal((1, k)).astype(mx.bfloat16)\n        ref = x.astype(mx.float32) @ mx.dequantize(w, s, b, group_size=GS, bits=4).astype(mx.float32).T\n        row = {"N": n, "K": k}\n        for rows in ROW_VARIANTS:\n            try:\n                y = qmv(x, w, s, b, rows).astype(mx.float32)\n                row[f"kernel_r{rows}_rel_err"] = float(mx.max(mx.abs(y - ref)) / mx.max(mx.abs(ref)))\n                row[f"kernel_r{rows}_ms"] = bench(lambda: qmv(x, w, s, b, rows))\n            except Exception as exc:  # noqa: BLE001 — a kernel that does not compile is the result\n                row[f"kernel_r{rows}_error"] = f"{type(exc).__name__}: {exc}"\n        y16 = _original(x, w, s, b, transpose=True, group_size=GS, bits=4).astype(mx.float32)\n        row["bf16_rel_err"] = float(mx.max(mx.abs(y16 - ref)) / mx.max(mx.abs(ref)))\n        row["qmm_bf16_ms"] = bench(lambda: _original(x, w, s, b, transpose=True, group_size=GS, bits=4))\n        x32, s32, b32 = x.astype(mx.float32), s.astype(mx.float32), b.astype(mx.float32)\n        row["qmm_fp32_ms"] = bench(lambda: _original(x32, w, s32, b32, transpose=True, group_size=GS, bits=4))\n        report["shapes"][name] = row\n        print(name, row, flush=True)\n    for key in [f"kernel_r{r}_ms" for r in ROW_VARIANTS] + ["qmm_bf16_ms", "qmm_fp32_ms"]:\n        rows = report["shapes"]\n        if all(key in rows[name] for name in shapes):\n            report[f"step_{key}"] = round(sum(rows[n][key] * c for n, c in per_layer.items()) * 36\n                                          + rows["lm_head"][key], 3)\n    with open(out, "w") as stream:\n        json.dump(report, stream, indent=1)\n    print(json.dumps({k: v for k, v in report.items() if k.startswith("step")}), flush=True)\n\n\ndef main():\n    mode = sys.argv[1]\n    if mode == "kernel":\n        return kernel_probe(sys.argv[2])\n    from mlx_lm import load\n\n    if mode == "e2e":\n        path, arm, out = sys.argv[2:5]\n        model, tokenizer = load(path)\n        report = measure(model, tokenizer, arm)\n    elif mode == "tp":\n        from mlx_lm.utils import sharded_load\n\n        path, arm, prefix = sys.argv[2:5]\n        group = mx.distributed.init(strict=True)\n        mx.set_default_device(mx.Device(mx.gpu, group.rank()))  # one rank, one card\n        model, tokenizer = sharded_load(path, tensor_group=group)\n        report = measure(model, tokenizer, arm)\n        report.update(rank=group.rank(), size=group.size())\n        out = f"{prefix}-rank{group.rank()}.json"\n    else:\n        raise SystemExit(__doc__)\n    report["mode"], report["model_path"] = mode, path\n    with open(out, "w") as stream:\n        json.dump(report, stream, indent=1)\n    print(json.dumps({k: v for k, v in report.items() if k != "tokens"}), flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'  # experiments/kaggle_compat/perf1.py
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 80 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

# -- B first: it is the cheapest and it decides whether the kernel arms run at all ----------
sh("kernel_probe", f"{PY} /tmp/perf1.py kernel {WORK}/kernel-probe.json", timeout=600)
# Fixed before the run: among row variants correct on every shape (rel. error <= 1e-2), take
# the fastest estimated decode step; run the kernel arms only if it beats float32
# `quantized_matmul` at the same shapes (PERF1-B kill criterion).
try:
    with open(f"{WORK}/kernel-probe.json") as stream:
        probe = json.load(stream)
    steps = {r: probe[f"step_kernel_r{r}_ms"] for r in (1, 2, 4, 8)
             if f"step_kernel_r{r}_ms" in probe
             and all(s.get(f"kernel_r{r}_rel_err", 1.0) <= 1e-2 for s in probe["shapes"].values())}
    rows = min(steps, key=steps.get) if steps else None
    kernel_ok = rows is not None and steps[rows] < probe["step_qmm_fp32_ms"]
except (OSError, ValueError, KeyError):
    rows, kernel_ok = None, False
report["kernel_rows"], report["kernel_ok"] = rows, kernel_ok
save()
if kernel_ok:
    env["PERF1_ROWS"] = str(rows)
ARMS = ["stock", "fp32"] + (["kernel"] if kernel_ok else [])

for key, model_id, revision in MODELS:
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    for arm in ARMS:
        sh(f"e2e_{key}_{arm}", f"{PY} /tmp/perf1.py e2e {path} {arm} {WORK}/e2e-{key}-{arm}.json",
           timeout=900)
        sh(f"tp_{key}_{arm}", f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring --python {PY} "
                              f"/tmp/perf1.py tp {path} {arm} {WORK}/tp-{key}-{arm}", timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
