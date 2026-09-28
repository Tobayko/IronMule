# DEMO4: the Apple live race (six questions at once, 2026-09-27) on a Kaggle Tesla T4, one model per
# notebook, for show videos. Private notebook, internet on. Quota: the user asked on 2026-09-28 for
# short runs of every open model; capped at 35 min.
#   * IronMule main 58cbb1d, no patch. Both arms go through IronMule's Runtime in their own process,
#     one after the other on GPU 0, as PORT1's `cross.py` defines them: off = baseline knobs,
#     interactive mode (one request after another) and MLX's own graph limits; on = the model's
#     opt-in numeric plan with the knobs its plan row was measured with, throughput mode (the
#     requests grouped). Grouping is batch-1 per request, not tensor batching, so on a T4 it lets
#     every answer advance together rather than finish sooner (PERF1 run 7: native 0.201 alone,
#     0.218 grouped, Qwen 3 8B).
#   * Scene 1: one question, 64 tokens, three answers per arm, interactive on both. Scene 2: the
#     Apple race's six questions submitted together, 48 tokens each, two rounds per arm. Warm-ups
#     at 8 tokens cover every prompt length first (stock compiles per new length).
#   * Every token keeps the time the engine produced it (service TTFT plus inter-token gaps),
#     from the moment its request set was submitted, and its decoded text so far.
#   * A demo, not a measurement: the ledger's paired runs are the evidence.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
LEAN = {"head_skip_prefill": True}
# The plan and knobs of each model's row in `ironmule/numeric_plans.py` (PORT2, PERF1).
MODELS = {
    "qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192",
                 "Qwen 3 8B 4-bit", "native", {}, {"enable_thinking": False}, "qualified"),
    "qwen3-14b": ("mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4",
                  "Qwen 3 14B 4-bit", "native", {}, {"enable_thinking": False}, "qualified"),
    "gemma3-12b": ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
                   "Gemma 3 12B 4-bit", "native", {}, {}, "qualified"),
    "gemma3-4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
                  "Gemma 3 4B 4-bit", "float32", TUNED, {}, "qualified"),
    "gemma4-e2b": ("mlx-community/gemma-4-e2b-it-4bit", "238767527555cb75a05732a84dff5d6ba0dd6809",
                   "Gemma 4 E2B 4-bit", "float32", {**TUNED, "fuse_projections": False}, {"enable_thinking": False}, "qualified"),
    "mistral-24b": ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    "2a1d5eabfc504747bdc24178394821a1efc0edde", "Mistral Small 3.2 24B 4-bit", "float32",
                    LEAN, {}, "qualified"),
    # No numeric plan pays on these two: exact keeps the arithmetic (PORT1 1B 0.550; PORT2 Llama float32
    # is 1.52x slower), so on is the tuned knobs alone and must stay token-identical.
    "gemma3-1b": ("mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
                  "Gemma 3 1B 4-bit", None, TUNED, {}, "exact"),
    "llama31-8b": ("mlx-community/Llama-3.1-8B-Instruct-4bit", "90215b22ec18e72f623dde2ea7af4097025160e2",
                   "Llama 3.1 8B 4-bit", None, TUNED, {}, "exact"),
    # Timed in PORT2 run 9b, but its float32 gate never ran (NEXT1-C): unqualified.
    "gemma4-e4b": ("mlx-community/gemma-4-e4b-it-4bit", "475b9088d29754a3379866cf5aeb6b41acd313c2",
                   "Gemma 4 E4B 4-bit", "float32", {**TUNED, "fuse_projections": False}, {"enable_thinking": False}, "unqualified"),
    # Its plan row is unqualified: the quality interval is too wide to pass or fail.
    "gptoss-20b": ("mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089",
                   "gpt-oss 20B MXFP4", "float32", LEAN, {"reasoning_effort": "low"}, "unqualified"),
    # gpt-oss again with room for its analysis channel, under its two faster plans; neither is qualified.
    "gptoss-20b-native": ("mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089",
                          "gpt-oss 20B MXFP4", "native", LEAN, {"reasoning_effort": "low"}, "unqualified"),
    "gptoss-20b-f16": ("mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089",
                       "gpt-oss 20B MXFP4", "float16", LEAN, {"reasoning_effort": "low"}, "unqualified"),
    # Families IronMule has not measured: `native` is architecture-agnostic, but no gate has run on them.
    "qwen25-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
                  "Qwen 2.5 7B 4-bit", "native", {}, {}, "unqualified"),
    "mistral-7b": ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "a4b8f870474b0eb527f466a03fbc187830d271f5",
                   "Mistral 7B v0.3 4-bit", "native", {}, {}, "unqualified"),
    "phi-4": ("mlx-community/phi-4-4bit", "fc0f8f23d369dc29b55cad1d65cb5bf0dcbee910",
              "Phi-4 14B 4-bit", "native", {}, {}, "unqualified"),
    # Gemma 4 thinks unless its template gets `enable_thinking=False` (DEMO4's first gemma4-e2b run spent
    # every token in `<|channel>thought`; an empty thought channel alone did not stop it). float16 is
    # qualified on E2B.
    "gemma4-e2b-f16": ("mlx-community/gemma-4-e2b-it-4bit", "238767527555cb75a05732a84dff5d6ba0dd6809",
                       "Gemma 4 E2B 4-bit", "float16", {**TUNED, "fuse_projections": False}, {"enable_thinking": False}, "qualified"),
    "gemma4-e4b-f16": ("mlx-community/gemma-4-e4b-it-4bit", "475b9088d29754a3379866cf5aeb6b41acd313c2",
                       "Gemma 4 E4B 4-bit", "float16", {**TUNED, "fuse_projections": False}, {"enable_thinking": False}, "unqualified"),
    "gemma4-e4b-native": ("mlx-community/gemma-4-e4b-it-4bit", "475b9088d29754a3379866cf5aeb6b41acd313c2",
                          "Gemma 4 E4B 4-bit", "native", {}, {"enable_thinking": False}, "unqualified"),
    "gemma4-e4b-qat-f16": ("mlx-community/gemma-4-E4B-it-qat-4bit", "0f35c6f6d386f7f74e628bd7c6526ce531212300",
                           "Gemma 4 E4B QAT 4-bit", "float16", {**TUNED, "fuse_projections": False}, {"enable_thinking": False}, "unqualified"),
}
# Longer answers for a model that thinks before it answers; one grouped round keeps the run short.
LONG = {"single_tokens": 128, "multi_tokens": 96, "rounds": 1}
OVERRIDES = {"gptoss-20b-native": LONG, "gptoss-20b-f16": LONG}
SELECT = "__SELECT__"  # one key of MODELS, set at submit time
MODEL_ID, REVISION, LABEL, DTYPE, KNOBS, TEMPLATE, STATUS = MODELS[SELECT]
CONFIG = {"dtype": DTYPE, "knobs": KNOBS, "template": TEMPLATE, "reps": 3, "rounds": 2, "warm_tokens": 8,
          "single": {"prompt": "Explain in two sentences why the sky is blue.", "max_tokens": 64},
          "multi": {"max_tokens": 48, "prompts": [
              "Name three rivers in Europe and one country each flows through.",
              "Explain in two sentences why the sky is blue.",
              "Write a short haiku about morning coffee.",
              "What are two benefits of drinking water regularly?",
              "Give three tips for a good night's sleep.",
              "Describe a cat to someone who has never seen one."]}}
override = OVERRIDES.get(SELECT, {})
CONFIG["single"]["max_tokens"] = override.get("single_tokens", CONFIG["single"]["max_tokens"])
CONFIG["multi"]["max_tokens"] = override.get("multi_tokens", CONFIG["multi"]["max_tokens"])
CONFIG["rounds"] = override.get("rounds", CONFIG["rounds"])
CHILD = '''
import json, sys, time
import mlx.core as mx
import ironmule
from ironmule import benchmark as bench
from ironmule.tune import load_engine

model, revision, arm, config, out = sys.argv[1], sys.argv[2], sys.argv[3], json.load(open(sys.argv[4])), sys.argv[5]
on = arm == "on"
started = time.time()
engine, tokenizer = load_engine(model, ironmule.Knobs(**config["knobs"]) if on else ironmule.BASELINE,
                                revision=revision, compute_dtype=config["dtype"] if on else None)
load_s = round(time.time() - started, 1)
with ironmule.Runtime(engine, tokenizer, model_id=model) as rt:
    eos = set(rt.backend.eos_ids)

    def run(questions, max_tokens, grouped):
        requests = [ironmule.Request(prompt_ids=rt.encode(q, **config["template"]), max_tokens=max_tokens,
                                     plan=ironmule.StrictOneShotPlan(), rid=f"q{i}") for i, q in enumerate(questions)]
        mode = ironmule.ThroughputMode() if grouped else ironmule.InteractiveMode()
        results, snapshot = bench._run(rt, ironmule, mode, requests)
        rows = []
        for question, result in zip(questions, results):
            metrics = result.metrics
            times = [metrics["service_ttft_ms"]]
            for gap in metrics["inter_token_ms"]:
                times.append(times[-1] + gap)
            visible = [t for t in result.tokens if t not in eos]
            chunks = [[round(times[k - 1], 1), tokenizer.decode(visible[:k])] for k in range(1, len(visible) + 1)]
            rows.append({"prompt": question, "tokens": [int(t) for t in result.tokens], "first_ms": round(times[0], 1),
                         "total_ms": round(times[-1], 1), "usage": {"completion_tokens": len(visible)},
                         "text": tokenizer.decode(visible), "chunks": chunks, "cumulative": True,
                         "stop": metrics.get("stop_reason")})
        print(arm, "grouped" if grouped else "one-by-one", round(snapshot["outer_wall_ms"]), "ms",
              [r["usage"]["completion_tokens"] for r in rows], flush=True)
        return rows, snapshot["outer_wall_ms"]

    single, multi = config["single"], config["multi"]
    warm = [run([single["prompt"]], config["warm_tokens"], False)[1],
            run(multi["prompts"], config["warm_tokens"], on)[1]]
    reps = [run([single["prompt"]], single["max_tokens"], False)[0][0] for _ in range(config["reps"])]
    rounds = [run(multi["prompts"], multi["max_tokens"], on) for _ in range(config["rounds"])]
shown = sorted(rounds, key=lambda r: r[1])[len(rounds) // 2]
try:
    peak = mx.get_peak_memory() / 1e9
except Exception:  # noqa: BLE001 - reported as unavailable, not zero
    peak = None
json.dump({"arm": arm, "load_s": load_s, "warm_walls_ms": warm, "peak_gb": peak, "reps": reps,
           "multi": shown[0], "multi_wall_ms": shown[1],
           "multi_rounds": [{"wall_ms": w, "rows": r} for r, w in rounds]}, open(out, "w"))
'''
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 35 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/race.py", "w") as stream:
    stream.write(CHILD)
with open("/tmp/demo4.json", "w") as stream:
    json.dump(CONFIG, stream)
report = {"schema": "ironmule.demo4-kaggle.v1", "commit": COMMIT, "select": SELECT, "model": [MODEL_ID, REVISION],
          "label": LABEL, "plan_status": STATUS, "config": CONFIG, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           CUDA_VISIBLE_DEVICES="0")


def save():
    with open(f"{WORK}/demo4-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
code, _ = sh("download", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                         f"print(s('{MODEL_ID}', revision='{REVISION}'))\"", timeout=1500)
if code == 0:
    # off pins MLX's own graph limits, as `cross.py`'s stock arm does; on keeps what IronMule sets.
    sh("race_off", f"MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=100 {PY} /tmp/race.py {MODEL_ID} {REVISION} off "
                   f"/tmp/demo4.json {WORK}/race-off.json", timeout=1500)
    sh("race_on", f"{PY} /tmp/race.py {MODEL_ID} {REVISION} on /tmp/demo4.json {WORK}/race-on.json", timeout=900)
try:
    off, on = (json.load(open(f"{WORK}/race-{arm}.json")) for arm in ("off", "on"))
    report["identical"] = {
        "single": [r["tokens"] for r in off["reps"]] == [r["tokens"] for r in on["reps"]],
        "multi": [[r["tokens"] for r in rnd["rows"]] for rnd in off["multi_rounds"]]
        == [[r["tokens"] for r in rnd["rows"]] for rnd in on["multi_rounds"]]}
    report["walls_ms"] = {arm: {"single": [r["total_ms"] for r in d["reps"]],
                                "multi": [rnd["wall_ms"] for rnd in d["multi_rounds"]], "peak_gb": d["peak_gb"]}
                          for arm, d in (("off", off), ("on", on))}
except (OSError, ValueError, KeyError) as exc:
    report["identical"] = {"error": repr(exc)}
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
print(json.dumps({k: report.get(k) for k in ("identical", "walls_ms")}, indent=1))
