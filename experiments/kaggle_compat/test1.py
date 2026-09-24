# TEST1: every test the repository can run off the target Mac, on a Kaggle T4 through MLX's
# CUDA backend, plus the listen-backlog probe (backlog 5 against 64) on the same host.
# Private notebook, internet on (pip, git, Hugging Face). No performance claim.
import json
import os
import re
import signal
import subprocess
import sys
import time

COMMIT = "f9e1d3062ba8d0bb6cf6c667e7f029157d5c80cc"
BEFORE = "b6886a0"  # the parent: stdlib listen backlog of 5
MODELS = {
    "4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
    "qwen35-9b": ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631"),
}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
OLD = "/tmp/IronMule-before"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter carries pip; the venv does not
DEADLINE = time.time() + 57 * 60
SATURATION = "tests/engine/test_product_http.py::test_handler_saturation_returns_429_then_recovers"
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.test1-kaggle.v1", "commit": COMMIT, "stages": {}}
# PYTHONNOUSERSITE: Kaggle's root user site holds a sitecustomize importing `wrapt` (DATA3).
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           IRONMULE_HOME="/tmp/ironmule-home", HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/test1-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd="/tmp", extra=None):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=dict(env, **(extra or {})), stdout=subprocess.PIPE,
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


def summary(out):
    """The pytest summary line and the failing node ids, as printed with -rfE."""
    lines = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed|error|skipped)", line)]
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)))
    return {"line": lines[-1].strip("= ") if lines else None, "failed": failed}


def pytest(name, args, timeout=1200, extra=None):
    code, out = sh(name, f"{PY} -m pytest {args} -rfEs -p no:cacheprovider --junitxml={WORK}/{name}.xml",
                   timeout=timeout, cwd=REPO, extra=extra)
    report[name] = summary(out) | {"exit": code}
    save()
    return report[name]


def overflows():
    with open("/proc/net/netstat") as stream:
        rows = [line.split() for line in stream if line.startswith("TcpExt:")]
    return int(rows[1][rows[0].index("ListenOverflows")])


def probe(label, cwd, runs, burners):
    """experiments/http_backlog/probe.sh, in Python: one pytest process per run."""
    hogs = [subprocess.Popen([SYS, "-c", "while True: pass"]) for _ in range(burners)]
    rows = []
    try:
        for _ in range(runs):
            if DEADLINE - time.time() < 120:
                break
            before, started = overflows(), time.time()
            proc = subprocess.run(f"{PY} -m pytest -n 0 -p no:cacheprovider {SATURATION}", shell=True, cwd=cwd,
                                  env=env, capture_output=True, text=True, timeout=180)
            rows.append({"wall_s": round(time.time() - started, 2), "overflows": overflows() - before,
                         "result": (proc.stdout.strip().splitlines() or ["?"])[-1]})
    finally:
        for hog in hogs:
            hog.kill()
    report.setdefault("backlog_probe", {})[label] = {"burners": burners, "runs": rows}
    save()
    print(f"== probe {label}: {rows}", flush=True)


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; nproc; free -g; "
          "df -h / | tail -1; uname -r; cat /proc/sys/net/ipv4/tcp_abort_on_overflow /proc/sys/net/core/somaxconn")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short "
            f"&& git -C {REPO} worktree add -q {OLD} {BEFORE} && git -C {OLD} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy 'ruff==0.16.6' 'matplotlib==3.11.1'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("mlx_gpu_smoke", f"""{PY} - <<'EOF'
import mlx.core as mx
print("mlx", mx.__version__, "device", mx.default_device(), "cuda", mx.cuda.is_available())
w = mx.random.normal((4096, 1024)); q, s, z = mx.quantize(w, group_size=64, bits=4)
x = mx.random.normal((1, 1024)); y = mx.quantized_matmul(x, q, s, z, transpose=True, group_size=64, bits=4)
ref = x @ mx.dequantize(q, s, z, group_size=64, bits=4).T
mx.eval(y, ref); print("qmm max_abs_diff", mx.abs(y - ref).max().item())
EOF""")
sh("download_4b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                  f"print(s('{MODELS['4b'][0]}', revision='{MODELS['4b'][1]}'))\"", timeout=900)

# The engine suite as pytest.ini configures it (-n auto, --dist loadfile), twice, then the
# real-model integration tests serially, then every failure alone and serially.
pytest("pytest_unit_1", "tests/engine -m 'not integration'")
pytest("pytest_unit_2", "tests/engine -m 'not integration'")
pytest("pytest_integration", "tests/engine -m integration -n 0", timeout=1800)
failed = sorted({node for key in ("pytest_unit_1", "pytest_unit_2", "pytest_integration")
                 for node in report.get(key, {}).get("failed", [])})
report["rerun_nodes"] = failed
for index, node in enumerate(failed):
    pytest(f"pytest_rerun_{index}", f"-n 0 '{node}'", timeout=600)
sh("collect_all", f"{PY} -m pytest tests --collect-only -q -n 0 -p no:cacheprovider | tail -3", cwd=REPO)

# TEST1 on this host: the parent commit's backlog of 5 against this commit's 64. First prove
# that each tree imports its own server, not the editable install's.
for label, tree in (("before", OLD), ("after", REPO)):
    sh(f"backlog_{label}", f"{PY} -c \"import sys; sys.path.insert(0, '.'); import ironmule_product.http_server as h; "
                           f"print(h.__file__, h._ProductHTTPServer.request_queue_size)\"", cwd=tree)
probe("backlog5_load8", OLD, 5, 8)
probe("backlog64_load8", REPO, 5, 8)
probe("backlog5_idle", OLD, 5, 0)
probe("backlog64_idle", REPO, 5, 0)

# The repository's other CI checks, and the evidence corpus from this clone.
sh("cli_smoke", "ironmule --help >/dev/null && ironmule info && ironmule models list; "
                "ironmule doctor; echo doctor_exit=$?", cwd="/tmp")
sh("ruff", f"{VENV}/bin/ruff check ironmule_product ironmule_inventory.py ironmule_cli.py friday_evidence/events.py "
           "friday_evidence/identity.py friday_evidence/statistics.py tools/product_bounded_bench.py "
           "tools/product_history_snapshot.py --select F", cwd=REPO)
sh("figures", f"{PY} tools/make_figures.py --check > /tmp/figures.log 2>&1; rc=$?; "
               "grep -v findfont /tmp/figures.log; exit $rc", cwd=REPO)
sh("ssot", f"{PY} tools/ssot.py --database /tmp/ssot.sqlite3 build && "
           f"{PY} tools/ssot.py --database /tmp/ssot.sqlite3 verify --check-sources", cwd=REPO)

# Last and largest: the opt-in Qwen hybrid-cache gate. PERF1 run 13 found Qwen 3.5 greedy
# decoding non-deterministic with CUDA graphs on, so a failure is rerun with graphs off.
code, out = sh("download_qwen35_9b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                     f"print(s('{MODELS['qwen35-9b'][0]}', revision='{MODELS['qwen35-9b'][1]}'))\"",
               timeout=1200)
if code == 0:
    qwen = {"IRONMULE_QWEN_MODEL": out.strip().splitlines()[-1]}
    first = pytest("pytest_qwen", "tests/engine/test_qwen_hybrid_integration.py -m integration -n 0",
                   timeout=1200, extra=qwen)
    if first.get("exit") not in (0, None):
        pytest("pytest_qwen_graphs0", "tests/engine/test_qwen_hybrid_integration.py -m integration -n 0",
               timeout=1200, extra=dict(qwen, MLX_USE_CUDA_GRAPHS="0"))
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
