# Clean-up check: the engine suite and the repository's CI checks on commit 299300a0ef5f1306dc2b73d8e3b73644861aef48
# (dead Q4 helpers and archived Kaggle templates removed, repository map added).
# Kaggle CPU notebook, no GPU quota; MLX's CPU build. No performance claim.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "299300a0ef5f1306dc2b73d8e3b73644861aef48"
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 40 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.cleanup-check.v1", "commit": COMMIT, "stages": {}}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           IRONMULE_HOME="/tmp/ironmule-home", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/cleanup-result.json", "w") as stream:
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


sh("env", "nproc; free -g; uname -r")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}' 'mlx[cpu]>=0.32,<0.33' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy 'ruff==0.16.6' 'matplotlib==3.11.1'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)
sh("pytest_q4_serial", f"{PY} -m pytest -n 0 tests/engine/test_q4_corpus.py tests/engine/test_q4_optimizer.py "
                       "tests/engine/test_docs_links.py -rfEs -p no:cacheprovider", timeout=900, cwd=REPO)
sh("ruff", f"{VENV}/bin/ruff check ironmule_product ironmule_inventory.py ironmule_cli.py friday_evidence/events.py "
           "friday_evidence/identity.py friday_evidence/statistics.py tools/product_bounded_bench.py "
           "tools/product_history_snapshot.py --select F", cwd=REPO)
sh("cli_smoke", "ironmule --help >/dev/null && ironmule info", cwd="/tmp")
sh("figures", f"{PY} tools/make_figures.py --check > /tmp/figures.log 2>&1; rc=$?; "
              "grep -v findfont /tmp/figures.log; exit $rc", cwd=REPO)
sh("ssot", f"{PY} tools/ssot.py --database /tmp/ssot.sqlite3 build && "
           f"{PY} tools/ssot.py --database /tmp/ssot.sqlite3 verify --check-sources", cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
