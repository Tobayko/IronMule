"""PI1 on a second machine: a Kaggle CPU session (no GPU, no GPU quota). Builds llama.cpp with IronMule's kernels
(tools/build_ggml.py), downloads Qwen3-4B Q4_K_M, and runs lab/pi1/real.py with the engine's defaults and with the
shipped Pilot beside 0, 1 (busy, and busy half of every 20 ms) and 2 foreign processes, then lab/pi1/ab_idle.py
(the Pilot attached and detached in turn in one process). Writes /kaggle/working/pi1_cpu.json.
Usage on Kaggle: the kernel runs this file with the commit in MG2_COMMIT."""
import json, os, subprocess, sys, time
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time()}


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def save():
    (W / "pi1_cpu.json").write_text(json.dumps(OUT, indent=1))


def main():
    OUT["cpu"] = subprocess.run("lscpu | head -20; nproc; free -g | head -2", shell=True, capture_output=True,
                                text=True).stdout
    save()
    sh(f"cd {W} && curl -sSL https://github.com/Tobayko/IronMule/archive/{COMMIT}.tar.gz | tar xz && mv IronMule-{COMMIT} repo")
    repo = W / "repo"
    # the engine files without the package's MLX imports: a bare package beside the scripts
    pkg = W / "run" / "ironmule"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("ggml_engine.py", "speculation.py", "ggml_agent.py", "pilot.py", "draft_model.py", "pilot_weights.npz"):
        (pkg / name).write_bytes((repo / "ironmule" / name).read_bytes())
    binary_module = repo / "ironmule" / "ggml_binary.py"
    if binary_module.is_file():
        (pkg / binary_module.name).write_bytes(binary_module.read_bytes())
    devices_module = repo / "ironmule_devices.py"
    if devices_module.is_file():
        (pkg.parent / devices_module.name).write_bytes(devices_module.read_bytes())
    lab = W / "run" / "lab" / "pi1"
    lab.mkdir(parents=True, exist_ok=True)
    for name in ("real.py", "ab_idle.py"):
        (lab / name).write_bytes((repo / "lab" / "pi1" / name).read_bytes())
    t0 = time.time()
    built = subprocess.run(f"cd {repo} && {sys.executable} tools/build_ggml.py --dest {W}/ggml", shell=True,
                           capture_output=True, text=True)
    OUT["build_s"] = round(time.time() - t0, 1)
    OUT["build_tail"] = (built.stdout + built.stderr)[-1500:]
    save()
    exports = dict(line.split(" ", 1)[1].split("=", 1) for line in built.stdout.splitlines() if line.startswith("export "))
    env = {**os.environ, **exports, "IRONMULE_GGML_COLLECTIVE": "0", "ROW1_GGML": "B16"}
    gguf = W / "Qwen3-4B-Q4_K_M.gguf"
    sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    OUT["runs"] = {}
    for mode in ("pilot", "default"):
        rows = []
        for hogs, tokens in (("0,1,d0.5", "64"), ("2", "16")):
            out = W / f"real_{mode}_{hogs.replace(',', '_')}.json"
            done = subprocess.run([sys.executable, str(lab / "real.py"), str(gguf), mode, str(out)], capture_output=True,
                                  text=True, timeout=5400,
                                  env={**env, "REAL_HOGS": hogs, "REAL_NS": "1,8", "REAL_TOKENS": tokens,
                                       **({"REAL_BLOCKS": "1"} if hogs == "2" else {})})
            rows += [json.loads(l) for l in done.stdout.splitlines() if l.startswith('{"hogs"')]
            if done.returncode:
                OUT.setdefault("errors", []).append(done.stderr[-800:])
            OUT["runs"][mode] = rows
            save()
    done = subprocess.run([sys.executable, str(lab / "ab_idle.py"), str(gguf)], capture_output=True, text=True,
                          timeout=3600, env=env)
    OUT["ab_idle"] = done.stdout.splitlines()[-6:]
    OUT["finished"] = time.time()
    save()


main()
