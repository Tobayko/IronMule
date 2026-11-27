# Kaggle script kernel (GPU T4): EM1 A/B -- token embedding on the GPU against llama.cpp's CPU placement, same waves,
# tokens compared. Pinned to a pushed commit; reuses the CUDA build of the HC1 kernel (attached as a kernel source), so
# only the em1 patch's two files are compiled. Elsewhere (a RunPod pod): EM1_WORK=/work/em1 EM1_ARCH=86 python THIS.
import json
import os
import subprocess

SHA = os.environ.get("EM1_SHA", "__SHA__")
W = os.environ.get("EM1_WORK", "/kaggle/working")
ARCH = os.environ.get("EM1_ARCH", "75")
os.makedirs(W, exist_ok=True)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q https://github.com/Tobayko/IronMule.git {W}/IronMule && cd {W}/IronMule && git checkout -q {SHA} && git log --oneline -1")
os.chdir(f"{W}/IronMule")
sh("nvidia-smi -L; nproc; pip install -q jinja2 numpy \"mlx[cpu]>=0.32,<0.33\" 2>&1 | tail -1")
prior = subprocess.run("ls /kaggle/input/*/ggml-cuda75.tar 2>/dev/null | head -1", shell=True, capture_output=True,
                       text=True).stdout.strip()
if prior:
    sh(f"mkdir -p ~/.cache/ironmule && tar -xf {prior} -C ~/.cache/ironmule && echo reused {prior}")
sh(f"python tools/build_ggml.py --cuda {ARCH} --jobs {os.cpu_count()} > {W}/build.log 2>&1 || (tail -30 {W}/build.log; exit 1); tail -4 {W}/build.log")
if ARCH == "75":
    sh(f"tar -cf {W}/ggml-cuda75.tar -C ~/.cache/ironmule ggml")
models = f"{W}/models"
sh(f"python lab/portable_gpu/fetch.py qwen3-4b --dir {models} > {W}/fetch.log 2>&1; tail -2 {W}/fetch.log")
tree = subprocess.run("ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-* | head -1", shell=True,
                      capture_output=True, text=True).stdout.split()[0]
env = (f"PYTHONPATH={W}/IronMule IRONMULE_GGML_BINDINGS={tree} LLAMA_CPP_LIB_PATH={tree}/vendor/llama.cpp/build/bin "
       "IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUBLAS_MIN_ROWS=512")
model = f"{models}/Qwen3-4B-Q4_K_M.gguf"
for arm in ("1", "0", "1", "0"):  # alternating, two runs each
    tag = f"em1_{arm}_{len([f for f in os.listdir(W) if f.startswith(f'em1_{arm}_')])}"
    sh(f"{env} IRONMULE_GGML_EM1={arm} python lab/em1/em1_probe.py {model} {W}/{tag}.json 2>&1 "
       f"| grep '^EM1\\|Error\\|Traceback' | tee -a {W}/em1.txt")
runs = {f: json.load(open(f"{W}/{f}")) for f in sorted(os.listdir(W)) if f.startswith("em1_") and f.endswith(".json")}
base = runs.get("em1_0_0.json")
for name, run in runs.items():
    if base is not None:
        same = sum(run["tokens"][k] == base["tokens"][k] for k in base["tokens"])
        rows = sum(a == b for k in base["tokens"] for a, b in zip(run["tokens"][k], base["tokens"][k]))
        total = sum(len(base["tokens"][k]) for k in base["tokens"])
        line = f"TOKENS {name} against em1_0_0: waves identical {same}/{len(base['tokens'])}, answers {rows}/{total}"
        print(line, flush=True)
        open(f"{W}/em1.txt", "a").write(line + "\n")
# the HC1 probe on this build too (the embedding on the GPU leaves the CPU threads nothing to do in a step)
for cfg in ("", "IRONMULE_GGML_EM1=0", "IRONMULE_GGML_THREADS=1 IRONMULE_GGML_POLL=0"):
    sh(f"{env} {cfg} python lab/hc1/hc1_probe.py {model} 2>&1 | grep '^HC1\\|Error\\|Traceback' "
       f"| sed 's/^/{cfg or 'default'} /' | tee -a {W}/em1.txt")
