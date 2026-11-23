# Kaggle script kernel (GPU T4): HC1 probe for two threadpool settings. Pinned to a pushed commit.
import os, subprocess
SHA = "__SHA__"
W = "/kaggle/working"
def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
sh(f"git clone -q https://github.com/Tobayko/IronMule.git {W}/IronMule && cd {W}/IronMule && git checkout -q {SHA} && git log --oneline -1")
os.chdir(f"{W}/IronMule")
sh("nvidia-smi -L; nproc; pip install -q jinja2 numpy 2>&1 | tail -1")
sh(f"python tools/build_ggml.py --cuda 75 > {W}/build.log 2>&1 || (tail -30 {W}/build.log; exit 1); tail -3 {W}/build.log")
sh(f"python lab/portable_gpu/fetch.py qwen3-4b --dir {W}/models > {W}/fetch.log 2>&1; tail -2 {W}/fetch.log")
tree = subprocess.run("ls -d ~/.cache/ironmule/ggml/*/cuda/llama_cpp_python-* 2>/dev/null || ls -d ~/.cache/ironmule/ggml/*/llama_cpp_python-* | head -1",
                      shell=True, capture_output=True, text=True).stdout.split()[0]
env = f"IRONMULE_GGML_BINDINGS={tree} LLAMA_CPP_LIB_PATH={tree}/vendor/llama.cpp/build/bin IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUBLAS_MIN_ROWS=512"
model = f"{W}/models/Qwen3-4B-Q4_K_M.gguf"
for cfg in ("", "IRONMULE_GGML_THREADS=1 IRONMULE_GGML_POLL=0", "IRONMULE_GGML_POLL=0", "IRONMULE_GGML_THREADS=2"):
    sh(f"{env} {cfg} python lab/hc1/hc1_probe.py {model} 2>&1 | grep '^HC1\\|Error\\|Traceback' | tee -a {W}/hc1.txt")
