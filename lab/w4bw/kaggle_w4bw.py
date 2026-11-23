# Kaggle script kernel (GPU L4, machine_shape NvidiaL4): W4R bandwidth bench (lab/w4bw/bench.py), pinned to a commit.
import os
import subprocess

SHA = os.environ.get("W4BW_SHA", "__SHA__")
ARGS = os.environ.get("W4BW_ARGS", "__ARGS__")
W = "/kaggle/working"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q https://github.com/Tobayko/IronMule.git {W}/IronMule && cd {W}/IronMule && git checkout -q {SHA} && git log --oneline -1")
os.chdir(f"{W}/IronMule")
sh("nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv,noheader; nproc; nvcc --version | tail -2")
cc = subprocess.run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True, capture_output=True,
                    text=True).stdout.split()
if not cc or float(cc[0]) < 8.0:  # W4R needs sm_80+: a T4 fallback stops here, before any build
    raise SystemExit(f"W4BW: this session's GPU (compute capability {cc}) cannot run W4R")
sh("pip install -q numpy \"mlx[cpu]>=0.32,<0.33\" 2>&1 | tail -1")
sh(f"PYTHONPATH={W}/IronMule python lab/w4bw/bench.py {W}/w4bw {'' if ARGS.startswith('__') else ARGS} 2>&1 | tee {W}/w4bw.log")
