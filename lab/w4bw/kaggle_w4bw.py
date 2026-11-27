# Kaggle script kernel (GPU T4 or L4): W4R bandwidth bench (lab/w4bw/bench.py), pinned to a commit. On a T4 the
# kernels run through T4W (Turing fallbacks in native/kernels/w4mma.cu): Turing numbers, not an A100's.
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
if not cc or float(cc[0]) < 7.5:  # W4R: sm_80+, or Turing through T4W (m16n8k8, plain copies, no MQ1)
    raise SystemExit(f"W4BW: this session's GPU (compute capability {cc}) cannot run W4R")
sh("pip install -q numpy \"mlx[cpu]>=0.32,<0.33\" 2>&1 | tail -1")
sh(f"PYTHONPATH={W}/IronMule python lab/w4bw/bench.py {W}/w4bw {'' if ARGS.startswith('__') else ARGS} 2>&1 | tee {W}/w4bw.log")
