# Train the hardware controller on a Linux host

IronMule can run a GGUF chat model on the CPU, on one GPU, on independent GPU replicas, or in one engine split over visible GPUs. The operator supplies a model with a chat template. At startup `--placement auto` checks which devices have enough free memory and qualifies their actual GGUF execution. It keeps a CPU worker if system memory permits. The scope is one host per controller. Independent servers learn their own costs.

Each request stays on one worker for its lifetime. The C++17 controller predicts completion time from the prompt length in characters, the output limit and the number of requests already on each worker. It can reserve a fast worker for a possible second, longer request. The existing Pilot inside each worker controls the finer decisions, including thread count, decode width and prefill budget. It already learns from step-level measurements and runs replay during idle periods. [Per-step Pilot and previous RL training](../docs/AGENT.md)

## Start on a CPU server, RunPod or Kaggle

Use a source checkout on Linux with Python 3.12, a C++17 compiler, CMake, Git and `patch`. A CUDA development image needs a driver and `nvcc` that support the card. The GGUF backend runs through llama.cpp. The MLX CPU wheel covers the Python package's MLX dependency on these hosts. On Kaggle, select an available accelerator, put the GGUF in a notebook input and use a writable state directory under `/kaggle/working`. On RunPod, put the model and state on a volume that outlives the pod. Persist the state directory to resume after a session ends. [RunPod Pods](https://docs.runpod.io/pods/overview) · [Kaggle notebooks](https://www.kaggle.com/docs/notebooks)

```bash
python -m pip install -e '.[cpu]' 'cmake>=3.22,<4' ninja diskcache jinja2
python tools/build_ggml.py --auto --env-file ./ggml.env
source ./ggml.env
export MODEL=/absolute/path/to/model.gguf
export IRONMULE_HOME="$PWD/ironmule-state"
ironmule hardware calibrate --gguf "$MODEL" --minutes 120
ironmule serve --gguf "$MODEL" --placement auto
```

`calibrate` initializes the state and runs until the hardware comparison is complete or the post-startup time budget expires. It uses the production workers without opening an HTTP port. You can also start an ordinary server immediately with `ironmule setup --mode server` and `ironmule serve --gguf "$MODEL" --placement auto`. It begins by routing to the shortest queue and gathers the same evidence in gaps in demand. The GGUF path, build, state directory and relevant runtime settings must match for the next start to reuse evidence. `CUDA_VISIBLE_DEVICES` restricts NVIDIA devices. `IRONMULE_GGML_GPU_LAYERS=0` selects the CPU.

For a small first run, the measured CPU trace used the following public GGUF. Set `MODEL` to the downloaded file before calibrating.

```bash
curl -fL 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf' -o smollm2-135m.gguf
echo '2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d  smollm2-135m.gguf' | sha256sum -c -
export MODEL="$PWD/smollm2-135m.gguf"
```

## What happens overnight

1. After 15 seconds without admissions, the controller checks for outstanding requests, host CPU and memory pressure where exposed, and utilization of the selected accelerators. It records completed, paired trials on the qualified workers. An incoming request cancels active probes and invalidates any partial pair. Probes use public generated fixtures. Only lengths from recent caller prompts and their output limits influence workload selection.
2. Eight workloads combine two prompt sizes, two output limits and different queue states. The order of workers rotates between trials. The first three rounds supply the C++ trainer with 24 measured costs per worker. It fits a ridge model of log latency and evaluates three ways to price the next arrival. The native planner plays out placing the current request on each candidate and the next one on its best remaining lane. This is a bounded model-based decision with a two-request horizon.
3. The final two rounds provide 16 new paired measurements per worker. The same frozen policy has to reduce total completion time by at least 5% in **each** round. Total time to first token must hold or improve and the compared routes must generate the same token counts on each fixture. The GGUF answer-quality gate has already run per worker. Expect the reference to win on some machines. One usable worker produces a latency model and always keeps its sole route.
4. At admission, the learned policy applies inside measured prompt/output ranges and queue configurations when all its workers are available. Elsewhere the shortest accepting queue serves. Three consecutive observed completion times above twice the prediction or a failed chosen backend withdraw the policy. The next idle campaign starts from the reference. A sustained change of workload sizes or a six-hour refresh interval also starts a new campaign. The state keeps recent comparison outcomes. A completed campaign sleeps between refreshes.
5. `GET /health` reports `placement.phase`, trial count, remaining idle seconds and the trace path. The last foreground decision names the worker, the reason and, for learned routes, its predicted completion time. `POST /v1/ironmule/learning` with `{"action":"pause"}` or `{"action":"resume"}` controls placement probes. Pausing persists and serves the reference. A backend computation or already launched GPU kernel can take until its next cancellation point to yield.

These measurements compare finite local fixtures, under queue depths zero and one. The workload planner caps fixtures at 4,096 prompt characters and 128 output tokens. Read throughput and long-context tails from a separate serving benchmark with representative concurrent traffic. A deployment with more work waiting still serves through the reference and the per-worker Pilot.

A real workload transition was also exercised on two separately qualified configurations of the same CPU. The first campaign kept the reference when output lengths differed. After 32 short requests the policy retrained, passed both comparison rounds and routed a subsequent foreground request to the three-thread worker. The two holdout completion-time ratios were 0.550 and 0.530. That result applies to these short CPU fixtures. [Measurements and reproduction](../docs/LAB.md#2026-11-23-hardware-placement-on-a-linux-cpu)

## Training time

A complete local CPU run with SmolLM2-135M took **79 seconds** with a cached quality gate, including **75 seconds** of idle calibration. Replaying its trace through the C++ trainer and holdout took **0.54 ms**. [Measured trace](../lab/placement/results/smollm2-135m-cpu.json)

A campaign uses **40 × number of workers** timed worker measurements. Each includes the extra load request in loaded cases. The first 24 sets fit the policy and the final 16 sets qualify it. The native fit and scoring run on a CPU core. Device time is spent running the GGUF. Build, download and the per-worker startup quality gates have their own cost.

| Host example | Worker measurements | Idle campaign at an assumed mean of 5–15 s per worker measurement |
| --- | ---: | ---: |
| RunPod, one GPU and a CPU worker | 80 | 8–21 min |
| Kaggle, two GPUs and a CPU worker | 120 | 11–31 min |
| One shared GPU engine or CPU only | 40 | 4–11 min |

The ranges include 15 seconds of initial quiet and about one second between completed sets. They exclude contention, interrupted trials, setup and startup gates. For a large model, a slow CPU replica can dominate. With 60 seconds per CPU measurement and 4 per GPU measurement, one CPU and one GPU need about **44 idle minutes**. The running service replaces these assumptions with observed timings in `placement.remaining_idle_seconds`.

```bash
ironmule hardware estimate --devices 2 --seconds-per-arm 8 --startup-minutes 20.35
ironmule hardware benchmark
ironmule hardware train --trace lab/placement/results/smollm2-135m-cpu.json --output ./placement-report.json
```

The first command gives about 32 minutes for that particular input. The second times the native fit on 4,096 synthetic timing records. The third retrains and replays the included real CPU trace offline. To train on your machine, substitute the trace path printed by your calibration. Online activation uses the local trace and recomputes its holdout comparison when restored. A changing topology, driver or kernel/quality binding starts a different trace. The earlier neural Pilot's teacher/PPO run took 47 minutes on two Kaggle T4s. The earlier RunPod A100 Qwen3-32B session spent 1,221 seconds in startup. They measure different jobs. [Pilot training](../docs/AGENT.md) · [A100 session and CPU calibration](../docs/LAB.md)

## Kernel generation and hardware scope

The automatic native build detects visible CUDA compute capabilities or ROCm targets and can build a CPU backend. At application startup the CUDA generator can emit, compile and time kernel variants for supported operations on a previously unseen card. On SM80+ the W4 path evaluates quantized matrix kernels against a float64 reference and selects observed winners by shape. The generated library and dispatch table are bound to the GPU, driver, compiler, GGUF and generator sources. A failed W4 attempt can try the general CUDA generator, then use the backend kernels that pass the quality gate. The W4 source and its licensed headers are included in the wheel.

Independent replicas each need room for the whole model and the KV cache. When replicas do not fit but aggregate visible device memory looks sufficient, IronMule tries the GGML layer-split worker. The real backend decides whether the topology works. A new vendor ISA needs a compatible backend and toolchain. Hardware discovery happens on process startup. Other servers and GPU hotplug during a live request are outside this placement protocol. [CUDA multi-GPU systems](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

The design borrows the idea of planning from measured experience from Dyna and measuring candidate schedules on the target hardware from Ansor. The shipped controller implements the small estimator and paired comparison above. [Sutton, integrated modeling and control](https://proceedings.neurips.cc/paper_files/paper/1990/hash/d9fc5b73a8d78fad3d6dffe419384e70-Abstract.html) · [Zheng et al., Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)
