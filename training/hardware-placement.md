# Train the hardware controller on a Linux host

IronMule can run a GGUF chat model on the CPU, on one GPU, on independent GPU replicas, or in one engine split over visible GPUs. The operator supplies a model with a chat template. Run `ironmule serve --gguf MODEL.gguf --placement auto` after building the local backend. On a fresh machine the command initializes server state. At startup it checks which devices have enough free memory and qualifies their actual GGUF execution. It keeps a CPU worker if system memory permits. The scope is one host per controller. Independent servers learn their own costs.

Each request stays on one worker for its lifetime. The hardware controller trains and decides on the CPU. Its C++17 latency model predicts completion time from the prompt length in characters, the output limit and the number of requests already on each worker. It can reserve a fast worker for a possible second, longer request. The existing Pilot inside each worker controls the finer decisions, including thread count, decode width and prefill budget. It already learns from step-level measurements and runs replay during idle periods. [Per-step Pilot and previous RL training](../docs/AGENT.md)

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

`calibrate` initializes the state and runs until the hardware comparison is complete or the post-startup time budget expires. It uses the production workers without opening an HTTP port. You can start serving immediately with `ironmule serve --gguf "$MODEL" --placement auto`. Open `http://127.0.0.1:8080/` for the chat application or use its OpenAI-compatible API. `GET /v1/models` includes the loaded GGUF. It begins by routing to the shortest queue and gathers the same evidence in gaps in demand. The GGUF path, build, state directory and relevant runtime settings must match for the next start to reuse evidence. `CUDA_VISIBLE_DEVICES` restricts NVIDIA devices. `IRONMULE_GGML_GPU_LAYERS=0` selects the CPU.

For a small first run, the measured CPU trace used the following public GGUF. Set `MODEL` to the downloaded file before calibrating.

```bash
curl -fL 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf' -o smollm2-135m.gguf
echo '2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d  smollm2-135m.gguf' | sha256sum -c -
export MODEL="$PWD/smollm2-135m.gguf"
```

## What happens overnight

1. A new server starts probing after 15 seconds of spare capacity. It records the length of actual empty periods. Once it has observed eight gaps, it can start after two seconds if the lower quartile of the recent gaps leaves time for a case. It uses the last measured case duration, initially assuming two seconds. All workers must be accepting and empty, with host and accelerator pressure below their limits. Each worker also has its own per-step Pilot. An incoming request cancels placement probes, discards the partial comparison and restores the usual wait for the next attempt. A current GPU kernel may take until its next cancellation point to yield.
2. Eight fixtures cover prompt length, output budget and queues. Once a queue pattern appears at least four times, the next campaign can replay up to four recurring patterns with at most four total requests ahead. Each arm recreates the background requests on all affected workers. For deeper queues the latency model prices the immediate request. All candidate workers run the same public fixture, in rotating order. The first 16 cases train a native latency predictor and, for competing routes, a contextual bandit from scratch. In the bandit, each action selects a worker. Its C++17 update increases the probability of actions with higher measured reward. Reward reflects completion-time gains, penalizes slower first tokens and penalizes different output or stopping. Because idle trials measure every arm, the trainer can evaluate alternative actions without exploring on a caller. Probe timings may teach the per-step cost model. Probe arrivals are excluded from the worker's foreground traffic history.
3. Eight additional cases select between the bandit and three two-request lookahead settings of the latency model. The winner is fitted on all 24 training cases and frozen. The last two rounds have eight new, paired cases each. Compared with the shortest queue, the policy must save at least 5% in total completion time in **both** rounds, preserve or improve total first-token time, and match visible answers, output counts and stop reasons for every selected route. Response digests cover generated public fixtures. Only lengths and output limits from caller messages inform the workload plan. Earlier traces replay offline under their original protocol. Changing the queue protocol starts a fresh local campaign.
4. Foreground requests use the trained policy inside its measured workload range. Unsupported queue shapes or unavailable workers use the shortest accepting queue. Three consecutive completion times above twice the prediction, or a failed selected worker, withdraw the policy. A sustained change in request sizes or the six-hour refresh restarts measurement in spare capacity. One eligible worker keeps its route while the internal Pilot controls the engine.
5. `GET /health` reports the training phase, idle threshold, interruptions, remaining idle seconds, last routing decision and trace path under `placement`. The estimate uses median case times for observed workloads. Pausing and resuming placement are available through `POST /v1/ironmule/learning` with `{"action":"pause"}` or `{"action":"resume"}`. Completed trials are checkpointed atomically. The next start reconstructs the policy and rechecks its holdouts for the same machine, model, runtime and training implementation.

The initial campaign compares an empty queue and one request ahead. Subsequent campaigns can measure recurring foreground backlogs at the same request lengths. Only an observed and measured queue configuration is eligible for learned routing. The workload planner caps fixtures at 4,096 prompt characters and 128 output tokens. Read throughput and long-context tails from a separate serving benchmark with representative concurrent traffic. A deployment with more work waiting still serves through the reference and the per-worker Pilot.

An earlier token-count-qualified workload transition was exercised on two separately qualified configurations of the same CPU. The first campaign kept the reference when output lengths differed. After 32 short requests the policy retrained, passed both comparison rounds and routed a subsequent foreground request to the three-thread worker. The two holdout completion-time ratios were 0.550 and 0.530. That result applies to these short CPU fixtures. [Measurements and reproduction](../docs/LAB.md#2026-11-23-hardware-placement-on-a-linux-cpu)

## Training time

An earlier complete local CPU run with SmolLM2-135M took **79 seconds** with a cached quality gate, including **75 seconds** of idle calibration. Replaying its trace through the C++ trainer and holdout took **0.54 ms**. [Measured trace](../lab/placement/results/smollm2-135m-cpu.json)

A campaign uses **40 × number of workers** timed worker measurements. Each includes the extra load request in loaded cases. The first 24 sets fit the policy and the final 16 sets qualify it. The native fit and scoring run on a CPU core. Device time is spent running the GGUF. Build, download and the per-worker startup quality gates have their own cost.

| Host example | Worker measurements | Idle campaign at an assumed mean of 5–15 s per worker measurement |
| --- | ---: | ---: |
| RunPod, one GPU and a CPU worker | 80 | 8–21 min |
| Kaggle, two GPUs and a CPU worker | 120 | 11–31 min |
| One shared GPU engine or CPU only | 40 | 4–11 min |

The two-GPU example assumes a model that fits on each card. A 32B checkpoint split across both cards counts as one GPU worker. The ranges include 15 seconds of initial quiet and about one second between completed sets. They exclude contention, interrupted trials, setup and startup gates. For a large model, a slow CPU replica can dominate. With 60 seconds per CPU measurement and 4 per GPU measurement, one CPU and one GPU need about **44 idle minutes**. The running service replaces these assumptions with observed timings in `placement.remaining_idle_seconds`.

```bash
ironmule hardware estimate --devices 2 --seconds-per-arm 8 --startup-minutes 20.35
ironmule hardware estimate --seconds-per-worker 4 60 --startup-minutes 20.35
ironmule hardware estimate --trace /path/to/current-placement-trace.json
ironmule hardware benchmark
ironmule hardware train --trace lab/placement/results/smollm2-135m-cpu.json --output ./placement-report.json
```

The first command gives about 32 minutes for that particular input. The asymmetric estimate gives roughly 64 minutes including startup. A live trace estimates the time needed for its remaining cases. The benchmark times the latency fit on 4,096 synthetic records. The train command retrains and replays the included real CPU trace offline. To train on your machine, substitute the trace path printed by your calibration. Online activation uses the local trace and recomputes its holdout comparison when restored. A changing topology, driver or kernel/quality binding starts a different trace. The earlier neural Pilot's teacher/PPO run took 47 minutes on two Kaggle T4s. The earlier RunPod A100 Qwen3-32B session spent 1,221 seconds in startup. They measure different jobs. [Pilot training](../docs/AGENT.md) · [A100 session and CPU calibration](../docs/LAB.md)

## Kernel generation and hardware scope

The automatic build queries visible CUDA capabilities and checks installed compilers for native support for every requested GPU. If the accelerator build fails, auto mode reports the error and builds the CPU backend. ROCm builds target locally enumerated gfx architectures. [NVIDIA compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) At application startup the CUDA generator can emit, compile and time kernel variants for supported operations on a previously unseen card. On SM80+ the W4 path evaluates quantized matrix kernels against a float64 reference and selects observed winners by shape. The generated library and dispatch table are bound to the GPU, driver, compiler, GGUF and generator sources. A failed W4 attempt can try the general CUDA generator, then use the backend kernels that pass the quality gate. The W4 source and its licensed headers are included in the wheel.

A ROCm replica is identified by PCI address. The qualification child and the serving worker each verify that the HIP visibility mask resolves to that same physical board. The idle controller reads the Linux driver's busy percentage only for participating AMD GPUs. Unavailable readings defer calibration. AMD uses the kernels in the compiled HIP backend. A real AMD replica comparison is still required. [HIP device visibility](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html) · [Linux amdgpu activity readings](https://docs.kernel.org/gpu/amdgpu/thermal.html)

Independent replicas each need room for the whole model and the KV cache. When replicas do not fit but aggregate visible device memory looks sufficient, IronMule tries the GGML layer-split worker. The real backend decides whether the topology works. A new vendor ISA needs a compatible backend and toolchain. Hardware discovery happens on process startup. Other servers and GPU hotplug during a live request are outside this placement protocol. [CUDA multi-GPU systems](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

The design borrows the idea of planning from measured experience from Dyna and measuring candidate schedules on the target hardware from Ansor. The shipped controller implements the small estimator and paired comparison above. [Sutton, integrated modeling and control](https://proceedings.neurips.cc/paper_files/paper/1990/hash/d9fc5b73a8d78fad3d6dffe419384e70-Abstract.html) · [Zheng et al., Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)
