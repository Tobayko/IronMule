# Training and measuring the agent

This folder holds what trains, simulates and measures the IronMule agent ([docs/AGENT.md](../docs/AGENT.md)).
The agent itself ships in `ironmule_product/`. Nothing here is imported by the product.

| File | What it does |
| :-- | :-- |
| `ag1_online.py` | The agent, unchanged, against a simulated server priced by a machine's fitted costs. Live traces, idle replays and qualification run on a simulated clock. `--prompts` adds real prompt texts with one limit for all (AG2). |
| `rl2_headroom.py` | The hardware harness. It runs the product's own service with the agent attached, records the online phase, then compares the default, the fixed rule and the frozen agent on held-out traces. `--plan ag1 --profile NAME` picks a protocol, `--write-kernel` writes a Kaggle kernel, and `--analyse` judges a report. |
| `ag4_routing.py` | AG4 step 2: the product's router between a single path and a stacked engine on one card, against each path alone, priced by a hardware measurement. |
| `rl4_h9_replay.py` | Trains two dispatch checkpoints from zero, selects one on validation traffic, and evaluates both against the fixed arms, simple rules and the published checkpoint on unseen quiet, burst and spike traces. |
| `rl4_agent.py` | RL4, the offline REINFORCE agent whose policy format the product's learned dispatch (LD1) and the agent share. |

Their tests are in [tests/agent/](../tests/agent/). The tests run without a model or GPU, also in CI:

```bash
python -m pytest tests/agent
```

## Replaying the H9 dispatch training

Run these from the repository root. Each command trains two controllers from zero for 16,000 episodes per seed on one CPU core. They use simulated requests and the published timing coefficients for the two-T4 Qwen3-1.7B H9 pool.

```sh
python -m training.rl4_h9_replay --out rl4-replay.json
python -m training.rl4_h9_replay --width-followup --out rl6-replay.json
```

Sixty validation traces per phase select one seed. Each frozen controller then runs on 128 unseen traces per phase, sharing the same arrivals with every baseline. The follow-up reads the first study's archived checkpoint, adds 16 trainable group-width weights, and uses a new test set. The objective adds mean time to first token and mean time to completion.

The width controller selected a cap of two on every held-out decision with at least two requests waiting. During spikes this cut simulated cost by 6.01% against the first study's frozen controller, while mean time to first token rose from 8.32 to 11.15 simulated seconds. Against the best fixed policy in each phase, cost rose 0.29% in quiet traffic and fell 0.62% in bursts and 0.88% in spikes. Both studies saved their checkpoints for replay in the simulator. Hardware qualification would require fresh on-device measurements. [First study](../evidence/runtime/rl4-h9-holdout.json) · [Width follow-up](../evidence/runtime/rl6-h9-width-holdout.json)

## Forecasting the next native workload

Use the normal local GGML bindings and a SmolLM2-135M-Instruct Q4_K_M GGUF. The study opens the native engine on
two CPU threads, checks that widths four and eight produce identical calibration output, and observes 864 real
requests. Each set of eight arrives together. The three workloads have short prompts, longer distinct prompts or
longer prompts with a shared session prefix. All have a three-token generation limit. The controller sees timings
and token counts. Phase labels and copy targets stay in the study.

```sh
IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_GGML_THREADS=2 IRONMULE_GGML_PREFIX_REUSE=1 IRONMULE_PILOT=0 \
  python training/ag10_replay_study.py --gguf /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --state-dir ./new-ag10-state --output ./ag10-replay.json
```

The sampler forecast all 18 held-out blocks before they arrived. It found matching history for eight.
Across four traffic measurements, mean absolute log₂ forecast error fell from 1.010 to 0.976 (3.33%).
Short requests improved from 0.962 to 0.786 and distinct long prompts from 0.734 to 0.723. Follow-ups with
shared prefixes worsened from 1.333 to 1.419. All 576 held-out first words passed the copy check. All requests
ran at width four, providing 1,401 measured processing steps. The agent rebuilt to width eight after the last
block. Measuring the serving effect of that switch requires subsequent traffic. [Per-block evidence and fitted weights](../evidence/runtime/smollm2-135m-cpu-dream-replay.json)

## The workflow

Every step below follows [AGENTS.md](../AGENTS.md), "Performance and evidence". A result counts only when it was
measured on the hardware it claims, and its protocol was fixed before its data.

1. **Write the protocol first.** Write it in [docs/BACKLOG.md](../docs/BACKLOG.md), under the agent's entry (AG1,
   AG2): what changes, the profile, decisions, kill rule and expected session hours. A change after data is
   labelled post hoc.
2. **Simulate.** A gate passes in simulation before hardware is spent, for example S-LP:

   ```bash
   python training/ag1_online.py --traces 300 --out sim.json
   ```

   ```bash
   python training/ag1_online.py --traces 300 --prompts prompts.jsonl --limit 512 --out sim.json
   ```

   `--profile v-g12` mirrors a hardware protocol's per-card sizes for a dry run of its online phase.

   A prompts file has one `{"prompt": ..., "tokens": ...}` per line. It is built from a public dataset, kept
   outside the repository, and never committed.
3. **Smoke locally.** A smoke checks the tool and measures nothing:

   ```bash
   python training/rl2_headroom.py --plan ag1 --profile cpu-lp --smoke --model mlx-community/SmolLM2-135M-Instruct-8bit --out smoke.json
   ```

4. **Run on Kaggle.** Push the commit first, then write a kernel pinned to its full SHA:

   ```bash
   python training/rl2_headroom.py --write-kernel kernel/ --owner tobayko --commit FULL_SHA --plan ag1 --profile t4-g12-v
   ```

   ```bash
   kaggle kernels push -p kernel/
   ```

   For a smoke kernel, add `'--smoke', '--model', MODEL` to the last line of `kernel/run.py`, and add `-smoke` to
   the id in `kernel/kernel-metadata.json`. Kaggle runs spend the owner's weekly GPU quota. State the expected
   session hours first, and poll the status at most every two minutes.
5. **Analyse and record.** Read a report with the harness's own judge:

   ```bash
   python -I training/rl2_headroom.py --analyse rl2_headroom_report.json
   ```

   Downloaded output goes into a new, empty directory outside the repository and is read with `python -I`.
   Record the verdict under the entry, and note that the research repository's ledger entry is still owed.
   Archive a finished kernel's source and output in `tobayko/ironmule-notebook-archive` before deleting it.

## Profiles (`--plan ag1 --profile ...`)

| Profile | Machine and model | What it tests |
| :-- | :-- | :-- |
| `t4`, `t4b`-`t4d` | T4, Qwen3 1.7B, tuned engine | AG1 V-V4: order by limit and width |
| `cpu`, `cpu2`, `cpu3` | CPU, SmolLM2-135M | AG1 V-CPU, V5-CPU: order only (capacity 1) |
| `cpu-lp` | CPU, SmolLM2-135M, Dolly prompts | AG2 smoke path (real prompts, one limit) |
| `t4-g12-lp` | 2 x T4, Gemma 3 12B `native`, Dolly | AG2 gate (smoke) |
| `t4-g12-v` | 2 x T4, Gemma 3 12B `native`, Dolly, limit 256 | AG2 V-G12 (replaced before its data by KB2) |

A new protocol gets a new profile on fresh seeds. A test checks that no two profiles share a seed.
