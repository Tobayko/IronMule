# Training and measuring the agent

This folder holds what trains, simulates and measures the IronMule agent ([docs/AGENT.md](../docs/AGENT.md)).
The agent itself ships in `ironmule_product/`. Nothing here is imported by the product.

| File | What it does |
| :-- | :-- |
| `ag1_online.py` | The agent, unchanged, against a simulated server priced by a machine's fitted costs. Live traces, idle replays and qualification run on a simulated clock. `--prompts` adds real prompt texts with one limit for all (AG2). |
| `rl2_headroom.py` | The hardware harness. It runs the product's own service with the agent attached, records the online phase, then compares the default, the fixed rule and the frozen agent on held-out traces. `--plan ag1 --profile NAME` picks a protocol, `--write-kernel` writes a Kaggle kernel, and `--analyse` judges a report. |
| `rl4_agent.py` | RL4, the offline REINFORCE agent whose policy format the product's learned dispatch (LD1) and the agent share. |

Their tests are in [tests/agent/](../tests/agent/). The tests run without a model or GPU, also in CI:

```bash
python -m pytest tests/agent
```

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
| `t4-g12-v` | 2 x T4, Gemma 3 12B `native`, Dolly, limit 256 | AG2 V-G12, the hardware gain |

A new protocol gets a new profile on fresh seeds. A test checks that no two profiles share a seed.
