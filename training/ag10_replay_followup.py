"""Continue a completed native replay study from an exact local checkpoint and test its width proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ag10_replay_study as study

ag, ge = study.ag, study.ge
WORDS = ("mint", "cream", "navy", "teal", "plum", "peach", "lime", "beige")
ARMS = ("agent", "w4", "w8")


def run_arm(engine, agent, waves):
    elapsed, emitted, correct = 0.0, 0, 0
    times, answers = [], []
    post_first_s, after_first_tokens = 0.0, 0
    for jobs, targets in waves:
        live = [ge.Job(job.ids, job.limit) for job in jobs]
        arrived = time.monotonic_ns()
        engine.run(live)
        finished = max(job.finished_ns for job in live)
        elapsed += (finished - arrived) / 1e9
        post_first_s += (finished - min(job.first_ns for job in live)) / 1e9
        for job, target in zip(live, targets):
            if not (job.reason in ("eos", "length") and arrived <= job.first_ns <= job.finished_ns and job.tokens):
                raise RuntimeError("incomplete native request")
            times.append((job.first_ns - arrived) / 1e6)
            if agent is not None:
                agent.observe_request(arrived / 1e9, len(job.ids), len(job.tokens), len(job.ids) - job.reused,
                                      first=job.first_ns / 1e9, finished=job.finished_ns / 1e9)
            first = re.search(r"\w+", study.output_text(engine, job.tokens).casefold())
            correct += int(bool(first and first.group() == target))
            emitted += len(job.tokens)
            after_first_tokens += len(job.tokens) - 1
            answers.append((job.reason, job.tokens))
    return {"width": engine.seqs, "requests": len(times), "generated": emitted, "correct": correct,
            "elapsed_s": elapsed, "tokens_per_s": emitted / elapsed, "ttft_ms": times,
            "ttft_p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
            "post_first_s": post_first_s, "post_first_tokens": after_first_tokens,
            "output_digest": hashlib.sha256(json.dumps(answers).encode()).hexdigest()}


def run(args):
    if args.state_dir.exists() or args.output.exists():
        raise ValueError("choose a fresh state directory and output file")
    if ge.THREADS != 2 or ge.gpu_layers(ge.bindings()) or not ge.PREFIX_REUSE:
        raise ValueError("the study requires two CPU threads, local CPU inference and prefix reuse")
    prior = json.loads(args.prior_report.read_text())
    if prior.get("schema") != "ironmule.ag10.replay-study.v1" or prior["blocks"][-1].get("switched_to") != "w8":
        raise ValueError("the source study must end with an autonomous switch to width eight")
    files = list((args.prior_state / "ggml-agent").glob("*.json"))
    if len(files) != 1:
        raise ValueError("provide the unique exact local study state")
    checkpoint = json.loads(files[0].read_text())
    binding = {"gguf_sha256": study.digest(args.gguf), "provider": "cpu", "threads": 2, "study": "ag10-replay-v1",
               "machine": ge.machine_key()}
    if checkpoint.get("binding") != binding or binding["gguf_sha256"] != prior["gguf_sha256"]:
        raise ValueError("model or machine binding changed")
    for source in ("ironmule/ggml_engine.py", "ironmule/ggml_agent.py"):
        if prior["source_sha256"][source] != study.digest(ROOT / source):
            raise ValueError("engine or controller code differs from the measured revision")
    if study.digest(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libllama.so") != prior["native_llama_sha256"]:
        raise ValueError("different native executable")
    if not (checkpoint["requests"] == sum(row["requests"] for row in prior["blocks"])
            and checkpoint["current"] == "w8" and checkpoint["trial"] == {"from": "w4", "since": checkpoint["requests"]}):
        raise ValueError("checkpoint does not contain the measured pending trial")
    calibration = ge.measure_widths(str(args.gguf), {}, study.CONTEXT, [4, 8], steps=3, repeats=2, long_ctx=96,
                                    experience={"prompt_tokens": 64, "concurrency": 8, "widths": [4, 8]})
    if not calibration.get("output_digests") or calibration["output_digests"] != prior["calibration_output_digests"]:
        raise ValueError("calibration outputs changed")
    configs = ag.configs_from_measurement([calibration], study.CONTEXT)
    dest = args.state_dir / "ggml-agent" / files[0].name
    dest.parent.mkdir(parents=True)
    shutil.copyfile(files[0], dest)
    agent = ag.GgmlAgent(args.state_dir, binding, configs, "w4", seed=study.SEED)
    if not (agent.path == dest and agent.requests == checkpoint["requests"] and agent.current == "w4"
            and agent.live == checkpoint["live"] and agent.trial is None):
        raise ValueError("continuation failed to restore the exact checkpoint under the normal restart guard")
    report = {"schema": "ironmule.ag10.width-continuation.v1", "source_study_sha256": study.digest(args.prior_report),
              "source_checkpoint_sha256": study.digest(files[0]), "gguf_sha256": binding["gguf_sha256"],
              "native_llama_sha256": prior["native_llama_sha256"], "context": study.CONTEXT, "threads": 2,
              "verified_calibration_outputs": calibration["output_digests"], "resumed_incumbent": agent.current,
              "original_proposal": "w8", "restored_requests": agent.requests,
              "restored_live_steps": dict(agent.live), "source_sha256": {k: study.digest(ROOT / k) for k in
                  ("training/ag10_replay_study.py", "training/ag10_replay_followup.py",
                   "ironmule/ggml_agent.py", "ironmule/ggml_engine.py")}, "blocks": []}
    with ExitStack() as cleanup:
        engines = {}
        for arm in ARMS:
            width = 8 if arm == "w8" else 4
            engines[arm] = ge.GgmlEngine(str(args.gguf), seqs=width, seq_ctx=study.CONTEXT, threads=2, speculate=False)
            cleanup.callback(engines[arm].close)
        engines["agent"].observer = agent.observe_step
        phases = [("reacquisition", "followup")] * 3
        for phase in ("short", "document", "followup", "short"):
            phases.extend([("holdout", phase)] * 3)
        for index, (role, phase) in enumerate(phases):
            waves = [study.prepare_wave(engines["agent"], role, phase, 1000 + index, wave, vocabulary=WORDS)
                     for wave in range(4)]
            order = list(ARMS[index % 3:] + ARMS[:index % 3])
            answers = {}
            for arm in order:
                answers[arm] = run_arm(engines[arm], agent if arm == "agent" else None, waves)
            row = {"block": index, "role": role, "phase": phase, "order": order, "arms": answers}
            target = agent.idle()
            if agent.history and agent.history[-1]["requests"] == agent.requests:
                row["post_decision"] = agent.history[-1]["action"]
                row["reason"] = agent.history[-1].get("reason")
            if target:
                before = time.monotonic()
                engines["agent"].reconfigure(configs[target]["seqs"], configs[target]["settings"])
                seconds = time.monotonic() - before
                agent.switched(target, seconds=seconds)
                row.update(switched_to=target, switch_s=seconds)
            report["blocks"].append(row)
            print(index, role, phase, answers["agent"]["width"], row.get("post_decision"), target, flush=True)
            report.update(status="running", live_steps=dict(agent.live))
            study.save(args.output, report)
    report.update(status="complete", final_serving=agent.current, trial=agent.trial, rejected=sorted(agent.rejected),
                  final_world_models=agent.status()["world_model"])
    study.save(args.output, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("gguf", "prior-report", "prior-state", "state-dir", "output"):
        p.add_argument("--" + option, required=True, type=Path)
    run(p.parse_args())
