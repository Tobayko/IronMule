"""Import paired serving measurements and target-specific prompt experiments into the existing experience archive."""

import json

from lab.pk6.native_dispatch.score import run
from training.experience_archive import record


def experiences(root, source, digest):
    path = root / source
    result = json.loads(path.read_bytes())
    folder = path.parent
    if path.name == "comparison.json":
        if result != run(folder / "http"):
            raise ValueError(
                "native dispatch comparison differs from its raw stream receipts"
            )
        hardware, base, gguf, awq, version, seeds = result["binding"]
        for arm, run_result in result["arms"].items():
            for width, metrics in run_result["metrics"].items():
                external = arm == "vllm"
                yield record(
                    "native_dispatch_http",
                    source,
                    digest,
                    f"{arm}/{width}",
                    "l40s-native-dispatch",
                    hardware=hardware,
                    base_commit=base,
                    model=awq if external else gguf,
                    quantization="official AWQ-Marlin" if external else "Q4_K_M",
                    software={
                        "engine": "vllm-" + version if external else "IronMule",
                        "process": arm,
                    },
                    workload={
                        "concurrency": int(width),
                        "output_limit": 32,
                        "seeds": seeds if external else [seeds[int(arm[1:]) - 1]],
                    },
                    metrics=[metrics],
                    quality=run_result["quality"],
                    memory_mib=run_result["resources"]["peak_gpu_mib"],
                    profiles=[run_result["proof"]],
                    reason=run_result["excluded_from_promotion"]
                    or "measured product comparison",
                    authorship="Codex wrote the Ada candidates. IronMule ranked the native Q4_K tile order.",
                    uncertainty="System comparison on one board with distinct official quantizations. Diagnostic and rejected paths supply no production labels.",
                )
    elif path.name == "chunk-plan.json":
        from lab.pk6.native_dispatch.chunk_plan import propose

        if result != propose(folder / "http/primary"):
            raise ValueError("unified chunk forecast differs from its device traces")
        yield record(
            "native_chunk_forecast",
            source,
            digest,
            "budget",
            "l40s-native-dispatch",
            hardware=result["hardware"],
            model=result["gguf"],
            quantization="Q4_K_M",
            decision={"rows": result["rows"]},
            metrics=result["simulated"],
            provenance="simulated",
            reason="ranked prefill slices from timed mixed calls",
            uncertainty=result["scope"],
            authorship="IronMule fitted its CPU step-cost model and chose the experiment. Codex adapted the shared-pool scheduler.",
        )
        yield record(
            "native_chunk_step_cost",
            source,
            digest,
            "prediction-error",
            "l40s-native-dispatch",
            hardware=result["hardware"],
            model=result["gguf"],
            quantization="Q4_K_M",
            metrics=[result["model"]["holdout_mean_absolute_error_s"]],
            workload={
                "train_calls": result["model"]["train_calls"],
                "holdout_calls": result["model"]["holdout_calls"],
            },
            reason="predicted durations on an independent server restart and input seed",
            uncertainty="Real native calls from the stock deployment. End-to-end improvement requires a candidate run.",
        )
    elif path.name == "order.json":
        yield record(
            "native_prompt_development_order",
            source,
            digest,
            "target",
            result["order_id"],
            hardware=result["hardware"],
            base_commit=result["base_commit"],
            model=result["gguf"],
            quantization="Q4_K_M",
            workload={"request_ids_matched": result["matched_request_ids"]},
            provenance="simulated",
            metrics=result["options"],
            decision=result["target"],
            reason="ranked device-cost sensitivity",
            authorship="IronMule selected the measured target. Codex wrote the replacement function.",
            uncertainty="The assumed 10 percent device-time gain ranks experiments. It is not a measured saving.",
        )
    elif path.name == "fit.json":
        for i, row in enumerate(result["rows"]):
            yield record(
                "native_operator_pilot",
                source,
                digest,
                str(i),
                str(row["shape"][1:3]),
                hardware=result["target"]["hardware"],
                model=result["target"]["gguf"],
                quantization="Q4_K_M",
                software={
                    "ggml_cuda": result["target"]["ggml_cuda"],
                    "weights": result["weights_sha256"],
                },
                workload={"shape": row["shape"], "partition": row["set"]},
                metrics=[row["median_us"]],
                decision=row["trained_profile"],
                reason="measured operator profile replay",
                authorship="IronMule fitted its CPU profile head from synchronized native matmuls.",
                uncertainty="Operator times include GGML graph calls. Product qualification is separate.",
            )
    elif path.name == "result.json" and path.parent.name == "gate-fix":
        gate = result["gate"]
        for i, trial in enumerate(gate.get("tried", [])):
            yield record(
                "native_numeric_qualification",
                source,
                digest,
                str(i),
                "l40s-explicit-cublas-128",
                hardware=result["hardware"],
                model=result["gguf"],
                quantization="Q4_K_M",
                profiles=[trial["settings"]],
                metrics=[{"ppl_ratio_interval": trial.get("interval")}],
                quality=trial.get("factual"),
                decision="qualified" if trial["passes"] else "rejected",
                reason=gate.get("reason"),
                authorship="Codex corrected the identity check. IronMule evaluated the requested override.",
                uncertainty="One target and gate corpus. A successful factual check alone cannot qualify a CUDA profile.",
            )
    elif path.name == "result.json":
        family = path.parent.name
        for i, cell in enumerate(result["cells"]):
            yield record(
                "native_prompt_arithmetic",
                source,
                digest,
                str(i),
                f"l40s-arithmetic-{cell['n']}-{cell['k']}",
                hardware=result.get("hardware", result.get("board")),
                model=result.get("weights", result.get("gguf_sha256")),
                quantization=cell["type"],
                software={"family": family, "library": result.get("library_sha256")},
                workload={k: cell[k] for k in ("n", "k", "b")},
                metrics=cell["variants"] if "variants" in cell else [cell],
                reason="kernel declined this cell"
                if cell.get("rc")
                else "standalone operator screen",
                authorship="Codex implemented the experiment. The GPU ran the real GGUF tensor.",
                uncertainty="Warm arithmetic on sampled channels and generated activations. Full serving and quality require separate qualification.",
            )
