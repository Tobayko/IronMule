"""Index the mixed-prefill forecast separately from its measured device outcome."""

import json

from lab.pk6.mixed_progress import grade, plan
from training.experience_archive import record


def experiences(root, source, digest):
    path = root / source
    saved = json.loads(path.read_bytes())
    if path.name == "plan.json":
        if saved != plan.plan():
            raise ValueError("archived row selection disagrees with the CPU replay")
        for row in saved["simulated"]:
            yield record(
                "mixed_prefill_forecast",
                source,
                digest,
                f"rows/{row['rows']}",
                saved["source"],
                hardware={"gpu": "NVIDIA A100-SXM4-80GB"},
                provenance="simulated",
                profiles=[{"row_budget": row["rows"]}],
                metrics=[row["forecast"]],
                workload={
                    "width": 32,
                    "output_tokens": 32,
                    "prompt_lengths": [33, 444],
                },
                decision=row["rows"] == saved["selected_rows"],
                reason="ranked_for_hardware_test",
                authorship="IronMule StepCost ranked row budgets for the scheduler written by Codex.",
                uncertainty=saved["uncertainty"],
                trace_sha256=saved["source_trace_sha256"],
            )
        return
    if path.name == "failed-start.json":
        device = json.loads((path.parent / "target.json").read_bytes())
        attempts = saved["gate_attempts"]
        libraries = saved["two_compiled_libraries"]
        if (
            saved["ready"]
            or not any(
                a["kernel_mode"] == "generated"
                and a["diagnostic"]["code"] == "out_of_memory"
                for a in attempts
            )
            or libraries["raw_engine_w4"]["mtime_unix"]
            <= libraries["calibration_w4"]["mtime_unix"]
        ):
            raise ValueError("unbound kernel fallback failure")
        yield record(
            "packed_kernel_retry_failure",
            source,
            digest,
            "original-R1",
            "l40s-mixed-prefill-progress",
            hardware=device["hardware"],
            model=device["gguf_sha256"],
            quantization="Q4_K_M",
            base_commit=device["source"],
            profiles=["locally tuned W4", "stock retry"],
            error=attempts,
            reason="out_of_memory_then_raw_w4_build_in_stock_retry",
            workload={"phase": "before HTTP readiness"},
            quality=None,
            libraries=libraries,
            cost_reference=str((path.parent / "cost.json").relative_to(root)),
            authorship="Codex diagnosed the two-library retry. IronMule refused the generated kernel.",
            uncertainty="Operator stopped the mislabeled retry before readiness. Its time to completion is unknown.",
        )
        return
    observed = grade.grade(path.parent)
    if saved != observed:
        raise ValueError("archived decision disagrees with hardware replay")
    target = observed["target"]
    software = json.loads((path.parent / "software.json").read_bytes())
    for key, counts in observed["comparisons"].items():
        candidate, reference, width = key.split("/")
        seed = str(grade.SEEDS[int(candidate[1]) - 1])
        external = reference == "vllm"
        # The paired arms, seeds and width cells share a single board and episode.
        yield record(
            "mixed_prefill_http_comparison",
            source,
            digest,
            key,
            "l40s-mixed-prefill-progress",
            hardware=target["hardware"],
            software=software,
            base_commit=target["source"],
            model={
                "native": target["gguf_sha256"],
                "reference": target["awq_manifest_sha256"]
                if external
                else target["gguf_sha256"],
            },
            quantization={
                "candidate": "Q4_K_M",
                "reference": "AWQ-Marlin" if external else "Q4_K_M",
            },
            workload={
                "width": int(width),
                "max_tokens": 32,
                "seed": int(seed),
                "repeats": 3,
            },
            profiles=[candidate, reference],
            metrics=[
                counts,
                {
                    "candidate": observed["median_by_arm_width"][candidate][width],
                    "reference": grade.summarize(
                        [
                            grade.measure(w)
                            for w in grade.lines(
                                path.parent / f"waves-{reference}.jsonl"
                            )
                            if w["seed"] == int(seed)
                            and w["width"] == int(width)
                            and w["repeat"] < 3
                        ]
                    ),
                },
            ],
            quality={
                "separate_width32_graded_tasks": {
                    arm: observed["quality"][arm][seed]
                    for arm in (candidate, reference)
                }
            },
            memory_mib={
                arm: observed["resources"][arm]["peak_gpu_mib"]
                for arm in (candidate, reference)
            },
            cost_reference=str((path.parent / "cost.json").relative_to(root)),
            reason="different_native_profiles"
            if counts.get("same_native_profile") is False
            else observed["decision"],
            authorship="Codex implemented mixed scheduling. IronMule ranked its experimental row limit.",
            uncertainty="One device and synthetic chat traffic. Mixed-call evidence is bound to measured HTTP waves. "
            "Cross-engine comparisons use different quantizations.",
        )
