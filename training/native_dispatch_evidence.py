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
    elif path.name == "capacity-control.json":
        if result != run(folder / "controls/capacity"):
            raise ValueError(
                "capacity comparison differs from its raw streaming receipts"
            )
        for seed in (1, 2):
            h, r = f"H{seed}", f"R{seed}"
            if h not in result["arms"] or r not in result["arms"]:
                continue
            candidate, reference = (result["arms"][key] for key in (h, r))
            for width in (1, 8, 32):
                pair = result["pairs"][f"{h}/{r}/{width}"]
                if (
                    not pair["same_fixed_work"]
                    or not candidate["proof"]["verified_compact_pool"]
                ):
                    raise ValueError(
                        "unmatched capacity workload or missing physical pool"
                    )
                yield record(
                    "native_long_document_capacity",
                    source,
                    digest,
                    f"{h}/{r}/{width}",
                    "l40s-capacity-long-documents",
                    hardware=result["binding"][0],
                    base_commit=result["binding"][1],
                    model=result["binding"][2],
                    quantization="Q4_K_M",
                    workload={
                        "concurrency": width,
                        "seed": result["binding"][-1][seed - 1],
                        "document_words": 900,
                        "output_limit": 32,
                    },
                    decision={"reference_cells": 65536, "tested_cells": 16384},
                    metrics=[
                        {
                            "candidate": candidate["metrics"][str(width)],
                            "reference": reference["metrics"][str(width)],
                            "paired_ratios": pair["candidate_over_reference"],
                            "reference_peak_serving_mib": reference["resources"][
                                "peak_gpu_mib"
                            ],
                        }
                    ],
                    quality={
                        "candidate": candidate["quality"],
                        "reference": reference["quality"],
                    },
                    memory_mib=candidate["resources"]["peak_gpu_mib"],
                    reason="matched long-document pressure and repeated-prefix holdout",
                    authorship="IronMule sized the pool from measured request demand. Codex integrated and measured the physical allocation.",
                    uncertainty="Three waves per width on this physical GPU. Factual checks used their separate fixed corpus; there is no vLLM result for this document workload.",
                )
    elif path.name == "document-retrieval.json":
        from lab.pk6.native_dispatch.document_quality import SEEDS, assess

        if result != assess(folder / "controls/document-retrieval"):
            raise ValueError("document retrieval differs from its original requests")
        for arm, trial in result.items():
            yield record(
                "native_document_retrieval",
                source,
                digest,
                arm,
                "l40s-capacity-long-documents",
                hardware=trial["hardware"],
                base_commit=trial["source"],
                model=trial["model"],
                quantization="Q4_K_M",
                workload={
                    "document_seeds": list(SEEDS),
                    "document_words": 900,
                    "concurrency": 32,
                },
                profiles=[{"arm": arm, "cells": 16384 if arm == "H2" else 65536}],
                quality={
                    "correct_by_wave": trial["correct_by_wave"],
                    "requests_per_wave": 32,
                },
                reason="two exact-retrieval waves followed by the short factual gate",
                authorship="Codex designed this holdout after the measured compact-pool speedup.",
                uncertainty="One cold process per profile, two document seeds, retrieval from synthetic inventory logs.",
            )
    elif path.name == "draft-load-plan.json":
        from lab.pk6.native_dispatch.draft_load import propose

        if result != propose(folder / "http", folder / "draft-metadata.json"):
            raise ValueError(
                "draft-load proposal differs from measured target verifications"
            )
        yield record(
            "native_draft_load_proposal",
            source,
            digest,
            "four-slots",
            "l40s-draft-load",
            hardware=result["hardware"],
            model=result["gguf"],
            quantization="Q4_K_M",
            workload={"draft": result["draft"]},
            decision={"slots": result["slots"], "max_running": result["max_running"]},
            provenance="simulated",
            reason="bounded follow-up from observed draft coverage and footprint",
            authorship="Codex designed the catch-up scheduling experiment for IronMule to evaluate.",
            uncertainty=result["scope"],
        )
    elif path.name == "worker-plan.json":
        from lab.pk6.native_dispatch.w4_profile import select

        screen = folder / "followup"
        if result != select(screen):
            raise ValueError(
                "bounded W4 proposal differs from its four measured workers"
            )
        for name, sha in result["sources"].items():
            worker = json.loads((screen / name).read_bytes())
            for width in (1, 8):
                samples = [wave for wave in worker["waves"] if wave["width"] == width]
                yield record(
                    "native_bounded_w4_worker",
                    source,
                    digest,
                    f"{name}/{width}",
                    "l40s-bounded-w4-fixed-work",
                    hardware=result["hardware"],
                    model=result["gguf"],
                    quantization="Q4_K_M",
                    software={
                        "library": result["library_sha256"]
                        if name.startswith("worker-M")
                        else "stock",
                        "source_receipt": sha,
                    },
                    workload={
                        "width": width,
                        "seed": worker["seed"],
                        "output_limit": 32,
                    },
                    metrics=samples,
                    quality=worker["quality"],
                    startup_s=worker["startup_s"],
                    repacked_bytes=worker["native_repacked_bytes"],
                    authorship="IronMule selected complete tensor geometries from the measured memory budget. Codex integrated the product experiment.",
                    reason="measured fixed-output native worker",
                    uncertainty="Repeated equal-length prompts on one L40S. Server queues, streaming, controller overhead and the product quality gate require a separate run.",
                )
    elif path.name.startswith("qualification-"):
        target = json.loads((folder / "target.json").read_bytes())
        arm = path.stem.removeprefix("qualification-")
        if arm not in target["order"] or len(result["log_sha256"]) != 64:
            raise ValueError("qualification receipt missing its process binding")
        for i, gate in enumerate(result["gates"]):
            if gate["passes"]:
                continue
            diagnostic = gate.get("diagnostic") or {}
            yield record(
                "native_rejected_startup",
                source,
                digest,
                f"{arm}/{i}",
                f"l40s-qualification-{arm[:-1]}",
                hardware=target["hardware"],
                base_commit=target["source"],
                model=target["gguf_sha256"],
                quantization="Q4_K_M",
                software={
                    "engine_sha256": target["engine_sha256"],
                    "kernel_mode": gate["kernel_mode"],
                },
                decision="rejected",
                reason=diagnostic.get("code") or "quality gate failed",
                workload={"process": arm, "stage": diagnostic.get("stage")},
                metrics=diagnostic.get("candidates", []),
                quality=gate.get("interval"),
                authorship="IronMule qualified the requested profile and recorded the outcome.",
                uncertainty="A later successful fallback describes the profile that served. Its throughput cannot be attributed to the rejected profile.",
            )
    elif path.name.startswith("shape-"):
        protocol = json.loads((folder / "protocol.json").read_bytes())
        if result["seed"] not in protocol["seeds"] or set(
            map(int, result["cells"])
        ) != set(protocol["widths"]):
            raise ValueError("native grid cells differ from the device protocol")
        for width, choices in result["cells"].items():
            if set(choices) != set(protocol["modes"]):
                raise ValueError("missing native grid comparison")
            yield record(
                "native_cuda_grid",
                source,
                digest,
                width,
                f"l40s-native-grid-{result['n']}-{result['k']}",
                hardware=protocol["hardware"],
                model=protocol["gguf"],
                quantization="Q4_K",
                software={
                    "backend": protocol["backend"],
                    "reference_cuda": protocol["reference_cuda"],
                },
                workload={
                    "tensor": result["tensor"],
                    "n": result["n"],
                    "k": result["k"],
                    "rows": int(width),
                    "activation_seed": result["seed"],
                    "weight_copies": result["weight_copies"],
                },
                metrics=[
                    {"mode": mode, **reading} for mode, reading in choices.items()
                ],
                reason="measured complete graph for each native CUDA variant",
                authorship="Codex wrote the dispatch variants. IronMule used timed HTTP calls to select the follow-up widths.",
                uncertainty="Two seeds for one geometry stay in the same training group. End-to-end throughput requires an independent serving measurement.",
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
    elif path.name == "fit.json" and folder.name == "pool-pilot":
        from lab.pk6.replay_http import checksum

        if checksum(folder / "pilot.npz") != result["weights_sha256"] or any(
            checksum(folder.parent / name) != bound
            for name, bound in result["files"].items()
        ):
            raise ValueError("changed pool controller or training measurements")
        for row in result["scores"]:
            workload = row["workload"]
            seed = row["server_seed_index"]
            width = row["width"]
            yield record(
                "native_pool_pilot_replay",
                source,
                digest,
                f"{workload}/{seed}/{width}",
                f"l40s-pool-{workload}-process-seed-{seed}",
                hardware=result["hardware"],
                base_commit=result["source"],
                model=result["model"],
                quantization="Q4_K_M",
                software={
                    "controller": result["weights_sha256"],
                    "ggml_cuda": result["native"],
                },
                workload={
                    "inputs": workload,
                    "split": row["split"],
                    "concurrency": width,
                },
                decision=row["trained"],
                metrics=[row],
                reason="grouped CPU replay of measured complete server profiles",
                authorship="IronMule fitted its Pilot on measured pressure and mixed workloads.",
                uncertainty="One GPU, two process seeds. The simple length rule is included for comparison. Runtime release is scored separately.",
            )
    elif path.name == "pool-restart.json":
        from lab.pk6.replay_http import checksum

        archive = folder / "controls/pool-restart"
        if result != run(archive):
            raise ValueError("pool restart differs from measured requests")
        for receipt in sorted(archive.rglob("decision.json")):
            if not (receipt.parent / "http").is_dir():
                continue
            choice = json.loads(receipt.read_bytes())
            arm = choice["arm"]
            if not (receipt.parent / "http" / f"end-{arm}.json").exists():
                continue
            actual = result["arms"][arm]
            if (
                choice["policy_sha256"] != checksum(folder / "pool-pilot/fit.json")
                or choice["hardware"] != result["binding"][0]
            ):
                raise ValueError("restart used a different controller or GPU")
            yield record(
                "native_pool_pilot_serving",
                source,
                digest,
                f"{receipt.parent.name}/{arm}",
                "l40s-capacity-long-documents",
                hardware=choice["hardware"],
                base_commit=result["binding"][1],
                model=choice["gguf"],
                quantization="Q4_K_M",
                workload={
                    "concurrency": choice["decision"]["width"],
                    "sample_seed": choice["seed"],
                },
                decision=choice["decision"]["profile"],
                profiles=[actual["proof"]],
                metrics=[
                    actual["metrics"],
                    {
                        "prelaunch_s": choice["prelaunch_s"],
                        "startup_s": actual["startup_s"],
                    },
                ],
                quality=actual["quality"],
                memory_mib=actual["resources"]["peak_gpu_mib"],
                authorship="IronMule selected its pool before serving. Codex implemented the comparator and withdrawal path.",
                reason=choice["decision"]["reason"],
                uncertainty="Same physical L40S and archived inputs, with a fresh serving worker.",
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
