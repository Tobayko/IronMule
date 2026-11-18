"""Regular, agent-enabled HTTP qualification of the low-width Q4_K kernel after A/B/A native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

from ironmule.factual_gate import MAX_TOKENS, FactCase, cases, correct
from ironmule_product import compare, compare_servers, compare_tokenizer
from ironmule_product.state import _atomic_write
from lab.kg4 import scoped_native as native_gate
from lab.portable_gpu import mixed_traffic, production_compare
from lab.portable_gpu.prefill_http import Profile, read

ARMS = ("stock", "scoped", "stock-closing")
WIDTHS = (8, 32)
FACTS = (
    ("What is the capital city of Canada?", "Ottawa"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What is the capital city of Egypt?", "Cairo"),
    ("What is the chemical symbol for helium?", "He"),
    ("How many sides does a pentagon have? Use digits.", "5"),
)


def inputs(seed, rep, width):
    s = seed + rep * 1009 + width
    perf = compare.performance_prompts(s, width, width)
    q = []
    for case in cases(s):
        prompt, answer, forms = case.prompt, case.expected, case.accepted_forms
        if case.category == "facts":
            i = int(case.key.rsplit("-", 1)[-1])
            question, answer = FACTS[i]
            prompt, forms = question + " Reply with the answer only.", ()
        # Distinct rendering per wave even for the fixed public retention questions.
        prompt = f"Task sheet {s}. " + prompt
        q.append(FactCase(case.key, case.category, prompt, answer, forms))
    return perf, q


def scope_delta(before, after, candidate):
    if not candidate:
        if before is not None or after is not None:
            raise ValueError("stock_worker_has_scoped_kernel")
        return {}
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or before.get("mode") != "low-width8"
        or after.get("mode") != "low-width8"
        or before.get("q4_buckets") != after.get("q4_buckets")
        or not after["q4_buckets"]
        or not set(after["q4_buckets"]) <= {6, 7, 8}
    ):
        raise ValueError("unverified_worker_scope")
    result = {k: after[k] - before[k] for k in ("low_runs", "wide_runs", "switches")}
    if any(type(x) is not int or x < 0 for x in result.values()):
        raise ValueError("worker_scope_restarted")
    return {
        **result,
        "q4_buckets": after["q4_buckets"],
        "parked_before": before["parked"],
        "parked_after": after["parked"],
    }


def dispatch(hits, scope, *, candidate, width, performance):
    values = hits["kg4"]
    if (
        len(values) != 64
        or len(hits["cublas"]) != 1
        or any(type(x) is not int or x < 0 for x in values)
        or hits["cublas"] != [0]
    ):
        return False
    if not candidate:
        return not any(values)
    allowed = scope["q4_buckets"]
    if any(x and (i + 1) not in allowed for i, x in enumerate(values)):
        return False
    # Regular HTTP arrivals can start narrow even in the width-32 cell. Only the covered
    # B6..8 buckets and the pressure telemetry can explain such a hit, never an arbitrary B32.
    if any(values) and (scope["low_runs"] < 1 or scope["switches"] < 1):
        return False
    if not performance:
        return True
    if width == 8:
        return sum(values) > 0 and scope["wide_runs"] == 0
    return scope["wide_runs"] > 0


def requests(rows, cases_=None):
    result = []
    for i, row in enumerate(rows):
        if not math.isfinite(row["ttft_ms"]) or row["ttft_ms"] <= 0:
            raise ValueError("invalid_per_request_ttft")
        record = {
            k: row[k]
            for k in (
                "tokens",
                "prompt_tokens",
                "ttft_ms",
                "visible_ttft_ms",
                "finish_reason",
            )
        }
        record["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        if cases_ is not None:
            case = cases_[i]
            record.update(
                key=case.key, category=case.category, correct=correct(case, row["text"])
            )
        result.append(record)
    return result


def compatible(ggml, server, binding, arm, worker_pid):
    from ironmule.long_prefill_study import uuid_hash

    b = ggml.get("experiment_binding") or {}
    if (
        ggml.get("prefill_batch", {}).get("enabled") is not False
        or b.get("provider") != "cuda"
        or b.get("gguf_sha256") != binding["gguf_sha256"]
        or b.get("shared") is not False
        or b.get("settings", {}).get("kv_unified") is not False
        or b.get("gpu_layers") != 999
        or b.get("kg_sha256") != (binding["plugin_sha256"] if arm == "scoped" else None)
        or uuid_hash(server["device"]) != binding["gpu"]["uuid_sha256"]
        or uuid_hash(b.get("cuda_uuid") or "") != binding["gpu"]["uuid_sha256"]
    ):
        raise ValueError("invalid_serving_execution_binding")
    placement = ggml.get("kg4_placement") or {}
    if placement.get("offloaded_layers") != [65, 65] or placement.get(
        "worker_pid"
    ) != int(worker_pid):
        raise ValueError("unverified_serving_layer_placement")
    return {
        k: b[k]
        for k in (
            "gguf_sha256",
            "implementation",
            "study_native_runtime_sha256",
            "threads",
            "seqs",
            "seq_ctx",
            "settings",
            "shared",
            "cap",
            "gpu_layers",
            "provider",
            "speculate",
            "max_rows",
            "prefix_reuse",
            "prompt_budget",
            "has_drafter",
            "split",
            "host_sha256",
        )
    }


def aggregate(report):
    reasons = set()
    tables = {
        a: {(v["repeat"], v["width"]): v for v in report["rows"] if v["arm"] == a}
        for a in ARMS
    }
    cells = {(r, w) for r in range(report["repeats"]) for w in WIDTHS}
    if any(set(t) != cells for t in tables.values()) or len(report["rows"]) != 3 * len(
        cells
    ):
        return False, ["incomplete_http_cells"], []
    metrics = []
    for i, w in sorted(cells):
        a, b, c = (tables[arm][i, w] for arm in ARMS)
        if (
            not a["prompt_ids"] == b["prompt_ids"] == c["prompt_ids"]
            or not a["binding"] == b["binding"] == c["binding"]
        ):
            reasons.add("unmatched_serving_conditions")
        for row in (a, b, c):
            for kind in ("performance", "factual"):
                wave = row[kind]
                if not dispatch(
                    wave["hits"],
                    wave["scope"],
                    candidate=row["arm"] == "scoped",
                    width=w,
                    performance=kind == "performance",
                ):
                    reasons.add(f"dispatch_{kind}_w{w}")
        for kind in ("performance", "factual"):
            work = [
                [
                    (v["tokens"], v["finish_reason"], v["text_sha256"])
                    for v in arm[kind]["requests"]
                ]
                for arm in (a, b, c)
            ]
            if not work[0] == work[1] == work[2]:
                reasons.add(f"generations_{kind}_w{w}")
        aa, bb, cc = (
            [v["correct"] for v in arm["factual"]["requests"]] for arm in (a, b, c)
        )
        if not aa == bb == cc or any(
            not any(
                v["correct"] and v["category"] == cat for v in b["factual"]["requests"]
            )
            for cat in ("facts", "arithmetic", "extraction")
        ):
            reasons.add(f"facts_w{w}")
        perf = [arm["performance"]["metrics"] for arm in (a, b, c)]
        rate = perf[1]["tokens_per_second"] / max(
            p["tokens_per_second"] for p in (perf[0], perf[2])
        )
        ttft = perf[1]["ttft_p95_ms"] / min(
            p["ttft_p95_ms"] for p in (perf[0], perf[2])
        )
        if rate < (1.03 if w == 8 else 0.97) or ttft > native_gate.P95_LIMIT:
            reasons.add(f"throughput_or_latency_w{w}")
        metrics.append(
            {"repeat": i, "width": w, "throughput_ratio": rate, "p95_ttft_ratio": ttft}
        )
    return not reasons, sorted(reasons), metrics


def run(args, report, save, deadline):
    receipts = [
        json.loads(p.read_bytes())
        for p in (args.receipt_a, args.receipt_b, args.receipt_c)
    ]
    native = native_gate.qualify(*receipts)
    if native["eligible"] is not True:
        report.update(status="skipped", reasons=["native_receipt", *native["reasons"]])
        return
    # Import capsule-only fixed helpers after native qualification; no model load on a negative result.
    from http_kernel import worker
    from select_layout import HERE, gpu, sha

    from ironmule import ggml_engine, kg4_scope
    from native import delta

    binding = native["binding"]
    plugin = HERE / "r1.so"
    lib = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"
    if (
        os.environ.get("GGML_CUDA_DISABLE_GRAPHS") != "1"
        or gpu() != binding["gpu"]
        or sha(lib) != binding["native"]["backend/CUDA"]
        or sha(plugin) != binding["plugin_sha256"]
        or sha(Path(ggml_engine.__file__)) != binding["engine_sha256"]
        or sha(Path(kg4_scope.__file__)) != binding["scope_source_sha256"]
        or sha(Path(native_gate.__file__)) != binding["harness_sha256"]
    ):
        raise ValueError("local_native_identity")
    artifacts = compare_servers.prepare_artifacts(args)
    expected = {
        **compare_tokenizer._QWEN_FILES,
        "config.json": compare_tokenizer.TOKENIZERS["qwen3-32b"][2],
    }
    if (
        artifacts["tokenizer_files"] != expected
        or args.gguf_sha256 != binding["gguf_sha256"]
    ):
        raise ValueError("local_model_tokenizer_identity")
    seed = binding["seed"] + 200003
    native_prompts = set()
    for r in range(native_gate.REPEATS):
        for w in WIDTHS:
            native_prompts.update(
                compare.performance_prompts(binding["seed"] + r * 1009 + w, w, w)
            )
    native_prompts.update(case.prompt for case in cases(receipts[0]["facts_seed"]))
    distinct = set()
    groups = {}
    for r in range(args.repeats):
        for w in WIDTHS:
            perf, qs = groups[r, w] = inputs(seed, r, w)
            prompts = perf + [q.prompt for q in qs]
            if len(set(prompts)) != len(prompts) or set(prompts) & (
                distinct | native_prompts
            ):
                raise ValueError("overlapping_holdouts")
            distinct.update(prompts)
    (
        args.provider,
        args.context,
        args.concurrency,
        args.seed,
        args.requests,
        args.max_tokens,
    ) = "cuda", 2048, list(WIDTHS), seed, 8, 32
    report.update(
        native_binding=binding,
        tokenizer_sha256=args.tokenizer_sha256,
        heldout_seed=seed,
        distinct_requests=len(distinct),
        mixed={},
        status="running",
    )
    save()
    arm_seconds = []
    for arm_index, arm in enumerate(ARMS):
        started = time.monotonic()
        directory = args.state_dir / arm
        directory.mkdir(mode=0o700, parents=True)
        args.agent_dir = directory / "agent"
        args.log_dir = directory / "logs"
        args._compare_workdir = None
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "IRONMULE_KG_LIB",
                "IRONMULE_KG4_SCOPE",
                "IRONMULE_KG5",
                "IRONMULE_GGML_PREFILL_RECEIPT",
                "IRONMULE_GGML_SEQS",
                "IRONMULE_PILOT",
            }
        }
        env.update(
            IRONMULE_KG="0",
            IRONMULE_GGML_GPU_LAYERS="999",
            IRONMULE_GGML_PREFILL_BATCH="0",
            IRONMULE_PREFILL_STUDY="1",
            IRONMULE_GGML_STREAMS="1",
            IRONMULE_GGML_THREADS="2",
            IRONMULE_KG4_PLACEMENT="1",
        )
        for source, label in (
            ("IRONMULE_KG4_COUNTER_DIR", "kg4"),
            ("IRONMULE_CUBLAS_COUNTER_DIR", "cublas"),
        ):
            (directory / label).mkdir()
            env[source] = str((directory / label).resolve())
        if arm == "scoped":
            env.update(
                IRONMULE_KG="1",
                IRONMULE_KG_LIB=str(plugin),
                IRONMULE_KG4_SCOPE="low-width8",
            )
        with (
            patch.dict(os.environ, env, clear=True),
            compare_servers.open_server("ironmule", args, 0, deadline) as server,
        ):
            production_compare.check_server("ironmule_gguf", server, args, None)
            compare.stream("ironmule", server, compare.WARMUP, 8, deadline)
            compare_servers.refresh_identity("ironmule", server, deadline)
            if server.get("autonomous") is not True:
                raise ValueError("agent_inactive")
            worker_id, _ = worker()
            mapped = Path("/proc", worker_id[0], "maps").read_text().splitlines()
            paths = {
                Path(v.split(maxsplit=5)[5])
                for v in mapped
                if len(v.split(maxsplit=5)) == 6
                and v.split(maxsplit=5)[5].startswith("/")
            }
            mapped_native = {
                sha(p)
                for p in paths
                if p.is_file()
                and (p.name.startswith(("libllama", "libggml")) or p == plugin)
            }
            if not set(binding["native"].values()) <= mapped_native or (
                binding["plugin_sha256"] in mapped_native
            ) != (arm == "scoped"):
                raise ValueError("actual_worker_dsos")
            profile = Profile("stock")
            for r in range(args.repeats):
                for w in WIDTHS if r % 2 == 0 else WIDTHS[::-1]:
                    prompts, qs = groups[r, w]
                    row = {
                        "repeat": r,
                        "width": w,
                        "arm": arm,
                        "offloaded_layers": [65, 65],
                    }
                    proof = production_compare.proof(
                        "ironmule",
                        server,
                        args,
                        prompts + [c.prompt for c in qs],
                        deadline,
                    )
                    row["prompt_ids"] = proof["prompt_ids_sha256"]
                    seen = []
                    for label, items, limit in (
                        ("performance", prompts, 32),
                        ("factual", [c.prompt for c in qs], MAX_TOKENS),
                    ):
                        ggml = read(server, "/health", deadline)["ggml"]
                        row_binding = compatible(
                            ggml, server, binding, arm, worker_id[0]
                        )
                        scope_before = ggml.get("kg4_scope")
                        snap = profile.before(server, deadline)
                        who, before = worker()
                        rows, elapsed = compare.batch(
                            "ironmule",
                            server,
                            items,
                            w,
                            limit,
                            deadline,
                            request_id_prefix=f"scoped-{r}-{w}-{label}",
                        )
                        who_after, after = worker()
                        trace = profile.after(server, deadline, snap, rows)
                        ggml = read(server, "/health", deadline)["ggml"]
                        if (
                            who != worker_id
                            or who_after != who
                            or compatible(ggml, server, binding, arm, worker_id[0])
                            != row_binding
                        ):
                            raise ValueError("serving_worker_changed")
                        if "binding" in row and row["binding"] != row_binding:
                            raise ValueError("different_quality_context")
                        row["binding"] = row_binding
                        seen += [s["prompt_tokens"] for s in rows]
                        row[label] = {
                            "hits": delta(before, after),
                            "scope": scope_delta(
                                scope_before, ggml.get("kg4_scope"), arm == "scoped"
                            ),
                            "requests": requests(
                                rows, qs if label == "factual" else None
                            ),
                            "metrics": compare.performance_metrics(rows, elapsed),
                            "timing": trace["intervals"],
                        }
                    if proof["prompt_tokens"] != seen:
                        raise ValueError("served_token_count")
                    report["rows"].append(row)
                    save()
            arm_seconds.append(time.monotonic() - started)
            if arm in ("stock", "scoped"):
                mixed = {"status": "skipped", "reason": "budget"}
                reserve = (len(ARMS) - arm_index - 1) * max(arm_seconds)
                if deadline - time.monotonic() - reserve > 46:
                    try:
                        first_id, before = worker()
                        prior = read(server, "/health", deadline)["ggml"]
                        mixed = {
                            "status": "measured",
                            "result": mixed_traffic.measure(
                                "ironmule",
                                server,
                                seed + 400009,
                                min(deadline, time.monotonic() + 45),
                                gguf=args.gguf,
                            ),
                        }
                        after_id, after = worker()
                        current = read(server, "/health", deadline)["ggml"]
                        if (
                            first_id != worker_id
                            or after_id != worker_id
                            or compatible(prior, server, binding, arm, first_id[0])
                            != compatible(current, server, binding, arm, after_id[0])
                        ):
                            raise ValueError("mixed_worker_changed")
                        mixed["hits"] = delta(before, after)
                        mixed["scope"] = scope_delta(
                            prior.get("kg4_scope"),
                            current.get("kg4_scope"),
                            arm == "scoped",
                        )
                        for request in mixed["result"]["requests"]:
                            request["text_sha256"] = hashlib.sha256(
                                request.pop("text").encode()
                            ).hexdigest()
                    except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
                        mixed = {"status": "incomplete", "reason": type(exc).__name__}
                report["mixed"][arm] = mixed
                save()
    report["eligible"], report["reasons"], report["paired"] = aggregate(report)
    report["mixed_paired"] = all(
        report["mixed"].get(k, {}).get("status") == "measured"
        for k in ("stock", "scoped")
    )
    if report["mixed_paired"]:
        try:
            paired = mixed_traffic.pair(
                report["mixed"]["scoped"]["result"], report["mixed"]["stock"]["result"]
            )
            if "ratio_order" in paired:
                paired["ratio_order"] = ["scoped", "stock"]
            report["mixed_comparison"] = paired
        except compare.Unverified:
            report["mixed_paired"] = False
            report["mixed_comparison"] = {"status": "unpaired_inputs"}
    report["status"] = "measured"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "gguf",
        "tokenizer",
        "receipt-a",
        "receipt-b",
        "receipt-c",
        "state-dir",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--budget-seconds", type=int, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--startup-timeout", type=int, default=1200)
    args = p.parse_args(argv)
    if (
        not 1 <= args.budget_seconds <= 3600
        or not 1 <= args.repeats <= 5
        or args.startup_timeout < 1
    ):
        p.error("invalid duration or repeats")
    deadline = time.monotonic() + args.budget_seconds
    args.output.parent.mkdir(exist_ok=True, parents=True)
    os.close(os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    report = {
        "schema": "ironmule.kg4-scoped-http.v1",
        "status": "starting",
        "eligible": False,
        "repeats": args.repeats,
        "reasons": [],
        "rows": [],
    }

    def save():
        _atomic_write(args.output, report)

    save()
    try:
        run(args, report, save, deadline)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - preserve the numeric receipt on any failure
        report.update(status="incomplete", eligible=False, error=type(exc).__name__)
        if (
            isinstance(exc, ValueError)
            and str(exc).isascii()
            and str(exc).replace("_", "").isalnum()
        ):
            report["reasons"].append(str(exc))
        return 1
    finally:
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
