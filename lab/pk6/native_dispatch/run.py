"""Run complete stock, candidate and comparator processes against fixed rendered requests."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from ironmule.ggml_profiles import FLAGS
from lab.pk6.mixed_progress import trial
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save
from lab.portable_gpu.fetch_awq import verify

SEEDS = (180341, 943819)
DRAFT_SHA256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"


def rejected_first_seed(folder, arm):
    if arm not in {
        prefix + "2" for prefix in ("F", "I", "M", "J", "Y", "W", "S", "Z", "T", "X")
    }:
        return None
    first = arm[:-1] + "1"
    ready = folder / f"ready-{first}.json"
    if not (folder / f"end-{first}.json").is_file() or not ready.is_file():
        return None
    fallback = json.loads(ready.read_bytes())["health"].get("kernel_fallback") or {}
    rejected = any(
        attempt.get("passes") is False
        and (attempt.get("diagnostic") or {}).get("code") == "quality_rejected"
        for attempt in fallback.get("attempts", ())
    )
    if fallback.get("serving") == "stock" and rejected:
        return {
            "first": first,
            "ready_sha256": checksum(ready),
            "reason": "same program rejected by the fixed model quality gate before serving",
        }
    return None


def main(*, comparator="awq_marlin"):
    parser = argparse.ArgumentParser(description=__doc__)
    for field, default in (
        ("models", "/models"),
        ("state", "/work/state"),
        ("vllm", "/opt/vllm034"),
    ):
        parser.add_argument("--" + field, type=Path, default=Path(default))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--routing-receipt", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--pool-plan", type=Path)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs=2, default=SEEDS)
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument(
        "--vllm-recheck-reason",
        help="record what decision needs a fresh comparator process",
    )
    args = parser.parse_args()
    from lab.pk6.native_dispatch.reference import ARCHIVE, installation, lookup

    archive = args.reference_archive or ARCHIVE
    reason = (args.vllm_recheck_reason or "").strip()
    if (
        "vllm" in args.arms
        and not (archive / "vllm-reference.json").is_file()
        and not reason
    ):
        parser.error(
            "archived vLLM evidence missing: pass --reference-archive from the study checkout or record a --vllm-recheck-reason"
        )
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    hw = trial.board()
    gguf = args.models / trial.GGUF_NAME
    if checksum(gguf) != GGUF:
        raise ValueError("unexpected GGUF")
    draft = None
    if any(a.startswith(("K", "O", "C")) for a in args.arms):
        draft = args.models / "Qwen3-0.6B-Q8_0.gguf"
        if checksum(draft) != DRAFT_SHA256:
            raise ValueError("foreign draft model")
    draft_load = None
    if any(a.startswith("C") for a in args.arms):
        if not args.policy:
            raise ValueError("missing measured draft-load proposal")
        draft_load = json.loads(args.policy.read_bytes())
        if (
            not draft_load["selected"]
            or draft_load["hardware"] != hw
            or draft_load["gguf"] != GGUF
            or draft_load["draft"] != DRAFT_SHA256
            or (draft_load["slots"], draft_load["max_running"]) != (4, 8)
        ):
            raise ValueError("foreign drafting profile")
    policy_decision = None
    if any(a.startswith("Z") for a in args.arms):
        if not args.routing_receipt or not args.lib:
            raise ValueError("routed arm requires its target-specific build receipt")
        receipt = json.loads(args.routing_receipt.read_bytes())
        cuda = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"
        if (
            receipt["kind"] != "routed"
            or receipt["hardware"] != hw
            or receipt["gguf"] != GGUF
            or receipt["ggml_cuda"] != checksum(cuda)
            or receipt["library_sha256"] != checksum(args.lib)
        ):
            raise ValueError("foreign routed kernel")
        from lab.pk6.native_dispatch.live_policy import choose

        if not args.policy:
            raise ValueError("the CPU Pilot needs the measured policy bundle")
        policy_decision = choose(args.policy, hw, checksum(cuda), receipt)
        if policy_decision["profile"] != "routed":
            raise ValueError(
                "Pilot retained the reference; no routed qualification requested"
            )
    if any(a.startswith("L") for a in args.arms):
        from lab.pk6.native_dispatch.live_policy import choose

        if not args.policy:
            raise ValueError("the CPU Pilot needs the measured policy bundle")
        policy_decision = choose(
            args.policy,
            hw,
            checksum(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"),
        )
        if policy_decision["profile"] != "stock":
            raise ValueError("use the qualified routed build for this policy")
    pool = None
    if any(a.startswith(("H", "W")) for a in args.arms):
        pool = json.loads(args.pool_plan.read_bytes())
        if (
            pool["hardware"] != hw
            or pool["gguf"] != GGUF
            or pool["context"] != 2048
            or not 4096 <= pool["cells"] <= 65536
            or pool["cells"] % 256
        ):
            raise ValueError("unknown workload or device for compact KV candidate")
    packed_receipt = None
    if any(a.startswith("W") for a in args.arms):
        if not args.routing_receipt or not args.lib:
            raise ValueError("missing measured packed program")
        packed_receipt = json.loads(args.routing_receipt.read_bytes())
        if (
            packed_receipt["kind"] != "packed-marlin"
            or packed_receipt["hardware"] != hw
            or packed_receipt["gguf"] != GGUF
            or packed_receipt["pool_cells"] != pool["cells"]
            or packed_receipt["library_sha256"] != checksum(args.lib)
            or packed_receipt["ggml_cuda"]
            != checksum(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so")
        ):
            raise ValueError("foreign packed kernel or KV capacity")
    bounded_w4 = None
    if any(a.startswith("S") for a in args.arms):
        if not args.routing_receipt or not args.lib:
            raise ValueError("missing bounded W4 build")
        bounded_w4 = json.loads(args.routing_receipt.read_bytes())
        folder = args.lib.resolve().parent
        if (
            not bounded_w4["selected"]
            or bounded_w4["hardware"] != hw
            or bounded_w4["gguf"] != GGUF
            or checksum(args.lib) != bounded_w4["counter_library"]
            or checksum(folder / "w4-original.so") != bounded_w4["library_sha256"]
            or checksum(folder / "selected.table") != bounded_w4["table_sha256"]
        ):
            raise ValueError("unverified bounded W4 program or table")
    chunk_plan = None
    if any(a.startswith("Q") for a in args.arms):
        if args.policy is None:
            raise ValueError("missing measured device forecast")
        chunk_plan = json.loads(args.policy.read_bytes())
        if (
            chunk_plan["hardware"] != hw
            or chunk_plan["gguf"] != GGUF
            or chunk_plan["rows"] not in (256, 384, 512, 768, 1024, 1536)
        ):
            raise ValueError("foreign chunked prefill forecast")
    trial.SEEDS = tuple(args.seeds)
    reference = (
        lookup(archive, artifact=comparator) if "vllm" not in args.arms else None
    )
    comparator_install = None
    if reference:
        version = reference["version"]
        awq_digest = reference["quantized_model"]
    else:
        awq_digest = verify(args.models / "qwen3-32b-awq")["manifest_sha256"]
        version = subprocess.check_output(
            [
                str(args.vllm / "bin/python"),
                "-c",
                "import importlib.metadata as m; print(m.version('vllm'))",
            ],
            text=True,
        ).strip()
        if version != "0.34.0":
            raise ValueError("unexpected comparator")
        if "vllm" in args.arms:
            comparator_install = installation(
                args.vllm / "bin/python", args.models / "tokenizers/qwen3-32b"
            )
    save(
        args.out / "target.json",
        {
            "hardware": hw,
            "gguf_sha256": GGUF,
            "awq_manifest_sha256": awq_digest,
            **(
                {"external_reference_waves_sha256": reference["waves_sha256"]}
                if reference
                else {}
            ),
            "seeds": trial.SEEDS,
            "order": args.arms,
            "vllm_version": version,
            **({"vllm_installation": comparator_install} if comparator_install else {}),
            "source": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "engine_sha256": checksum(Path("ironmule/ggml_engine.py")),
            "library_sha256": checksum(args.lib) if args.lib else None,
            "threshold": args.threshold,
            "cpu_policy": policy_decision,
            "draft_gguf_sha256": DRAFT_SHA256 if draft else None,
            "pool_plan": pool,
            **({"draft_load_plan": draft_load} if draft_load else {}),
            "packed_program": packed_receipt,
            "shared_chunk_plan": chunk_plan,
            **({"bounded_w4": bounded_w4} if bounded_w4 else {}),
        },
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 45
    reused = []
    skipped = {}
    for arm in args.arms:
        rejection = rejected_first_seed(args.out, arm)
        if rejection:
            skipped[arm] = rejection
            continue
        if arm == "vllm":
            target = json.loads((args.out / "target.json").read_bytes())
            known = lookup(archive, target, artifact=comparator)
            if known and not reason:
                save(args.out / "reused-vllm.json", known)
                reused.append(arm)
                continue
            if reason:
                save(
                    args.out / "vllm-recheck.json",
                    {"reason": reason, "previous": known},
                )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "qualification window exhausted before starting another server"
            )
        for flag in (
            *FLAGS,
            "IRONMULE_LAB_ACTIVITY",
            "IRONMULE_LAB_DRAFT_MIX",
            "IRONMULE_GGML_DRAFT_SLOTS",
            "IRONMULE_GGML_DRAFT_MAX_RUNNING",
            "IRONMULE_LAB_MMQ_TILE",
            "IRONMULE_LAB_POOL_CELLS",
            "IRONMULE_LAB_SHARED_CHUNK",
            "IRONMULE_CUBLAS_MIN_ROWS",
            "IRONMULE_LAB_FP8",
            "IRONMULE_LAB_SHARED_PROFILE",
            "IRONMULE_LAB_STREAM_ALTERNATIVE",
            "IRONMULE_LAB_KERNEL_TRACE",
        ):
            os.environ.pop(flag, None)
        os.environ.update(
            IRONMULE_KG="0",
            IRONMULE_KG_LIB="",
            IRONMULE_GGML_PACK="1",
            IRONMULE_GGML_UBATCH="8192",
            IRONMULE_CUDA_GRAPHS_ROWS="32",
        )
        if arm.startswith(("K", "O", "C")):
            os.environ["IRONMULE_LAB_DRAFT_MIX"] = (
                "context_first" if arm.startswith("O") else "model_first"
            )
        if arm.startswith("C"):
            os.environ.update(
                IRONMULE_LAB_DRAFT_MIX="limited_catchup",
                IRONMULE_GGML_DRAFT_SLOTS=str(draft_load["slots"]),
                IRONMULE_GGML_DRAFT_MAX_RUNNING=str(draft_load["max_running"]),
            )
        if arm.startswith("E"):
            # Exercise normal automatic generation on this board, sharing its real persistent calibration state.
            for key in (
                "IRONMULE_KG",
                "IRONMULE_KG_LIB",
                "IRONMULE_GGML_PACK",
                "IRONMULE_GGML_UBATCH",
                "IRONMULE_CUDA_GRAPHS_ROWS",
                "IRONMULE_W4",
                "IRONMULE_W4_TABLE",
            ):
                os.environ.pop(key, None)
        if arm.startswith("Q"):
            os.environ["IRONMULE_LAB_SHARED_CHUNK"] = str(chunk_plan["rows"])
        if arm.startswith(("H", "W")):
            os.environ["IRONMULE_LAB_POOL_CELLS"] = str(pool["cells"])
        if arm.startswith(("N", "G")):
            os.environ["IRONMULE_LAB_STREAM_ALTERNATIVE"] = "plain"
            if arm.startswith("G"):
                os.environ.update(IRONMULE_PS1="1", IRONMULE_PK6="1")
        if arm.startswith("X"):
            os.environ.update(
                IRONMULE_LAB_MMQ_TILE="grid", IRONMULE_LAB_KERNEL_TRACE="1"
            )
        if arm.startswith("T"):
            os.environ.update(IRONMULE_LAB_MMQ_TILE="96", IRONMULE_LAB_KERNEL_TRACE="1")
        if arm.startswith(("U", "V")):
            os.environ["IRONMULE_LAB_SHARED_PROFILE"] = "wide"
        if arm.startswith("D"):
            os.environ["IRONMULE_LAB_ACTIVITY"] = str(args.out)
        elif arm.startswith(("P", "B")):
            os.environ["IRONMULE_CUBLAS_MIN_ROWS"] = str(args.threshold)
        elif arm.startswith(("F", "V", "I", "M", "Z", "J", "Y", "W")):
            if not args.lib:
                raise ValueError("missing measured kernel library")
            os.environ.update(
                IRONMULE_LAB_KERNEL_TRACE="1",
                IRONMULE_KG_LIB=str(args.lib.resolve()),
                IRONMULE_LAB_FP8=str(args.threshold),
            )
        elif arm.startswith("S"):
            os.environ.update(
                IRONMULE_LAB_KERNEL_TRACE="1",
                IRONMULE_KG_LIB=str(args.lib.resolve()),
                IRONMULE_W4_TABLE=str(args.lib.resolve().with_name("selected.table")),
            )
        elif arm != "vllm" and not arm.startswith(
            ("Q", "R", "U", "N", "G", "T", "X", "A", "L", "H", "E", "K", "O", "C")
        ):
            raise ValueError("unregistered arm")
        if not arm.startswith("S") and bounded_w4:
            os.environ.pop("IRONMULE_W4_TABLE", None)
        if arm.startswith(("K", "O", "C")):
            trial.measure(args, arm, hw, 0, deadline, draft=draft)
        else:
            trial.measure(args, arm, hw, 0, deadline)
    completed = {
        "hardware": trial.board(),
        "order": [a for a in args.arms if a not in reused and a not in skipped],
    }
    if reused:
        completed.update(requested=args.arms, reused=reused)
    save(args.out / "completed.json", completed)


if __name__ == "__main__":
    with Path(
        os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
    ).open("a") as lock:
        while True:
            remaining = float(os.environ["RP_DEADLINE"]) - time.time() - 45
            if remaining <= 0:
                raise TimeoutError("device wait exhausted the reserved window")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(remaining, 1.0))
        main()
