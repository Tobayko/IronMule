"""Try the real local servers before spending a full comparison budget on them."""
from __future__ import annotations

import sys
import time

from ironmule_product import compare, compare_servers
from ironmule_product.state import _atomic_write


def run(args, save):
    deadline = time.monotonic() + args.budget_seconds
    report = {"schema": "ironmule.compare.readiness.v1", "status": "running", "attempts": [],
              "ready_engines": [], "vllm_mode": None}

    def attempt(engine, eager=False):
        args.vllm_enforce_eager = eager
        mode = "enforce_eager" if eager else "default"
        row = {"engine": engine, "mode": mode, "status": "running", "stage": "starting"}
        report["attempts"].append(row)
        save(report)
        try:
            compare.remaining(deadline)
            with compare_servers.open_server(engine, args, 0, deadline) as server:
                row["stage"] = "chat"
                save(report)
                reply = compare.stream(engine, server, "Count from one to five, in words.", 4, deadline)
                row.update(completion_tokens=reply["tokens"], prompt_tokens=reply["prompt_tokens"],
                           finish_reason=reply["finish_reason"], stage="identity")
                compare_servers.refresh_identity(engine, server, deadline)
                if not server.get("placement_verified") or not server.get("version"):
                    row["code"] = "placement_or_version_unverified"
                elif engine == "ironmule" and not server.get("autonomous"):
                    row["code"] = "autonomous_agent_unverified"
                elif engine == "llama_cpp" and not server.get("baseline_stock"):
                    row["code"] = "stock_unverified"
                if "code" in row:
                    row["status"] = "failed"
            if "code" not in row:
                row.update(status="ready", stage="finished")
                report["ready_engines"].append(engine)
                if engine == "vllm":
                    report["vllm_mode"] = mode
        except Exception as exc:
            # Only export the launcher's enumerated diagnostics. HTTP errors can contain prompts or log paths.
            if isinstance(exc, compare_servers.StartupError):
                code = exc.code
                row["startup_stage"] = exc.stage
            elif isinstance(exc, TimeoutError):
                code = "timeout"
            else:
                code = {"starting": "launch_failed", "chat": "chat_failed", "identity": "identity_failed"}[row["stage"]]
            row.update(status="failed", code=code)
        save(report)
        return row

    save(report)
    for engine in args.engines:
        result = attempt(engine)
        if engine == "vllm" and result.get("code") == "cuda_graph_or_compiler" and time.monotonic() < deadline:
            attempt(engine, eager=True)
    ready = len(report["ready_engines"])
    report["status"] = "ready" if ready == len(args.engines) else ("partial" if ready else "failed")
    save(report)
    return report


def main(argv=None):
    args = compare.arguments(argv)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = run(args, lambda report: _atomic_write(args.output, report))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
