"""Serve entry point. Reads the device profile, refuses to invent one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from friday_calibrate.profile import HISTORY, profile_for
from friday_runtime_core.history import HistoryError, RuntimeHistory

from .scope import live_machine_sha256
from .server import Server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3"
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def load_profile(database: str | Path, *, model_id: str | None = None):
    """This machine's profile for *this* model, or ``None`` — never a guess.

    ``None`` means the pair (machine, model) was never calibrated here, and
    serving says so instead of running on a profile measured for something else.
    """

    try:
        with RuntimeHistory.open(HISTORY, database, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
    except (HistoryError, OSError):
        return None
    return profile_for(rows, machine_sha256=live_machine_sha256(), model_id=model_id)


def _offer_calibration(database: str | Path, model_id: str, *, run_it: bool):
    """No profile for this machine and model. Say so; measure only if asked.

    A calibration run holds the GPU for roughly a quarter of an hour, so it is
    not something to start behind someone's back on the first launch. Without
    the flag this prints the one command that fixes it and serves the baseline,
    which is the honest state: nothing here was verified on this machine yet.
    """

    print(f"This machine has no calibration profile for {model_id}.")
    if not run_it:
        print("  Serving the safe baseline -- no knob was verified here.")
        print(f"  To measure it (~15 min on AC power, GPU held):")
        print(f"    python tools/run_calibration.py run --execute --pairs 6 --model {model_id}")
        return None

    print("  Measuring this machine now (~15 min). Ctrl-C aborts; nothing is written.")
    from friday_calibrate.cli import main as calibrate_main

    code = calibrate_main(
        ["run", "--execute", "--pairs", "6", "--model", model_id, "--database", str(database)]
    )
    if code != 0:
        print("  Calibration did not complete; serving the baseline.")
        return None
    return load_profile(database, model_id=model_id)


def _latch(database: str | Path):
    """Persist the breaker into the same hash chain the profile lives in."""

    from friday_calibrate.profile import FAILURE_KIND
    from friday_runtime_core.breaker import PersistentLatch
    from friday_runtime_core.provenance import ProvenanceSpec, collect_provenance

    spec = ProvenanceSpec(
        runtime_id=HISTORY.runtime_id,
        code_directories=("friday_serve", "friday_calibrate", "friday_runtime_core"),
        spec_files=("requirements-apple-silicon.txt",),
    )

    def load():
        with RuntimeHistory.open(HISTORY, database, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
        failures = [row for row in rows if row["record_kind"] == FAILURE_KIND]
        return failures[-1]["report"].get("reason") if failures else None

    def append(reason):
        with RuntimeHistory.open(HISTORY, database) as history:
            history.persist(
                {
                    "schema_version": HISTORY.schema_version,
                    "runtime_id": HISTORY.runtime_id,
                    "kind": FAILURE_KIND,
                    "run_id": f"serve-failure-{reason}",
                    "status": "measurement_failed",
                    "reason": reason,
                    "formal_claim": False,
                },
                collect_provenance(spec, require_clean=False),
            )

    return PersistentLatch(load, append)


def _prewarm_hardware(backend: Any, profile: Any) -> None:
    """Pre-warm Metal GPU shader compilation and pin memory to eliminate cold first-request latency."""
    import time
    import mlx.core as mx
    from .dispatch import knobs_for

    try:
        from .environment_tuning import tune_runtime_environment

        env_info = tune_runtime_environment()
        qos_str = "P-Core QoS Active" if env_info.get("qos_interactive") else "Standard QoS"
        print(
            f"✓ Environment Tuned: {qos_str} | UMA: {env_info.get('uma_gb', 0):.0f} GB | "
            f"Metal Cache: {env_info.get('metal_cache_limit_gb', 0):.0f} GB | "
            f"Wired: {env_info.get('metal_wired_limit_gb', 0):.0f} GB"
        )
    except Exception as exc:
        print(f"⚠️ Environment tuning skipped: {exc}")

    try:
        t0 = time.perf_counter()
        knobs = knobs_for(profile)
        dummy_ids = backend.encode("Friday Apple Silicon Pre-Warmup")
        engine = backend._engine(knobs)
        capacity = 128
        state, token = engine._prefill(dummy_ids, capacity)
        body = engine._body(capacity, 1)
        for _ in range(4):
            out = body(token, state)
            token, state = engine._picks(out)[:, -1:], out[1]
        leaves = backend._leaves(state) if hasattr(backend, "_leaves") else []
        mx.eval(token, *leaves)
        mx.synchronize()
        warm_ms = (time.perf_counter() - t0) * 1000
        print(f"✓ Hardware Pre-Warmup complete in {warm_ms:.1f} ms. Metal JIT shaders primed.")
    except Exception as exc:
        print(f"⚠️ Pre-warmup warning (ignored): {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", allow_abbrev=False)

    generate = commands.add_parser("generate", allow_abbrev=False)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=32)
    generate.add_argument("--execute", action="store_true")

    http_cmd = commands.add_parser("serve", allow_abbrev=False)
    http_cmd.add_argument("--host", default="127.0.0.1")
    http_cmd.add_argument("--port", type=int, default=8080)
    http_cmd.add_argument("--dashboard", action="store_true", default=True)
    http_cmd.add_argument("--no-dashboard", action="store_false", dest="dashboard")
    http_cmd.add_argument("--interactive", action="store_true", default=True, help="Live interactive terminal cockpit (default: True in TTY)")
    http_cmd.add_argument("--no-interactive", action="store_false", dest="interactive", help="Disable live terminal cockpit")
    http_cmd.add_argument("--concurrency", type=int, default=None, help="Maximum concurrent requests (default: auto-adaptive, 8 for 1B, 4 for 4B, 2 for 12B)")
    http_cmd.add_argument("--model", default=None, help="Model ID or local path")
    http_cmd.add_argument(
        "--calibrate-if-unknown",
        action="store_true",
        help="Measure this machine and model first when no profile exists yet (~15 min GPU)",
    )
    http_cmd.add_argument(
        "--throttle",
        choices=("auto", "off"),
        default="auto",
        help="Step back while the Mac is busy, on battery or thermally limited (default: auto)",
    )
    http_cmd.add_argument("--dual-model", action="store_true", help="Also load the 1B model co-resident (a second LLM in unified memory); off by default")

    args = parser.parse_args(argv)
    # The profile is bound to one machine and one model, so the model has to be
    # known before it is looked up -- ``serve`` overrides it on the subparser.
    target_model = args.model or DEFAULT_MODEL
    profile = load_profile(args.database, model_id=target_model)

    if args.command == "status":
        from .dispatch import explain

        described = explain(profile)
        described["database"] = args.database
        described["model_id"] = target_model
        described["machine_sha256"] = live_machine_sha256()
        described["serves"] = "baseline" if profile is None else "device_profile_dispatch"
        if profile is None:
            described["reason"] = "no_profile_for_this_machine_and_model"
        _print(described)
        return 0

    if args.command == "serve":
        if profile is None:
            profile = _offer_calibration(
                args.database, target_model, run_it=args.calibrate_if_unknown
            )

        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from .http_server import create_server
        from .ironmule_backend import IronMuleBackend
        from .rl_controller import AdaptiveRLController
        from .telemetry import get_global_tracker
        from .terminal_dashboard import render_cockpit

        rl_path = PROJECT_ROOT / ".friday-data" / "rl-controller.json"
        rl_shadow_path = PROJECT_ROOT / ".friday-data" / "rl-shadow-decisions.jsonl"
        # Shadow only: the controller logs the action it would pick; serving
        # applies device-profile knobs and never updates the weights here.
        rl_ctrl = AdaptiveRLController.load(rl_path, shadow_log_path=rl_shadow_path)

        def _adaptive_concurrency(m_id: str) -> int:
            m = m_id.lower()
            return 8 if "1b" in m else (2 if "12b" in m else 4)

        concurrency = args.concurrency if args.concurrency is not None else _adaptive_concurrency(target_model)

        from .throttle import Throttle, set_global_throttle

        throttle = set_global_throttle(
            Throttle(enabled=args.throttle == "auto", max_width=concurrency)
        ).start()
        if throttle.enabled:
            print("✓ Considerate mode on: pace and batch width follow this Mac's idle load.")

        print(f"Loading {target_model} into Apple Silicon Unified Memory (Adaptive Concurrency: {concurrency})...")
        backend = IronMuleBackend.load(target_model)
        _prewarm_hardware(backend, profile)

        alternate_backends = {}
        if getattr(args, "dual_model", False) and "1b" not in target_model.lower():
            try:
                model_1b_id = "mlx-community/gemma-3-1b-it-4bit"
                print(f"Pre-warming secondary model ({model_1b_id}) for Dual-Model Co-Residency...")
                backend_1b = IronMuleBackend.load(model_1b_id)
                _prewarm_hardware(backend_1b, None)
                alternate_backends[model_1b_id] = backend_1b
                alternate_backends["gemma-1b"] = backend_1b
                alternate_backends["1b"] = backend_1b
                print("✓ Dual-Model Co-Residency Active: 1B + 4B pinned simultaneously in Unified Memory.")
            except Exception as exc:
                print(f"⚠️ Secondary model load skipped: {exc}")

        server = Server(
            backend,
            profile,
            latch=_latch(args.database),
            rl_controller=rl_ctrl,
            alternate_backends=alternate_backends,
        )
        tracker = get_global_tracker()
        tracker.set_server_info(args.host, args.port)

        is_interactive = bool(args.dashboard and getattr(args, "interactive", True) and sys.stdout.isatty())
        httpd = create_server(
            server,
            host=args.host,
            port=args.port,
            telemetry_tracker=tracker,
            enable_dashboard=args.dashboard,
            interactive_dashboard=is_interactive,
            max_concurrency=concurrency,
        )
        if not is_interactive:
            print(f"⚡ Friday Server running on http://{args.host}:{args.port}")
            print(f"   OpenAI API: http://{args.host}:{args.port}/v1/chat/completions")
            print(f"   Terminal Dashboard: http://{args.host}:{args.port}/dashboard")
            print(render_cockpit(tracker, colored=True))

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
            if is_interactive:
                sys.stdout.write("\n⚡ Friday server shut down cleanly.\n")
                sys.stdout.flush()
        return 0

    if not args.execute:
        _print(
            {
                "state": "not_released",
                "hint": "pass --execute; generation loads the local model",
                "profile": None if profile is None else profile.profile_id,
                "would_use": [] if profile is None else list(profile.verified_knobs()),
            }
        )
        return 78

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from .ironmule_backend import IronMuleBackend

    backend = IronMuleBackend.load(args.model)
    server = Server(backend, profile, latch=_latch(args.database))
    result = server.generate(args.prompt, args.max_tokens)
    _print(
        {
            "state": "generated",
            "plan": result.plan,
            "reason": result.reason,
            "knobs": dict(result.knobs),
            "tokens": len(result.tokens),
            "token_sha256": result.token_sha256,
            "prefill_ms": result.prefill_ns / 1e6,
            "decode_ms": result.decode_ns / 1e6,
            "text": result.text,
            "formal_claim": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
