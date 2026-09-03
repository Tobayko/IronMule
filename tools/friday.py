#!/usr/bin/env python3
"""Single entry point for the Friday Hardware-Aware AI Runtime on Apple Silicon.

Quickstart:
    ./friday serve                              Start OpenAI-compatible server with Live-Cockpit
    ./friday autotune                           Auto-tune hardware knobs on real Apple Silicon (<15s)
    ./friday status                             Inspect device profile, knobs, and runtime health
    ./friday doctor                             Verify Apple Silicon Metal GPU & environment readiness
    ./friday monitor [--port 8080]              Remote live terminal telemetry monitor
    ./friday <tool> --execute                   Run a paired hardware measurement tool
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent

PRIMARY_COMMANDS = {
    "serve": "Start OpenAI-compatible HTTP/SSE server with Terminal Live-Cockpit",
    "autotune": "Universal Hardware Auto-Tuner: safely calibrates hardware knobs in <15s",
    "status": "Everything on one screen: hardware facts, verified knobs, runtime state",
    "doctor": "Preflight check: verifies Apple Silicon Metal GPU, Python, and memory",
    "monitor": "Remote Interactive Terminal Cockpit: in-place telemetry at 10-20 FPS",
}

TOOLS = {
    "loop": (
        "optimization_loop.py",
        "Self-optimization loop: explores execution plans, refines, confirms its own winner",
    ),
    "dispatch": (
        "measure_dispatch_plan.py",
        "Measure one execution plan against a baseline, paired, with a frozen threshold",
    ),
    "cooldown": (
        "measure_cooldown_effect.py",
        "Characterize how an idle pause slows the next operation",
    ),
    "aa": (
        "run_h0_aa.py",
        "Run the preregistered H0 A/A null control (calibration; no optimization)",
    ),
    "model-loop": (
        "model_loop.py",
        "H2: a local model proposes execution plans, the harness judges them (needs mlx-lm)",
    ),
    "codegen": (
        "codegen_loop.py",
        "H2 full: a local model writes execution plans, sandboxed and judged (needs mlx-lm)",
    ),
    "roofline": (
        "measure_roofline.py",
        "Is inference memory-bound or compute-bound? Decides which optimizations can help (needs mlx-lm)",
    ),
    "fusion": (
        "measure_fusion_layer.py",
        "A layer over an unmodified model: fuse its forward pass, measure the gain (needs mlx-lm)",
    ),
    "guard": (
        "run_h01_guard.py",
        "Verify the H0.1 analysis core stays stdlib-only (no MLX, NumPy or sockets)",
    ),
    "evidence": (
        "evidence.py",
        "Verify or display the append-only H1/H2 evidence history (no GPU)",
    ),
    "autotune": (
        "autotune.py",
        "Universal Hardware Auto-Tuner: safely calibrates and certifies hardware knobs in <45s",
    ),
    "serve": (
        "run_serve.py",
        "OpenAI-compatible HTTP/SSE server with Terminal Live-Cockpit for LLM inference",
    ),
    "monitor": (
        "monitor.py",
        "Interactive IronMule Live Terminal Monitor: in-place, flicker-free telemetry cockpit at 10-20 FPS",
    ),
}


def _shared():
    """The preconditions every measuring tool shares, loaded from one place."""

    return _load(TOOLS_DIR / "_bench.py")


def _load(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_list() -> int:
    print("🐎 Friday — Hardware-Aware Self-Optimizing AI Runtime on Apple Silicon\n")
    print("Core Commands:")
    pwidth = max(len(name) for name in PRIMARY_COMMANDS)
    for name, desc in PRIMARY_COMMANDS.items():
        print(f"  ./friday {name:<{pwidth}}  {desc}")

    print("\nExploratory Measurement Tools (require --execute):")
    m_tools = {k: v for k, v in TOOLS.items() if k not in PRIMARY_COMMANDS}
    mwidth = max(len(name) for name in m_tools)
    for name, (script, description) in m_tools.items():
        print(f"  ./friday {name:<{mwidth}}  {description}")

    print("\nQuickstart:  ./friday serve")
    print("Auto-tune:   ./friday autotune")
    print("Status:      ./friday status")
    return 0


def cmd_doctor() -> int:
    """Report whether this machine can run the measurements, and what is missing."""

    findings: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str, fatal: bool = True) -> None:
        findings.append({"check": name, "ok": ok, "detail": detail, "fatal": fatal})

    check(
        "python",
        sys.version_info[:2] >= (3, 12),
        f"{sys.version_info.major}.{sys.version_info.minor}, need >= 3.12",
    )

    try:
        import mlx.core as mx

        from importlib.metadata import version

        mx.eval(mx.array([1.0]) + 1)
        check("mlx", True, f"{version('mlx')}, Metal device responds")
    except Exception as exc:
        check("mlx", False, f"unavailable: {type(exc).__name__}")

    try:
        from importlib.metadata import version

        check("numpy", True, version("numpy"))
    except Exception:
        check("numpy", False, "missing")

    try:
        from importlib.metadata import version

        check("mlx-lm", True, f"{version('mlx-lm')} (only needed for model tests)", fatal=False)
    except Exception:
        check("mlx-lm", False, "missing; only needed for model tests", fatal=False)

    # Mains power is a measurement requirement, not a nicety: on battery macOS
    # caps the GPU power budget, so runs are neither comparable nor gentle.
    source = _shared().read_power_source()
    check(
        "power",
        source == "ac_power",
        {"ac_power": "AC power",
         "battery_power": "on battery -- measurements are refused",
         "unknown": "cannot determine -- measurements are refused"}[source],
    )

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    check("disk", free_gb > 5, f"{free_gb:.1f} GB free", fatal=False)

    width = max(len(str(f["check"])) for f in findings)
    for finding in findings:
        mark = "ok  " if finding["ok"] else ("FAIL" if finding["fatal"] else "warn")
        print(f"  [{mark}] {str(finding['check']):<{width}}  {finding['detail']}")

    blocking = [f for f in findings if not f["ok"] and f["fatal"]]
    print()
    if blocking:
        print(f"Not ready: {', '.join(str(f['check']) for f in blocking)}")
        return 1
    print("Ready to measure.")
    return 0


def cmd_status(rest: list[str]) -> int:
    """Everything relevant on one screen: device, knobs, runtime, runs, open work.

    Replaces twelve loopback web dashboards. Line-oriented on purpose - a
    full-screen TUI reads far worse to a screen reader than plain scrollback,
    and it is the leaner build besides.
    """

    known = {"--json", "--plain", "--decisions", "--signals", "--all"}
    unknown = [item for item in rest if item not in known]
    if unknown:
        print(json.dumps({"error": "unknown option", "given": unknown,
                          "known": sorted(known)}))
        return 64
    sys.path.insert(0, str(PROJECT_ROOT))
    import time

    from friday_runtime_core import status as ui
    from friday_runtime_core import status_sources as src
    from friday_runtime_core.provenance import hardware_facts

    profile = src.device_profile()
    everything = "--all" in rest
    sections = [
        ui.device_section(hardware_facts(), profile),
        ui.knob_section(profile, src.KNOWN_KNOBS),
        ui.runtime_section(profile, src.circuit_reason()),
        ui.runs_section(src.recent_runs()),
    ]
    # The two sections that carry what a web dashboard used to: shown on
    # request, so the default page stays one screen.
    if everything or "--signals" in rest:
        sections.append(ui.signal_section(src.h0_board()))
    if everything or "--decisions" in rest:
        sections.append(ui.decisions_section(src.optimizer_decisions()))
    sections.append(ui.open_section(src.open_backlog_entries()))
    snapshot = ui.Status(generated_at=time.strftime("%Y-%m-%d %H:%M"), sections=tuple(sections))
    if "--json" in rest:
        print(json.dumps(snapshot.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    colour = ui.use_colour() and "--plain" not in rest
    sys.stdout.write(ui.render(snapshot, colour=colour))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(__doc__.strip())
        print()
        return cmd_list()
    command, rest = args[0], args[1:]
    if command == "list":
        return cmd_list()
    if command == "doctor":
        return cmd_doctor()
    if command == "status":
        return cmd_status(rest)
    if command == "serve" and (not rest or rest[0] not in {"status", "generate", "serve"}):
        rest = ["serve"] + rest
    if command == "autotune" and "--execute" not in rest and "-h" not in rest and "--help" not in rest:
        rest = ["--execute"] + rest
    if command not in TOOLS:
        print(json.dumps({"error": "unknown tool", "tool": command, "known": sorted(TOOLS)}))
        return 64
    module = _load(TOOLS_DIR / TOOLS[command][0])
    # Not every tool takes arguments; the guard and the A/A sequencer are
    # parameterless by design, so calling them with an empty list must still work.
    import inspect

    if inspect.signature(module.main).parameters:
        return int(module.main(rest) or 0)
    if rest:
        print(json.dumps({"error": "tool takes no arguments", "tool": command, "given": rest}))
        return 64
    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
