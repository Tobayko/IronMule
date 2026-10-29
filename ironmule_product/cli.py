"""Portable CLI entry points for configuration and the local model service."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from .errors import InvalidRequest, ModelNotFound, ProductError, StateError
from .state import ProductStore
from .types import ModelSpec


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _parser(command: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"ironmule {command}", description=description, allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, help="dedicated product state directory (default: IRONMULE_HOME or ~/.ironmule/product)")
    return parser


def setup(argv: list[str]) -> int:
    parser = _parser("setup", "Initialize local product settings without loading or downloading a model.")
    parser.add_argument("--mode", choices=("desktop", "server"), default="desktop")
    args = parser.parse_args(argv)
    store = ProductStore(args.state_dir)
    value = store.setup(args.mode)
    _print({"settings": value, "state_dir": str(store.root),
            "next": "ironmule models add <cached-model-id>", "models_loaded": False})
    return 0


def inventory_rows(roots: list[Path] | None = None) -> list[dict[str, Any]]:
    from ironmule_inventory import discover_models

    if roots is not None:
        return discover_models(roots, loader="mlx_lm")
    rows = discover_models(loader="mlx_lm")
    local = Path(__file__).resolve().parents[1] / ".friday-data" / "models" / "hub"
    if local.is_dir():
        rows.extend(discover_models([local], loader="mlx_lm"))
    return sorted(rows, key=lambda row: (row["model_id"], row["revision"], row["snapshot_path"]))


def models(action: str, argv: list[str]) -> int:
    parser = _parser(f"models {action}", "Manage local model registrations; removing a registration keeps weight files.")
    if action in ("add", "remove"):
        parser.add_argument("model", help="exact model id from `ironmule models list`")
    if action == "add":
        parser.add_argument("--revision", help="exact cached snapshot revision")
        parser.add_argument("--cache-root", action="append", type=Path)
        parser.add_argument("--download", action="store_true", help="explicitly download this model through Hugging Face before registration")
    args = parser.parse_args(argv)
    store = ProductStore(args.state_dir)
    store.settings()  # A missing setup is an actionable error, not an implicit write.
    if action == "registered":
        _print({"models": [spec.as_dict() for spec in store.models()], "models_loaded": False})
        return 0
    if action == "remove":
        removed = store.remove_model(args.model)
        _print({"model": args.model, "registration_removed": removed, "weights_deleted": False})
        return 0 if removed else 1
    if action != "add":
        raise InvalidRequest("unknown model registry command")
    if args.download:
        if args.cache_root and len(args.cache_root) != 1:
            raise InvalidRequest("a download requires at most one explicit cache root")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise InvalidRequest("model download requires the huggingface_hub dependency") from exc
        try:
            snapshot_download(args.model, revision=args.revision,
                              cache_dir=str(args.cache_root[0]) if args.cache_root else None,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"])
        except Exception as exc:
            # Third-party exceptions may include credentials or signed URLs.
            raise InvalidRequest("model download failed; check model access, revision and network connectivity") from exc
    rows = [row for row in inventory_rows(args.cache_root)
            if row["model_id"] == args.model and row["status"] == "available"
            and (args.revision is None or row["revision"] == args.revision)]
    if not rows:
        raise ModelNotFound("no complete local snapshot matched; use models list, --revision or an explicit --download")
    revisions = {row["revision"] for row in rows}
    if len(revisions) != 1:
        raise InvalidRequest("multiple cached revisions matched; select one with --revision")
    row = rows[0]
    spec = ModelSpec.from_dict(row)
    store.register_model(spec)
    _print({"registered": spec.as_dict(), "warnings": row.get("warnings", []),
            "models_loaded": False, "hardware_qualified": False})
    return 0


def status(argv: list[str]) -> int:
    parser = _parser("status --product", "Inspect product configuration without initializing a model.")
    parser.add_argument("--product", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON is the default output")
    args = parser.parse_args(argv)
    store = ProductStore(args.state_dir)
    _print({"schema": "ironmule.product_status.v1", "settings": store.settings(),
            "models": [spec.as_dict() for spec in store.models()],
            "optimization": store.optimization_status(), "hardware_qualified": False})
    return 0


class IdleTuner:
    """START1: tune this machine for the served model in idle windows, never beside a loaded worker.

    One GPU cannot tune and serve at once (the timings would carry the users' load, and
    `gpu_busy()` refuses). So the idle task releases the worker, runs `ironmule tune` (planner,
    bounded, paired confirmation) as its own process group, stops it the moment a request is
    waiting, and hands the service a fresh worker, which loads whatever profile was confirmed.
    The tune stores a profile only after its confirmation, so an interruption stores nothing.
    """

    POLL_S = 0.5
    MAX_FAILURES = 2

    def __init__(self, service: Any, spec: Any, make_backend: Any, *, popen: Any = subprocess.Popen,
                 log: Any = print, cuda_device: str | None = None) -> None:
        self.service, self.spec, self.make_backend, self.popen, self.log = service, spec, make_backend, popen, log
        # HW1: a server pinned to one GPU tunes on that GPU, not on whichever CUDA lists first.
        self.environ = None if cuda_device is None else {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_device}
        self.done = False
        self.failures = 0
        tool = [sys.executable, "-c", "import sys, ironmule_cli; sys.exit(ironmule_cli.main(sys.argv[1:]))", "tune",
                "--model", spec.model_id]
        self.show = tool + ["--show"]
        self.tune = tool + ["--explorer", "planner", "--budget-minutes", "5"]

    def _confirmed(self) -> bool:
        shown = subprocess.run(self.show, capture_output=True, text=True, timeout=120, env=self.environ)
        return shown.returncode == 0 and "no profile for this hardware yet" not in shown.stdout

    def __call__(self, preempted: Any) -> None:
        if self.done or self.failures >= self.MAX_FAILURES:
            return
        if self._confirmed():
            self.done = True
            return
        self.log(json.dumps({"idle_tune": "started", "model": self.spec.model_id}), flush=True)
        backend = self.service.backend
        if backend is not None:
            backend.close()  # frees the GPU; a waiting request is served by the fresh worker below
        outcome = "interrupted"
        try:
            process = self.popen(self.tune, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=self.environ)
            while process.poll() is None:
                if preempted():
                    _stop_group(process)
                    break
                time.sleep(self.POLL_S)
            else:
                outcome = "done" if process.returncode == 0 else "failed"
        finally:
            fresh = self.make_backend()
            self.service.backend = fresh
            fresh.start()  # load before the scheduler takes the waiting requests
        if outcome == "done":
            self.done = True
        elif outcome == "failed":
            self.failures += 1
        self.log(json.dumps({"idle_tune": outcome, "model": self.spec.model_id}), flush=True)


def _stop_group(process: Any) -> None:
    """Stop a tune and every child it started (screening and confirmation processes)."""
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def serve(argv: list[str]) -> int:
    parser = _parser("serve", "Serve a registered local model through the OpenAI-compatible API.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", help="registered model to keep loaded")
    choice.add_argument("--no-model", action="store_true", help="control plane only; generation returns unavailable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key-env", default="IRONMULE_API_KEY", help="environment variable holding the API token")
    parser.add_argument("--tls-cert", type=str)
    parser.add_argument("--tls-key", type=str)
    parser.add_argument("--engine", choices=("reference", "tuned", "rust2", "stock", "ironmule"), default="reference",
                        help="stock/ironmule are the earlier names of reference/tuned. "
                             "experimental: stream from IronMule's engine with this machine's confirmed "
                             "`ironmule tune` profile (the untuned engine where none exists); not yet faster "
                             "than reference everywhere, and its tokens follow IronMule's engine (SERVE6, EXACT1). "
                             "rust2 (CUDA, Qwen3 dense 4-bit): the rust2 binary named by IRONMULE_RUST2_BINARY, "
                             "with the DSpark draft in IRONMULE_RUST2_DRAFT if set (RUST2-SRV)",)
    parser.add_argument("--no-thinking", action="store_true",
                        help="answer without a thinking section where the model's chat template has one "
                             "(start sets it on a machine without a GPU, CPU5)")
    parser.add_argument("--chat-reuse", action="store_true",
                        help="reuse each chat's earlier turns instead of recomputing them (--engine tuned only); "
                             "exact within its own plan, which differs from the strict one (SERVE7)")
    parser.add_argument("--compute-dtype", choices=("float32", "native", "dequantize"), default=None,
                        help="opt-in numeric plan for GPUs that emulate bf16 (NVIDIA below Ampere), or "
                             "dequantize on a machine without a GPU; changes output; see `ironmule plans`")
    parser.add_argument("--learning", choices=("off", "active", "paused"), default="off",
                        help="RSI3-LIVE: learn group width and gather wait from this server's own traffic "
                             "(cost model, sequential RL, dreaming); only a qualified policy replaces the reference. "
                             "Pause/resume at runtime via POST /v1/ironmule/learning or the chat page")
    parser.add_argument("--no-dreaming", action="store_true",
                        help="with --learning: the same learner without model rollouts (comparison arm)")
    parser.add_argument("--no-replay", action="store_true",
                        help="with --learning: no idle replay qualification (an external harness qualifies, P3)")
    parser.add_argument("--gpus", default=None, metavar="all|0,1",
                        help="HW1: one model worker per listed CUDA GPU (`all`: every GPU this machine shows), "
                             "each request on the GPU with the shortest queue. Each worker loads the whole model; "
                             "live learning and idle tuning run only with one GPU. Default: one worker, as before")
    parser.add_argument("--dispatch-policy", default=None, metavar="POLICY.json",
                        help="LD1: opt-in learned dispatch (where, in which order and how wide requests run; "
                             "never what they answer). Needs --dispatch-evidence for this model, revision and "
                             "GPUs; without matching evidence the reference dispatch serves and /health says why")
    parser.add_argument("--dispatch-evidence", default=None, metavar="EVIDENCE.json",
                        help="LD1: the passed hardware comparison that qualifies --dispatch-policy")
    parser.add_argument("--schedule", choices=("auto", "reference"), default="auto",
                        help="RL6: `auto` runs a fixed schedule (group width, shortest output limit first) that "
                             "ships with the hardware comparison that qualified it, only on exactly the engine, "
                             "model, revision, GPU and GPU count it was measured on, and only when no serving "
                             "policy is stored; `reference` keeps the default schedule. /health shows which serves")
    args = parser.parse_args(argv)
    args.engine = {"stock": "reference", "ironmule": "tuned"}.get(args.engine, args.engine)  # earlier names
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    if args.engine != "reference" and args.compute_dtype is not None:
        parser.error("--compute-dtype runs on the stock engine only")  # stock = reference
    if args.gpus is not None and args.no_model:
        parser.error("--gpus places model workers; --no-model starts none")
    if (args.dispatch_policy is None) != (args.dispatch_evidence is None):
        parser.error("--dispatch-policy and --dispatch-evidence go together")
    if args.dispatch_policy is not None and args.learning != "off":
        parser.error("--dispatch-policy and --learning both decide groups; use one")
    from .backend import MLXWorkerClient
    from .calibration import CalibrationFailure, model_lease
    from .http_server import create_server
    from .service import ProductService

    store = ProductStore(args.state_dir)
    store.settings()
    backend = service = server = None
    lease = None
    lease_entered = False
    previous_term = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        spec = store.model(args.model) if args.model else None
        if spec is not None:
            lease = model_lease(store)
            try:
                lease.__enter__()
                lease_entered = True
            except CalibrationFailure as exc:
                raise InvalidRequest("model resources are in use by another IronMule operation") from exc
            if args.engine != "reference" and args.compute_dtype is not None:
                raise InvalidRequest(f"--engine {args.engine} and --compute-dtype cannot be combined yet")
            if args.chat_reuse and args.engine != "tuned":
                raise InvalidRequest("--chat-reuse needs --engine tuned")
            devices = [None]
            if args.gpus is not None:
                from .device_pool import cuda_devices, select_devices
                devices = select_devices(args.gpus, cuda_devices())
            if len(devices) > 1 and args.learning != "off":
                raise InvalidRequest("--learning models one worker; use it with one GPU (HW1)")
            def make_backend(device=devices[0]):
                return MLXWorkerClient(spec, compute_dtype=args.compute_dtype, chat_reuse=args.chat_reuse,
                                       thinking=not args.no_thinking, cuda_device=device,
                                       execution_variant="reference" if args.engine == "reference" else "current_engine",
                                       **({"engine_configuration": "rust2"} if args.engine == "rust2" else {}))
            if len(devices) > 1:
                from .device_pool import start_pool
                service = start_pool(store, spec, devices, make_backend, ProductService)
            else:
                backend = make_backend()
                backend.start()
        if service is None:
            service = ProductService(store, backend=backend, spec=spec)
            if spec is not None:
                service.make_backend = make_backend  # STAB1: a lost worker is replaced, within bounds
        if spec is not None and args.learning != "off":
            service.learning = _live_learning(store, service, spec, args)
        wants_schedule = (args.schedule == "auto" and args.gpus is not None and args.learning == "off"
                          and args.dispatch_policy is None)
        binding = None
        if spec is not None and (args.dispatch_policy is not None or wants_schedule):
            from .device_pool import cuda_devices, gpu_name
            pinned = [device for device in devices if device is not None] or cuda_devices()[:1]
            binding = {"model_id": spec.model_id, "revision": spec.revision, "engine": args.engine,
                       "gpu": gpu_name(pinned) if pinned else None, "devices": len(devices)}
        if binding is not None and wants_schedule:
            from .serving_policy import qualified_schedule, stored_policy_exists
            qualified = None if stored_policy_exists(store.root) else qualified_schedule(binding)
            if qualified is not None:  # RL6, confirmed by LD1-V: only on the setup it was measured on
                for shard in getattr(service, "shards", [service]):
                    shard.policy = qualified[0]
        if binding is not None and args.dispatch_policy is not None:
            from .learned_dispatch import LearnedDispatch
            dispatch = LearnedDispatch.load(args.dispatch_policy, args.dispatch_evidence, binding,
                                            kill_path=store.root / "dispatch-withdrawals.json")
            if hasattr(service, "attach"):
                service.attach(dispatch)
            else:
                service.dispatch = dispatch
        if (spec is not None and args.engine == "tuned" and len(devices) == 1
                and (service.learning is None or not service.learning.eligible)):
            # Idle tuning unloads the worker; learning keeps the serving model loaded (RSI3-LIVE).
            service.idle_task = IdleTuner(service, spec, lambda: MLXWorkerClient(
                spec, execution_variant="current_engine", chat_reuse=args.chat_reuse,
                thinking=not args.no_thinking, cuda_device=devices[0]), cuda_device=devices[0])
        server = create_server(service, host=args.host, port=args.port,
                               api_key=os.environ.get(args.api_key_env),
                               tls_cert=args.tls_cert, tls_key=args.tls_key)
        print(json.dumps({"service": "ironmule", "host": server.server_address[0],
                          "port": server.server_address[1], "ready": service.health()["ready"]}), flush=True)
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if server is not None:
            server.server_close()
        elif service is not None:
            service.close()
        elif backend is not None:
            backend.close()
        if lease_entered:
            lease.__exit__(None, None, None)
    return 0


def _runtime_versions() -> dict[str, str | None]:
    """Experience is bound to the software that produced it, not only to the model."""
    from importlib import metadata

    out = {}
    for name in ("ironmule", "mlx", "mlx-lm"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _live_learning(store: ProductStore, service: Any, spec: ModelSpec, args: Any) -> Any:
    """RSI3-LIVE learner bound to this hardware, model, plan and schedule; see live_learning."""
    from .live_learning import LiveLearning
    from .readiness import hardware_identity

    backend = service.backend
    hardware = hardware_identity()
    payload = backend.ready_payload or {}
    binding = {"hardware": {"chip_name": hardware["chip_name"], "gpu_devices": hardware["gpu_devices"]},
               "device": backend.device, "model": {"id": spec.model_id, "revision": spec.revision},
               "plan": {"execution_variant": backend.execution_variant,
                        "engine_configuration": backend.engine_configuration, "engine": payload.get("engine"),
                        "compute_dtype": backend.compute_dtype, "chat_reuse": backend.chat_reuse,
                        "thinking": backend.thinking},
               "streams_batches": backend.streams_batches, "runtime": _runtime_versions(),
               "worker_sources_sha256": payload.get("sources_sha256")}
    policy = service.policy
    # The scheduler never runs a group wider than the stored policy allows, so neither may an action.
    learning = LiveLearning(store.root, binding, capacity=min(backend.batch_capacity, policy.max_width),
                            reference_width=policy.max_width, reference_gather_ms=policy.gather_ms, join=policy.join,
                            max_pending=service.settings["max_pending"],
                            request_timeout_s=service.settings["request_timeout_s"], mode=args.learning,
                            dreaming=not args.no_dreaming, replay=not args.no_replay)
    print(json.dumps({"learning": learning.summary(), "reason": learning.reason}), flush=True)
    return learning


DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"  # 3.4 GB, not gated; every IronMule study measured it
CPU_MODEL = "mlx-community/Qwen3-0.6B-4bit"  # the model CPU3 measured; 4B answers far too slowly on a CPU
CPU_REQUEST_TIMEOUT_S = 900


def _cpu_only() -> bool:
    """No Apple Silicon and no NVIDIA driver: MLX can only use the CPU (CPU4). install.sh asks the same."""
    import platform
    import shutil
    return not ((sys.platform == "darwin" and platform.machine() == "arm64") or shutil.which("nvidia-smi"))


def _open_when_ready(url: str) -> None:
    import time
    import urllib.request
    import webbrowser

    for _ in range(600):  # the page answers once the model is loaded; give it five minutes
        try:
            urllib.request.urlopen(url, timeout=1).close()
        except OSError:
            time.sleep(0.5)
            continue
        webbrowser.open(url)
        return


def start(argv: list[str]) -> int:
    parser = _parser("start", "Set up, get one model (asking before any download), serve it and open the chat page.")
    parser.add_argument("--model", help=f"model to chat with (default: the one already registered, else {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--yes", action="store_true", help="download the model without asking")
    parser.add_argument("--no-browser", action="store_true", help="only print the chat address")
    parser.add_argument("--engine", choices=("reference", "tuned", "rust2"), default="reference",
                        help="see `ironmule serve --help`")
    parser.add_argument("--chat-reuse", action="store_true", help="see `ironmule serve --help`")
    parser.add_argument("--learning", choices=("off", "active", "paused"), default="off",
                        help="see `ironmule serve --help`")
    args = parser.parse_args(argv)
    store = ProductStore(args.state_dir)
    cpu_only = _cpu_only()
    try:
        store.settings()
    except StateError:
        store.setup("desktop")
        if cpu_only:  # a CPU answer takes minutes; the 120 s default would cut every one off
            store.set_request_timeout(CPU_REQUEST_TIMEOUT_S)
    state = ["--state-dir", str(store.root)]
    registered = [spec.model_id for spec in store.models()]
    model = args.model or (registered[0] if registered else CPU_MODEL if cpu_only else DEFAULT_MODEL)
    if model not in registered:
        add = [model, *state]
        if not any(row["model_id"] == model and row["status"] == "available" for row in inventory_rows()):
            try:
                answer = "y" if args.yes else input(f"Download {model} from Hugging Face now? [Y/n] ")
            except EOFError:
                answer = "n"
            if answer.strip().lower() not in ("", "y", "yes"):
                print("ironmule: nothing downloaded; answer yes, pass --yes, or pick a cached model with --model",
                      file=sys.stderr)
                return 1
            add.append("--download")
        with contextlib.redirect_stdout(io.StringIO()):  # the registry's JSON is noise here
            models("add", add)
    plan = []
    if cpu_only:
        try:
            answer = input("No GPU found, so this runs on the CPU. Expand the 4-bit weights to float32 "
                           "(about 6x faster here; answers can differ slightly)? [Y/n] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() in ("", "y", "yes"):
            plan = ["--compute-dtype", "dequantize"]
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Loading {model}. The chat opens at {url} once it is ready; Ctrl+C stops IronMule.", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    engine = (["--engine", args.engine] if args.engine != "reference" else []) + (
        ["--chat-reuse"] if args.chat_reuse else []) + (
        ["--no-thinking"] if cpu_only else []) + (  # CPU5: thinking costs minutes on a CPU
        ["--learning", args.learning] if args.learning != "off" else [])
    return serve(["--model", model, "--port", str(args.port), *engine, *plan, *state])


def optimize(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: ironmule optimize {run|status|history|pause|resume} [options]")
        print("run waits for real host readiness and executes a fixed, bounded calibration; no automatic activation")
        return 0
    action, rest = argv[0], argv[1:]
    if action not in ("run", "status", "history", "pause", "resume"):
        raise InvalidRequest("unknown optimization command")
    parser = _parser(f"optimize {action}", "Local automatic calibration with a verified metadata history.")
    parser.add_argument("--json", action="store_true", help="include the full report (other commands always return JSON)")
    if action == "run":
        parser.add_argument("--model", required=True, help="registered local model id")
        parser.add_argument("--wait-ready", type=float, default=300, help="maximum readiness wait, 0..900 seconds; eligibility thresholds do not change")
        parser.add_argument("--readiness-only", action="store_true", help="verify readiness waiting without loading any model")
    if action == "history":
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--after-seq", type=int, default=0)
        parser.add_argument("--run-id")
    args = parser.parse_args(rest)
    from .calibration import CalibrationJob, history as read_history, status as read_status
    from friday_evidence.events import EventJournalError

    store = ProductStore(args.state_dir)
    try:
        if action == "status":
            _print(read_status(store))
        elif action == "history":
            _print(read_history(store, limit=args.limit, after_seq=args.after_seq, run_id=args.run_id))
        elif action in ("pause", "resume"):
            _print(store.set_optimization_paused(action == "pause"))
        else:
            def progress(event):
                if event["kind"] in ("run_started", "worker_started", "deferred", "run_finished"):
                    print(json.dumps({"event": event["kind"], "run_id": event["run_id"]}), file=sys.stderr, flush=True)

            job = CalibrationJob(store, args.model, max_wait_s=args.wait_ready,
                                 readiness_only=args.readiness_only, on_event=progress)
            previous = signal.getsignal(signal.SIGTERM)

            def stop(_signum, _frame):
                job.cancel.set()
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
            try:
                report = job.run()
            finally:
                signal.signal(signal.SIGTERM, previous)
            result = report if args.json else {key: report.get(key) for key in
                     ("run_id", "status", "model_id", "error_code", "error_type", "error_stage", "error_detail", "evaluation", "activation_allowed")}
            _print(result)
            return 0 if report["status"] in ("measured", "ready") else 3 if report["status"] == "deferred" else 1
    except EventJournalError as exc:
        raise InvalidRequest("optimization history is unavailable or failed integrity checks") from exc
    return 0


def dream_cmd(argv: list[str]) -> int:
    """RSI4: learn the serving policy from this state's request log (see serving_policy)."""
    from .serving_policy import ServingLog, dream, load_policy, save_policy

    parser = _parser("dream", "Learn group width, joining and gather wait from logged traffic (RSI4)")
    parser.add_argument("--capacity", type=int, default=8, help="the backend's group capacity to dream within")
    parser.add_argument("--dry-run", action="store_true", help="print the choice without storing it")
    args = parser.parse_args(argv)
    store = ProductStore(args.state_dir)
    records = ServingLog(store.root).read()
    policy, evidence = dream(records, load_policy(store.root), capacity=args.capacity)
    if not args.dry_run and evidence.get("reason") is None:
        save_policy(store.root, policy, evidence)
    _print({"policy": {"max_width": policy.max_width, "join": policy.join, "gather_ms": policy.gather_ms,
                       "source": policy.source}, "stored": not args.dry_run and evidence.get("reason") is None,
            "evidence": evidence})
    return 0


def dispatch(command: str, argv: list[str]) -> int:
    try:
        if command == "start":
            return start(argv)
        if command == "setup":
            return setup(argv)
        if command == "serve":
            return serve(argv)
        if command == "optimize":
            return optimize(argv)
        if command == "status":
            return status(argv)
        if command == "dream":
            return dream_cmd(argv)
        if command == "models" and argv:
            return models(argv[0], argv[1:])
        raise InvalidRequest("unknown product command")
    except ProductError as exc:
        print(f"ironmule: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("ironmule: local service configuration or access failed", file=sys.stderr)
        return 1
