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


def _qualified_knob_options(spec: Any, devices: list[str | None]) -> dict[str, Any]:
    """KP1: the worker option for a shipped class-A knob set bound to exactly this setup, else nothing."""
    from .knob_profile import framework_versions, qualified_knobs, shipped_knobs

    if not shipped_knobs():
        return {}
    from .device_pool import cuda_devices, gpu_name

    pinned = [device for device in devices if device is not None] or cuda_devices()[:1]
    found = qualified_knobs({"model_id": spec.model_id, "revision": spec.revision, "engine": "tuned",
                             "gpu": gpu_name(pinned) if pinned else None, **framework_versions()})
    if found is None:
        return {}
    knobs, record = found
    return {"engine_knobs": {"knobs": knobs, "source": f"qualified:{record['experiment']}"[:64]}}


def _engine_options(engine: str, width: int | None = None) -> dict[str, Any]:
    """The worker client's options that `--engine` sets beyond the execution variant; `width` is the stacked
    engine's (AG5: the agent's measured best width; None keeps 8)."""
    from .backend import DENSE_STARTUP_TIMEOUT

    if engine == "rust2":
        return {"engine_configuration": "rust2"}
    if engine == "dense":  # KB1: its probe compiles every kernel shape before the worker is ready
        return {"engine_configuration": "dense_stacked", "startup_timeout": DENSE_STARTUP_TIMEOUT,
                **({"dense_width": width} if width is not None else {})}
    return {}



def _ggml_agent_plan(width: dict[str, Any], root: Any, flash_only: bool = False) -> dict[str, Any] | None:
    """AG10: the configurations the ggml engine's agent may serve -- every width SEQ1 measured on this machine with the
    settings the quality gate passed -- and the one SEQ1's rule chose as the incumbent. None with fewer than two."""
    from ironmule.ggml_agent import configs_from_measurement

    if not width.get("step_s") or os.environ.get("IRONMULE_GGML_AGENT", "1") == "0":
        return None
    measured = width.get("measured") or [{"settings": width.get("settings") or {}, "step_s": width["step_s"],
                                          "step_one_s": width.get("step_one_s") or {}}]
    configs = configs_from_measurement(measured, int(width["seq_ctx"]))
    if flash_only:  # SP1: the shared pool needs flash attention (FA1u); widths size its pool
        configs = {k: c for k, c in configs.items() if c["settings"].get("flash_attn")}
    served = {k: True for k, v in (width.get("settings") or {}).items() if v}
    incumbent = next((k for k, c in configs.items()
                      if c["seqs"] == int(width["sequences"]) and c["settings"] == served), None)
    if len(configs) < 2 or incumbent is None:
        return None
    return {"configs": configs, "incumbent": incumbent, "root": str(root)}

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
    parser.add_argument("--gguf", default=None,
                        help="GG1 (with --engine ggml, CPU only): the GGUF llama.cpp serves for the registered model; "
                             "it must give the model's token ids and pass the quality gate on this machine (its "
                             "perplexity within 0.5 %% of the model widened to float32), or the server refuses. "
                             "`tools/gguf_from_mlx.py` builds one from the registered MLX model")
    parser.add_argument("--engine", choices=("reference", "tuned", "rust2", "dense", "ggml", "stock", "ironmule"),
                        default=None,
                        help="default reference; with --agent on and no --compute-dtype the agent chooses the plan and "
                             "engine itself on this machine (AG4). stock/ironmule are the earlier names of "
                             "reference/tuned. "
                             "experimental: stream from IronMule's engine with this machine's confirmed "
                             "`ironmule tune` profile (the untuned engine where none exists); not yet faster "
                             "than reference everywhere, and its tokens follow IronMule's engine (SERVE6, EXACT1). "
                             "rust2 (CUDA, Qwen3 dense 4-bit): the rust2 binary named by IRONMULE_RUST2_BINARY, "
                             "with the DSpark draft in IRONMULE_RUST2_DRAFT if set (RUST2-SRV). "
                             "dense (CUDA below compute capability 8, Qwen 3 bf16 or Gemma 3 at 4 bit; on a machine without a "
                             "GPU, Llama and Qwen checkpoints on the CPU stacked plan, CPU6): the opt-in dense "
                             "numeric plan, up to eight requests in one decode step through IronMule's own kernels; "
                             "answers differ from the other engines' but never with the group (KB1, KB2)",)
    parser.add_argument("--no-thinking", action="store_true",
                        help="answer without a thinking section where the model's chat template has one "
                             "(start sets it on a machine without a GPU, CPU5)")
    parser.add_argument("--chat-reuse", action="store_true",
                        help="reuse each chat's earlier turns instead of recomputing them (--engine tuned only); "
                             "exact within its own plan, which differs from the strict one (SERVE7)")
    parser.add_argument("--compute-dtype", choices=("float32", "float16", "native", "dequantize"), default=None,
                        help="opt-in numeric plan for GPUs that emulate bf16 (NVIDIA below Ampere), or "
                             "dequantize on a machine without a GPU; changes output; see `ironmule plans`")
    parser.add_argument("--speculate", choices=("auto", "context", "off"), default="auto",
                        help="SP2 (the CPU stacked engine): verify drafts copied from the request's own prompt and "
                             "answer, several positions in one step, with how many learned while it serves; every "
                             "token stays the engine's own. Opt-in: `context` with --engine dense (or the agent's "
                             "stacked engine); `auto` and `off` leave it off, since its hardware test SP2-H did not "
                             "pass its bound. A plan whose verify window is not bit-identical on this machine never "
                             "speculates; /health says what serves")
    parser.add_argument("--cpu-blas", choices=("stock", "openblas"), default="stock",
                        help="CPU6 (Linux x86-64, no GPU): `openblas` runs MLX's float32 matrix products on an "
                             "OpenBLAS found on this machine (NumPy's bundled one first) instead of the reference BLAS "
                             "the mlx-cpu wheel ships; several times faster with --compute-dtype dequantize or float32. "
                             "Opt-in: a different summation order can change answers. /health says which BLAS serves "
                             "and, if the optimised one could not, why")
    parser.add_argument("--learning", choices=("off", "active", "paused"), default="off",
                        help="RSI3-LIVE: learn group width and gather wait from this server's own traffic "
                             "(cost model, sequential RL, dreaming); only a qualified policy replaces the reference. "
                             "Pause/resume at runtime via POST /v1/ironmule/learning or the chat page")
    parser.add_argument("--no-dreaming", action="store_true",
                        help="with --learning: the same learner without model rollouts (comparison arm)")
    parser.add_argument("--no-replay", action="store_true",
                        help="with --learning: no idle replay qualification (an external harness qualifies, P3)")
    parser.add_argument("--agent", choices=("off", "on"), default="off",
                        help="AG1: the IronMule agent decides each group's order and width, learns what this "
                             "machine's groups cost from every request, trains a schedule while idle and serves it "
                             "only after it beat the incumbent in a replayed comparison on this server. Never "
                             "changes an answer; one worker; not with --learning or --dispatch-policy")
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
    parser.add_argument("--knobs", choices=("auto", "reference"), default="auto",
                        help="KP1: `auto` gives --engine tuned the knob set (compiled step, fewer host reads) that "
                             "ships with the hardware comparison that qualified it, only on exactly the model, "
                             "revision, GPU and MLX versions it was measured on, only knobs that leave every answer "
                             "unchanged, and only when no tune profile is stored; `reference` keeps the engine's "
                             "default knobs")
    args = parser.parse_args(argv)
    if args.gguf is not None and args.engine is None:  # a GGUF names the ggml engine (GG1); AG10 is its agent
        args.engine = "ggml"
    agent_chooses = args.agent == "on" and args.engine is None and args.compute_dtype is None and not args.no_model
    args.engine = {"stock": "reference", "ironmule": "tuned", None: "reference"}.get(args.engine, args.engine)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    if args.engine != "reference" and args.compute_dtype is not None:
        parser.error("--compute-dtype runs on the stock engine only")  # stock = reference
    if args.gpus is not None and args.no_model:
        parser.error("--gpus places model workers; --no-model starts none")
    if args.speculate == "context" and args.engine not in ("dense", "ggml") and not agent_chooses:
        parser.error("--speculate context needs --engine dense, --engine ggml or --agent on (SP2)")
    if (args.engine == "ggml") != (args.gguf is not None):
        parser.error("--engine ggml and --gguf go together (GG1)")
    if args.cpu_blas != "stock" and args.gpus is not None:
        parser.error("--cpu-blas applies to CPU workers; --gpus pins workers to GPUs")
    if (args.dispatch_policy is None) != (args.dispatch_evidence is None):
        parser.error("--dispatch-policy and --dispatch-evidence go together")
    if args.dispatch_policy is not None and args.learning != "off":
        parser.error("--dispatch-policy and --learning both decide groups; use one")
    if args.agent == "on" and (args.learning != "off" or args.dispatch_policy is not None):
        parser.error("--agent decides groups itself; use it without --learning and --dispatch-policy")
    from .backend import MLXWorkerClient
    from .calibration import CalibrationFailure, model_lease
    from .http_server import create_server
    from .service import ProductService

    store = ProductStore(args.state_dir)
    store.settings()
    backend = service = server = agent_setup = None
    ggml_options: dict[str, Any] = {}
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
            agent_setup = None
            if agent_chooses:  # AG4: the agent sets up plan and engine itself, once per machine and model
                agent_setup = _agent_setup(spec)
                args.engine = agent_setup["engine"]
                args.compute_dtype = agent_setup["compute_dtype"] if args.engine == "reference" else None
                if agent_setup.get("cpu_blas") and args.gpus is None:  # CPU6 step 3: the BLAS it chose and gated
                    args.cpu_blas = agent_setup["cpu_blas"]
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
            if args.engine == "route" and len(devices) > 1:  # only the agent chooses `route`
                agent_setup = _fallback_setup(agent_setup, "its two paths share one card, and this server runs on "
                                              "several", agent_setup.get("compute_dtype"))
                args.engine, args.compute_dtype = "reference", agent_setup["compute_dtype"]
            knob_options = _qualified_knob_options(spec, devices) if (
                args.knobs == "auto" and args.engine == "tuned") else {}
            # AG5: the stacked engine serves at the width the agent measured best on this card (None keeps 8).
            stacked_width = (agent_setup or {}).get("width") if args.engine in ("dense", "route") else None

            ggml_options: dict[str, Any] = {}
            if args.engine == "ggml":  # GG1: llama.cpp's kernels, served only behind the gate (owner, 2026-11-10)
                from ironmule.ggml_engine import available, gate as ggml_gate

                reason = available()
                if reason is not None:
                    raise InvalidRequest(f"--engine ggml: {reason}")
                checked = ggml_gate(spec.snapshot_path, args.gguf, cache=Path(store.root) / "ggml-gate")
                print(json.dumps({"ggml_gate": {k: checked.get(k) for k in ("ratio", "interval", "passes", "reason")}}),
                      flush=True)
                if checked.get("passes") is not True:
                    raise InvalidRequest("--engine ggml: the GGUF did not pass the quality gate on this machine "
                                         f"({checked.get('reason') or checked.get('interval')})")
                if not os.environ.get("IRONMULE_GGML_SEQS"):  # SEQ1: the engine chooses its width on this machine
                    from ironmule.ggml_engine import choose_sequences

                    width = choose_sequences(args.gguf, checked.get("settings"), cache=Path(store.root) / "ggml-gate",
                                             device=devices[0])
                    os.environ["IRONMULE_GGML_SEQS"] = str(width["sequences"])  # the worker and its client follow it
                    checked = {**checked, "sequences": width}
                    shared = checked.get("shared") or {}
                    if shared.get("passes") and os.environ.get("IRONMULE_GGML_SHARED", "1") != "0":
                        from ironmule.ggml_engine import LLAMA_MAX_SEQ, fa1_version

                        if fa1_version() >= 3:  # SP1: one pool, prefixes shared; twice the requests at once
                            cap = int(os.environ.get("IRONMULE_GGML_SHARED_CAP") or min(LLAMA_MAX_SEQ // 2,
                                                                                      2 * int(width["sequences"])))
                            checked["shared_serving"] = {"settings": shared["settings"], "cap": cap,
                                                         "ratio": shared.get("ratio"),
                                                         "interval": shared.get("interval")}
                            print(json.dumps({"ggml_shared_pool": checked["shared_serving"]}), flush=True)
                    plan = _ggml_agent_plan(width, store.root, flash_only=bool(checked.get("shared_serving")))
                    if plan is not None:  # AG10: the agent may switch among the measured widths while it serves
                        checked["agent"] = plan
                        os.environ["IRONMULE_GGML_SEQS"] = str(max(c["seqs"] for c in plan["configs"].values()))
                    if checked.get("shared_serving"):  # the service feeds up to the pool's running requests
                        os.environ["IRONMULE_GGML_SEQS"] = str(max(int(os.environ["IRONMULE_GGML_SEQS"]),
                                                                   checked["shared_serving"]["cap"]))
                    print(json.dumps({"ggml_sequences": {k: width.get(k) for k in ("sequences", "fitting", "rates",
                                                                                   "reason")}}), flush=True)
                ggml_options = {"engine_configuration": "ggml", "ggml_gguf": str(Path(args.gguf).resolve()),
                                "ggml_gate": checked, **({"speculate": True} if args.speculate != "off" else {})}

            def make_backend(device=devices[0]):
                # SP2: speculation on the stacked engine where asked. SP2-H (2026-11-10) did not pass its bound, so
                # `auto` leaves it off, also where the agent chose that engine (BACKLOG SP2, Tier 0)
                speculate = args.engine == "dense" and args.speculate == "context"
                if args.engine == "ggml":
                    # MG2: with --gpus each worker sees one GPU and offloads every layer to it (data parallel)
                    return MLXWorkerClient(spec, thinking=not args.no_thinking, execution_variant="current_engine",
                                           cuda_device=device, **ggml_options)
                return MLXWorkerClient(spec, compute_dtype=args.compute_dtype, chat_reuse=args.chat_reuse,
                                       thinking=not args.no_thinking, cuda_device=device,
                                       cpu_blas=None if args.cpu_blas == "stock" else args.cpu_blas,
                                       execution_variant="reference" if args.engine == "reference" else "current_engine",
                                       **_engine_options(args.engine, stacked_width), **knob_options,
                                       **({"speculate": True} if speculate else {}))
            def start_serving() -> tuple[Any, Any]:
                """(service, backend) for the engine `args` names now."""
                if args.engine == "route":  # AG4 step 2: the single path and the stacked engine on one card
                    # AG6: the router learns its serving mode (route, stacked or single) by dreaming (AG6-H)
                    from ironmule_product.path_router import PolicyRouter

                    from .device_pool import start_paths
                    device = devices[0]
                    blas = None if args.cpu_blas == "stock" else args.cpu_blas
                    single = lambda: MLXWorkerClient(spec, compute_dtype=agent_setup["compute_dtype"],  # noqa: E731
                                                     thinking=not args.no_thinking, cuda_device=device, cpu_blas=blas)
                    stacked = lambda: MLXWorkerClient(spec, thinking=not args.no_thinking,  # noqa: E731
                                                      cuda_device=device, execution_variant="current_engine",
                                                      cpu_blas=blas, **_engine_options("dense", stacked_width))
                    label = device if device is not None else "0"
                    return start_paths(store, spec, [single, stacked], [f"{label}:single", f"{label}:stacked"],
                                       ProductService, PolicyRouter.from_setup(agent_setup)), None
                if len(devices) > 1:
                    from .device_pool import start_pool
                    return start_pool(store, spec, devices, make_backend, ProductService), None
                started = make_backend()
                try:
                    started.start()
                except BaseException:
                    started.close()
                    raise
                return None, started

            try:
                service, backend = start_serving()
            except Exception as exc:  # noqa: BLE001 - only the agent's own choice falls back
                if not agent_chooses:
                    raise  # a plan the caller named is kept, and its failure reported
                # The agent's choice did not start here (a worker that ran out of memory, a kernel that failed):
                # its gated numeric plan on the reference engine serves instead, else stock; never silently.
                tried, failure = {(args.engine, args.compute_dtype)}, exc
                for compute_dtype in (agent_setup.get("compute_dtype"), None):
                    if ("reference", compute_dtype) in tried:
                        continue
                    tried.add(("reference", compute_dtype))
                    agent_setup = _fallback_setup(agent_setup, failure, compute_dtype)
                    args.engine, args.compute_dtype = "reference", compute_dtype
                    try:
                        service, backend = start_serving()
                        break
                    except Exception as again:  # noqa: BLE001 - the next fallback, or the last failure below
                        failure = again
                else:
                    raise failure
        if service is None:
            service = ProductService(store, backend=backend, spec=spec)
            if spec is not None:
                service.make_backend = make_backend  # STAB1: a lost worker is replaced, within bounds
        if spec is not None and args.learning != "off":
            service.learning = _live_learning(store, service, spec, args)
        if spec is not None and args.agent == "on" and agent_setup is not None:
            for each in [service, *getattr(service, "shards", [])]:
                each.agent_setup = {k: agent_setup.get(k) for k in ("engine", "compute_dtype", "reason", "decision",
                                                                    "numeric", "stacked", "levers", "fallback_from",
                                                                    "cpu_blas", "blas")}
        if spec is not None and args.agent == "on" and args.engine == "route":
            print(json.dumps({"agent": {"serving": "two paths on one card; the agent dreams whether to route, stack or "
                                                   "serve single", "reason": agent_setup["reason"]}}), flush=True)
        elif spec is not None and args.agent == "on" and args.engine == "ggml":
            # AG10: the ggml engine batches continuously and reuses sequences; its own agent (in the worker) learns,
            # dreams and switches its width and attention; the group agent (AG1) would price the wrong engine
            plan = (ggml_options.get("ggml_gate") or {}).get("agent")
            print(json.dumps({"agent": {"serving": "the ggml engine's own agent (AG10)" if plan else
                                        "the ggml engine's measured width (SEQ1)",
                                        "configurations": sorted((plan or {}).get("configs", {}))}}), flush=True)
        elif spec is not None and args.agent == "on" and args.engine == "dense":
            # AG4: the stacked engine serves its measured schedule (w8, slot reuse); the online agent does not
            # attach yet, because its world model prices one group at a time, not slot reuse (AG4, step 3).
            print(json.dumps({"agent": {"serving": "the chosen stacked engine's own schedule", "reason":
                                        "the online agent does not model slot reuse yet"}}), flush=True)
        elif spec is not None and args.agent == "on":
            shards = getattr(service, "shards", None)
            try:
                if shards:  # AG2: one agent per GPU, bound to its card, ordering and learning from that card's queue
                    from .agent import share_lengths

                    agents = [_agent(store, shard, spec, card=device) for shard, device in zip(shards, service.devices)]
                    share_lengths(agents)  # answer lengths depend on the model, not the card
                    service.agents = agents
                    for shard, agent in zip(shards, agents):
                        shard.agent = shard.dispatch = agent
                else:
                    agent = _agent(store, service, spec)
                    service.agent = service.dispatch = agent
            except Exception as exc:  # noqa: BLE001 - the server's own schedule serves without the online agent
                print(json.dumps({"agent": {"serving": "the server's own schedule", "reason":
                                            f"the online agent did not start ({type(exc).__name__}: {exc})"[:300]}}),
                      flush=True)
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
        if (spec is not None and args.engine == "tuned" and len(devices) == 1 and service.agent is None
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


def _blas_binding(backend: Any) -> dict[str, Any]:
    """CPU6: a worker asked to run on the optimised BLAS binds learning and evidence to it (nothing otherwise, so
    every other binding, and its digest, is unchanged)."""
    asked = getattr(backend, "cpu_blas", None)
    return {} if asked is None else {"cpu_blas": asked}


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
                        "thinking": backend.thinking, **_blas_binding(backend)},
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


def _agent_setup(spec: ModelSpec) -> dict[str, Any]:
    """AG4: the agent's own plan and engine for this machine and model (`ironmule.plan_choice`), measured once on
    the device and stored with its evidence; printed, and shown in `/health` as `agent_setup`. It never stops the
    server: whatever fails in the measurement leaves the reference engine on the checkpoint's dtype, with why."""
    from ironmule.autopilot import _probe_out_of_process
    from ironmule.hw import probe
    from ironmule.plan_choice import choose_setup
    from ironmule.tune import resolve_local_model

    try:
        hardware = _probe_out_of_process(probe)
        identity = resolve_local_model(spec.model_id, revision=spec.revision).identity
        setup = choose_setup(spec.model_id, hardware["fingerprint"], identity.identity_sha256,
                             log=lambda line: print(json.dumps({"agent_setup": line}), flush=True))
        if setup.get("engine") not in ("reference", "dense", "route"):
            raise ValueError(f"unknown engine {setup.get('engine')!r}")
    except Exception as exc:  # noqa: BLE001 - the agent's setup is an optimisation; the reference path is not
        setup = {"engine": "reference", "compute_dtype": None, "width": None,
                 "reason": f"the agent's setup failed ({type(exc).__name__}: {exc})"[:300]
                           + "; the reference engine serves on the checkpoint's dtype"}
    print(json.dumps({"agent_setup": {k: setup.get(k) for k in ("engine", "compute_dtype", "cpu_blas", "reason")}}),
          flush=True)
    return setup


def _fallback_setup(setup: dict[str, Any], cause: BaseException | str, compute_dtype: str | None) -> dict[str, Any]:
    """The agent's own choice cannot serve here; the reference engine serves `compute_dtype` (the agent's gated
    numeric plan, or None for the checkpoint's dtype) instead. Printed, and kept in `/health` (`fallback_from`)."""
    why = cause if isinstance(cause, str) else f"{type(cause).__name__}: {cause}"
    reason = (f"the agent chose {setup.get('engine')}, which did not start ({why})"[:300]
              + f"; the reference engine serves on {compute_dtype or 'the checkpoint dtype'}")
    print(json.dumps({"agent_setup": {"fallback": reason}}), flush=True)
    return {**setup, "engine": "reference", "compute_dtype": compute_dtype, "width": None, "reason": reason,
            "fallback_from": setup.get("fallback_from") or setup.get("engine")}


def _agent(store: ProductStore, service: Any, spec: ModelSpec, card: str | None = None) -> Any:
    """AG1: the agent bound to this hardware, model, plan and software; its learning lives in the state directory.
    On several GPUs each card gets its own (`card`, AG2), so their learning and evidence never mix."""
    from .agent import Agent
    from .readiness import hardware_identity

    backend = service.backend
    if getattr(backend, "reuses_slots", False) is True:
        raise InvalidRequest("--agent prices one group at a time; the dense plan's slot reuse is not modelled yet")
    hardware = hardware_identity()
    binding = {"hardware": {"chip_name": hardware["chip_name"], "gpu_devices": hardware["gpu_devices"]},
               "device": backend.device, "model": {"id": spec.model_id, "revision": spec.revision},
               "plan": {"execution_variant": backend.execution_variant,
                        "engine_configuration": backend.engine_configuration, "compute_dtype": backend.compute_dtype,
                        "chat_reuse": backend.chat_reuse, "thinking": backend.thinking, **_blas_binding(backend)},
               "runtime": _runtime_versions()}
    if card is not None:
        binding["card"] = card
    policy = service.policy
    capacity = service._batch_capacity()
    # what the agent's evidence was measured against: a changed schedule or capacity starts it fresh
    binding.update(capacity=capacity, schedule={"max_width": policy.max_width, "gather_ms": policy.gather_ms,
                                                "order": policy.order, "source": policy.source})
    from .hardware_profile import device_name, profile

    apple = str(hardware["chip_name"] or "").startswith("Apple")  # Apple's GPU is named by its chip
    name = (device_name(card or getattr(backend, "cuda_device", None)) if backend.device == "gpu" and not apple
            else hardware["chip_name"])
    agent = Agent(store.root / "agent", binding, capacity=capacity, device=backend.device,
                  reference={"width": capacity, "sjf": policy.order == "sjf", "gather_ms": policy.gather_ms},
                  deadline_s=service.settings["request_timeout_s"] or 120.0,
                  hardware=profile(name, spec.weight_bytes or spec.snapshot_path, backend.compute_dtype))
    print(json.dumps({"agent": {k: agent.status()[k] for k in ("serving", "reason", "traffic")}}), flush=True)
    return agent


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
    parser.add_argument("--agent", choices=("off", "on"), default="off", help="see `ironmule serve --help`")
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
        ["--learning", args.learning] if args.learning != "off" else []) + (
        ["--agent", args.agent] if args.agent != "off" else [])
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
