"""Portable CLI entry points for configuration and the local model service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

from .errors import InvalidRequest, ModelNotFound, ProductError
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
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    from .backend import MLXWorkerClient
    from .http_server import create_server
    from .service import ProductService

    store = ProductStore(args.state_dir)
    store.settings()
    backend = service = server = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        spec = store.model(args.model) if args.model else None
        if spec is not None:
            backend = MLXWorkerClient(spec)
            backend.start()
        service = ProductService(store, backend=backend, spec=spec)
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
    return 0


def dispatch(command: str, argv: list[str]) -> int:
    try:
        if command == "setup":
            return setup(argv)
        if command == "serve":
            return serve(argv)
        if command == "status":
            return status(argv)
        if command == "models" and argv:
            return models(argv[0], argv[1:])
        raise InvalidRequest("unknown product command")
    except ProductError as exc:
        print(f"ironmule: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("ironmule: local service configuration or access failed", file=sys.stderr)
        return 1
