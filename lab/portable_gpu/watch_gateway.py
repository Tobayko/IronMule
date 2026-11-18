"""Embed two read-only capability routes in the portable GPU job's bootstrap gateway."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ironmule_product.live_page import LIVE_PAGE
from ironmule_product.live_view import PROVIDER_EVENTS, benchmark

MODELS = {"qwen3-06b": "Qwen3-0.6B", "qwen3-32b": "Qwen3-32B"}
HARDWARE = re.compile(r"NVIDIA (?:[ABH]\d{2,3}|L\d{2}S?|(?:GeForce )?RTX [0-9]{4})(?:[- ](?:SXM[0-9]?|PCIe|NVL|Ada|Generation|HBM3e?|[0-9]{2,3}GB))*", re.IGNORECASE)
SHARED_PAGE = LIVE_PAGE.replace('<html lang="en">', '<html lang="en" data-watch="shared">').encode()
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def _lifecycle(root):
    try:
        with (root / "status.json").open("rb") as source:
            raw = source.read(4097)
            modified = os.fstat(source.fileno()).st_mtime
        if len(raw) <= 4096:
            status = json.loads(raw)
            if isinstance(status, dict):
                return status, datetime.fromtimestamp(modified, timezone.utc).isoformat()
    except (OSError, ValueError, RecursionError):
        pass
    return {}, None


def public_snapshot(root):
    life, stamp = _lifecycle(root)
    phase = life.get("stage")
    if not isinstance(phase, str) or phase not in PROVIDER_EVENTS:
        phase = None
    large = phase in {"fetch_qwen3_32b", "tokenizer_qwen3_32b", "compare_qwen3_32b", "finished"}
    model = "qwen3-32b" if large else "qwen3-06b"
    summary = benchmark(root / (model + ".json"))
    if phase == "finished" and summary["status"] == "idle":
        model = "qwen3-06b"
        summary = benchmark(root / (model + ".json"))
    failed = life.get("state") == "error"
    if summary["status"] == "idle":
        state = "error" if failed else "preparing" if phase and phase != "finished" else "unavailable"
        summary.update(status=state, stage="preflight" if state == "preparing" else None, updated_at=stamp)
    elif failed and summary["status"] == "running" or phase == "finished" and summary["status"] == "running":
        summary.update(status="error", verdict="incomplete")
    # The capability exposes only a product name, never a local model path, host ID or accelerator UUID.
    device = summary.get("device")
    if not isinstance(device, str) or not HARDWARE.fullmatch(device):
        try:
            with (root / "device.txt").open("rb") as source:
                device = source.read(256).decode("ascii").split(",", 1)[0].strip()
        except (OSError, UnicodeError):
            device = ""
    summary.update(model=MODELS[model] if phase or summary.get("model") else None,
                   device=device if HARDWARE.fullmatch(device) else None,
                   provider={"name": "Runpod", "event": PROVIDER_EVENTS.get(phase), "updated_at": stamp})
    return {"schema": "ironmule.live.v1", "observed_at": time.time(), "runtime": {"status": "detached", "samples": []},
            "benchmark": summary}


def watch_routes(handler, root, watch_id, result_token):
    """Return True only for a configured independent capability's exact GET paths."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{32,80}", watch_id) or secrets.compare_digest(watch_id, result_token):
        return False
    url = f"/watch/{watch_id}/"
    if handler.path not in (url, url + "data"):
        return False
    if handler.path == url:
        content_type, body = "text/html; charset=utf-8", SHARED_PAGE
    else:
        content_type, body = "application/json", json.dumps(public_snapshot(root), allow_nan=False).encode()
    handler.send_response(200)
    for header, value in (("Content-Type", content_type), ("Content-Length", str(len(body))),
                          ("Cache-Control", "private, no-store"), ("Content-Security-Policy", CSP),
                          ("Referrer-Policy", "no-referrer"), ("X-Robots-Tag", "noindex, nofollow, noarchive"),
                          ("X-Content-Type-Options", "nosniff")):
        handler.send_header(header, value)
    handler.end_headers()
    handler.wfile.write(body)
    return True


def render(template):
    """Build one bootstrap script; the temporary zip supplies only stdlib-dependent monitor sources."""
    here = Path(__file__).resolve()
    source = here.parents[2]
    archive = io.BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        bundle.writestr("ironmule_product/__init__.py", "")
        for name in ("live_view", "live_page"):
            bundle.write(source / "ironmule_product" / (name + ".py"), f"ironmule_product/{name}.py")
        bundle.write(here, "watch_gateway.py")
    encoded = base64.b64encode(archive.getvalue()).decode("ascii")
    setup = ("import base64\nimport sys\nfrom tempfile import TemporaryDirectory\n"
             "_watch_dir = TemporaryDirectory()\n"
             "_watch_zip = pathlib.Path(_watch_dir.name) / 'watch.zip'\n"
             f"_watch_zip.write_bytes(base64.b64decode({encoded!r}))\n"
             "sys.path.insert(0, str(_watch_zip))\n"
             "from watch_gateway import watch_routes\n"
             "_watch_id = os.environ.get('WATCH_ID', '')\n\n")
    marker, get = "class Handler(", "    def do_GET(self):\n"
    if template.count(marker) != 1 or template.count(get) != 1:
        raise ValueError("bootstrap gateway must contain one Handler and one do_GET")
    return template.replace(marker, setup + marker).replace(
        get, get + "        if watch_routes(self, ROOT, _watch_id, TOKEN):\n            return\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.template.read_text()))


if __name__ == "__main__":
    main()
