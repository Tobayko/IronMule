"""File transport between the development evaluator and a separate hosted Codex worker.

The host consumes author/request.json and submits the worker's reply with this module.
Replaying the evaluation reuses the response bound to those exact input bytes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


class PendingAuthor(RuntimeError):
    """A persisted request is waiting for the host to return a result."""


def digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def write_once(path, data):
    """A crash exposes either a complete JSON document or an empty destination."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                if json.loads(path.read_text()) != data:
                    raise ValueError(f"conflicting record: {path.name}") from None
            folder = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
        finally:
            temp.unlink(missing_ok=True)


def checked_proposal(proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"definition", "rationale"}:
        raise ValueError("expected exactly a function definition and rationale")
    for value in proposal.values():
        if not isinstance(value, str) or not value.strip() or len(value.encode()) > 128 * 1024:
            raise ValueError("empty or oversized proposal field")
    return proposal


def read_request(directory: Path):
    request = json.loads((directory / "request.json").read_text())
    request_id = request.pop("request_id")
    if request.get("schema") != "ironmule.hosted-codex.request.v1" or digest(request) != request_id:
        raise ValueError("request digest or schema mismatch")
    return request_id, request


def read_response(directory: Path, request_id: str):
    response = json.loads((directory / "response.json").read_text())
    proposal = checked_proposal(response["proposal"])
    if (response["schema"] != "ironmule.hosted-codex.response.v1" or response["request_id"] != request_id
            or response["proposal_sha256"] != digest(proposal) or not response["worker_receipt"] or not response["model"]):
        raise ValueError("hosted Codex response does not match the order and source")
    return proposal


def hosted_proposal(state: Path, order: dict, source_text: str, *, knowledge=None) -> dict:
    """Publish once and return a checked reply; the caller may retry after PendingAuthor."""
    directory = state / "author"
    directory.mkdir(mode=0o700, exist_ok=True)
    payload = {"schema": "ironmule.hosted-codex.request.v1", "order": order, "source": source_text,
               "instructions": "Return a dedented replacement definition and a rationale. Preserve the "
               "function signature, behavior and request isolation. The editor preserves decorators. "
               "Change only the specified function. Leave measurement, quality and budget gates intact."}
    if knowledge is not None:
        payload["knowledge"] = knowledge
    request_id = digest(payload)
    write_once(directory / "request.json", {"request_id": request_id, **payload})
    path = directory / "response.json"
    if not path.exists():
        raise PendingAuthor(f"hosted Codex request {request_id} is waiting at {directory / 'request.json'}")
    return read_response(directory, request_id)


def respond(state, expected_id, proposal, receipt, model):
    """Record a host-attributed reply. The receipt is operator metadata, not an attestation."""
    directory = state / "author"
    request_id, _ = read_request(directory)
    if expected_id != request_id:
        raise ValueError("request digest mismatch")
    if not receipt.strip() or not model.strip():
        raise ValueError("worker attribution is required")
    write_once(directory / "response.json", {
        "schema": "ironmule.hosted-codex.response.v1", "request_id": request_id,
        "proposal_sha256": digest(checked_proposal(proposal)), "proposal": proposal,
        "worker_receipt": receipt, "model": model,
    })


def consume_cli(state: Path, *, retry_preflight=False):
    """Consume one immutable request on an included ChatGPT subscription, resuming receipts without a second call."""
    state = Path(state).resolve()
    directory = state / "author"
    request_id, request = read_request(directory)
    work = directory / "cli"
    work.mkdir(mode=0o700, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "response.json").exists():
            return read_response(directory, request_id)
        done = work / "completed.json"
        if not done.exists():
            dispatch = work / "dispatch.json"
            if dispatch.exists():
                raise RuntimeError("CLI dispatch already attempted; no successful exit receipt is available")
            attempts = list(work.glob("preflight-*.json"))
            if attempts and not retry_preflight:
                raise RuntimeError("CLI preflight already attempted; inspect the login and explicitly retry preflight")
            write_once(work / f"preflight-{len(attempts) + 1}.json", {"request_id": request_id, "time": time.time()})
            env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")}

            def invoke(argv, **kwargs):
                return subprocess.run(["codex", *argv], text=True, check=True, cwd=work, env=env,
                                      stdout=subprocess.PIPE, **kwargs).stdout.strip()

            auth = invoke(["login", "status"], timeout=30, stderr=subprocess.STDOUT)
            if "using ChatGPT" not in auth:
                raise RuntimeError("included ChatGPT login required for unattended authoring")
            catalog = json.loads(invoke(["debug", "models"], timeout=30))
            models = [m["slug"] for m in catalog["models"] if m.get("visibility") == "list"]
            preferred = ("gpt-6.2-sol", "gpt-6.1-sol", "gpt-6-sol", "gpt-6.1-terra", "gpt-5.6-sol")
            model = next((slug for slug in preferred if slug in models), None)
            if model is None:
                raise RuntimeError("no supported Codex coding model in the signed-in account catalog")
            schema = {"type": "object", "properties": {"definition": {"type": "string"},
                      "rationale": {"type": "string"}}, "required": ["definition", "rationale"],
                      "additionalProperties": False}
            write_once(work / "schema.json", schema)
            native = request["order"]["target"]["path"].endswith((".cu", ".cuh", ".cc", ".cpp"))
            settings = ["-c", "forced_login_method=chatgpt"]
            if native:
                settings += ["-c", "model_reasoning_effort=high"]
            prompt = ("Complete the following IronMule development order using its included source. Return one "
                      "replacement definition in the source language, and its rationale. Keep the signature, "
                      "behavior, workload, and evaluator fixed. Preserve the original if no valid optimization "
                      "is available. Use only the provided material and return the requested JSON.\n" + json.dumps(request))
            write_once(dispatch, {"request_id": request_id, "order_id": request["order"].get("order_id"),
                                  "model": model, "catalog": models, "time": time.time(), "payer": "ChatGPT session"})
            with (work / "stderr.log").open("w") as log:
                invoke(["exec", "--ignore-user-config", "--ephemeral", "-s", "read-only", *settings, "-m", model,
                        "--skip-git-repo-check", "--output-schema", str(work / "schema.json"),
                        "-o", str(work / "reply.tmp"), "-"], input=prompt, timeout=480, stderr=log)
            proposal = checked_proposal(json.loads((work / "reply.tmp").read_text()))
            write_once(done, {"request_id": request_id, "model": model, "exit_code": 0, "proposal": proposal})
        success = json.loads(done.read_text())
        sent = json.loads((work / "dispatch.json").read_text())
        if (success["request_id"] != request_id or sent["request_id"] != request_id or success["exit_code"] != 0
                or sent["model"] != success["model"]):
            raise ValueError("successful CLI exit is not bound to this order and model")
        respond(state, request_id, success["proposal"], "codex-cli/" + digest(success), success["model"])
        return read_response(directory, request_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--consume-cli", action="store_true", help="answer one pending request using the logged-in subscription")
    parser.add_argument("--retry-preflight", action="store_true", help="retry a preflight that never dispatched")
    parser.add_argument("--request-id")
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--worker-receipt")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.consume_cli:
        if any((args.request_id, args.proposal, args.worker_receipt, args.model)):
            parser.error("choose one of CLI consumption and hosted reply submission")
        consume_cli(args.state, retry_preflight=args.retry_preflight)
        return
    if args.retry_preflight or not all((args.request_id, args.proposal, args.worker_receipt, args.model)):
        parser.error("a hosted reply requires --request-id, --proposal, --worker-receipt and --model")
    respond(args.state, args.request_id, json.loads(args.proposal.read_text()),
            args.worker_receipt, args.model)


if __name__ == "__main__":
    main()
