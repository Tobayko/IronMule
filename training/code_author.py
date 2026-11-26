"""File transport between the development evaluator and a separate hosted Codex worker.

The host consumes author/request.json and submits the worker's reply with this module.
Replaying the evaluation reuses the response bound to those exact input bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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
    response = json.loads(path.read_text())
    proposal = checked_proposal(response["proposal"])
    if (response["schema"] != "ironmule.hosted-codex.response.v1"
            or response["request_id"] != request_id or response["proposal_sha256"] != digest(proposal)
            or not response["worker_receipt"] or not response["model"]):
        raise ValueError("hosted Codex response does not match the order and source")
    return proposal


def respond(state, expected_id, proposal, receipt, model):
    """Record a host-attributed reply. The receipt is operator metadata, not an attestation."""
    directory = state / "author"
    payload = json.loads((directory / "request.json").read_text())
    request_id = payload.pop("request_id")
    if expected_id != request_id or digest(payload) != expected_id:
        raise ValueError("request digest mismatch")
    if not receipt.strip() or not model.strip():
        raise ValueError("worker attribution is required")
    write_once(directory / "response.json", {
        "schema": "ironmule.hosted-codex.response.v1", "request_id": request_id,
        "proposal_sha256": digest(checked_proposal(proposal)), "proposal": proposal,
        "worker_receipt": receipt, "model": model,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--worker-receipt", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    respond(args.state, args.request_id, json.loads(args.proposal.read_text()),
            args.worker_receipt, args.model)


if __name__ == "__main__":
    main()
