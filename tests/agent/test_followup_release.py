import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from lab.pk1 import followup_contract as rules, followup_release as adapter, restart_order
from training import code_author, restart_evidence
from training.code_trial import read, save


def test_followup_control_rejects_missing_gate_and_changed_author_bytes(tmp_path):
    scope = Path(__file__).resolve().parents[2] / "lab/autonomous/results/a100-restart" / rules.PARENT
    packet = read(scope / "request.json")
    source = restart_evidence.candidate_bytes((scope / "reference-source.txt").read_text(), packet, read(scope / "order.json"),
                                               read(scope / "response.json")["proposal"]).decode()
    original = tmp_path / adapter.release.SOURCE
    original.parent.mkdir()
    original.write_text(source)
    nodes = next(c.body for c in ast.parse(source).body if isinstance(c, ast.ClassDef) and c.name == "GgmlEngine")
    chosen = next(f for f in nodes if isinstance(f, ast.FunctionDef) and not f.decorator_list and adapter.sites(original)["GgmlEngine." + f.name][1])
    target = {"class": "GgmlEngine", "function": chosen.name, "path": adapter.release.SOURCE, "line": chosen.lineno}
    order = {"schema": rules.SCHEMA, "base_commit": rules.BASE, "parent_order": rules.PARENT, "control_receipt_sha256": rules.CONTROL,
             "source_sha256": adapter.PR60_SOURCE, "weights_sha256": rules.HEAD, "allowed_paths": [adapter.release.SOURCE],
             "target": target, "selected": target, "candidates": [{"target": target, "eligible": True}], "contract": rules.contract_digest()}
    order["order_id"] = restart_order.order_digest(order)
    save(tmp_path / "order.json", order)
    lines = source.splitlines(keepends=True)
    fragment = textwrap.dedent("".join(lines[chosen.lineno - 1:chosen.end_lineno]))
    state = tmp_path / "authoring"
    state.mkdir()
    with pytest.raises(code_author.PendingAuthor):
        code_author.hosted_proposal(state, order, fragment)
    rid, _ = code_author.read_request(state / "author")
    amended = fragment.splitlines(keepends=True)
    amended.insert(chosen.body[0].lineno - chosen.lineno, "    pass\n")
    proposal = {"definition": "".join(amended).strip(), "rationale": "Test-only pass statement under an unchanged declaration."}
    code_author.respond(state, rid, proposal, "local-test", "test-fixture")
    candidate = state / "candidate.py"
    projected = "".join(lines[:chosen.lineno-1]) + textwrap.indent(proposal["definition"], " " * chosen.col_offset) + "\n" + "".join(lines[chosen.end_lineno:])
    candidate.write_text(projected)
    args = SimpleNamespace(order=tmp_path / "order.json", checkout=tmp_path, trial=tmp_path / "absent",
                           candidate_source=candidate, control=scope / "control.json", weights=scope / "startup-head.npz", state=tmp_path / "publish", publish=True)
    assert adapter.register(order, tmp_path) == "qwen-followup-" + chosen.name
    assert adapter.authorship(args, order) == proposal["rationale"]
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.control(args)
    assert not args.state.exists()
    candidate.write_text(projected + "\n")
    with pytest.raises(ValueError):
        adapter.authorship(args, order)
