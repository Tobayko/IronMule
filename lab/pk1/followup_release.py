"""Publish an attributed single-method improvement over the actual merged PR60 Qwen runtime."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import textwrap

from lab.pk1 import followup_contract as rules
from lab.pk1 import followup_trial as trial
from lab.pk1.post_restart_profile import sites
from training import scoped_release as release
from training.code_author import read_request, read_response
from training.code_trial import read
from training.trial_release import require


PR60_SOURCE = "80a41c59bfe8088b305529cca5164134c25d4b5a0ce9572877cb25a2b7d269fd"


def register(order, checkout):
    """Register only the exact measured PR60 method in this adapter process."""
    trial.validate_order(order)
    target = order["target"]
    name = target["function"]
    method = target["class"] + "." + name
    base = checkout / release.SOURCE
    point = sites(base).get(method)
    require(release.checksum(base) == order["source_sha256"] == PR60_SOURCE and point and point[:2] == (target["line"], True)
            and order["selected"] == target and order["candidates"] and order["candidates"][0]["target"] == target
            and order["candidates"][0]["eligible"] is True, "target differs from the observed, eligible source site")
    adapter = "qwen-followup-" + name
    spec = (rules.BASE, method)
    require(release.ADAPTERS.setdefault(adapter, spec) == spec, "conflicting follow-up method registration")
    release.validate_order(adapter, order)
    return adapter


def authorship(args, order):
    src = args.checkout / release.SOURCE
    candidate = args.candidate_source
    trial.source_check(src, candidate, order)
    inbox = candidate.parent / "author"
    request_id, request = read_request(inbox)
    answer = read_response(inbox, request_id)
    text = src.read_text()
    lines = text.splitlines(keepends=True)
    target = order["target"]
    klass, = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == target["class"]]
    node, = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == target["function"]]
    fragment = textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))
    require(request["order"] == order and request["source"] == fragment, "author was given different source or order")
    replacement = textwrap.dedent(answer["definition"]).strip()
    edited, = ast.parse(replacement).body
    require(isinstance(edited, ast.FunctionDef) and not edited.decorator_list and edited.name == node.name,
            "different or decorated replacement")
    before, after = ast.dump(node), ast.dump(edited)
    # Compare the whole function declaration, including return annotation and type parameters.
    node.body = edited.body
    require(ast.dump(node) == after and before != after, "changed declaration or identical proposal")
    expected = "".join(lines[:node.lineno - 1]) + textwrap.indent(replacement, " " * node.col_offset) + "\n" + "".join(lines[node.end_lineno:])
    require(expected.encode() == candidate.read_bytes(), "candidate differs from authored function")
    compile(expected, release.SOURCE, "exec")
    return answer["rationale"]


def approved(args):
    order = read(args.order)
    register(order, args.checkout)
    require(release.checksum(args.control) == rules.CONTROL and release.checksum(args.weights) == rules.HEAD,
            "different parent release or trained policy")
    parent = read(args.control)
    require(parent["publication"]["pr"] == 60 and parent["publication"]["candidate"] == rules.BASE
            and parent["order"]["order_id"] == rules.PARENT and parent["qualification"]["candidate_sha256"] == order["source_sha256"],
            "parent source differs from the actual merged and qualified PR")
    observed = trial.qualification(args)
    release.validate(observed)
    require(observed["reference_sha256"] == parent["qualification"]["candidate_sha256"]
            and observed["candidate_sha256"] == release.checksum(args.candidate_source), "qualified follow-up source differs")
    return observed


def control(args):
    order = read(args.order)
    adapter = register(order, args.checkout)
    rationale = authorship(args, order)
    return release.publish(args.state, order, args.candidate_source, adapter=adapter,
                           verify=lambda: approved(args), rationale=rationale, submit=args.publish)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("state", "order", "checkout", "candidate-source", "trial", "control", "weights"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    print(json.dumps(control(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
