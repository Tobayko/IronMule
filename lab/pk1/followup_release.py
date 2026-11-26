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


SAMPLING = "ironmule.qwen-calibration-sampling-order.v1"


def verified_sampling(args, order):
    """Rebuild historical prices, prospective selection and independent holdouts before accepting the exact order."""
    from training import calibration_sampling as sampling, calibration_stability as stability

    root = Path(__file__).resolve().parents[2]
    data = getattr(args, "sampling_evidence", None) or root / "lab/autonomous/results"
    history = data / "a100-restart" / rules.PARENT
    previous = data / "a100-calibration-stability"
    current = data / "a100-calibration-sampling"
    trace = root / "lab/pk1/results/pod32/raw/ab2"
    # Price the original provider quote and conservative disk bound, as in its frozen acquisition.
    forecast = stability.forecast(history, args.control, args.checkout, [trace / "trace_k1.txt", trace / "trace_k2.txt"], 2.201904761904762)
    require(forecast == read(previous / "prediction.json"), "historical phase observations or acquisition cost differ")
    reproduced = sampling.reconstruct(previous, current / "prediction.json", [current / f"{s}.json" for s in (97351, 97352)],
                                      args.control, args.weights, args.checkout)
    require(reproduced == order and reproduced["schema"] == SAMPLING, "no matching prospective full-work sampling opportunity")
    return order["order_id"]


def register(order, checkout):
    """Compatibility entry for the exclusive-time scope. A sampling order needs its protected proof."""
    return _register(order, checkout, sampling_order=None)


def _register(order, checkout, *, sampling_order):
    """Register only the exact measured PR60 method in this adapter process."""
    trial.validate_order(order)
    target = order["target"]
    name = target["function"]
    method = target["class"] + "." + name
    base = checkout / release.SOURCE
    point = sites(base).get(method)
    require(release.checksum(base) == order["source_sha256"] == PR60_SOURCE and point and point[0] == target["line"],
            "different merged PR60 site or source")
    if order["schema"] == SAMPLING:
        require(sampling_order == order["order_id"], "unverified sampling opportunity")
        adapter = "qwen-calibration-" + name
    else:
        require(point[1] is True and order["selected"] == target and order["candidates"]
                and order["candidates"][0]["target"] == target and order["candidates"][0]["eligible"] is True,
                "target differs from observed eligible source site")
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
    sampling_order = verified_sampling(args, order) if order["schema"] == SAMPLING else None
    _register(order, args.checkout, sampling_order=sampling_order)
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
    if sampling_order:
        observed = {**observed, "binding": {**observed["binding"], "verified_sampling_order": sampling_order,
                    "sampling_profile_sha256": release.checksum(args.trial / "sampling-profile.json"),
                    "sampling_parent_receipt_sha256": release.checksum(args.trial / "sampling-parent-receipt.json")}}
    return observed


def register_receipt(order, checkout, receipt_path, expected_digest):
    """A target receives a hash-pinned merged receipt from the independent control host."""
    require(release.checksum(receipt_path) == expected_digest, "unverified control receipt")
    receipt = read(receipt_path)
    require(receipt["order"] == order, "control receipt is for a different source order")
    release.validate(receipt["qualification"])
    adapter = _register(order, checkout, sampling_order=receipt["qualification"]["binding"].get("verified_sampling_order"))
    require(receipt["adapter"] == adapter, "different scoped target release")
    return adapter


def control(args):
    order = read(args.order)
    # Validate attribution before the independent scoring and Git stage, including dry runs.
    rationale = authorship(args, order)
    observed = approved(args)
    adapter = _register(order, args.checkout, sampling_order=observed["binding"].get("verified_sampling_order"))
    return release.publish(args.state, order, args.candidate_source, adapter=adapter,
                           verify=lambda: observed, rationale=rationale, submit=args.publish)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("state", "order", "checkout", "candidate-source", "trial", "control", "weights"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--sampling-evidence", type=Path, help="portable archive root containing the protected sampling and parent records")
    parser.add_argument("--publish", action="store_true")
    print(json.dumps(control(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
