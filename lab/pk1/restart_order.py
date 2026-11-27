"""Propose a startup experiment from repeated measured phase traces and two local mask observations."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tarfile
from itertools import pairwise
from pathlib import Path

from lab.pk1.native_trial import FLAGS, checksum
from lab.portable_gpu.bench import save
from training.code_author import hosted_proposal, write_once

SCHEMA = "ironmule.qwen-restart.v1"
PATH = "ironmule/ggml_engine.py"
ARCHIVE = "lab/pk1/results/pod37/pk6_all.tgz"
PHASE = re.compile(r"^# init ([a-z_0-9]+) begin ([\d.]+)$")


def read(path):
    return json.loads(path.read_bytes())


def order_digest(order):
    return hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode()).hexdigest()


def site(text):
    klass = next(s for s in ast.parse(text).body if isinstance(s, ast.ClassDef) and s.name == "GgmlEngine")
    return next(s for s in klass.body if isinstance(s, ast.FunctionDef) and s.name == "__init__")


def build(args):
    base, earlier = map(read, [args.canonical, args.previous])
    for order, path in [(base, args.canonical), (earlier, args.previous)]:
        if order["order_id"] != order_digest(order) or checksum(path.with_suffix(".profile.json")) != order["profile_sha256"]:
            raise ValueError("unbound local CUDA observation")
    b = base["binding"]
    if (base["profile_sha256"] == earlier["profile_sha256"] or any(earlier["binding"][k] != b[k] for k in
            ("gguf", "native", "fp", "table_sha256", "w4_sha256", "serving_sources", "tuner", "engine"))
            or any(b["mask"][k] != earlier["binding"]["mask"][k] for k in ("tokens_served", "q6k_tokens_served"))):
        raise ValueError("two matching real local startup choices required")
    if checksum(args.checkout / PATH) != b["serving_sources"][PATH]:
        raise ValueError("measured runtime source differs")
    raw = args.checkout / ARCHIVE
    phases = {}
    with tarfile.open(raw) as bundle:
        for arm in ("K1", "K2"):
            points = [m.groups() for line in bundle.extractfile(f"./trace_{arm}.txt").read().decode().splitlines()
                      if (m := PHASE.fullmatch(line))]
            for (name, begin), (_, end) in pairwise(points):
                phases.setdefault(name, []).append(float(end) - float(begin))
    # Only phases with an observed, comparable retained result and a usable replay mechanism can be proposed.
    eligible = {"measure_kg": "two matching on-device mask profiles", "measure_pk2": None}
    flags = read(args.canonical.with_suffix(".profile.json"))["reference_flags"]
    if any(flags.get(k) != FLAGS[k] for k in ("PS1", "PS1_WARM", "PK6", "GGML_PACK", "GGML_UBATCH", "CUDA_GRAPHS_ROWS")):
        raise ValueError("capture used a different serving bundle")
    choices = sorted((min(values), name) for name, values in phases.items() if len(values) == 2 and eligible.get(name))
    if not choices or choices[-1][0] <= 1:
        raise ValueError("insufficient historical phase time to merit a hardware experiment")
    selected = choices[-1][1]
    node = site((args.checkout / PATH).read_text())
    order = {"schema": SCHEMA, "evaluator": SCHEMA, "base_commit": base["base_commit"],
             "target": {"path": PATH, "class": "GgmlEngine", "function": "__init__", "line": node.lineno},
             "allowed_paths": [PATH], "binding": b, "source_sha256": checksum(args.checkout / PATH),
             "evidence": {"historical_artifact": ARCHIVE, "historical_sha256": checksum(raw),
                          "historical_startup_seconds": phases, "archive_arms": ["K1", "K2"],
                          "local_profile_ids": [earlier["order_id"], base["order_id"]],
                          "local_mask_measurements": [earlier["binding"]["mask"], b["mask"]],
                          "resolved_source_phase": selected, "manual_prefill_threshold": 128,
                          "eligible": eligible, "provenance": "Historical timings are from two earlier A100s starts. "
                              "These local mask summaries are fresh. Current cold-start and serving gains are unmeasured."},
             "hypothesis": f"Reduce repeated {selected} time on a cold engine restart of an already qualified exact "
                 "GPU/model/profile, while keeping ordinary serving and every missing/stale receipt fallback intact.",
             "contract": {"cpu_source": "replace only the named GgmlEngine.__init__ definition; same signature and fallbacks",
                          "cold": "fresh restricted process to completion of first fixed 32-client, 32-token mixed wave",
                          "warm": "second mixed wave and 18 factual checks on the same engine",
                          "gate": "two A/A observations then six alternating pairs, matching outputs/work, all cold wins, "
                                  "median cold gain >=2% and exceeds observed A/A variation, warm median time and p95 TTFT "
                                  "ratios <=1.02, host and device memory <=1.05x; signed stale-profile negative must fall back",
                          "scope": "repeat verified deployment startup only; no claim of faster warm generation"},
             "receipt_protocol": {
                 "path_env": "IRONMULE_KG_RESTART_RECEIPT", "sha256_env": "IRONMULE_KG_RESTART_SHA256",
                 "schema": "ironmule.w4-restart-qualified.v1",
                 "fields": "schema, binding, mask, created_s, quality. binding has model_sha256, model_stat (list of "
                     "st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns from the verified mounted GGUF), native "
                     "(ggml_binary.identity()), fp (kernel_gen.fingerprint()), w4_sha256, table_sha256, settings "
                     "(self.settings), seqs, seq_ctx, threads, max_rows, speculate, gpu_layers, affinity (sorted list), "
                     "flags (exact effective full native execution flag dict), and measure_kg_sha256 "
                     "(sha256(inspect.getsource(GgmlEngine.measure_kg).encode())). Native flags bound by name in "
                     "binding.flags: " + ", ".join("IRONMULE_" + k for k in FLAGS) + ".",
                 "quality": "quality.completed == 32, quality.facts == 18, quality.output_sha256 is 64 hex. "
                     "mask is the original complete GgmlEngine.measure_kg() dict including both admitted width lists. "
                     "Both outputs are already qualified in an unmodified reference on this machine. Only accept age "
                     "0..86400 seconds, a bound external SHA, valid values, equal model stat or independently rehash the "
                     "GGUF, identical CUDA/native/build/env/geometry and W4+table bytes. Binding is from reference, not "
                     "from candidate. The trusted broker selects/pins the receipt. Never trust arbitrary caller files.",
                 "fallback": "On missing/malformed/stale/nonmatching receipt, width/mask setter/getter discrepancy, "
                     "disabled kernel_mask, KG4 explicit scope, or any missing identity, retain normal upstream behavior. "
                     "No partial mask installation: restore original registers before measuring if application failed. "
                     "Successful replay must set/read back both mask registers and restore the runtime state the measured "
                     "path normally leaves. Emit self.kg['restart_profile_reused']=True ONLY on positive replay; do not "
                     "change other performance, quality, calibration or budget criteria. The worker has no writable store."}}
    order["order_id"] = order_digest(order)
    save(args.output, order)


def author(orderfile, checkout, destination):
    order = read(orderfile)
    path = checkout / PATH
    if order["order_id"] != order_digest(order) or order["schema"] != SCHEMA or checksum(path) != order["source_sha256"]:
        raise ValueError("different source or order")
    source = path.read_text()
    node = site(source)
    import textwrap
    original = ast.get_source_segment(source, node)
    lines = source.splitlines(keepends=True)[node.lineno-1:node.end_lineno]
    fragment = textwrap.dedent("".join(lines))
    destination.parent.mkdir(parents=True, exist_ok=True)
    answer = hosted_proposal(destination.parent, order, fragment, knowledge={
        "calibration": ast.get_source_segment(source, next(n for n in ast.walk(ast.parse(source))
                                   if isinstance(n, ast.FunctionDef) and n.name == "measure_kg")),
        "source": PATH, "source_sha256": checksum(path)})
    edited = textwrap.dedent(answer["definition"]).strip()
    body = ast.parse(edited).body
    if (len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != node.name or body[0].decorator_list
            or ast.dump(body[0].args) != ast.dump(node.args) or ast.dump(body[0]) == ast.dump(ast.parse(fragment).body[0])):
        raise ValueError("changed signature or no candidate")
    all_lines = source.splitlines(keepends=True)
    candidate = "".join(all_lines[:node.lineno-1]) + textwrap.indent(edited, " " * node.col_offset) + "\n" + "".join(all_lines[node.end_lineno:])
    compile(candidate, PATH, "exec")
    if not original or destination.exists() and destination.read_text() != candidate:
        raise ValueError("conflicting source experiment")
    destination.write_text(candidate)
    write_once(destination.parent / "authorship.json", {"order": order["order_id"], "source_sha256": checksum(destination),
               "rationale": answer["rationale"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    new = sub.add_parser("plan")
    for key in ("canonical", "previous"):
        new.add_argument("--" + key, type=Path, required=True)
    code = sub.add_parser("author")
    code.add_argument("--order", type=Path, required=True)
    for cmd in (new, code):
        cmd.add_argument("--checkout", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "plan":
        build(a)
    else:
        author(a.order, a.checkout, a.output)


if __name__ == "__main__":
    main()
