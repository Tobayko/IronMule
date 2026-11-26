"""Rank on-device startup experiments from the two ordinary PR57 server traces."""
import argparse
import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path

from lab.pk6 import warm_result
from training.code_author import hosted_proposal, read_request, read_response, write_once, PendingAuthor
from training.code_trial import git, read
from training.experiment_planner import digest

METHOD_PATH = "ironmule/ggml_engine.py"
ROOT = Path(__file__).resolve().parents[2]
STAGE = re.compile(r"^# init (\w+)(?: begin)? ([\d.]+)$")


def last_init(trace):
    lines = trace.read_text().splitlines()
    start = max(i for i, line in enumerate(lines) if line.startswith("# init warm_up begin "))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("# init result "))
    marks = [(i, m[1], float(m[2])) for i in range(start, end)
             if (m := STAGE.fullmatch(lines[i]))]
    phases = {}
    for (i, name, t0), (j, _, t1) in zip(marks, marks[1:]):
        phases[name] = {"elapsed_s": t1 - t0, "calls": [
            [float(cols[1]), int(cols[2]), int(cols[3])] for line in lines[i+1:j]
            if line and not line.startswith("#") and (cols := line.split())]}
    return {"phases": phases, "result": json.loads(lines[end].removeprefix("# init result ")),
            "trace_sha256": warm_result.checksum(trace)}


def source_methods(code):
    engine, = (n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == "GgmlEngine")
    return {n.name: n for n in engine.body if isinstance(n, ast.FunctionDef)}


def rank(directory, measurements, source):
    verified = warm_result.result(directory)
    if verified["status"] != "complete":
        raise ValueError("need a complete correct external reference series")
    by_start = verified["by_start"]
    methods = source_methods(source)
    samples = [read(measurements / f"init-{label}.json") for label in ("K1", "K2")]
    candidates = []
    for name in sorted(set.intersection(*(set(s["phases"]) for s in samples))):
        if name not in methods:
            continue
        rows = [s["phases"][name] for s in samples]
        times = [r["elapsed_s"] for r in rows]
        # A sensitivity model for measurement loops. Retain a full warmup and at least
        # two paid observations for every complete profile; never remove a quality check.
        default = next((v.value for key, v in zip(reversed(methods[name].args.args), reversed(methods[name].args.defaults))
                        if key.arg == "repeats" and isinstance(v, ast.Constant)), None)
        saving = max(0, (min(times) * (default - 2) / (1 + default)) - abs(times[0]-times[1])) if default and default >= 3 else 0.
        geometry = [[(r[1], r[2]) for r in phase["calls"]] for phase in rows]
        repeated = bool(geometry[0] and geometry[0] == geometry[1])
        prior = {"measure_kg": "existing device-mask replay release PR60",
                 "measure_padding": "terminal rejected same-family source order 4e87c90d",
                 "measure_pass_costs": "varying full price vectors and observed post-serving drift"}.get(name)
        candidates.append({"method": name, "line": methods[name].lineno,
                           "elapsed_s": times, "calls": [len(r["calls"]) for r in rows],
                           "same_call_geometry": repeated, "original_repeats": default,
                           "conditional_saving_s": saving, "prior": prior,
                           "eligible": repeated and prior is None and saving >= .02 * by_start["K2"]["startup_s"]})
    candidates.sort(key=lambda x: (-int(x["eligible"]), -x["conditional_saving_s"], x["method"]))
    selected = next((r for r in candidates if r["eligible"]), None)
    return {"schema": "ironmule.dynamic-calibration-search.v1", "kind": "simulated",
            "base_commit": verified["source_commit"], "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "hardware": verified["hardware"], "gguf": verified["gguf_sha256"],
            "inputs": {tag: warm_result.checksum(measurements / f"init-{tag}.json") for tag in ("K1", "K2")},
            "measured_http_sha256": verified["source_files"], "candidates": candidates,
            "selected": selected, "uncertainty": "Two complete starts on one machine. Sensitivity is a projection. Prefix measurements can select a different winner; prior sampling failed at full-worker scale."}


def commission(directory, observations, out):
    source = git("show", warm_result.SOURCE + ":" + METHOD_PATH, cwd=ROOT) + "\n"
    forecast = rank(directory, observations, source)
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "forecast.json", forecast)
    winner = forecast["selected"]
    if not winner:
        return {"status": "deferred"}
    name = winner["method"]
    node = source_methods(source)[name]
    # Send full baseline method and the original data; the author supplies the implementation.
    code = ast.get_source_segment(source, node)
    order = {"schema": "ironmule.local-calibration-source-order.v1", "base_commit": forecast["base_commit"],
             "allowed_paths": [METHOD_PATH], "target": {"path": METHOD_PATH, "function": "GgmlEngine."+name, "line": node.lineno},
             "hypothesis": "Reduce the cost of this measured calibration on repeated fresh ordinary target starts with a guarded local decision. Preserve the complete quality checks and competing execution choices.",
             "origin": digest(forecast), "source_sha256": forecast["source_sha256"],
             "scope": "An experiment-specific explicit opt-in only, named IRONMULE_TRIAL_LOCAL_CALIBRATION. Outside it preserve the full original behavior, including alternate input arguments and unsupported geometry.",
             "constraints": ["Use only locally timed complete alternatives. Do not import or trust measurements from another host, hardcode 128 as the decision, skip NLL or alter numerical acceptance thresholds.",
                             "Retain all result fields, full original measurements whenever uncertain, hardware setter and cleanup semantics. No new imports, files, I/O, planner or evaluator changes.",
                             "Limit the change to the supplied method. Use a conservative rule that can decline. Aim to save the cost represented in the two source traces."],
             "measurements": [{k:v for k,v in read(observations / f"init-{t}.json").items() if k == "result"} for t in ("K1","K2")],
             "forecast": winner, "prior_rejection": "A separate early-sampling change in measure_padding cut native calls but lost on complete workers. No source reattempt for that site."}
    order["order_id"] = digest(order)
    write_once(out / "order.json", order)
    try:
        hosted_proposal(out, order, code)
    except PendingAuthor:
        return {"status": "requested", "request": str(out/"author/request.json")}
    return {"status": "answered", "order": order["order_id"]}


def materialize(folder):
    """Rebuild the exact, attributed source artifact from the immutable Git parent and reply."""
    order = read(folder / "order.json")
    request_id, request = read_request(folder / "author")
    source = git("show", order["base_commit"] + ":" + METHOD_PATH, cwd=ROOT) + "\n"
    if (order["base_commit"] != warm_result.SOURCE or order != request["order"]
            or hashlib.sha256(source.encode()).hexdigest() != order["source_sha256"]
            or digest({k:v for k,v in order.items() if k != "order_id"}) != order["order_id"]):
        raise ValueError("different order, baseline or source")
    name = order["target"]["function"].split(".")[-1]
    node = source_methods(source)[name]
    proposal = read_response(folder / "author", request_id)
    replacement = textwrap.dedent(proposal["definition"]).strip()
    generated, = ast.parse(replacement).body
    if (not isinstance(generated, ast.FunctionDef) or generated.name != name or generated.decorator_list
            or ast.dump(generated.args) != ast.dump(node.args) or node.lineno != order["target"]["line"]):
        raise ValueError("generated source is outside the ordered method")
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [textwrap.indent(replacement," " * node.col_offset) + "\n"]
    result = "".join(lines)
    compile(result,METHOD_PATH,"exec")
    return result.encode()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    project = sub.add_parser("project")
    project.add_argument("trace", type=Path)
    project.add_argument("out", type=Path)
    run = sub.add_parser("commission")
    for key in ("directory", "observations", "out"):
        run.add_argument(key, type=Path)
    build = sub.add_parser("materialize")
    build.add_argument("folder", type=Path)
    build.add_argument("output", type=Path)
    a = p.parse_args()
    if a.cmd == "project":
        write_once(a.out, last_init(a.trace))
    elif a.cmd == "materialize":
        with a.output.open("xb") as stream:
            stream.write(materialize(a.folder))
    else:
        print(json.dumps(commission(a.directory, a.observations, a.out)))
