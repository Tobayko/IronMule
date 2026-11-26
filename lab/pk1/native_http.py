"""Qualify and serve a published width-32 W4 variant through the product HTTP service on its bound target."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

from ironmule.ggml_profiles import FLAGS
from ironmule_product import compare
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import InvalidRequest
from ironmule_product.http_server import create_server
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec
from lab.pk1 import native_release
from lab.pk1 import native_trial as trial
from lab.rp.restricted_worker import open_worker
from training.code_trial import PAIRS, judge, read, runtime_env, save
from training.trial_release import deployment, require

WORKER = Path(__file__).with_name("native_http_worker.py")
SEEDS = tuple(70441 + n for n in range(PAIRS))
GENERATIONS = (32, 8, 1)
PROMPT_BYTES = 1536


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def locked_receipt(state):
    with deployment(state) as d:
        pin = read(d / "target.json")
        content = Path(pin["path"]).read_bytes()
        require(hashlib.sha256(content).hexdigest() == pin["sha256"], "published native control receipt changed")
        receipt = json.loads(content)
        pub, selected = receipt["publication"], read(d / "selection.json")
        require(receipt["qualification"]["passed"] and pub["order_id"] == receipt["order"]["order_id"]
                and pub["base"] == receipt["order"]["base_commit"] and type(pub["pr"]) is int and pub["pr"] > 0,
                "missing merged native qualification")
        return receipt, pin["sha256"], selected["arm"] == "cand" and not selected["withdrawn"]


def code_binding(args, receipt_sha):
    # The two Python checkouts may differ (original engine, newer protected control host).
    # Bind their actual product code, including scheduler and protocol, in addition to native target_files.
    root = Path(__file__).resolve().parents[2]
    sources = {name: {str(p.relative_to(base)): trial.checksum(p) for p in sorted((base / "ironmule_product").glob("*.py"))}
               for name, base in [("broker", root), ("worker", args.checkout)]}
    return digest({"receipt": receipt_sha, "product": sources, "adapter": trial.checksum(__file__),
                   "entry": trial.checksum(WORKER), "suite": SEEDS, "widths": GENERATIONS, "max_tokens": 32,
                   "broker_python": sys.version, "broker_affinity": sorted(os.sched_getaffinity(0)),
                   "helpers": {path: trial.checksum(root / path) for path in ("training/code_trial.py",
                               "lab/rp/restricted_worker.py", "ironmule/factual_gate.py", "lab/pk1/native_release.py")}})


def env_for(args, binary, table):
    env = {k: v for k, v in runtime_env("cuda").items() if k not in ("HOME", "TMPDIR")}
    return {**env, **{"IRONMULE_" + k: v for k, v in trial.FLAGS.items()},
            "PYTHONPATH": str(args.checkout), "IRONMULE_KG_LIB": str(binary), "IRONMULE_W4_TABLE": str(table),
            "IRONMULE_GGML_SEQS": "32", "IRONMULE_GGML_SEQ_CTX": "2048", "IRONMULE_ADMISSION_LEARN": "0",
            "IRONMULE_PILOT_LEARN": "0"}


@contextmanager
def isolated_inputs(args, receipt, cand):
    binary, expected = native_release.target_files(args, receipt, cand)
    with tempfile.TemporaryDirectory(dir=args.worker_root, prefix="http-") as tmp:
        stage = Path(tmp)
        stage.chmod(0o755)
        inputs, writable = stage / "inputs", stage / "out"
        inputs.mkdir(mode=0o755)
        inputs.chmod(0o755)
        writable.mkdir(mode=0o700)
        os.chown(writable, args.worker_uid, args.worker_uid)
        lib, table = inputs / "active.so", inputs / "dispatch.table"
        profile = inputs / "execution.json"
        profile.write_text(json.dumps({"binding": receipt["binding"], "flags": list(FLAGS), "binary": expected,
                                      "table": receipt["order"]["binding"]["table_sha256"]}, allow_nan=False))
        for origin, dest in [(binary, lib), (args.table, table), (WORKER, inputs / WORKER.name),
                             (Path(trial.__file__), inputs / "native_trial.py")]:
            shutil.copyfile(origin, dest)
            dest.chmod(0o444)
        profile.chmod(0o444)
        require(trial.checksum(lib) == expected and trial.checksum(table) == receipt["order"]["binding"]["table_sha256"],
                "changed immutable HTTP input")
        env = env_for(args, lib, table)
        # Python lists its search directory to locate packages. Expose a minimal source view,
        # without granting access to the baseline checkout's Git configuration or other files.
        source_view = inputs / "source"
        source_view.mkdir(mode=0o755)
        source_view.chmod(0o755)
        sources = ("ironmule", "ironmule_product", "friday_evidence", "ironmule_devices.py",
                   "ironmule_inventory.py", "ironmule_controller.py", "ironmule_cli.py")
        for name in sources:
            (source_view / name).symlink_to(args.checkout / name)
        env["PYTHONPATH"] = str(source_view)
        paths = [inputs, args.gguf, *(args.checkout / name for name in sources),
                 *(Path(env[k]).resolve() for k in ("IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH"))]
        for path in [Path(sys.prefix), Path(sys.base_prefix), *(Path(x) for x in env.get("LD_LIBRARY_PATH", "").split(":") if x)]:
            real = path.resolve(strict=True)
            if not real.is_relative_to("/usr") and real not in paths:
                paths.append(real)
        yield SimpleNamespace(lib=lib, binary=expected, readonly=paths, env=env, args=args, out=writable,
                              command=[str(Path(sys.executable).resolve()), str(inputs / WORKER.name)],
                              profile=profile)


def launch(layout, mode, tail, timeout):
    timeout = min(timeout, int(layout.args.deadline - time.monotonic()))
    if timeout < 1:
        raise TimeoutError("HTTP campaign deadline expired")
    return open_worker([*layout.command, mode, str(layout.args.gguf), str(layout.profile), *tail],
                       env=layout.env, cwd=layout.out, output=layout.out, readonly=layout.readonly,
                       timeout=timeout, uid=layout.args.worker_uid, gid=layout.args.worker_uid,
                       text=False, stderr=subprocess.PIPE if mode == "serve" else None)


def product_gate(layout):
    # Only the isolated child maps this arm's native library, including during calibration.
    output = layout.out / "quality.json"
    with launch(layout, "gate", [str(output)], layout.args.timeout) as proc:
        proc.stdin.close()
        proc.stdout.close()
        proc.wait(timeout=min(layout.args.timeout, max(1.0, layout.args.deadline - time.monotonic())))
        if proc.returncode:
            raise ValueError(f"restricted product quality gate failed ({proc.returncode})")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quality.json"
        trial.copy_result(output, path, layout.args.worker_uid)
        gate = read(path)
    require(gate.get("passes") is True and gate["kernel_profile"]["kernel_sha256"] == layout.binary
            and gate["kernel_profile"]["gguf_sha256"] == native_release.QWEN32, "GGUF product gate did not pass")
    return gate


class ReleaseClient(MLXWorkerClient):
    def _launch_worker(self, arguments, environment):
        # Only the product protocol arguments cross this boundary. The outer server's environment stays here.
        tail = arguments[arguments.index("--spec"):]
        return self.resources.enter_context(launch(self.layout, "serve", tail, self.layout.args.timeout))


def in_scope(request):
    return (len(request.messages) == 1 and request.messages[0][0] == "user" and request.temperature == 0
            and request.top_p == 1 and not request.stop and request.max_tokens == 32
            and len(request.messages[0][1].encode()) <= PROMPT_BYTES)


def reference_endpoint(url, model, own_port):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port or parsed.port == own_port
            or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("reference must name a separate local product server")
    base = "http://127.0.0.1:" + str(parsed.port)
    def get(path):
        # No credential, proxy or redirect is needed to probe this dedicated local listener.
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", parsed.port, timeout=3)
        try:
            conn.request("GET", path)
            res = conn.getresponse()
            require(res.status == 200, "ordinary reference is unavailable")
            return json.loads(res.read(1 << 20))
        finally:
            conn.close()
    health, models = get("/health"), get("/v1/models")["data"]
    require(health.get("ready") and health.get("ggml", {}).get("gate", {}).get("passes") is True
            and not health.get("w4_release") and any(m.get("id") == model and m.get("revision") == "sha256:" + native_release.QWEN32
                                                     and m.get("loaded") for m in models), "reference model or quality gate differs")
    return base + "/v1/chat/completions"


class ScopedService(ProductService):
    reference_url = None
    instance = None

    def reference_for(self, request):
        if self.reference_url and not in_scope(request):
            return reference_endpoint(self.reference_url, self.spec.model_id, self.listen_port)
        return None

    def health(self):
        result = super().health()
        if self.instance:
            result["w4_release"] = self.instance
        return result

    def stream(self, request):
        if not in_scope(request):
            raise InvalidRequest("dedicated 32-token endpoint supports one short greedy user message")
        return super().stream(request)


@contextmanager
def server(args, receipt, cand, gate, port=0):
    from ironmule.gguf_tokenizer import gguf_model_id

    with ExitStack() as stack:
        layout = stack.enter_context(isolated_inputs(args, receipt, cand))
        trusted = gate["kernel_profile"]
        require(trusted["kernel_sha256"] == layout.binary and trusted["flags"] == {k: layout.env[k] for k in FLAGS}
                and gate["settings"] == receipt["binding"]["engine"], "HTTP quality receipt belongs to another arm")
        spec = ModelSpec(gguf_model_id(str(args.gguf)), "sha256:" + native_release.QWEN32, str(args.gguf), args.gguf.stat().st_size)
        client = ReleaseClient(spec, startup_timeout=args.timeout, execution_variant="current_engine", thinking=False,
                               engine_configuration="ggml", speculate=True, ggml_gguf=str(args.gguf),
                               ggml_gate=gate, ggml_sequences=32)
        client.layout, client.resources = layout, stack
        client.start()
        stack.callback(client.close)
        product = ProductStore(Path(stack.enter_context(tempfile.TemporaryDirectory(dir=args.state))))
        product.setup("server")
        service = ScopedService(product, backend=client, spec=spec)
        stack.callback(service.close)
        service.instance = {"arm": "cand" if cand else "ref", "binary": layout.binary, "pr": receipt["publication"]["pr"],
                            "order_id": receipt["order"]["order_id"]}
        if run_tag := os.environ.get("IRONMULE_HTTP_WITNESS"):
            service.instance["broker_instance"] = run_tag
        service.reference_url = getattr(args, "reference_url", None)
        if service.reference_url:
            reference_endpoint(service.reference_url, spec.model_id, port)
        http = create_server(service, host="127.0.0.1", port=port)
        service.listen_port = http.server_address[1]
        stack.callback(http.server_close)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        stack.callback(thread.join)
        def drain():
            # The listener stops accepting before the normal worker is closed.
            http.shutdown()
            end = min(args.deadline, time.monotonic() + 30)
            while client.ready and service.outstanding() and time.monotonic() < end:
                time.sleep(.05)
        stack.callback(drain)
        yield {"base_url": "http://127.0.0.1:" + str(http.server_address[1]), "model_id": spec.model_id}, service, client


def witness(batch):
    return [[row["text"], row["finish_reason"], row["prompt_tokens"], row["tokens"]] for row in batch]


def measure(address, service, client, seed, binding, deadline):
    from ironmule.factual_gate import MAX_TOKENS, cases, correct
    from ironmule_devices import ggml_devices

    cells, gpu, samples = {}, [], []
    for width in GENERATIONS:
        prompts = compare.performance_prompts(seed, width, width)
        rows, elapsed = compare.batch("ironmule", address, prompts, width, 32, deadline)
        ttft = sorted(r["ttft_ms"] for r in rows)[math.ceil(.95 * width) - 1] / 1000
        cells[str(width)] = {"wall_s": elapsed, "ttft_s": ttft, "output_sha256": digest(witness(rows)),
                             "completed": len(rows), "tokens": sum(r["tokens"] for r in rows)}
        samples.extend(service.live_samples()[-width:])
        devices = ggml_devices()
        require(len(devices) == 1 and devices[0]["provider"] == "cuda", "expected one visible CUDA device")
        gpu.append(devices[0]["total_memory"] - devices[0]["free_memory"])
    facts = cases(seed)
    rows, _ = compare.batch("ironmule", address, [case.prompt for case in facts], len(facts), MAX_TOKENS, deadline)
    require(len(rows) == 18 and all(correct(q, row["text"]) for q, row in zip(facts, rows, strict=True)),
            "HTTP factual quality gate failed")
    rss = next(int(line.split()[1]) for line in Path(f"/proc/{client._process.pid}/status").read_text().splitlines()
               if line.startswith("VmHWM:"))
    return {**cells["32"], "binding": binding, "workload": {"seed": seed, "widths": list(GENERATIONS), "max_tokens": 32},
            "evaluator_sha256": trial.checksum(__file__), "correct": 32, "facts": witness(rows),
            "device_used_bytes": max(gpu), "peak_rss_kib": rss, "cells": cells, "samples": samples}


def verdict(pairs):
    from ironmule.factual_gate import cases, correct
    for ref, cand in pairs:
        facts = cases(ref["workload"]["seed"])
        if (ref["facts"] != cand["facts"] or len(facts) != len(ref["facts"]) or len(facts) != 18
                or not all(correct(q, row[0]) for q, row in zip(facts, ref["facts"], strict=True))):
            return {"passed": False, "reason": "changed or incorrect factual answers"}
        if not all(type(r["device_used_bytes"]) is int and r["device_used_bytes"] > 0 for r in (ref, cand)):
            return {"passed": False, "reason": "unavailable GPU memory"}
        if cand["device_used_bytes"] > 1.05 * ref["device_used_bytes"]:
            return {"passed": False, "reason": "GPU memory grew beyond five percent"}
        for width in map(str, GENERATIONS):
            a, b = ref["cells"][width], cand["cells"][width]
            if not all(math.isfinite(row[k]) and row[k] > 0 for row in (a, b) for k in ("wall_s", "ttft_s", "tokens")):
                return {"passed": False, "reason": "invalid HTTP observation"}
            if any(a[k] != b[k] for k in ("output_sha256", "completed", "tokens")) or a["completed"] != int(width):
                return {"passed": False, "reason": "changed HTTP answers, stop reasons or token counts"}
    for width in ("1", "8"):
        for metric in ("wall_s", "ttft_s"):
            ratios = [c["cells"][width][metric] / r["cells"][width][metric] for r, c in pairs]
            if statistics.median(ratios) > 1.05:
                return {"passed": False, "reason": f"HTTP width {width} {metric} regressed"}
    return judge(pairs)


def roundtrip(server_address, service, client, seeds, binding, args):
    # Every new process receives the same untimed warmup on both arms before matched holdouts.
    compare.batch("ironmule", server_address, compare.performance_prompts(80413, 32, 32), 32, 32,
                  args.deadline)
    return [measure(server_address, service, client, seed, binding, args.deadline) for seed in seeds]


def update_selection(args, selection, *, promote=False):
    with deployment(args.state) as d:
        path = d / "http-selection.json"
        current = read(path) if path.exists() else {}
        if promote and (current.get("withdrawn") or read(d / "selection.json").get("arm") != "cand"):
            raise ValueError("HTTP or native selection was withdrawn while qualifying")
        save(path, selection)


def withdraw(args, reason):
    update_selection(args, {"arm": "ref", "withdrawn": True, "reason": reason})


def sample_shape(sample):
    return (int(sample["prompt_tokens"]) // 32, int(sample["completion_tokens"]), int(sample["queue_ms"]) // 100)


def live_bounds(samples):
    bounds = {}
    for row in samples:
        if not row["ttft_ms"] or not row["tokens_per_second"] or row["queue_ms"] is None:
            continue
        key = str(sample_shape(row))
        seconds = row["completion_tokens"] / row["tokens_per_second"]
        old_time, old_ttft, n = bounds.get(key, [0, 0, 0])
        bounds[key] = [max(seconds, old_time), max(row["ttft_ms"], old_ttft), n + 1]
    return {k: v[:2] for k, v in bounds.items() if v[2] >= 6}


def missed(row, bounds):
    try:
        envelope = bounds.get(str(sample_shape(row)))
        if not envelope:
            return None
        return (row["completion_tokens"] / row["tokens_per_second"] > envelope[0] * 1.05
                or row["ttft_ms"] > envelope[1] * 1.05)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def assess(evidence, key, pin):
    pairs, samples = [], []
    for half in (0, 1):
        ref, cand = (read(evidence / f"{arm}-{half}.json") for arm in ("ref", "cand"))
        pairs.extend(zip(ref["rows"], cand["rows"], strict=True))
        samples.extend(s for row in ref["rows"] for s in row["samples"])
    require([r["workload"]["seed"] for r, _ in pairs] == list(SEEDS)
            and all(r["binding"] == key and r["evaluator_sha256"] == trial.checksum(__file__)
                    for pair in pairs for r in pair), "incomplete or foreign HTTP holdouts")
    score, bounds = verdict(pairs), live_bounds(samples)
    if score["passed"] and not bounds:
        score = {"passed": False, "reason": "no comparable live HTTP samples for regression detection"}
    return {"binding": key, "control_receipt": pin, "decision": score, "reference_bounds": bounds,
            "quality_sha256": {arm: trial.checksum(evidence / f"gate-{arm}.json") for arm in ("ref", "cand")}}


def qualify(args, d, receipt, pin, key):
    require(locked_receipt(args.state)[2], "native variant must already be active on this target")
    selected = d / "http-selection.json"
    if selected.exists() and read(selected).get("withdrawn"):
        raise ValueError("HTTP deployment has been withdrawn")
    save(d / "http-trial.json", {"binding": key, "control_receipt": pin})
    evidence = d / ("http-" + key)
    evidence.mkdir(mode=0o700, exist_ok=True)
    gates = {}
    for arm in ("ref", "cand"):
        path = evidence / f"gate-{arm}.json"
        if not path.exists():
            with isolated_inputs(args, receipt, arm == "cand") as layout:
                save(path, product_gate(layout))
        gates[arm] = read(path)
    decision_path = evidence / "decision.json"
    if not decision_path.exists():
        # ABBA fresh worker starts. A half contains three distinct seeds. Save complete halves atomically.
        for half, arm in ((0, "ref"), (0, "cand"), (1, "cand"), (1, "ref")):
            path = evidence / f"{arm}-{half}.json"
            if path.exists():
                continue
            seeds = SEEDS[:PAIRS // 2] if half == 0 else SEEDS[PAIRS // 2:]
            with server(args, receipt, arm == "cand", gates[arm]) as (address, service, client):
                rows = roundtrip(address, service, client, seeds, key, args)
                for row in rows:
                    require(row["binding"] == key, "incorrect HTTP observation binding")
                save(path, {"rows": rows})
        result = assess(evidence, key, pin)
        save(decision_path, result)
    result = read(decision_path)
    accepted = result["decision"].get("passed") is True
    if accepted:
        require(assess(evidence, key, pin) == result, "HTTP decision no longer matches the measured holdouts")
    update_selection(args, {"arm": "cand" if accepted else "ref", "binding": key, "control_receipt": pin,
                            "withdrawn": not accepted}, promote=accepted)
    return result


def live(args, d, receipt, pin, key):
    evidence = d / ("http-" + key)
    result = read(evidence / "decision.json")
    require(result["binding"] == key and result["control_receipt"] == pin, "HTTP target binding changed")
    if result["decision"].get("passed") and assess(evidence, key, pin) != result:
        withdraw(args, "HTTP holdout reports changed since qualification")
    deadline = time.monotonic() + args.lifetime
    stop = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    starts = []
    served = 0
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            selected = read(d / "http-selection.json")
            active = (selected.get("arm") == "cand" and not selected.get("withdrawn") and locked_receipt(args.state)[2]
                      and selected.get("binding") == key and selected.get("control_receipt") == pin
                      and result["decision"].get("passed") is True)
            arm = "cand" if active else "ref"
            try:
                gate = read(evidence / f"gate-{arm}.json")
                with server(args, receipt, active, gate, args.port) as (address, service, client):
                    print(json.dumps({"ready": True, "arm": arm, "url": address["base_url"], "pr": receipt["publication"]["pr"]}), flush=True)
                    starts.append({"arm": arm, "worker": service.instance or {}, "url": address["base_url"]})
                    if getattr(args, "canary", False):
                        from ironmule.factual_gate import cases, correct
                        q = cases(75287)[0]
                        canary = compare.stream("ironmule", address, q.prompt, 32, args.deadline)
                        require(correct(q, canary["text"]), "published endpoint failed its ordinary completion")
                        starts[-1]["external_completion"] = witness([canary])
                    since, misses = time.time(), 0
                    while not stop.wait(.5) and time.monotonic() < deadline:
                        if not client.ready or not service.ready():
                            raise ValueError("product worker exited")
                        if active:
                            desired = read(d / "http-selection.json")
                            if desired.get("arm") != "cand" or not locked_receipt(args.state)[2]:
                                break
                            for row in service.live_samples():
                                if row["at"] <= since:
                                    continue
                                since = row["at"]
                                loss = missed(row, result["reference_bounds"])
                                if loss is not None:
                                    misses = misses + 1 if loss else 0
                                if misses >= 3:
                                    raise ValueError("three matched HTTP requests exceeded measured reference bounds")
                    served += service.health()["completed_requests"]
            except Exception as exc:
                if not active:
                    raise
                withdraw(args, str(exc)[:300])
    finally:
        signal.signal(signal.SIGTERM, previous)
    return {"control_receipt": pin, "binding": key, "starts": starts, "completed": served,
            "selection": read(d / "http-selection.json")}


def exercise(args, receipt, pin, key):
    """Use a fresh supervisor on an independent copy of the selected state. The real selection stays available."""
    from ironmule.factual_gate import cases, correct
    from ironmule.gguf_tokenizer import gguf_model_id
    original = args.state / "deployment"
    with tempfile.TemporaryDirectory(dir=args.state) as tmp:
        shadow = Path(tmp)
        d = shadow / "deployment"
        d.mkdir()
        for name in ("http-selection.json", "target.json", "selection.json"):
            shutil.copyfile(original / name, d / name)
        shutil.copytree(original / ("http-" + key), d / ("http-" + key))
        require(read(d / "http-selection.json")["arm"] == "cand", "HTTP candidate was not admitted")
        cmd = [sys.executable, "-m", "lab.pk1.native_http", "serve", "--state", str(shadow), "--port", str(args.port),
               "--lifetime", str(args.timeout), "--timeout", str(args.timeout), "--scoped-only", "--worker-uid", str(args.worker_uid)]
        for k in ("checkout", "gguf", "trial", "candidate_source", "reference_lib", "table", "worker_root"):
            cmd += ["--" + k.replace("_", "-"), str(getattr(args, k))]
        # The serving supervisor owns the driver process. Its worker still goes through open_worker.
        with tempfile.TemporaryFile() as log:
            run_tag = secrets.token_hex(16)
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, env={**os.environ, "IRONMULE_HTTP_WITNESS": run_tag})
            try:
                proofs = []
                base = "http://127.0.0.1:" + str(args.port)
                model = gguf_model_id(str(args.gguf))
                for arm in ("cand", "ref"):
                    binary = receipt["build"]["binary"] if arm == "cand" else receipt["order"]["binding"]["w4_sha256"]
                    while proc.poll() is None and time.monotonic() < args.deadline:
                        try:
                            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                                health = json.load(resp)
                            if (health.get("ready") and health.get("w4_release") == {"arm": arm, "binary": binary,
                                    "pr": receipt["publication"]["pr"], "order_id": receipt["order"]["order_id"], "broker_instance": run_tag}):
                                break
                        except OSError:
                            pass
                        time.sleep(.5)
                    else:
                        raise ValueError("protected HTTP supervisor did not reach the requested arm")
                    fact = cases(74241)[0]
                    row = compare.stream("ironmule", {"base_url": base, "model_id": model}, fact.prompt, 32, args.deadline)
                    require(correct(fact, row["text"]), "external HTTP holdout answer failed")
                    proofs.append({"arm": arm, "binary": binary, "output": witness([row]), "correct": True})
                    if arm == "cand":
                        withdraw(SimpleNamespace(state=shadow), "bounded HTTP rollback exercise")
                require(proofs[0]["output"] == proofs[1]["output"] and read(d / "http-selection.json")["withdrawn"],
                        "rollback changed external completion or failed to persist")
                return {"control_receipt": pin, "binding": key, "responses": proofs, "withdrawn": True}
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--state", type=Path, required=True)
    for operation in ("qualify", "serve", "exercise"):
        cmd = sub.add_parser(operation)
        for key in ("state", "checkout", "gguf", "trial", "candidate-source", "reference-lib", "table", "worker-root"):
            cmd.add_argument("--" + key, type=Path, required=True)
        cmd.add_argument("--worker-uid", type=int, default=60000)
        cmd.add_argument("--timeout", type=int, default=1800, choices=range(1, 3601), metavar="SECONDS")
        if operation in ("exercise", "serve"):
            cmd.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=8230)
        if operation == "serve":
            cmd.add_argument("--canary", action="store_true", help="send one externally measured factual request at startup")
            cmd.add_argument("--report", type=Path, help="atomically publish the final serving receipt")
            route = cmd.add_mutually_exclusive_group(required=True)
            route.add_argument("--reference-url", help="checked ordinary loopback product endpoint for other valid requests")
            route.add_argument("--scoped-only", action="store_true", help="dedicated width-32 profile endpoint")

            cmd.add_argument("--lifetime", type=int, default=600)
    args = parser.parse_args(argv)
    if args.operation == "rollback":
        withdraw(args, "explicit HTTP rollback")
        return
    for name, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve(strict=name != "report"))
    if args.operation == "serve" and not 1 <= args.lifetime <= args.timeout:
        parser.error("use a bounded lifetime at most the isolated worker timeout")
    args.deadline = time.monotonic() + args.timeout
    receipt, pin, _ = locked_receipt(args.state)
    key = code_binding(args, pin)
    d = args.state / "deployment"
    with (d / ".http-device.lock").open("a") as device:
        fcntl.flock(device, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.operation == "qualify":
            try:
                result = qualify(args, d, receipt, pin, key)
            except Exception as exc:
                path = d / ("http-" + key)
                if (path / "gate-ref.json").exists() and not (path / "decision.json").exists():
                    save(path / "decision.json", {"binding": key, "control_receipt": pin,
                         "decision": {"passed": False, "reason": str(exc)[:300]}, "reference_bounds": {}})
                    withdraw(args, "HTTP qualification failed")
                    result = read(path / "decision.json")
                else:
                    raise
            print(json.dumps(result))
        elif args.operation == "exercise":
            print(json.dumps(exercise(args, receipt, pin, key)))
        else:
            result = live(args, d, receipt, pin, key)
            if args.report:
                save(args.report, result)


if __name__ == "__main__":
    main()
