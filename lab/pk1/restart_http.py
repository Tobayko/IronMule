"""Measure, then serve a reviewed startup source through the ordinary local product HTTP service."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from ironmule.factual_gate import MAX_TOKENS, cases
from ironmule.ggml_profiles import FLAGS
from ironmule_product import compare
from lab.pk1 import native_http as http
from lab.pk1 import native_trial
from lab.pk1 import restart_release, restart_trial
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read, save
from training.trial_release import require

WORKER = Path(__file__).with_name("restart_http_worker.py")
SEEDS = tuple(range(95111, 95117))


@contextmanager
def inputs(args, receipt, candidate, *, gate_cache=None):
    q = receipt["qualification"]
    cache = args.trial / "receipt.json"
    require(receipt["adapter"] == restart_release.ADAPTER and q["evaluator_sha256"] == restart_release.evaluator(),
            "unbound native startup release")
    bound = q["binding"]["runtime"]
    if candidate:
        require(release.checksum(cache) == q["binding"]["cache_receipt_sha256"], "foreign startup cache")
        learned = read(cache)
        require(learned["binding"] == bound and 0 <= time.time() - learned["created_s"] <= 86400, "expired startup receipt")
    tree = release.projected_tree(receipt, args.checkout, args.worker_root, args.candidate_source if candidate else None)
    with tempfile.TemporaryDirectory(dir=args.worker_root) as tmp:
        stage = Path(tmp)
        stage.chmod(0o711)
        public, writable = stage / "inputs", stage / "out"
        public.mkdir()
        public.chmod(0o755)
        if gate_cache:
            writable = gate_cache
            writable.parent.mkdir(mode=0o755, exist_ok=True)
            writable.parent.chmod(0o711)
            require(not writable.is_symlink(), "invalid product quality checkpoint path")
        writable.mkdir(mode=0o700, exist_ok=bool(gate_cache))
        os.chown(writable, args.worker_uid, args.worker_uid)
        writable.chmod(0o700)
        for path in [WORKER, http.WORKER, Path(native_trial.__file__), restart_trial.HARNESS]:
            shutil.copyfile(path, public / path.name)
            (public / path.name).chmod(0o444)
        lib = public / "active.so"
        table = public / "dispatch.table"
        for src, dest, expected in ((args.reference_lib, lib, bound["w4_sha256"]), (args.table, table, bound["table_sha256"])):
            shutil.copyfile(src, dest)
            dest.chmod(0o444)
            require(release.checksum(dest) == expected, "changed startup library or table")
        env = http.env_for(SimpleNamespace(checkout=tree), lib, table)
        env["IRONMULE_GGML_THREADS"] = str(bound["threads"])
        if candidate:
            dst = public / "restart.json"
            shutil.copyfile(cache, dst)
            dst.chmod(0o444)
            env.update(IRONMULE_KG_RESTART_RECEIPT=str(dst), IRONMULE_KG_RESTART_SHA256=release.checksum(dst))
        profile = public / "expected.json"
        profile.write_text(json.dumps({"runtime": bound, "candidate": candidate, "vendor": q["binding"]["vendor"],
                "mask_bits": q["binding"]["mask_bits"], "pilot": q["binding"]["pilot_sha256"],
                "source": q["candidate_sha256" if candidate else "reference_sha256"], "binary": bound["w4_sha256"],
                "table": bound["table_sha256"], "flags": list(FLAGS),
                "binding": {"profile": {}, "model_sha256": bound["model_sha256"], "native": bound["native"], "engine": bound["settings"]}}))
        profile.chmod(0o444)
        paths = [public, tree, args.gguf, *(Path(env[k]).resolve() for k in ("IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH"))]
        for item in [sys.prefix, sys.base_prefix, *env.get("LD_LIBRARY_PATH", "").split(":")]:
            if item and not Path(item).resolve().is_relative_to("/usr"):
                paths.append(Path(item).resolve())
        yield SimpleNamespace(args=args, binary=bound["w4_sha256"], source_sha256=q["candidate_sha256" if candidate else "reference_sha256"], command=[str(Path(sys.executable).resolve()), str(public / WORKER.name)],
                              env=env, out=writable, profile=profile, readonly=paths)


def key_for(args, receipt):
    from training import pilot_restart_profiles
    from ironmule import pilot
    files = [Path(__file__), WORKER, http.WORKER, Path(http.__file__), Path(release.__file__),
             Path(pilot_restart_profiles.__file__), Path(pilot.__file__),
             *sorted((Path(__file__).resolve().parents[2] / "ironmule_product").glob("*.py"))]
    return http.digest({"control": args.receipt_sha256, "order": receipt["order"]["order_id"], "seeds": SEEDS, "head": args.head_sha256, "geometry": args.geometry_sha256,
                        "code": {p.name: release.checksum(p) for p in files}})


def wave(address, seed, deadline):
    rows, wall = compare.batch("ironmule", address, [q.prompt for q in cases(seed)], 18, MAX_TOKENS, deadline)
    return http.graded_wave(seed, http.witness(rows), wall, [r["ttft_ms"] for r in rows])


def byte_shape(lengths):
    ordered = sorted(n.bit_length() for n in lengths)
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def measured_geometry(trial):
    """Derive coarse input-size and concurrency bounds from the two actual graded reference cold starts."""
    manifest = read(trial / "result.json")
    require(manifest.get("passed") is True, "startup reference geometry needs a qualified native trial")
    shapes, origins = [], []
    for i in range(2):
        name = f"facts-ref-{i}.json"
        path = trial / name
        sha = release.checksum(path)
        require(manifest["measurements_sha256"][name] == sha, "changed factual reference measurement")
        first = read(path)["factual_first"][0]
        require(len(first["answers"]) == 18 and first["correct"] == [True] * 18, "incomplete reference workload")
        lengths = [len(q.prompt.encode()) for q in cases(first["seed"])]
        shapes.append(byte_shape(lengths))
        origins.append(sha)
    return {"count": 18, "max_tokens": 32, "size_unit": "UTF-8 byte bit length",
            "minimum": [min(col) for col in zip(*shapes, strict=True)],
            "maximum": [max(col) for col in zip(*shapes, strict=True)], "reference_sha256": origins}


def supported_batch(lengths, geometry):
    if len(lengths) != geometry["count"] or any(n <= 0 for n in lengths):
        return False
    return all(lo <= n <= hi for lo, n, hi in zip(geometry["minimum"], byte_shape(lengths), geometry["maximum"], strict=True))


@contextmanager
def cold_endpoint(args, receipt, gates, *, withdrawn, ablate=False, port=0):
    from ironmule.gguf_tokenizer import gguf_model_id
    from ironmule_product.errors import BackendUnavailable
    from ironmule_product.http_server import create_server

    model = gguf_model_id(str(args.gguf))
    geometry = args.geometry_record
    selection = {}
    with ExitStack() as stack:
        class ColdProduct:
            backend = None
            client = None
            failure = None
            def __init__(self):
                self.lock = threading.Lock()
                self.arrivals = threading.Condition()
                self.seen = {}
                self.last_arrival = 0.
                self.family = None
                self.collection_deadline = None
                self.first_request_id = None
                self.first_arrived_ns = None
            def classify(self, request):
                with self.arrivals:
                    if self.family is None:
                        if self.collection_deadline is None:
                            self.first_arrived_ns = time.monotonic_ns()
                            self.first_request_id = request.request_id
                            self.collection_deadline = time.monotonic() + .25
                        length = len(request.messages[0][1].encode())
                        if (request.request_id in self.seen or request.model != model or not http.in_scope(request)
                                or not geometry["minimum"][0] <= length.bit_length() <= geometry["maximum"][-1]):
                            self.family = "unknown"
                        self.seen[request.request_id] = length
                        self.last_arrival = time.monotonic()
                        self.arrivals.notify_all()
                        while self.family is None:
                            now = time.monotonic()
                            wait_until = self.collection_deadline
                            if len(self.seen) > geometry["count"] or now >= wait_until:
                                self.family = "unknown"
                            elif len(self.seen) == geometry["count"]:
                                quiet_until = self.last_arrival + .01
                                if now >= quiet_until:
                                    self.family = "factual-first-18" if supported_batch(self.seen.values(), geometry) else "unknown"
                                wait_until = min(wait_until, quiet_until)
                            if self.family is None:
                                self.arrivals.wait(max(0., wait_until - now))
                            else:
                                self.arrivals.notify_all()
                    return self.family
            def ready(self):
                return self.failure is None and (self.backend is None or self.backend.ready())
            def health(self):
                return self.backend.health() if self.backend else {"ready": self.ready(), "cold": True}
            def models(self):
                return self.backend.models() if self.backend else [{"id": model, "revision": "sha256:" + http.native_release.QWEN32, "loaded": False}]
            def reference_for(self, req):
                if getattr(args, "reference_url", None) and not http.in_scope(req):
                    return http.reference_endpoint(args.reference_url, model, self.port)
                return None
            def cancel(self, request_id):
                if self.backend:
                    return self.backend.cancel(request_id)
            def close(self):
                # ExitStack owns the protected process and real product service.
                return None
            def __getattr__(self, name):
                if self.backend is None:
                    raise AttributeError(name)
                return getattr(self.backend, name)
            def note(self, *a, **kw):
                if self.backend:
                    self.backend.note(*a, **kw)
            def stream(self, req):
                if not http.in_scope(req):
                    from ironmule_product.errors import InvalidRequest
                    raise InvalidRequest("dedicated startup endpoint requires a short greedy user request")
                actual_family = self.classify(req) if self.backend is None else self.family
                with self.lock:
                    if self.failure:
                        raise BackendUnavailable(self.failure)
                    if not self.backend:
                        prior = choose(args, receipt, withdrawn=withdrawn(), previous=True, family=actual_family)
                        policy = choose(args, receipt, withdrawn=withdrawn(), ablate=ablate, family=actual_family)
                        candidate = policy["profile"] == "candidate"
                        arm = "candidate" if candidate else "reference"
                        try:
                            _, self.backend, self.client = stack.enter_context(http.server(
                                args, receipt, candidate, gates[arm], self.port, inputs=inputs, listen=False,
                                settings=receipt["qualification"]["binding"]["runtime"]["settings"]))
                        except Exception as exc:
                            self.failure = "protected product startup failed: " + type(exc).__name__
                            raise BackendUnavailable(self.failure) from exc
                        selection.update(policy=policy, prior=prior, arm=arm, first_request_id=self.first_request_id, family=actual_family, joined_requests=len(self.seen),
                                         byte_shape=byte_shape(self.seen.values()), geometry_sha256=args.geometry_sha256,
                                         request_received_monotonic_ns=self.first_arrived_ns, worker_ready_monotonic_ns=time.monotonic_ns(),
                                         request_to_ready_s=(time.monotonic_ns() - self.first_arrived_ns) / 1e9,
                                         worker_pid=self.client._process.pid)
                        observer = getattr(args, "on_startup_selection", None)
                        if observer:
                            observer(dict(selection))
                return self.backend.stream(req)
        outer = ColdProduct()
        listener = create_server(outer, host="127.0.0.1", port=port)
        outer.port = listener.server_address[1]
        stack.callback(listener.server_close)
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        try:
            yield {"base_url": f"http://127.0.0.1:{outer.port}", "model_id": model}, outer, selection
        finally:
            listener.shutdown()
            thread.join()
            end = min(args.deadline, time.monotonic() + 30)
            while outer.backend and outer.backend.outstanding() and time.monotonic() < end:
                time.sleep(.05)


def measure(args, receipt, candidate, gate, seed):
    from ironmule_devices import ggml_devices
    start = time.monotonic()
    with cold_endpoint(args, receipt, {"candidate" if candidate else "reference": gate}, withdrawn=lambda: False,
                       ablate=not candidate) as (address, outer, selection):
        # The listener is ready, but no worker exists until the first real POST enters ColdProduct.stream().
        require(outer.backend is None and not selection, "cold worker started before the request")
        first = wave(address, seed, args.deadline)
        completed = time.monotonic() - start
        policy = selection["policy"]
        require(selection["arm"] == ("candidate" if candidate else "reference"), "trained startup model retained its reference")
        startup = selection["request_to_ready_s"]
        service, client = outer.backend, outer.client
        warm = wave(address, seed ^ 733, args.deadline)
        rows, wall = compare.batch("ironmule", address, compare.performance_prompts(seed, 32, 32), 32, 32, args.deadline)
        perf = compare.performance_metrics(rows, wall)
        peak = next(int(line.split()[1]) for line in Path(f"/proc/{client._process.pid}/status").read_text().splitlines()
                    if line.startswith("VmHWM:"))
        dev, = ggml_devices()
        require(dev["provider"] == "cuda", "target is no longer the CUDA worker")
        return {"seed": seed, "startup_decision": policy, "cold_admission": selection, "source_sha256": receipt["qualification"]["candidate_sha256" if candidate else "reference_sha256"],
                "startup_s": startup, "cold_completed_s": completed, "cold": first, "warm": warm,
                "perf_completed": len(rows), "perf_output_sha256": http.digest(http.witness(rows)),
                "perf_tokens": perf["completion_tokens"], "perf_wall_s": wall, "perf_ttft_s": perf["ttft_p95_ms"] / 1000,
                "peak_rss": peak, "device_bytes": dev["total_memory"] - dev["free_memory"], "samples": service.live_samples()}


def score(pairs, qualification):
    require(len(pairs) == len(SEEDS), "incomplete cold/warm product pairs")
    metrics = []
    for seed, (ref, cand) in zip(SEEDS, pairs, strict=True):
        for row, arm in ((ref, "reference"), (cand, "candidate")):
            require(row["seed"] == seed and row["source_sha256"] == qualification[arm + "_sha256"]
                    and row["startup_decision"]["profile"] == arm and row["startup_decision"]["previous"] is False
                    and row["startup_decision"]["ablate_weights"] is (arm == "reference")
                    and row["cold_admission"]["request_received_monotonic_ns"] < row["cold_admission"]["worker_ready_monotonic_ns"]
                    and row["cold_admission"]["policy"] == row["startup_decision"] and row["cold_admission"]["joined_requests"] == 18
                    and row["startup_decision"]["family"] == "factual-first-18"
                    and math.isfinite(row["startup_decision"]["cpu_ms"]) and row["startup_decision"]["cpu_ms"] >= 0
                    and row["perf_completed"] == 32 and all(math.isfinite(row[k]) and row[k] > 0 for k in
                    ("perf_tokens", "perf_wall_s", "perf_ttft_s", "peak_rss", "device_bytes", "startup_s", "cold_completed_s")),
                    "unbound product measurement")
            for name, value in (("cold", seed), ("warm", seed ^ 733)):
                w = row[name]
                require(w == http.graded_wave(value, w["responses"], w["wall_s"], w["first_token_ms"]) and w["correct"] == 18,
                        "product factual answers or timing failed")
            require(row["cold_completed_s"] >= row["cold"]["wall_s"] >= row["startup_s"], "invalid startup clock")
        require(all(ref[k] == cand[k] for k in ("perf_output_sha256", "perf_tokens"))
                and all(ref[k]["responses"] == cand[k]["responses"] for k in ("cold", "warm")), "different product work")
        metrics.append({"cold_correct_rate_ratio": ref["cold_completed_s"] / cand["cold_completed_s"],
                        "cold_p95_ratio": cand["cold"]["ttft_p95_s"] / ref["cold"]["ttft_p95_s"],
                        "warm_correct_rate_ratio": cand["warm"]["correct_completions_per_s"] / ref["warm"]["correct_completions_per_s"],
                        "warm_p95_ratio": cand["warm"]["ttft_p95_s"] / ref["warm"]["ttft_p95_s"],
                        "perf_time_ratio": cand["perf_wall_s"] / ref["perf_wall_s"], "perf_p95_ratio": cand["perf_ttft_s"] / ref["perf_ttft_s"],
                        "rss_ratio": cand["peak_rss"] / ref["peak_rss"], "gpu_ratio": cand["device_bytes"] / ref["device_bytes"]})
    def median(metric):
        return statistics.median(r[metric] for r in metrics)
    passed = (all(r["cold_correct_rate_ratio"] > 1 and max(r["gpu_ratio"], r["rss_ratio"]) <= 1.05 for r in metrics)
              and median("cold_correct_rate_ratio") >= 1.02 and median("cold_p95_ratio") <= 1 and median("warm_correct_rate_ratio") >= 1
              and median("warm_p95_ratio") <= 1 and max(median("perf_time_ratio"), median("perf_p95_ratio")) <= 1.02)
    return {"passed": passed, "pairs": metrics}


def assess(evidence, receipt):
    pairs = [(read(evidence / f"reference-{s}.json"), read(evidence / f"candidate-{s}.json")) for s in SEEDS]
    geometry_path = evidence / "geometry.json"
    geometry = read(geometry_path)
    sha = release.checksum(geometry_path)
    for (ref, cand), seed in zip(pairs, SEEDS, strict=True):
        expected = byte_shape(len(q.prompt.encode()) for q in cases(seed))
        for arm in (ref, cand):
            seen = arm["cold_admission"]
            require(seen["geometry_sha256"] == sha and seen["byte_shape"] == expected
                    and all(lo <= n <= hi for lo, n, hi in zip(geometry["minimum"], expected, geometry["maximum"], strict=True)),
                    "product evidence does not match calibrated arrival geometry")
    decision = score(pairs, receipt["qualification"])
    names = ["geometry.json", *(f"gate-{arm}.json" for arm in ("reference", "candidate"))]
    names.extend(f"{arm}-{s}.json" for s in SEEDS for arm in ("reference", "candidate"))
    decision["integrity"] = {name: release.checksum(evidence / name) for name in names}
    decision["reference_bounds"] = http.live_bounds(s for ref, _ in pairs for s in ref["samples"])
    decision["passed"] = bool(decision["passed"] and decision["reference_bounds"])
    return decision


def qualify(args, receipt, evidence):
    require(release.selected(args.state, args.receipt_sha256) == "candidate", "startup release is inactive")
    for ablate, profile in ((True, "reference"), (False, "candidate")):
        decision = choose(args, receipt, withdrawn=False, ablate=ablate)
        require(decision["profile"] == profile, f"startup model selected {decision['profile']}: {decision['reason']}")
    with (evidence / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for arm in ("reference", "candidate"):
            dest = evidence / f"gate-{arm}.json"
            if not dest.exists():
                cache = args.worker_root / ("restart-gguf-gates-" + evidence.name.rsplit("-", 1)[-1]) / arm
                with inputs(args, receipt, arm == "candidate", gate_cache=cache) as layout:
                    output = layout.out / "quality.json"
                    output.unlink(missing_ok=True)
                    with http.launch(layout, "gate", [str(output)], args.timeout) as proc:
                        proc.stdin.close()
                        proc.stdout.close()
                        proc.wait(timeout=max(1., args.deadline - time.monotonic()))
                        require(proc.returncode == 0, f"restricted {arm} product gate exited {proc.returncode}")
                    with tempfile.TemporaryDirectory(dir=evidence) as tmp:
                        copied = Path(tmp) / "quality.json"
                        native_trial.copy_result(output, copied, args.worker_uid)
                        record = read(copied)
                    save(dest, record)
            gate = read(dest)
            require(gate.get("passes") is True and gate.get("source_sha256") == receipt["qualification"][arm + "_sha256"],
                    f"{arm} product gate failed: {gate.get('reason', 'source mismatch')}")
        for i, seed in enumerate(SEEDS):
            sequence = ("reference", "candidate") if i % 2 == 0 else ("candidate", "reference")
            for arm in sequence:
                dest = evidence / f"{arm}-{seed}.json"
                require(release.selected(args.state, args.receipt_sha256) == "candidate", "startup withdrawn during product qualification")
                if not dest.exists():
                    save(dest, measure(args, receipt, arm == "candidate", read(evidence / f"gate-{arm}.json"), seed))
        verdict = assess(evidence, receipt)
        save(evidence / "decision.json", verdict)
        return verdict


def choose(args, receipt, *, withdrawn, previous=False, ablate=False, family="factual-first-18"):
    from training.pilot_restart_profiles import select
    path = args.trial / "receipt.json"
    try:
        cached = release.checksum(path) == receipt["qualification"]["binding"]["cache_receipt_sha256"]
        cached = cached and 0 <= time.time() - read(path)["created_s"] <= 86400
    except (OSError, ValueError, KeyError, TypeError):
        cached = False
    return select(args.head, receipt, family=family, cache_ready=cached,
                  active=not withdrawn and release.selected(args.state, args.receipt_sha256) == "candidate",
                  previous=previous, ablate_weights=ablate, expected_sha256=args.head_sha256)


def exercise_wave(args, receipt, evidence, withdrawal, *, seed, lifetime):
    try:
        decision = read(evidence / "decision.json")
        require(decision["passed"] is True and assess(evidence, receipt) == decision, "product startup qualification changed")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        decision = {}
        save(withdrawal, {"withdrawn": True, "reason": str(exc)[:400]})
    start = time.monotonic()
    history = []
    while True:
        candidate = False
        try:
            gates = {arm: read(evidence / f"gate-{arm}.json") for arm in ("reference", "candidate")}
            with cold_endpoint(args, receipt, gates, withdrawn=withdrawal.exists, port=args.port) as (address, outer, selection):
                served = wave(address, seed, args.deadline)
                chosen, prior, arm = selection["policy"], selection["prior"], selection["arm"]
                candidate = arm == "candidate"
                service, client = outer.backend, outer.client
                require(served["correct"] == 18, "live product first batch failed factual grading")
                cold = time.monotonic() - start
                completion = {"arm": arm, "source_sha256": receipt["qualification"][arm + "_sha256"],
                              "work_sha256": http.digest(served["responses"]), "graded_correct": served["correct"],
                              "cold_to_18_correct_s": cold, "correct_completions_per_s_including_startup": 18 / cold,
                              "first_token_ms": served["first_token_ms"], "visible_url": address["base_url"],
                              "prior_decision": prior, "startup_decision": chosen, "cold_admission": selection}
                history.append(completion)
                since, misses = time.time(), 0
                end = time.monotonic() + lifetime
                while time.monotonic() < end:
                    require(service.ready() and client.ready, "product worker stopped")
                    if candidate and (withdrawal.exists() or release.selected(args.state, args.receipt_sha256) != "candidate"):
                        break
                    if candidate:
                        for row in service.live_samples():
                            if row["at"] > since:
                                since = row["at"]
                                miss = http.missed(row, decision["reference_bounds"])
                                if miss is not None:
                                    misses = misses + 1 if miss else 0
                                require(misses < 3, "three matching product requests exceeded reference bounds")
                    time.sleep(min(.5, max(0, end - time.monotonic())))
                else:
                    return {"arm": arm, "completions": history, "requests_served": service.health().get("completed_requests", 0)}
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            if withdrawal.exists():
                raise
            save(withdrawal, {"withdrawn": True, "reason": str(exc)[:400]})
        start = time.monotonic()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op", choices=("qualify", "serve", "exercise", "rollback"))
    for k in ("state", "trial", "checkout", "candidate-source", "reference-lib", "table", "gguf", "worker-root"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--geometry-sha256", required=True)
    p.add_argument("--worker-uid", type=int, default=60000)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--port", type=int, default=8232)
    p.add_argument("--lifetime", type=int, default=120, choices=range(0, 1801), metavar="0..1800")
    p.add_argument("--head", type=Path)
    p.add_argument("--head-sha256")
    p.add_argument("--reference-url", help="ordinary checked loopback endpoint for other requests")
    p.add_argument("--dedicated", action="store_true", help="a separate bounded lane for 32-token greedy requests")
    a = p.parse_args()
    require(0 < a.timeout <= 3600 and 0 < a.port <= 65535, "invalid product worker time or port")
    require(a.head and a.head_sha256, "missing pinned startup head")
    a.deadline = time.monotonic() + a.timeout
    receipt = release.pinned(a.state, a.receipt_sha256)
    require(release.checksum(a.geometry) == a.geometry_sha256, "changed release geometry")
    a.geometry_record = read(a.geometry)
    key = key_for(a, receipt)
    root = a.state / "deployment"
    evidence = root / ("restart-http-" + key)
    evidence.mkdir(exist_ok=True)
    write_once(evidence / "geometry.json", a.geometry_record)
    require(release.checksum(evidence / "geometry.json") == a.geometry_sha256, "changed stored product geometry")
    save(root / "restart-http.json", {"key": key, "receipt_sha256": a.receipt_sha256})
    withdrawn = evidence / "withdrawn.json"
    if a.op == "rollback":
        result = {"withdrawn": True, "reason": "explicit HTTP withdrawal"}
        save(withdrawn, result)
    elif a.op == "qualify":
        require(not withdrawn.exists(), "startup HTTP deployment previously withdrawn")
        try:
            result = qualify(a, receipt, evidence)
        except (ValueError, RuntimeError) as exc:
            result = {"passed": False, "reason": str(exc)[:400]}
            save(evidence / "decision.json", result)
            save(withdrawn, {"withdrawn": True, "reason": result["reason"]})
    else:
        require(a.head is not None and a.head_sha256 and (a.dedicated or a.reference_url), "missing trained head or product route")
        if a.op == "serve":
            from lab.pk1.restart_live import passive
            result = passive(a, receipt, evidence, withdrawn)
        else:
            require(not withdrawn.exists(), "startup HTTP already withdrawn")
            with tempfile.TemporaryDirectory(dir=a.state) as tmp:
                shadow = Path(tmp) / "withdrawn.json"
                before = exercise_wave(a, receipt, evidence, shadow, seed=96122, lifetime=0)
                save(shadow, {"withdrawn": True, "reason": "test persistent product withdrawal"})
                after = exercise_wave(a, receipt, evidence, shadow, seed=96122, lifetime=0)
                pair = [row["completions"][-1] for row in (before, after)]
                result = {"passed": before["arm"] == "candidate" and after["arm"] == "reference"
                          and pair[0]["work_sha256"] == pair[1]["work_sha256"], "before": pair[0], "after": pair[1]}
                if not result["passed"]:
                    save(withdrawn, {"withdrawn": True, "reason": "trained startup or product rollback check failed"})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
