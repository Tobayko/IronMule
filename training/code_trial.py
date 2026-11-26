"""One offline IronMule development order. The evaluator and its gates live in the control checkout."""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"ironmule/ggml_engine.py", "ironmule/gguf_tokenizer.py"}
PAIRS = 6


def run(argv, *, cwd=ROOT, timeout=120, **kwargs):
    return subprocess.run(argv, cwd=cwd, timeout=timeout, check=True, text=True,
                          stdout=subprocess.PIPE, **kwargs).stdout.strip()


def git(*args, cwd=ROOT):
    return run(["git", *args], cwd=cwd)


def save(path, data):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read(path):
    return json.loads(path.read_text())


def prepare(state, commit, name):
    tree = state / name
    if not tree.exists():
        git("clone", "--quiet", "--no-hardlinks", "--no-checkout", str(ROOT), str(tree))
    if not (tree / ".git").is_dir():
        raise ValueError("incomplete checkout")
    # Pin both independent checkouts; only the disposable candidate receives generated source.
    git("checkout", "--quiet", "--detach", commit, cwd=tree)
    return tree


def develop(order, tree, state):
    """One included-plan Codex call; no credentials or writable checkout in the prompt."""
    target = order["target"]
    path = target["path"]
    if path not in TARGETS or path not in order["allowed_paths"]:
        raise ValueError("target outside this evaluator")
    source = tree / path
    code = git("show", "HEAD:" + path, cwd=tree) + "\n"
    lines = code.splitlines(keepends=True)
    defs = [f for f in ast.walk(ast.parse(code)) if isinstance(f, ast.FunctionDef)
            and f.name == target["function"].split(".")[-1]]
    func = next((f for f in defs if min([f.lineno] + [d.lineno for d in f.decorator_list])
                 <= target["line"] <= f.lineno), None)
    if func is None:
        raise ValueError("profile location differs from pinned source")
    out = state / "proposal.json"
    if os.environ.get("IRONMULE_CODE_AUTHOR", "cli") == "hosted":
        from training.code_author import hosted_proposal
        proposal = hosted_proposal(state, order, code)
        save(out, proposal)
    if not out.exists():
        if (state / "dispatch.json").exists():
            raise RuntimeError("one authoring attempt was used; inspect the saved dispatch before a new order")
        if "using ChatGPT" not in run(["codex", "login", "status"], stderr=subprocess.STDOUT):
            raise RuntimeError("included ChatGPT access required; API billing is outside this adapter")
        catalog = json.loads(run(["codex", "debug", "models"]))
        models = [m["slug"] for m in catalog["models"] if m.get("visibility") == "list"]
        model = next(m for m in ("gpt-6.2-sol", "gpt-6.1-sol", "gpt-6-sol") if m in models)
        schema = {"type": "object", "properties": {"definition": {"type": "string"},
                  "rationale": {"type": "string"}}, "required": ["definition", "rationale"],
                  "additionalProperties": False}
        save(state / "schema.json", schema)
        prompt = ("You are IronMule's development worker. Return one optimized replacement Python function "
                  "definition for the measured target. Preserve its signature, semantics on all inputs, and "
                  "decorators (which the editor preserves for you). No new dependencies. No benchmark detection, "
                  "measurement changes, or skipped work. No shell or tools needed. Return a dedented def without "
                  "decorators. If no valid optimization exists return the original.\n"
                  + json.dumps(order) + "\nSource file:\n" + code)
        save(state / "dispatch.json", {"model": model, "catalog": models, "order_id": order["order_id"],
                                      "time": time.time(), "payer": "ChatGPT session"})
        with (state / "codex.log").open("w") as log:
            run(["codex", "exec", "--ignore-user-config", "--ephemeral", "-s", "read-only", "-m", model,
                 "--skip-git-repo-check", "--output-schema", str(state / "schema.json"), "-o",
                 str(state / "reply.tmp"), "-"], cwd=state, input=prompt, timeout=480, stderr=log)
        proposal = read(state / "reply.tmp")
        dispatch = read(state / "dispatch.json")
        dispatch["reply_sha256"] = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
        save(state / "dispatch.json", dispatch)
        save(out, proposal)
    if os.environ.get("IRONMULE_CODE_AUTHOR", "cli") != "hosted":
        actual = hashlib.sha256(json.dumps(read(out), sort_keys=True).encode()).hexdigest()
        if read(state / "dispatch.json").get("reply_sha256") != actual:
            raise ValueError("reply is not bound to the saved CLI dispatch")
    replacement = textwrap.dedent(read(out)["definition"]).strip()
    nodes = ast.parse(replacement).body
    if (len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef) or nodes[0].name != func.name
            or ast.dump(nodes[0].args) != ast.dump(func.args) or nodes[0].decorator_list):
        raise ValueError("invalid replacement function")
    lines[func.lineno - 1:func.end_lineno] = [textwrap.indent(replacement, " " * func.col_offset) + "\n"]
    new = "".join(lines)
    compile(new, path, "exec")
    source.write_text(new)
    return path


def judge(pairs):
    if len(pairs) != PAIRS:
        raise ValueError("incomplete paired observations")
    rates, latencies, memories = [], [], []
    for ref, cand in pairs:
        for row in (ref, cand):
            if not all(math.isfinite(row[k]) and row[k] > 0 for k in (
                    "wall_s", "ttft_s", "peak_rss_kib", "completed", "tokens")):
                raise ValueError("incomplete timing or resource observation")
        if ref["binding"] != pairs[0][0]["binding"] or ref.get("profiled") or cand.get("profiled"):
            return {"passed": False, "reason": "mixed binding or profiling active"}
        if any(ref[k] != cand[k] for k in ("output_sha256", "completed", "tokens", "binding", "workload", "evaluator_sha256")):
            return {"passed": False, "reason": "different work, output or execution conditions"}
        if not all(row["correct"] == row["completed"] for row in (ref, cand)):
            return {"passed": False, "reason": "quality"}
        rates.append(ref["wall_s"] / cand["wall_s"])
        latencies.append(cand["ttft_s"] / ref["ttft_s"])
        memories.append(cand["peak_rss_kib"] / ref["peak_rss_kib"])
    passed = min(rates) > 1 and statistics.median(rates) >= 1.02 and statistics.median(latencies) <= 1 \
        and max(memories) <= 1.05
    return {"passed": passed, "reason": "paired hardware gate", "rate_ratios": rates,
            "ttft_ratios": latencies, "peak_rss_ratios": memories}


def runtime_env(provider):
    environment = {"PYTHONPATH": "/trial", "HOME": "/tmp", "TMPDIR": "/tmp", "PATH": "/usr/local/bin:/usr/bin:/bin",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "IRONMULE_PILOT": "1",
                   "IRONMULE_GGML_GPU_LAYERS": "999" if provider == "cuda" else "0"}
    for k in ("IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(k):
            environment[k] = os.environ[k]
    return environment


def sandbox(tree, gguf, dest, provider, thread_candidate=None):
    """Read-only fixture, evaluator and candidate. No account credentials or campaign state in the worker."""
    python = Path(sys.executable)
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--proc", "/proc",
           "--dev", "/dev", "--tmpfs", "/tmp"]
    mounts = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/localtime",
                                "/etc/passwd", "/etc/group", "/etc/alternatives", "/etc/machine-id", "/sys", sys.base_prefix,
                                sys.prefix)]
    for key in ("IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH"):
        if os.environ.get(key):
            mounts.append(Path(os.environ[key]).resolve())
    for path in dict.fromkeys(mounts):
        if path.exists():
            cmd += ["--ro-bind", str(path), str(path)]
    if provider == "cuda":
        for dev in sorted(Path("/dev").glob("nvidia*")):
            cmd += ["--dev-bind", str(dev), str(dev)]
    cmd += ["--ro-bind", str(tree), "/trial", "--ro-bind", str(ROOT / "training/serving_trial.py"),
            "/trusted/serving_trial.py", "--ro-bind", str(gguf), "/models/" + gguf.name,
            "--bind", str(dest), "/output", "--chdir", "/trial"]
    if thread_candidate is not None:
        if provider != "cpu":
            raise ValueError("CPU-only scratch thread candidate")
        cmd += ["--ro-bind", str(thread_candidate.resolve(strict=True)), "/models/controller.npz",
                "--ro-bind", str(ROOT / "training/pilot_threads.py"), "/trusted/pilot_threads.py"]
    environment = runtime_env(provider)
    for k, v in environment.items():
        cmd += ["--setenv", k, v]
    return [*cmd, str(python)]


def measure(tree, args, seed, parent):
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        tmp = Path(tmp)
        weights = parent / "controller.npz"
        extra = ["--thread-candidate", "/models/controller.npz"] if weights.is_file() else []
        cmd = sandbox(tree, args.gguf.resolve(), tmp, args.provider, weights if extra else None)
        run([*cmd, "/trusted/serving_trial.py", "--gguf", "/models/" + args.gguf.name,
             "--require-provider", args.provider, "--out", "/output/trial.json", "--waves", "4",
             "--threads", "2", "--clients", "4", "--seed", str(seed), *extra], timeout=420)
        return read(tmp / "trial.json")


def verify_deliveries(row, seed):
    """Score the returned texts and reconstruct work totals outside the candidate process."""
    from training.serving_trial import COLORS, digest
    checksums = []
    ttft = []
    tokens = 0
    seconds = 0.0
    for i, wave in enumerate(row["waves"]):
        answers = wave["answers"]
        if len(answers) != 4 or wave["workload"] != ("short", "document")[i % 2]:
            raise ValueError("incomplete or different request wave")
        for lane, answer in enumerate(answers):
            expected = COLORS[(seed + i + lane) % len(COLORS)]
            first = re.search(r"\w+", answer["answer"].casefold())
            ids = answer["output_ids"]
            if (not first or first.group() != expected or answer["target"] != expected
                    or not isinstance(ids, list) or not all(type(t) is int and t >= 0 for t in ids)
                    or not 1 <= len(ids) <= 3 or answer["stop"] not in ("length", "eos")
                    or (answer["stop"] == "length" and len(ids) != 3)):
                raise ValueError("delivered answer failed external fixed-work check")
            tokens += len(ids)
        times = wave["ttfts_s"]
        if len(times) != 4 or not all(0 < t <= wave["wall_s"] for t in times):
            raise ValueError("invalid delivery timestamps")
        ttft.extend(times)
        seconds += wave["wall_s"]
        checksums.append(digest(answers))
    if (row["tokens"] != tokens or row["wall_s"] != seconds or row["output_sha256"] != digest(checksums)
            or row["ttft_s"] != sorted(ttft)[math.ceil(.95 * len(ttft)) - 1]):
        raise ValueError("whole-run metrics differ from individual deliveries")


def validate_arm(row, tree, order, seed, signature):
    expected = {name: hashlib.sha256((tree / "ironmule" / name).read_bytes()).hexdigest()
                for name in ("ggml_engine.py", "gguf_tokenizer.py", "pilot.py")}
    workload = {"seed": seed, "clients": 4, "waves": 4, "max_tokens": 3, "warmup_waves": 1,
                "context": 512, "phases": ["short", "document"]}
    if (row["evaluator_sha256"] != signature["evaluator_sha256"] or row["runtime_source_sha256"] != expected
            or row["workload"] != workload or row["completed"] != 16 or row["correct"] != 16
            or row["pilot_decisions"] <= 0 or row["profiled"] or row["schema"] != order["evaluator"]
            or len(row["waves"]) != 4 or any(w["completed"] != 4 or w["correct"] != 4 for w in row["waves"])):
        raise ValueError("unverified source, evaluator or incomplete work")
    verify_deliveries(row, seed)
    # Location-specific environment hashes can differ between discovery and isolation.
    # Actual settings, hardware, model and libraries must agree. Both timed arms must
    # additionally match *all* binding fields, including the environment hash.
    if any(row["binding"].get(k) != v for k, v in order["binding"].items() if k != "env_sha256"):
        raise ValueError("target differs from discovery")


def cycle(args):
    state = args.state.resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        order = read(args.order)
        binding = {"order_sha256": hashlib.sha256(args.order.read_bytes()).hexdigest(),
                   "base": git("rev-parse", "--verify", order["base_commit"] + "^{commit}"),
                   "provider": args.provider}
        if not re.fullmatch(r"[0-9a-f]{64}", order["order_id"]):
            raise ValueError("invalid development order id")
        context = state / "context.json"
        from training.experiment_planner import EVALUATOR, ORDER, digest
        if (order.get("schema") != ORDER or order.get("evaluator") != EVALUATOR
                or digest({k: v for k, v in order.items() if k != "order_id"}) != order["order_id"]):
            raise ValueError("invalid development order or incompatible evaluator")
        if context.exists() and read(context) != binding:
            raise ValueError("state belongs to another experiment")
        save(context, binding)
        reference = prepare(state, binding["base"], "reference")
        candidate = prepare(state, binding["base"], "candidate")
        if git("status", "--porcelain", cwd=reference):
            raise ValueError("dirty reference")
        path = develop(order, candidate, state)
        if not git("diff", "HEAD", "--name-only", cwd=candidate):
            save(state / "result.json", {"passed": False, "reason": "no source change"})
            return
        if git("diff", "HEAD", "--name-only", cwd=candidate) != path or git("ls-files", "--others", "--exclude-standard",
                                                                   cwd=candidate):
            raise ValueError("unexpected candidate changes")
        (state / "candidate.diff").write_text(git("diff", "HEAD", "--binary", cwd=candidate) + "\n")
        if args.author_only:
            return
        git("add", "--", path, cwd=candidate)
        ref_tree = git("rev-parse", "HEAD^{tree}", cwd=reference)
        cand_tree = git("write-tree", cwd=candidate)
        signature = {"reference_tree": ref_tree, "candidate_tree": cand_tree,
                     "evaluator_sha256": hashlib.sha256((ROOT / "training/serving_trial.py").read_bytes()).hexdigest(),
                     "gate_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "order_id": order["order_id"]}
        manifest = state / "evaluation.json"
        if manifest.exists() and read(manifest) != signature:
            raise ValueError("candidate, gate or evaluator changed; use a fresh evaluation directory")
        save(manifest, signature)
        with tempfile.TemporaryDirectory(dir=state) as tmp:
            cmd = sandbox(candidate, args.gguf.resolve(), Path(tmp), args.provider)
            run([*cmd, "-X", "pycache_prefix=/tmp/pycache", "-m", "compileall", "-q", "/trial/ironmule"])
            run([*cmd, "-m", "pytest", "-n", "0", "-p", "no:cacheprovider", "-q",
                 "tests/engine/test_gguf_tokenizer.py", "tests/engine/test_ggml_engine.py"], timeout=240)
        pairs = []
        for i in range(PAIRS):
            file = state / f"pair-{i}.json"
            if not file.exists():
                arms = [("reference", reference), ("candidate", candidate)]
                pair = {name: measure(tree, args, 8351 + i, state) for name, tree in
                        (reversed(arms) if i % 2 else arms)}
                save(file, {**signature, **pair})
            pair = read(file)
            if any(pair.get(k) != v for k, v in signature.items()):
                raise ValueError("cached pair differs from current experiment")
            for arm, tree in (("reference", reference), ("candidate", candidate)):
                validate_arm(pair[arm], tree, order, 8351 + i, signature)
            pairs.append((pair["reference"], pair["candidate"]))
        result = judge(pairs)
        result["serving_seconds_measured"] = sum(r["wall_s"] for pair in pairs for r in pair)
        result["worker_seconds_including_startup"] = sum(r["total_elapsed_s"] for pair in pairs for r in pair)
        result["pilot_in_timed_path"] = True
        save(state / "result.json", result)
        if result["passed"] and args.publish:
            publish(state, order, candidate, args.gguf)


def wait_for_checks(repo, number, commit):
    required = {"linux / py3.12 (MLX CPU)", "linux / wheel and quick start"}
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        pr = json.loads(run(["gh", "pr", "view", number, "--repo", repo, "--json", "headRefOid,statusCheckRollup"]))
        if pr["headRefOid"] != commit:
            raise ValueError("candidate changed while waiting for CI")
        checks = pr["statusCheckRollup"]
        states = [c.get("conclusion") or c.get("state") for c in checks]
        if any(v in {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"} for v in states):
            raise RuntimeError("GitHub checks failed")
        names = {c.get("name") or c.get("context") for c in checks}
        if required <= names and all(v == "SUCCESS" for v in states):
            return
        time.sleep(10)
    raise TimeoutError("qualification checks did not finish; resume from saved pairs")


def publish(state, order, candidate, gguf):
    """A qualification branch is a deployment record. Only its evaluated tree may be activated."""
    from training.trial_release import activate

    prefix = order["order_id"][:16]
    deployment = "qualification/" + prefix
    branch = "experiment/" + prefix
    repo = "Tobayko/IronMule"
    origin = "https://github.com/" + repo + ".git"
    base = git("rev-parse", "HEAD", cwd=candidate)
    body = ("## Problem\n\n" + order["hypothesis"] + "\n\n## Change\n\n" +
            read(state / "proposal.json")["rationale"] + "\n\n## Paired qualification\n\n" +
            "```json\n" + json.dumps(read(state / "result.json"), indent=2) + "\n```\n\n" +
            "Qualified only on the recorded model, hardware, and fixed copy workload. "
            "Rollback restores the previous checkout with training.trial_release rollback.")
    # commit-tree avoids changing the immutable detached base used for replay.
    stored = state / "release-commit.json"
    tree = git("write-tree", cwd=candidate)
    if stored.exists():
        commit = read(stored)["commit"]
        if git("rev-parse", commit + "^{tree}", cwd=candidate) != tree:
            raise ValueError("release content changed since qualification")
    else:
        msg = ("perf: reduce work in " + order["target"]["function"] +
               "\n\nVerified: six counterbalanced serving pairs and engine tests.\n\n"
               "Co-Authored-By: Codex <noreply@openai.com>\n")
        commit = run(["git", "commit-tree", tree, "-p", base], cwd=candidate, input=msg)
        save(stored, {"commit": commit})
    remote = git("ls-remote", "--heads", origin, "refs/heads/" + deployment, cwd=candidate).split()
    if not remote:
        git("push", origin, base + ":refs/heads/" + deployment, cwd=candidate)
    git("push", origin, commit + ":refs/heads/" + branch, cwd=candidate)
    prs = json.loads(run(["gh", "pr", "list", "--repo", repo, "--base", deployment, "--head", branch,
                          "--state", "all", "--json", "number,state,headRefOid"]))
    if not prs:
        if remote and remote[0] != base:
            raise ValueError("deployment branch moved before PR creation")
        run(["gh", "pr", "create", "--repo", repo, "--head", branch, "--base", deployment, "--title",
             "perf: optimize " + order["target"]["function"], "--body-file", "-"], input=body)
        prs = json.loads(run(["gh", "pr", "list", "--repo", repo, "--base", deployment, "--head", branch,
                              "--json", "number,state,headRefOid"]))
    if len(prs) != 1 or prs[0]["headRefOid"] != commit:
        raise ValueError("remote candidate does not match evaluated source")
    number = str(prs[0]["number"])
    if prs[0]["state"] != "MERGED":
        # Checks may not be registered immediately after publishing. An empty set cannot admit code.
        wait_for_checks(repo, number, commit)
        run(["gh", "pr", "merge", number, "--repo", repo, "--merge", "--match-head-commit", commit])
    status = activate(state, order, int(number), gguf)
    save(state / "activation.json", status)
    if status["withdrawn"]:
        raise RuntimeError("qualification passed, target activation withdrawn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--publish", action="store_true", help="submit a passing candidate for bound deployment")
    parser.add_argument("--author-only", action="store_true")
    cycle(parser.parse_args())


if __name__ == "__main__":
    main()
