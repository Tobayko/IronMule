"""Opt-in, bounded LoRA adaptation of registered local MLX chat models.

A new alias selects a fully materialized, independently loadable snapshot. The original
registration always serves the base. Rejected candidates never reach the model registry.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

from ironmule import factual_gate
from ironmule.hw import fingerprint
from ironmule.model_identity import build_model_identity
from ironmule.numeric_choice import _text_ids, gate_bos, nll_per_chunk, ratio_interval
from ironmule_product.calibration import CalibrationFailure, model_lease
from ironmule_product.errors import ProductError
from ironmule_product.state import ProductStore, _atomic_write, _locked, _read_json
from ironmule_product.types import ModelSpec


def examples(seed, count):
    """Synthetic supervision from integer operations and answers copied from generated records."""
    rng = random.Random(seed)
    for i in range(count):
        if i % 2:
            x, y = rng.sample(range(100000, 999999), 2)
            yield f"Record west: {x}. Record east: {y}. Give only the east record number.", str(y)
        else:
            a, b = rng.randrange(2, 145), rng.randrange(2, 41)
            sign, value = (("+", a + b), ("-", a - b), ("*", a * b))[i // 2 % 3]
            yield f"Calculate {a} {sign} {b}. Reply with just the result.", str(value)


def _train(spec, work, steps, seed):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx import optimizers
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import save_model
    from ironmule_product.worker import _load, _render

    model, tokenizer, _, _ = _load(spec.as_dict())
    mx.random.seed(seed)
    model.freeze()
    parameters = {"rank": 4, "dropout": 0.0, "scale": 4.0}
    linear_to_lora_layers(model, 1, parameters)
    trainable = tree_flatten(model.trainable_parameters())
    if not trainable or any(not name.endswith(("lora_a", "lora_b")) for name, _ in trainable):
        raise ValueError("model does not support training only LoRA weights")
    del trainable
    optimizer = optimizers.Adam(learning_rate=1e-4)

    def loss(net, prefix, suffix):
        tokens = mx.array([prefix + suffix])
        predicted = net(tokens[:, :-1])[:, len(prefix) - 1:, :].astype(mx.float32)
        return nn.losses.cross_entropy(predicted, tokens[:, len(prefix):]).mean()

    gradient = nn.value_and_grad(model, loss)
    for prompt, answer in examples(seed, steps):
        prefix = list(_render(tokenizer, [{"role": "user", "content": prompt}]))
        suffix = list(tokenizer.encode(answer, add_special_tokens=False)) + [tokenizer.eos_token_id]
        value, grad = gradient(model, prefix, suffix)
        optimizer.update(model, grad)
        mx.eval(model.trainable_parameters(), optimizer.state)
        if not math.isfinite(value.item()):
            raise ValueError("nonfinite training loss")

    leaves = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(work / "adapters.safetensors"), leaves)
    (work / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 1, "lora_parameters": parameters,
    }))
    fused = [(name, module.fuse()) for name, module in model.named_modules()
             if hasattr(module, "lora_a") and hasattr(module, "lora_b")]
    model.update_modules(tree_unflatten(fused))
    snapshot = work / "model"
    save_model(snapshot, model)
    source = Path(spec.snapshot_path)
    # Keep the original tokenizer/config bytes, including named chat templates from newer snapshots.
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".model", ".tiktoken", ".txt", ".jinja"}:
            if path.name != "model.safetensors.index.json":
                target = snapshot / path.relative_to(source)
                target.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(path, target)


def _measure(spec, seed):
    from mlx_lm import generate
    from ironmule_product.worker import _load, _render

    model, tokenizer, _, _ = _load(spec.as_dict())
    durations = []

    def answer(prompt):
        start = time.perf_counter()
        ids = _render(tokenizer, [{"role": "user", "content": prompt}])
        text = generate(model, tokenizer, prompt=ids, max_tokens=factual_gate.MAX_TOKENS)
        durations.append(time.perf_counter() - start)
        return text

    answer("Give the number one in digits.")
    durations.clear()
    scores = factual_gate.evaluate(answer, seed=seed)
    return {"scores": scores, "seconds": statistics.median(durations)}


def _perplexity(spec):
    from ironmule_product.worker import _load

    model, tokenizer, _, _ = _load(spec.as_dict())
    return nll_per_chunk(SimpleNamespace(model=model), _text_ids(tokenizer), bos=gate_bos(tokenizer))


def _verdict(reference, repeat, candidate, seed, base):
    comparison = factual_gate.compare(reference["scores"], repeat["scores"], candidate["scores"], seed=seed)
    retained = all(candidate["scores"][key] for key in candidate["scores"]
                   if reference["scores"][key] or repeat["scores"][key] or base["scores"][key])
    # Facts are stable retention probes. Only improvement on a fresh arithmetic/extraction task can qualify a revision.
    challenges = [case.key for case in factual_gate.cases(seed) if case.category != "facts"]
    better = any(candidate["scores"][key] and not reference["scores"][key] and not repeat["scores"][key]
                 for key in challenges)
    baseline = statistics.mean((reference["seconds"], repeat["seconds"]))
    ratio = candidate["seconds"] / baseline
    # Bound cumulative slowdown as well as a single update's cost.
    base_ratio = candidate["seconds"] / base["seconds"]
    passed = comparison["passes"] and retained and better and 0 < ratio <= 1.05 and 0 < base_ratio <= 1.05
    return {"passes": passed, "factual": comparison, "retained": retained, "improved": better,
            "perplexity": None, "base_perplexity": None,
            "median_latency_ratio": ratio, "base_latency_ratio": base_ratio,
            "latency_seconds": {"reference": reference["seconds"], "repeat": repeat["seconds"],
                                "candidate": candidate["seconds"], "base": base["seconds"]}}


def _digest(spec):
    return build_model_identity(spec.model_id, Path(spec.snapshot_path), spec.revision).model_manifest_sha256


def _active(store, base, alias, root):
    current = next((s for s in store.models() if s.model_id == alias), None)
    if current is None or current == replace(base, model_id=alias):
        return base
    if (Path(current.snapshot_path) != root / current.revision / "model" or _digest(current) != current.revision):
        raise ValueError("adapted model registration does not match its qualified snapshot")
    report = _read_json(root / current.revision / "report.json") or {}
    if not report.get("passes") or report.get("revision") != current.revision:
        raise ValueError("selected adapter has no matching qualification report")
    return current


def _cycle(store, base, current, root, alias, steps):
    train_seed = secrets.randbits(32)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        work = Path(temporary)
        started = time.perf_counter()
        _train(current, work, steps, train_seed)
        training_seconds = time.perf_counter() - started
        snapshot = work / "model"
        identity = build_model_identity(alias, snapshot)
        revision = identity.model_manifest_sha256
        weight_bytes = sum(p.stat().st_size for p in snapshot.glob("model*.safetensors"))
        candidate = ModelSpec(alias, revision, str(snapshot), weight_bytes)
        # Pick the challenge after freezing the actual serialized model. Never train on the retention questions.
        challenge = secrets.randbits(63)
        reference = _measure(current, challenge)
        result = _measure(candidate, challenge)
        repeat = _measure(current, challenge)
        base_result = reference if current == base else _measure(base, challenge)
        report = _verdict(reference, repeat, result, challenge, base_result)
        if report["passes"]:
            candidate_nll, reference_nll = _perplexity(candidate), _perplexity(current)
            base_nll = reference_nll if current == base else _perplexity(base)
            report["perplexity"] = ratio_interval(reference_nll, candidate_nll)
            report["base_perplexity"] = ratio_interval(base_nll, candidate_nll)
            report["passes"] = report["perplexity"]["passes"] and report["base_perplexity"]["passes"]
        report.update(revision=revision, parent=current.revision, train_seed=train_seed,
                      steps=steps, alias=alias, challenge_seed=challenge, training_seconds=training_seconds,
                      base_model=base.model_id, base_revision=base.revision, hardware=fingerprint())
        if report["passes"]:
            if revision == _digest(current):
                raise ValueError("candidate equals the incumbent")
            destination = root / revision
            if not destination.exists():
                _atomic_write(work / "report.json", report)
                work.rename(destination)
            else:
                previous = _read_json(destination / "report.json") or {}
                if (previous.get("revision") != revision or previous.get("parent") != current.revision
                        or not previous.get("passes")):
                    raise ValueError("conflicting adapter history")
            selected = replace(candidate, snapshot_path=str(destination / "model"))
            if _digest(selected) != revision:
                raise ValueError("candidate contents changed during qualification")
            store.register_model(selected)
        _atomic_write(root / "latest.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ironmule adapt", description=__doc__)
    parser.add_argument("action", choices=("train", "status", "rollback"))
    parser.add_argument("--model", required=True, help="registered base MLX chat model")
    parser.add_argument("--state-dir", type=Path, help="IronMule product state directory")
    parser.add_argument("--steps", type=int, default=24, help="gradient updates per candidate (1–1024)")
    parser.add_argument("--cycles", type=int, default=1, help="maximum candidates, stopping on first rejection (1–16)")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1024 or not 1 <= args.cycles <= 16:
        parser.error("steps must be 1–1024 and cycles 1–16")
    try:
        return _run(args)
    except (ProductError, CalibrationFailure, ValueError, OSError) as error:
        print(f"ironmule adapt: {error}", file=sys.stderr)
        return 1


def _run(args):
    store = ProductStore(args.state_dir)
    base = store.model(args.model)
    if store.root.resolve().is_relative_to(Path(base.snapshot_path).resolve()):
        raise ValueError("product state must be outside the base model snapshot")
    identity = build_model_identity(base.model_id, Path(base.snapshot_path), base.revision)
    alias = "local/adapt-" + identity.identity_sha256[:24]
    root = store.root / "adapters" / identity.identity_sha256
    with (model_lease(store) if args.action == "train" else nullcontext()), _locked(root):
        current = _active(store, base, alias, root)
        if args.action == "train":
            for _ in range(args.cycles):
                report = _cycle(store, base, current, root, alias, args.steps)
                print(json.dumps(report, allow_nan=False))
                if not report["passes"]:
                    break
                current = _active(store, base, alias, root)
        elif args.action == "rollback" and current != base:
            parent = _read_json(root / current.revision / "report.json")["parent"]
            target = replace(base, model_id=alias)
            if parent != base.revision:
                if len(parent) != 64 or any(ch not in "0123456789abcdef" for ch in parent):
                    raise ValueError("invalid parent revision")
                snapshot = root / parent / "model"
                weight_bytes = sum(p.stat().st_size for p in snapshot.glob("model*.safetensors"))
                target = ModelSpec(alias, parent, str(snapshot), weight_bytes)
                prior = _read_json(root / parent / "report.json") or {}
                if _digest(target) != parent or prior.get("revision") != parent or not prior.get("passes"):
                    raise ValueError("previous qualified snapshot is unavailable")
            store.register_model(target)
        current = _active(store, base, alias, root)
        print(json.dumps({"selected": current.revision, "serve_model": current.model_id,
                          "last_trial": _read_json(root / "latest.json")}))
    return 0
