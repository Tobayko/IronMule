"""A real tiny MLX model exercises LoRA fitting and the product worker reload; qualification is independent."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ironmule import adaptation as adapt
from ironmule import factual_gate
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec


@pytest.fixture
def tiny(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.train_from_iterator(["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*<>,.:?"],
                                  trainer=trainers.BpeTrainer(vocab_size=100, special_tokens=["<unk>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = ("{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}"
                               "{% if add_generation_prompt %}assistant: {% endif %}")
    config = dict(model_type="llama", hidden_size=16, num_hidden_layers=1, intermediate_size=24,
                  num_attention_heads=2, rms_norm_eps=1e-5, vocab_size=len(tokenizer), eos_token_id=1)
    source = tmp_path / "model"
    mx.random.seed(2718)
    save_model(source, Model(ModelArgs.from_dict(config)))
    tokenizer.save_pretrained(source)
    (source / "config.json").write_text(json.dumps(config))
    store = ProductStore(tmp_path / "state")
    store.setup()
    base = ModelSpec("local/test", "tiny", str(source), (source / "model.safetensors").stat().st_size)
    store.register_model(base)
    return store, base


def outcomes(seed=0):
    return {case.key: case.key.endswith("-0") for case in factual_gate.cases(seed)}


def test_quality_regression_and_latency_prevent_promotion():
    scores = outcomes()
    ref = {"scores": scores, "seconds": 1., "nll": [1.] * 16}
    better = {**ref, "scores": {**scores, "0:arithmetic-1": True}}
    assert adapt._verdict(ref, ref, better, 0, ref)["passes"]
    assert not adapt._verdict(ref, ref, ref, 0, ref)["passes"]
    assert not adapt._verdict(ref, ref, {**ref, "scores": {**scores, "0:facts-1": True}}, 0, ref)["passes"]
    assert not adapt._verdict(ref, ref, {**better, "seconds": 1.06}, 0, ref)["passes"]
    replacement = {**better, "scores": {**better["scores"], "0:facts-0": False, "0:facts-1": True}}
    assert not adapt._verdict(ref, ref, replacement, 0, ref)["passes"]


def test_fused_snapshot_selected_only_on_pass_and_rollback_reloads_base(tiny, monkeypatch):
    import mlx.core as mx
    from ironmule_product.worker import _load

    store, base = tiny
    original = adapt._digest(base)
    argv = ["--state-dir", str(store.root), "--model", base.model_id]
    baseline = {"scores": outcomes(), "seconds": 1.0, "nll": [1.0] * 16}
    # The omitted real scoring is covered by QG1; force a win to test the registry transition.
    def measure(spec, seed):
        scores = outcomes(seed)
        if spec != base:
            scores[next(key for key in scores if key.endswith("extraction-1"))] = True
        return {**baseline, "scores": scores}
    monkeypatch.setattr(adapt, "_measure", measure)
    monkeypatch.setattr(adapt, "_perplexity", lambda spec: [1. if spec == base else 1.1] * 16)
    assert adapt.main(["train", "--steps", "1", *argv]) == 0
    assert store.models() == [base]
    monkeypatch.setattr(adapt, "_perplexity", lambda spec: [1.] * 16)
    assert adapt.main(["train", "--steps", "2", *argv]) == 0
    selected, = [spec for spec in store.models() if spec != base]
    assert adapt._digest(selected) == selected.revision != original
    assert adapt._digest(base) == original
    assert (Path(selected.snapshot_path).parent / "adapters.safetensors").is_file()
    adapted, _, _, _ = _load(selected.as_dict())
    clean, _, _, _ = _load(base.as_dict())
    prompt = mx.array([[2, 3, 4]])
    assert not mx.array_equal(adapted(prompt), clean(prompt)).item()

    monkeypatch.setattr(adapt, "_measure", lambda spec, seed: {**baseline, "scores": outcomes(seed)})
    assert adapt.main(["train", "--steps", "1", *argv]) == 0
    assert store.model(selected.model_id) == selected
    assert adapt.main(["rollback", *argv]) == 0
    assert store.model(selected.model_id) == replace(base, model_id=selected.model_id)


def test_supervision_uses_new_verified_values():
    run = list(adapt.examples(91, 20))
    assert run != list(adapt.examples(90, 20))
    for prompt, target in run:
        if prompt.startswith("Calculate"):
            a, op, b = prompt.removeprefix("Calculate ").split(".")[0].split()
            assert int(target) == {"+": int(a)+int(b), "-": int(a)-int(b), "*": int(a)*int(b)}[op]
        else:
            assert f"east: {target}." in prompt
