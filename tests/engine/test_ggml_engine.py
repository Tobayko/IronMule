"""GG1: the ggml engine (`ironmule/ggml_engine.py`), its policy version 3, the GGUF builder and the product wiring.

The engine itself needs llama.cpp's bindings and a GGUF; those tests run where `IRONMULE_GGML_BINDINGS` (or an
installed `llama-cpp-python`) and `IRONMULE_TEST_GGUF` are set, and are skipped elsewhere."""

from __future__ import annotations

import collections
import importlib.util
import os
from pathlib import Path

import pytest

from ironmule import speculation as sp

ROOT = Path(__file__).resolve().parents[2]


def test_policy_v3_probes_a_size_before_its_measured_price_may_veto_it():
    policy = sp.StepPolicy(rows=16)
    for _ in range(2):  # two expensive steps of 3 rows: not yet a price
        policy.observe_step(3, 0.100, True)
    for _ in range(3):  # one row is measured: 5 ms
        policy.observe_step(1, 0.005, False)
    for _ in range(50):
        policy.observe_drafts(3, 2, 2)
    assert policy.choose([([1, 2, 3, 4], 3)]) != [0]  # 3 rows still priced by the prior slope: tried again
    policy.observe_step(3, 0.100, True)  # the third: now its median prices it, and it does not pay
    policy.observe_step(2, 0.100, True)
    policy.observe_step(2, 0.100, True)
    policy.observe_step(2, 0.100, True)
    assert policy.choose([([1, 2], 3)]) == [0]
    assert policy.summary()["policy_version"] == 3 and policy.summary()["step_ms_by_rows"]["3"] == 100.0


def test_policy_v3_prices_by_the_median_of_its_recent_steps():
    policy = sp.StepPolicy(rows=8)
    for seconds in (0.010, 0.011, 0.500, 0.012):  # one outlier does not move a median
        policy.observe_step(2, seconds, False)
    assert policy.price(2) == pytest.approx(0.012)
    for _ in range(40):
        policy.observe_step(2, 0.020, False)
    assert policy.price(2) == pytest.approx(0.020) and len(policy.times[2]) == policy.WINDOW


def _builder():
    spec = importlib.util.spec_from_file_location("gguf_from_mlx", ROOT / "tools" / "gguf_from_mlx.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_the_gguf_builder_maps_llama_parameters_to_llama_cpp_names():
    tool = _builder()
    assert tool.gguf_name("model.embed_tokens.weight") == "token_embd.weight"
    assert tool.gguf_name("model.norm.weight") == "output_norm.weight"
    assert tool.gguf_name("model.layers.3.input_layernorm.weight") == "blk.3.attn_norm.weight"
    assert tool.gguf_name("model.layers.3.post_attention_layernorm.weight") == "blk.3.ffn_norm.weight"
    assert tool.gguf_name("model.layers.12.self_attn.k_proj.weight") == "blk.12.attn_k.weight"
    assert tool.gguf_name("model.layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"
    assert tool.gguf_name("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_the_gguf_builder_permutes_query_rows_as_llama_cpp_does():
    np = pytest.importorskip("numpy")
    tool = _builder()
    w = np.arange(2 * 4 * 3).reshape(8, 3)  # 2 heads of 4 rows: halves interleaved per head
    out = tool.permute(w, 2)
    assert out[:4, 0].tolist() == [0, 6, 3, 9] and out[4:, 0].tolist() == [12, 18, 15, 21]
    assert tool.permute(np.arange(12).reshape(4, 3), 4, 2).shape == (4, 3)


def test_the_worker_client_serves_ggml_only_with_a_gguf_whose_gate_passed():
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.types import ModelSpec

    spec = ModelSpec("local/test", "revision", "/tmp/none", 1)
    with pytest.raises(ValueError, match="needs ggml_gguf"):
        MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml")
    with pytest.raises(ValueError, match="quality gate"):
        MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml", ggml_gguf="/x.gguf",
                        ggml_gate={"passes": False})
    client = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml",
                             ggml_gguf="/x.gguf", ggml_gate={"passes": True}, speculate=True)
    assert client.streams_batches and client.reuses_slots and client.batch_capacity == 8


def test_serve_refuses_ggml_without_a_gguf(tmp_path, capsys):
    from ironmule_product import cli

    for argv, message in ((["--engine", "ggml"], "--engine ggml and --gguf go together"),):
        with pytest.raises(SystemExit):
            cli.serve(["--model", "local/test", "--state-dir", str(tmp_path / "state"), *argv])
        assert message in capsys.readouterr().err


def test_health_says_ggml_answers_may_differ_and_shows_its_gate(tmp_path):
    from ironmule import ggml_gate_diagnostics as diag
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    class Backend:
        device, ready, engine_configuration, execution_variant = "cpu", True, "ggml", "current_engine"
        ggml_gate = {"ratio": 1.003, "interval": [1.001, 1.0048], "passes": True, "seconds": 70,
                     "streams": {"passes": False, "prefill_batch": {"matches": False, "calls": 2, "private": "text"},
                                 "diagnostic": {**diag.make("acceptance", "quality_rejected"), "log": "private"}},
                     "diagnostic": {**diag.make("acceptance", "quality_rejected"), "private_log": "private"},
                     "kernel_generation": {"status": "skipped", "reason": "cuda_backend_has_no_kg1_hook"}}
        speculate, speculation_status = True, {"policy_version": 3, "drafted": 4}
        ready_payload = {"engine": {"ggml": {"speculation": "context"}}}

        def close(self):
            pass

    store = ProductStore(tmp_path / "state")
    store.setup("server")
    service = ProductService(store, backend=Backend(), spec=None)
    try:
        health = service.health()
    finally:
        service.close()
    assert health["execution"] == "numeric@ggml" and health["ggml"]["gate"]["passes"] is True
    assert health["ggml"]["stream_qualification"] == {
        "passes": False, "grouped_greedy_match": False, "grouped_calls": 2,
        "diagnostic": diag.make("acceptance", "quality_rejected")}
    assert health["ggml"]["kernel_generation"] == Backend.ggml_gate["kernel_generation"]
    assert health["ggml"]["gate"]["diagnostic"] == diag.make("acceptance", "quality_rejected")
    assert health["speculation"]["serving"] == "context" and health["speculation"]["learned"]["policy_version"] == 3


GGUF = os.environ.get("IRONMULE_TEST_GGUF")


@pytest.fixture(scope="module")
def engine():
    from ironmule import ggml_engine

    if not GGUF or ggml_engine.available() is not None:
        pytest.skip("needs llama.cpp's bindings and IRONMULE_TEST_GGUF")
    built = ggml_engine.GgmlEngine(GGUF, speculate=True)
    yield built
    built.close()


def test_the_engine_serves_joins_and_cancels_and_frees_every_sequence(engine):
    from ironmule.ggml_engine import Job

    prompt = engine.tokenize("<|im_start|>user\nCount: one two three one two three<|im_end|>\n<|im_start|>assistant\n")
    seen: list[int] = []
    joined = [Job(list(prompt), 8)]
    stop = {"now": False}
    late = Job(list(prompt), 40, cancelled=lambda: stop["now"])

    def admit(free: int) -> list[Job]:
        out, joined[:] = joined[:], []
        if out:
            stop["now"] = False
            out.append(late)
        elif late.tokens:
            stop["now"] = True  # cancelled after its first tokens
        return out

    served = engine.run([Job(list(prompt), 16, emit=seen.append)], admit=admit)
    assert [len(job.tokens) <= job.limit for job in served] == [True, True, True]
    assert served[0].tokens == seen and served[0].reason in ("eos", "length") and served[2].reason == "cancelled"
    assert engine.free_slots() == engine.seqs


def test_the_first_start_tunes_eight_settings_and_the_gate_tries_them_fastest_first():
    from ironmule import ggml_engine

    assert len(ggml_engine.SETTINGS) == 8 and len({tuple(sorted(s.items())) for s in ggml_engine.SETTINGS}) == 8
    assert all(set(s) == {"flash_attn", "kv_unified", "kv_f32"} for s in ggml_engine.SETTINGS)
    assert "|" in ggml_engine.machine_key()


def test_the_detokenizer_pool_is_built_before_the_first_request():
    """GG2: eight streams' detokenizers exist at the worker's start; a stream takes one without building it."""
    from ironmule_product import worker

    built = []

    class Detokenizer:
        def reset(self):
            pass

    class Tokenizer:
        @property
        def detokenizer(self):
            built.append(1)
            return Detokenizer()

    pool = worker._DetokenizerPool(Tokenizer())
    pool.prepare(worker.STREAM_WIDTH)
    assert len(built) == 8
    taken = [pool.acquire() for _ in range(8)]
    assert len(built) == 8 and len({id(d) for d in taken}) == 8
    pool.acquire()
    assert len(built) == 9  # a ninth stream still builds its own


def test_the_gate_s_reference_is_llama_cpp_s_default_setting():
    from ironmule import ggml_engine

    assert ggml_engine.DEFAULT_SETTINGS in ggml_engine.SETTINGS and ggml_engine.ENGINE_VERSION >= 2
    assert not hasattr(ggml_engine, "GATE_REFERENCE")  # the float32-on-MLX reference is gone for this engine


def _fake_gate_children(monkeypatch, option_nll, *, factual_regression=False):
    """`gate`'s child processes, stubbed: the tuning puts flash attention first, the reference is 16 chunks of 2.0."""
    import json as _json
    import subprocess
    import types as _types

    from ironmule import ggml_engine

    fast = {"flash_attn": True, "kv_unified": False, "kv_f32": False}
    monkeypatch.setattr(ggml_engine, "file_sha256", lambda path: "0" * 64)

    def run(command, input=None, **kwargs):
        code = command[2]
        if code == ggml_engine.TUNE_CHILD:
            out = {"order": [fast, ggml_engine.DEFAULT_SETTINGS], "measured": []}
        elif code == ggml_engine.GATE_IDS:
            from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS
            out = {"ids": [1] * (CHUNKS * CHUNK_TOKENS + 1), "bos": 1, "facts_sha256": "a" * 64}
        else:
            settings = _json.loads(input)["settings"]
            from ironmule.factual_gate import _protocol
            protocol = _protocol(0, _json.loads(input)["kg5_shape"])
            facts = {case.key: True for case in protocol}
            if _json.loads(input)["kg5_shape"]:
                assert kwargs["env"]["IRONMULE_GGML_PROMPT_BUDGET"] == "32768"
                assert kwargs["env"]["IRONMULE_PILOT"] == "0"
            if _json.loads(input)["reference"]:
                assert kwargs["env"]["IRONMULE_KG_LIB"] == "" and kwargs["env"]["IRONMULE_KG5"] == "0"
            if factual_regression and (settings != ggml_engine.DEFAULT_SETTINGS or kwargs["env"].get("IRONMULE_KG_LIB")):
                facts[protocol[0].key] = False
            out = {"nll": [2.0] * 16 if settings == ggml_engine.DEFAULT_SETTINGS else option_nll,
                   "facts": facts, "repeat": facts if _json.loads(input)["reference"] else None}
        return _types.SimpleNamespace(returncode=0, stdout=_json.dumps(out) + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return fast


def test_gg3_a_llama_cpp_option_within_the_bound_of_its_default_serves(monkeypatch, tmp_path):
    """GG3 (owner, 2026-11-11): flash attention 0.2 % worse than llama.cpp's default is llama.cpp's quality."""
    from ironmule import ggml_engine

    fast = _fake_gate_children(monkeypatch, [2.0 + 0.002 + 0.0001 * (k % 3) for k in range(16)])
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert ggml_engine.OPTION_BOUND == 1.005 and result["bound"] == 1.005
    assert result["passes"] and result["settings"] == fast and 1.0 < result["ratio"] < 1.005


def test_gg3_an_option_beyond_the_bound_does_not_serve_and_the_default_does(monkeypatch, tmp_path):
    from ironmule import ggml_engine

    _fake_gate_children(monkeypatch, [2.0 + 0.008] * 16)
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert result["passes"] and result["settings"] == ggml_engine.DEFAULT_SETTINGS
    assert [row["passes"] for row in result["tried"]] == [False, True] and result["tried"][0]["ratio"] > 1.005


def test_qg1_rejects_fast_option_with_one_lost_answer_even_if_likelihood_matches(monkeypatch, tmp_path):
    from ironmule import ggml_engine

    _fake_gate_children(monkeypatch, [2.0] * 16, factual_regression=True)
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert result["passes"] and result["settings"] == ggml_engine.DEFAULT_SETTINGS
    assert result["tried"][0]["ratio"] == 1
    factual = result["tried"][0]["factual"]
    assert not factual["passes"] and not factual["by_category"]["arithmetic"]["passes"]


def test_qg1_kg5_requires_actual_candidate_dispatch(monkeypatch, tmp_path):
    from ironmule import ggml_engine

    _fake_gate_children(monkeypatch, [2.0] * 16)
    monkeypatch.setenv("IRONMULE_KG5", "1")
    monkeypatch.setenv("IRONMULE_KG5_SHAPE", "1024,32,8")
    monkeypatch.setattr(ggml_engine, "gpu_layers", lambda L=None: 999)
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert result["settings"] == ggml_engine.DEFAULT_SETTINGS
    assert result["tried"][0]["factual"]["passes"] and not result["tried"][0]["passes"]
    assert not result["streams"]["passes"]
    assert result["tried"][0]["factual"]["total"] == 32
    assert result["tried"][0]["factual"]["shape"] == "1024,32,8"


def test_qg1_cannot_choose_uncertified_generated_weights_at_default_settings(monkeypatch, tmp_path):
    from ironmule import ggml_engine

    _fake_gate_children(monkeypatch, [2.0] * 16, factual_regression=True)
    library = tmp_path / "kg.so"
    library.write_bytes(b"stand-in")
    monkeypatch.setenv("IRONMULE_KG_LIB", str(library))
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert not result["passes"]
    assert result["tried"][-1]["settings"] == ggml_engine.DEFAULT_SETTINGS
    assert not result["tried"][-1]["factual"]["passes"]
    diagnostic = result["diagnostic"]
    assert (diagnostic["stage"], diagnostic["code"]) == ("acceptance", "quality_rejected")
    assert len(diagnostic["candidates"]) == 2
    for candidate in diagnostic["candidates"]:
        assert candidate["nll_upper"] == 1.0 and candidate["bound"] == ggml_engine.OPTION_BOUND
        assert candidate["factual"]["arithmetic"] == {
            "total": 6, "reference_correct": 6, "repeat_correct": 6, "candidate_correct": 5}


@pytest.mark.parametrize("stage,returncode,body,expected", [
    ("tokenize", 1, "FileNotFoundError: fixture missing", "missing_fixture"),
    ("tuning", 0, "", "invalid_result"),
    ("reference", -9, "", "child_signal"),
    ("candidate", 1, "RuntimeError: CUDA error: out of memory", "out_of_memory"),
])
def test_gate_reports_the_actual_failed_child_without_its_text(monkeypatch, tmp_path, stage, returncode, body, expected):
    import json
    import subprocess
    from types import SimpleNamespace
    from ironmule import ggml_engine as ge

    _fake_gate_children(monkeypatch, [2.0] * 16)
    run = subprocess.run

    def fail_at(command, **kwargs):
        code = command[2]
        role = ("tokenize" if code == ge.GATE_IDS else "tuning" if code == ge.TUNE_CHILD
                else "reference" if json.loads(kwargs["input"])["reference"] else "candidate")
        if role == stage:
            return SimpleNamespace(returncode=returncode, stdout="/secret/run/private",
                                   stderr=body + " /secret/run/private")
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_at)
    result = ge.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    assert not result["passes"]
    diagnostic = result["diagnostic"]
    assert (diagnostic["stage"], diagnostic["code"], diagnostic["exit_code"]) == (stage, expected, returncode)
    assert "/secret/" not in json.dumps(result)


def _stub_engine(seqs: int = 4, seq_rm_ok: bool = True):
    """A GgmlEngine whose llama.cpp is a stub: every decode returns the same argmax (token 7), never end of text."""
    import threading
    import types

    import numpy as np

    from ironmule import ggml_engine as ge

    fed: list[list[tuple[int, int, int]]] = []
    removed: list[tuple[int, int, int]] = []
    logits = np.zeros((64, 16), dtype=np.float32)
    logits[:, 7] = 1.0

    class Batch:
        def __init__(self, n: int):
            self.token, self.pos, self.n_seq_id, self.logits = [0] * n, [0] * n, [0] * n, [0] * n
            self.seq_id, self.n_tokens = [[0] for _ in range(n)], 0

    def decode(ctx, b):
        fed.append([(b.token[i], b.pos[i], b.seq_id[i][0]) for i in range(b.n_tokens)])
        return 0

    def seq_rm(memory, seq, start, end):
        removed.append((seq, start, end))
        return seq_rm_ok or start < 0

    engine = ge.GgmlEngine.__new__(ge.GgmlEngine)
    engine.L = types.SimpleNamespace(
        llama_decode=decode, llama_memory_seq_rm=seq_rm, llama_vocab_is_eog=lambda vocab, token: False,
        llama_get_logits_ith=lambda ctx, i: np.ctypeslib.as_ctypes(logits[i]),
        llama_memory_clear=lambda memory, data: None)
    engine.np, engine.ctx, engine.vocab, engine.n_vocab, engine.memory = np, None, None, 16, None
    engine.seqs, engine.seq_ctx, engine.threads, engine.batch, engine.policy = seqs, 256, 1, Batch(256), None
    engine._free, engine._kept, engine.prefix_reuse = list(range(seqs)), {}, True
    engine.prompt_tokens = engine.reused_tokens = engine.steps = 0
    engine._lock = threading.Lock()
    return engine, fed, removed


def test_a_new_request_reuses_the_longest_kept_prefix_and_computes_the_rest():
    from ironmule.ggml_engine import Job

    engine, fed, removed = _stub_engine()
    shared = [1, 2, 3, 4, 5, 6]
    first = engine.run([Job(shared + [8, 9], 3)])[0]
    assert first.tokens == [7, 7, 7] and engine.free_slots() == 4
    # the finished request keeps every token that went through the model: its prompt and all but its last answer token
    assert engine._kept == {0: shared + [8, 9, 7, 7]} and removed[-1] == (0, 10, -1)
    fed.clear()
    second = engine.run([Job(shared + [10, 11], 2)])[0]
    prompt_rows = fed[0]
    assert [pos for _, pos, _ in prompt_rows] == [6, 7] and {seq for _, _, seq in prompt_rows} == {0}
    assert engine.reused_tokens == 6 and engine.prompt_tokens == 16 and second.tokens == [7, 7]
    fed.clear()
    engine.run([Job(shared + [10, 11], 2)])  # the same prompt again: all but its last token are kept
    assert [pos for _, pos, _ in fed[0]] == [7]


def test_prefix_reuse_falls_back_to_a_full_prompt_when_the_cache_refuses_a_partial_removal():
    from ironmule.ggml_engine import Job

    engine, fed, removed = _stub_engine(seq_rm_ok=False)
    engine.run([Job([1, 2, 3, 4], 2)])
    assert engine._kept == {}  # finish could not trim the sequence, so it was cleared and nothing is kept
    fed.clear()
    engine.run([Job([1, 2, 3, 5], 2)])
    assert [pos for _, pos, _ in fed[0]] == [0, 1, 2, 3] and engine.reused_tokens == 0


def test_prefix_reuse_off_computes_every_prompt():
    from ironmule.ggml_engine import Job

    engine, fed, _ = _stub_engine()
    engine.prefix_reuse = False
    engine.run([Job([1, 2, 3, 4], 2)])
    fed.clear()
    engine.run([Job([1, 2, 3, 4], 2)])
    assert [pos for _, pos, _ in fed[0]] == [0, 1, 2, 3] and engine._kept == {}


def test_reused_prompts_answer_as_fresh_ones_on_the_real_model(engine):
    from ironmule.ggml_engine import Job

    prompt = engine.tokenize("<|im_start|>user\nName three colours of the rainbow.<|im_end|>\n<|im_start|>assistant\n")
    engine._clear()
    fresh = engine.run([Job(list(prompt), 12)])[0].tokens
    before = engine.reused_tokens
    again = engine.run([Job(list(prompt), 12)])[0].tokens
    assert engine.reused_tokens - before == len(prompt) - 1 and again == fresh


def test_seq1_the_width_doubles_only_while_a_doubling_buys_ten_percent():
    from ironmule.ggml_engine import decide_width

    assert decide_width({8: 63.0, 16: 103.0, 32: 141.0}) == 32  # ROW1 round 5: 1.6x, then 1.4x
    assert decide_width({8: 100.0, 16: 105.0, 32: 200.0}) == 8  # 5 % is not worth the memory; it stops there
    assert decide_width({8: 100.0, 16: 112.0, 32: 120.0, 64: 200.0}) == 16
    assert decide_width({4: 10.0}) == 4


def test_seq1_only_widths_whose_kv_cache_fits_beside_the_model_and_a_reserve_are_tried():
    from ironmule.ggml_engine import fitting_widths

    gib, cell = 2**30, 147_456  # Qwen3-4B, f16 KV: 36 layers x 8 KV heads x (128 + 128) x 2 bytes
    assert fitting_widths(cell, 2048, int(4.3 * gib), 15 * gib, 15 * gib) == [8, 16, 32]
    assert fitting_widths(cell, 1024, int(4.3 * gib), 15 * gib, 15 * gib) == [8, 16, 32, 64]
    assert fitting_widths(cell, 2048, int(4.3 * gib), 8 * gib, 15 * gib) == [4]  # 8 sequences would not fit
    assert fitting_widths(cell, 2048, 7 * gib, 8 * gib, 15 * gib) == [1]  # nothing fits: one sequence, never zero


def test_seq1_the_choice_is_measured_once_and_remembered(monkeypatch, tmp_path):
    import subprocess
    import types

    from ironmule import ggml_engine

    calls = []

    def run(argv, **kw):  # the KV-size child, then the width child (it alone calls measure_widths)
        calls.append(argv[2])
        out = ('{"cell_bytes": 1000, "rates": {"8": 50.0, "16": 90.0, "32": 95.0, "64": 96.0}, "step_s": {}}'
               if "measure_widths" in argv[2] else '{"cell_bytes": 1000}')
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    (tmp_path / "m.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(ggml_engine, "memory_bytes", lambda: (16 * 2**30, 16 * 2**30))
    first = ggml_engine.choose_sequences(str(tmp_path / "m.gguf"), {"flash_attn": True}, cache=tmp_path / "c")
    again = ggml_engine.choose_sequences(str(tmp_path / "m.gguf"), {"flash_attn": True}, cache=tmp_path / "c")
    # AG10: the widths are measured under the gated setting and under llama.cpp's default, for the agent's choice
    assert first["sequences"] == 16 and first["fitting"] == [8, 16, 32, 64] and again == first and len(calls) == 3
    assert len(first["measured"]) == 2
    monkeypatch.setenv("IRONMULE_KG_LIB", str(tmp_path / "kg.so"))
    custom = ggml_engine.choose_sequences(str(tmp_path / "m.gguf"), {"flash_attn": True}, cache=tmp_path / "c")
    assert len(custom["measured"]) == 1, "the default setting is not prequalified with generated kernels"


def test_seq1_a_failed_measurement_keeps_eight_sequences(monkeypatch, tmp_path):
    import subprocess
    import types

    from ironmule import ggml_engine

    (tmp_path / "m.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="x"))
    result = ggml_engine.choose_sequences(str(tmp_path / "m.gguf"), {}, cache=tmp_path / "c")
    assert result["sequences"] == 8 and "8 sequences as before" in result["reason"]
    assert not (tmp_path / "c").exists()  # a failure is not remembered


def test_seq1_widths_are_measured_on_one_loaded_model(engine):
    from ironmule.ggml_engine import measure_widths

    out = measure_widths(GGUF, dict(engine.settings), 256, [1, 2], steps=2, repeats=1)
    assert out["cell_bytes"] > 0 and set(out["rates"]) == {"1", "2"} and all(r > 0 for r in out["rates"].values())
    assert all(out[key][width] > 0 for key in ("context_build_s", "context_release_s") for width in ("1", "2"))


def test_seq1_the_memory_seen_is_the_container_s_limit_with_its_file_cache_free(monkeypatch, tmp_path):
    from pathlib import Path as RealPath

    from ironmule import ggml_engine

    group = tmp_path / "sys/fs/cgroup/memory/box"
    group.mkdir(parents=True)
    (group / "memory.limit_in_bytes").write_text(str(14 * 2**30))
    (group / "memory.usage_in_bytes").write_text(str(5 * 2**30))
    (group / "memory.stat").write_text(f"total_cache {4 * 2**30}\ntotal_shmem {2**30}\ntotal_rss {2**30}\n")
    files = {"/proc/self/cgroup": "4:memory:/box\n0::/\n",
             "/proc/meminfo": "MemTotal: 16777216 kB\nMemAvailable: 16000000 kB\n"}

    def path(name, *rest):
        name = str(name)
        if name in files:
            target = tmp_path / name.strip("/").replace("/", "_")
            target.write_text(files[name])
            return RealPath(target)
        if name.startswith("/sys/fs/cgroup"):
            return RealPath(tmp_path / name.lstrip("/"), *rest)
        return RealPath(name, *rest)

    monkeypatch.setattr(ggml_engine, "Path", path)
    available, total = ggml_engine.memory_bytes()
    # used: 5 GiB minus 3 GiB of reclaimable file cache (4 GiB of cache, 1 GiB of it shared memory)
    assert total == 14 * 2**30 and available == 12 * 2**30


def _pool_engine(cells=1000, cap=4):
    """A SharedPool on a stand-in engine that records llama.cpp's memory calls."""
    import types

    from ironmule import ggml_engine as ge

    calls = []
    L = types.SimpleNamespace(llama_memory_seq_rm=lambda m, s, a, b: calls.append(("rm", s, a, b)) or True,
                              llama_memory_seq_cp=lambda m, a, b, p0, p1: calls.append(("cp", a, b, p0, p1)))
    engine = types.SimpleNamespace(L=L, memory=None, prefix_reuse=True)
    return ge.SharedPool(engine, cells, cap, 8, 2), calls


def test_sp1_a_request_shares_the_longest_prefix_any_sequence_holds():
    pool, calls = _pool_engine()
    system = list(range(100))
    a = pool.place(system + [500, 501], 10)
    assert a == (0, 0)  # nothing to share yet
    pool.sync(0, system + [500, 501, 7], 9)  # its prompt went through the model, one token generated
    b = pool.place(system + [600], 10)
    assert b == (1, 100) and ("cp", 0, 1, 0, 100) in calls
    pool.sync(1, system + [600], 10)
    assert pool.used == 104  # the 100 shared cells once, 3 of the first sequence's own, 1 of the second's
    pool.release(0, system + [500, 501, 7])
    pool.release(1, None)  # cancelled: its own cells go, the shared ones stay with the first (kept as a cache)
    assert pool.used == 103 and list(pool.kept) == [0] and pool.slots() == 4


def test_sp1_a_full_pool_evicts_finished_sequences_first_and_otherwise_makes_a_request_wait():
    pool, calls = _pool_engine(cells=300, cap=4)
    pool.place(list(range(200)), 10)
    pool.sync(0, list(range(200)), 10)
    pool.release(0, list(range(200)))  # 200 cells kept as a cache
    seq, keep = pool.place([9000 + i for i in range(150)], 20)  # needs 150 + 20 + 2; the cache must go
    assert keep == 0 and 0 not in pool.kept and ("rm", 0, -1, -1) in calls and pool.used == 0
    pool.sync(seq, [9000 + i for i in range(150)], 20)
    assert pool.place([7000 + i for i in range(150)], 20) is None  # 150 used + 22 reserved + 172 > 300: it waits


def test_kvd1_a_keep_budget_evicts_the_oldest_finished_sequences_when_a_request_ends():
    pool, calls = _pool_engine(cells=1000, cap=4)
    pool.keep_cells = 250
    for k in range(3):  # three finished requests of 100 cells each, nothing shared
        ids = [1000 * (k + 1) + i for i in range(100)]
        seq, _ = pool.place(ids, 5)
        pool.sync(seq, ids, 5)
        pool.release(seq, ids)
    assert pool.kept_cells() == 200 and list(pool.kept) == [1, 2] and ("rm", 0, -1, -1) in calls
    seq, keep = pool.place([2000 + i for i in range(100)] + [7], 5)  # a kept prefix still serves while it is kept
    assert keep == 100
    pool.keep_cells = None  # without a budget every finished sequence stays until the cells are needed
    pool.release(seq, [2000 + i for i in range(100)] + [7, 8])
    assert pool.kept_cells() == 202


def test_sp1_a_finished_sequence_is_continued_past_the_shared_prefix():
    pool, calls = _pool_engine()
    pool.place(list(range(50)) + [1, 2], 5)
    pool.sync(0, list(range(50)) + [1, 2, 3], 4)
    pool.release(0, list(range(50)) + [1, 2, 3])
    seq, keep = pool.place(list(range(50)) + [8], 5)
    assert (seq, keep) == (0, 50) and ("rm", 0, 50, -1) in calls and pool.used == 50


def test_sp4_drafts_come_from_other_requests_answers_in_their_own_bucket():
    from ironmule.speculation import COLLECTIVE_BUCKET, CollectiveIndex

    index = CollectiveIndex(keys=1000)
    index.add([5, 6, 7, 8, 9, 10, 11, 12])
    assert index.draft([1, 2, 6, 7, 8, 9], 3) == ([10, 11, 12], COLLECTIVE_BUCKET + 4)
    assert index.draft([1, 2, 3, 8, 9], 2) == ([10, 11], COLLECTIVE_BUCKET + 2)
    assert index.draft([1, 2, 3], 4) == ([], 0)
    index.add([1, 8, 9, 40, 41])  # the most recent answer wins for the same n-gram
    assert index.draft([0, 8, 9], 2) == ([40, 41], COLLECTIVE_BUCKET + 2)
    small = CollectiveIndex(keys=5)
    small.add(list(range(50)))
    assert len(small.table) == 5  # bounded, oldest keys go first


def test_pad1_fills_a_step_only_to_a_size_measured_cheaper():
    from ironmule.ggml_engine import pad_table

    # MG2 v4 on a T4: 4 rows 33 ms, 5-8 rows about 50 ms, more than 8 cheaper again (CUDA's matmul kernels)
    prices = {1: 0.0196, 2: 0.0210, 3: 0.0278, 4: 0.0331, 5: 0.0500, 6: 0.0510, 7: 0.0520, 8: 0.0525, 9: 0.0380,
              10: 0.0385, 16: 0.0410}
    table = pad_table(prices)
    assert table == {5: 9, 6: 9, 7: 9, 8: 9}
    assert pad_table({1: 0.10, 2: 0.11, 3: 0.12}) == {}  # a CPU: every larger step costs more
    assert pad_table({1: 0.10, 2: 0.097}) == {}  # within 5 %: not worth a guess


def test_pad1_guesses_change_no_answer_on_the_real_model(engine):
    from ironmule.ggml_engine import Job

    asks = ["Count from 1 to 20.", "Name three colours of the rainbow."]
    prompts = [engine.tokenize(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n") for q in asks]
    saved = getattr(engine, "pad_to", {})
    try:
        engine.pad_to = {}
        plain = [j.tokens for j in engine.run([Job(list(p), 12) for p in prompts])]
        engine.pad_to, engine.padded = {t: 9 for t in range(1, 9)}, 0
        padded = [j.tokens for j in engine.run([Job(list(p), 12) for p in prompts])]
        assert padded == plain and engine.padded > 0
    finally:
        engine.pad_to = saved


def test_the_common_prefix_matches_the_token_by_token_loop():
    import random

    from ironmule.ggml_engine import GgmlEngine
    from ironmule.draft_model import _common as drafter_common

    def loop(a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        return i

    rng = random.Random(3)
    for _ in range(3000):
        base = [rng.randrange(4) for _ in range(rng.randrange(0, 300))]
        a = base + [rng.randrange(4) for _ in range(rng.randrange(0, 80))]
        b = base[:rng.randrange(0, len(base) + 1)] + [rng.randrange(4) for _ in range(rng.randrange(0, 80))]
        if rng.random() < 0.2:
            b = tuple(b)
        assert GgmlEngine._common(a, b) == loop(a, b) == drafter_common(a, b)


def test_sp1_the_prefix_trie_picks_what_the_scan_of_every_sequence_picked():
    """The pool's trie against the scan it replaced (the longest common prefix over every held sequence, the earliest
    sequence among equals), through random places, syncs (growing and shrinking), releases and evictions."""
    import random

    from ironmule.ggml_engine import GgmlEngine

    def scan(pool, ids):
        best, match = None, 0
        for seq, held in pool.tokens.items():
            n = GgmlEngine._common(held, ids)
            if n > match:
                best, match = seq, n
        return best, match

    rng = random.Random(11)
    for _ in range(60):
        pool, _ = _pool_engine(cells=rng.choice([300, 2000]), cap=6)
        prefixes = [[rng.randrange(6) for _ in range(rng.randrange(1, 30))] for _ in range(4)]
        running = {}
        for _ in range(120):
            ids = rng.choice(prefixes) + [rng.randrange(6) for _ in range(rng.randrange(0, 15))]
            if len(ids) < 2:
                continue
            assert pool.longest(ids) == scan(pool, ids)
            op = rng.random()
            if op < 0.45 and pool.slots():
                placed = pool.place(ids, rng.randrange(1, 20))
                if placed is not None:
                    seq, keep = placed
                    running[seq] = list(ids)
                    pool.sync(seq, ids, 5)
            elif op < 0.75 and running:
                seq = rng.choice(list(running))
                held = running[seq]
                held = held[:rng.randrange(1, len(held) + 1)] + [rng.randrange(6) for _ in range(rng.randrange(0, 6))]
                running[seq] = held
                pool.sync(seq, held + [99], 3, length=len(held))
            elif running:
                seq = rng.choice(list(running))
                held = running.pop(seq)
                pool.release(seq, held if rng.random() < 0.7 else None)
            for seq, held in pool.tokens.items():  # the trie holds exactly what the pool's lists hold
                assert [node[3] for node in pool.trie.paths[seq]] == held


def test_st1_a_float32_kv_cache_ranks_first_only_well_ahead():
    """Two fresh starts on the same machine ranked f32 KV first and third on tuning noise; it served 30 % slower."""
    from ironmule.ggml_engine import rank_settings

    measured = [{"flash_attn": True, "kv_unified": True, "kv_f32": True, "score": 25.92},  # the noisy first start
                {"flash_attn": True, "kv_unified": True, "kv_f32": False, "score": 24.57},
                {"flash_attn": False, "kv_unified": False, "kv_f32": False, "score": 21.27}]
    assert [r["kv_f32"] for r in rank_settings(measured)] == [False, True, False]
    clearly = [dict(measured[0], score=30.0), measured[1]]
    assert rank_settings(clearly)[0]["kv_f32"] is True


def test_kg1_the_gate_is_keyed_by_the_kernel_library_it_ran_with(tmp_path, monkeypatch):
    from ironmule import ggml_engine as ge

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF weights")
    monkeypatch.delenv("IRONMULE_KG_LIB", raising=False)
    plain = ge.gate_key(str(gguf), str(gguf))
    library = tmp_path / "kg.so"
    library.write_bytes(b"one library")
    monkeypatch.setenv("IRONMULE_KG_LIB", str(library))
    with_library = ge.gate_key(str(gguf), str(gguf))
    library.write_bytes(b"another library")
    assert len({plain, with_library, ge.gate_key(str(gguf), str(gguf))}) == 3


def test_pi2_the_shared_pool_starts_prompts_within_a_set_budget(monkeypatch):
    from ironmule import ggml_engine
    from ironmule.ggml_engine import Job

    if not GGUF or ggml_engine.available() is not None:
        pytest.skip("needs llama.cpp's bindings and IRONMULE_TEST_GGUF")
    built = ggml_engine.GgmlEngine(GGUF, seqs=8, seq_ctx=512, speculate=False, shared=True, cap=8)
    try:
        def prompts(first: int) -> list[list[int]]:  # new prompts each time: the pool keeps what it computed
            return [built.tokenize(f"<|im_start|>user\nName a colour, number {i}.<|im_end|>\n<|im_start|>assistant\n")
                    for i in range(first, first + 4)]

        decode = built.L.llama_decode
        calls = {"n": 0}

        def counted(*args):
            calls["n"] += 1
            return decode(*args)

        monkeypatch.setattr(built.L, "llama_decode", counted)

        def steps(budget: int, first: int) -> tuple[int, list[list[int]]]:
            monkeypatch.setattr(ggml_engine, "PROMPT_BUDGET", budget)
            jobs = [Job(p, 4) for p in prompts(first)]
            calls["n"] = 0
            built.run(jobs)
            assert all(len(job.tokens) == 4 or job.reason == "eos" for job in jobs)
            return calls["n"], [job.tokens for job in jobs]

        together, _ = steps(0, 0)  # no budget: every prompt starts in the first step
        staggered, _ = steps(1, 10)  # a budget of one token: one prompt per step (at least one starts)
        assert staggered >= together + 3
    finally:
        built.close()


def test_kg3_with_one_kv_stream_per_sequence_a_new_prompt_gets_its_own_call():
    from ironmule.ggml_engine import Job

    engine, fed, _ = _stub_engine()
    engine.settings = {"kv_unified": False}  # one KV stream per sequence
    first = Job([1, 2, 3], 4)
    late = [Job([5, 6, 7, 8], 2)]

    def admit(room):
        return [late.pop()] if late and len(fed) >= 2 else []

    engine.run([first], admit=admit)
    for call in fed:  # no call mixes another sequence's single token with a prompt of several tokens
        seqs = {s for _, _, s in call}
        assert len(seqs) == 1 or len(call) == len(seqs), call
    assert any(len(call) == 4 and {s for _, _, s in call} == {call[0][2]} for call in fed)  # the late prompt alone
    engine2, fed2, _ = _stub_engine()
    engine2.settings = {"kv_unified": True}  # the shared pool keeps batching a prompt into the running step
    late = [Job([5, 6, 7, 8], 2)]

    def admit2(room):
        return [late.pop()] if late and len(fed2) >= 2 else []

    engine2.run([Job([1, 2, 3], 4)], admit=admit2)
    assert any(len({s for _, _, s in call}) == 2 and len(call) == 5 for call in fed2)


def _streams_engine():
    """The stub engine with one KV stream per sequence, which, as llama.cpp does, refuses a row that does not continue
    its stream (a sequence's first row in a call must sit right past what the stream holds)."""
    engine, fed, removed = _stub_engine()
    engine.settings = {"kv_unified": False}
    engine.L.llama_vocab_bos = lambda vocab: 1
    top: dict[int, int] = {}
    decode, rm = engine.L.llama_decode, engine.L.llama_memory_seq_rm

    def checked(ctx, b):
        first: dict[int, int] = {}
        for i in range(b.n_tokens):
            first.setdefault(b.seq_id[i][0], b.pos[i])
        for seq, pos in first.items():
            assert pos == top.get(seq, -1) + 1, (seq, pos, top.get(seq, -1))
        for i in range(b.n_tokens):
            top[b.seq_id[i][0]] = max(top.get(b.seq_id[i][0], -1), b.pos[i])
        return decode(ctx, b)

    def trimmed(memory, seq, start, end):
        ok = rm(memory, seq, start, end)
        if ok:
            top[seq] = -1 if start < 0 else min(top.get(seq, -1), start - 1)
        return ok

    engine.L.llama_decode, engine.L.llama_memory_seq_rm = checked, trimmed
    engine.L.llama_memory_seq_pos_max = lambda memory, seq: top.get(seq, -1)
    return engine, fed, removed


def test_step_observer_accounts_for_separate_prefills_and_decode_gap_fillers():
    from ironmule.ggml_engine import Job

    engine, calls, _ = _streams_engine()
    steps = []
    engine.observer = lambda *step: steps.append(step)
    # Prompt IDs are >= 10. Generated tokens are 7 and gap fillers use BOS (1).
    jobs = [Job([11, 12], 6), Job([13, 14], 2), Job([15, 16], 6)]
    engine.run(jobs)

    assert len(steps) == len(calls)
    assert any(all(t >= 10 for t, _, _ in call) for call in calls)
    assert any(any(t == 1 for t, _, _ in call) for call in calls)
    for (decode, prompt, _, seconds), call in zip(steps, calls, strict=True):
        assert prompt == sum(t >= 10 for t, _, _ in call)
        assert decode == len(call) - prompt
        assert seconds >= 0


def test_grouped_prefill_maps_each_argmax_to_its_own_stream_and_preserves_prefix_offsets():
    from ironmule.ggml_engine import Job

    engine, calls, _ = _streams_engine()
    engine.prefill_batch = True
    engine.n_vocab = 20
    logits = engine.np.zeros((4, 20), dtype=engine.np.float32)
    for seq in range(4):
        logits[seq, 10 + seq] = 1
    engine.L.llama_get_logits_ith = lambda ctx, i: engine.np.ctypeslib.as_ctypes(logits[calls[-1][i][2]])
    prompts = [[n] * 70 + [3, 4] for n in (14, 16, 15)]
    jobs = engine.run([Job(p, 1) for p in prompts])
    assert [j.tokens for j in jobs] == [[10], [11], [12]]
    calls.clear()
    observed = []
    engine.observer = lambda *step: observed.append(step)
    # Prefix tokens are already on each stream. All three new suffixes fit one native call.
    second = engine.run([Job(p[:-2] + [6, 8], 1) for p in prompts])
    assert [j.tokens for j in second] == [[10], [11], [12]]
    assert len(calls) == 1 and observed[0][:3] == (0, 6, 210)
    assert [(pos, seq) for _, pos, seq in calls[0]] == [(p, s) for s in range(3) for p in (70, 71)]
    assert engine.prefill_batch_calls == 2 and engine.prefill_batch_prompts == 6
    assert engine.prefill_batch_rows == 3 * 72 + 6
    assert engine.free_slots() == 4


def test_grouped_prefill_uses_contiguous_equal_suffixes_and_respects_native_row_limit():
    from ironmule.ggml_engine import Job

    engine, _, _ = _streams_engine()
    engine.seq_ctx = 10
    shape = [(0, 4, 1), (1, 3, 0), (3, 3, 0), (4, 3, 0), (5, 3, 0), (6, 3, 0), (7, 1, 0)]
    jobs = [(seq, Job([12] * length, 1), keep) for seq, length, keep in shape]
    assert [[seq for seq, _, _ in g] for g in engine._prefill_groups(jobs)] == [[0, 1], [3, 4, 5], [6], [7]]


def test_failed_grouped_prefill_releases_reserved_streams_for_the_next_request(monkeypatch):
    from ironmule.ggml_engine import Job

    engine, _, _ = _streams_engine()
    engine.prefill_batch = True
    with monkeypatch.context() as failing:
        failing.setattr(engine.L, "llama_decode", lambda ctx, batch: -1)
        with pytest.raises(RuntimeError, match="llama_decode failed"):
            engine.run([Job([12] * length, 1) for length in (2, 2, 3, 3)])
    assert engine.free_slots() == 4 and not engine._kept
    assert engine.run([Job([15, 16], 1)])[0].tokens == [7]


@pytest.mark.parametrize(("matches", "calls", "accepted"), [(True, 2, True), (False, 2, False), (True, 0, False)])
def test_grouped_prefill_gate_exercises_candidate_and_separates_the_stock_reference(
        monkeypatch, tmp_path, matches, calls, accepted):
    import json
    import subprocess
    from types import SimpleNamespace
    from ironmule import ggml_engine as ge
    from ironmule.numeric_choice import CHUNKS

    _fake_gate_children(monkeypatch, [2.0] * CHUNKS)
    probe = subprocess.run
    monkeypatch.setattr(ge, "machine_key", lambda: "fixture")
    monkeypatch.setattr(ge, "gpu_layers", lambda: 999)
    monkeypatch.setattr(ge, "fa1_version", lambda: 0)
    model = str(tmp_path / "model.gguf")
    stock_key = ge.gate_key(model, model)
    monkeypatch.setenv("IRONMULE_GGML_PREFILL_BATCH", "1")
    assert ge.gate_key(model, model) != stock_key
    hits = []

    def capture(command, **kwargs):
        done = probe(command, **kwargs)
        if command[2] == ge.GATE_CANDIDATE:
            payload = json.loads(kwargs["input"])
            if payload["reference"]:
                assert kwargs["env"]["IRONMULE_GGML_PREFILL_BATCH"] == "0"
                assert not payload["check_prefill_batch"]
            elif payload["check_prefill_batch"]:
                assert kwargs["env"]["IRONMULE_GGML_PREFILL_BATCH"] == "1"
                hits.append(payload["settings"])
                row = {**json.loads(done.stdout), "prefill_batch": {"matches": matches, "calls": calls}}
                return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr="")
        return done

    monkeypatch.setattr(subprocess, "run", capture)
    assert ge.gate(model, model, cache=tmp_path / "state")["passes"] is accepted
    assert hits


def test_separate_prefill_observes_only_new_tokens_and_the_prior_context(monkeypatch):
    from ironmule import ggml_engine

    engine, _, _ = _stub_engine()
    engine.seq_ctx = 4
    steps = []
    engine.observer = lambda *step: steps.append(step)
    ticks = iter((1.0, 1.5, 2.0, 2.25))
    monkeypatch.setattr(ggml_engine.time, "perf_counter", ticks.__next__)
    assert engine._prefill(list(range(10)), seq=0, start=3) == 7
    assert steps == [(0, 4, 3, .5), (0, 3, 7, .25)]


def test_kg3_a_free_id_between_running_streams_gets_placeholder_rows_that_are_removed_again():
    from ironmule.ggml_engine import Job

    engine, fed, removed = _streams_engine()
    a, b, c = Job([1, 2], 6), Job([3, 4], 2), Job([5, 6], 6)
    engine.run([a, b, c])  # b (the middle id) ends first; a and c run on around its free id
    assert len(b.tokens) == 2 and len(a.tokens) == len(c.tokens) == 6
    fillers = [call for call in fed if [s for _, _, s in call] == [0, 1, 2] and call[1][0] == 1]
    assert fillers  # one call over ids 0..2 with a placeholder for 1
    held = fillers[0][1][1]
    assert (1, held, -1) in removed  # the placeholder's cell is removed after the step


def test_kg3_a_request_that_takes_a_free_id_mid_run_is_not_overwritten_by_a_placeholder():
    from ironmule.ggml_engine import Job

    engine, fed, _ = _streams_engine()
    a, b, c = Job([1, 2], 8), Job([3, 4], 2), Job([5, 6], 8)
    late = [Job([7, 8, 9, 10, 11], 3)]

    def admit(room):  # once b's id is free, a new request takes it (the lowest) while a and c run on
        return [late.pop()] if late and len(b.tokens) == 2 and room else []

    engine.run([a, b, c], admit=admit)  # the stub refuses any row that would not continue its stream
    assert not late and all(len(j.tokens) == j.limit for j in (a, b, c))


def test_kg3_a_new_request_takes_the_lowest_free_stream():
    engine, _, _ = _stub_engine()
    engine.settings = {"kv_unified": False}
    engine._free, engine._kept = [3, 1, 2], {3: [9, 9, 9]}
    assert engine._take_sequence([5, 6, 7]) == (1, 0)
    engine._kept = {1: [5, 6, 8], 3: [5, 6, 7, 9]}
    assert engine._take_sequence([5, 6, 7, 10]) == (1, 2)
    engine._kept = {3: list(range(80))}
    assert engine._take_sequence(list(range(70)) + [5]) == (3, 70)  # a long kept prefix still wins


def test_pad3_streams_are_padded_evenly_to_the_size_measured_cheaper():
    from ironmule.ggml_engine import Job

    engine, fed, _ = _streams_engine()
    engine.pad_to, engine.padded = {4: 10}, 0  # 4 rows would be dear; 10 measured cheaper
    jobs = [Job([10 * k + 1, 10 * k + 2], 6) for k in range(4)]
    engine.run(jobs)
    padded = [call for call in fed if len(call) == 12]
    assert padded and engine.padded > 0  # 4 streams x 3 tokens: one pass, past the 10 measured cheaper
    for call in padded:
        counts = collections.Counter(s for _, _, s in call)
        assert set(counts.values()) == {3}  # every stream the same number of tokens, so the call does not split
    assert all(len(j.tokens) == 6 for j in jobs)


def test_pb1_on_a_gpu_with_one_kv_stream_per_request_the_pilots_prompt_budget_is_not_held(monkeypatch):
    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job

    monkeypatch.setattr(ge, "PROMPT_BUDGET", 0)

    class Pilot:  # the GPU Pilot's choice on a T4: a prompt budget of 32 tokens
        def decide(self, running, queued, waiting, ctx):
            return {"threads": 1, "cap": 32, "budget": 32, "draft": True}

        def observe(self, *args):
            pass

        def status(self):
            return {}

        def close(self):
            pass

    def first_call(gpu_layers):
        engine, fed, _ = _streams_engine()
        engine.pilot, engine.pool, engine.gpu_layers = Pilot(), None, gpu_layers
        engine.run([Job([10 * k + j for j in range(20)], 2) for k in range(4)])
        return {s for _, _, s in fed[0]}, len(fed)

    on_gpu, gpu_calls = first_call(1)
    on_cpu, cpu_calls = first_call(0)
    assert len(on_cpu) == 1  # on a CPU the budget holds: one 20-token prompt per step
    assert gpu_calls < cpu_calls  # on a GPU every prompt starts in the first step (each in its own call)


def test_kg3_on_a_gpu_the_gate_checks_one_kv_stream_per_request_and_a_cpu_gate_does_not(monkeypatch, tmp_path):
    from ironmule import ggml_engine

    _fake_gate_children(monkeypatch, [2.0 + 0.002] * 16)
    monkeypatch.setattr(ggml_engine, "gpu_layers", lambda L=None: 999)
    result = ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    streams = result["streams"]
    assert streams["passes"] and streams["settings"]["flash_attn"] and not streams["settings"]["kv_unified"]
    stored = next((tmp_path / "cache").glob("*.json"))
    old = {k: v for k, v in result.items() if k != "streams"}
    stored.write_text(__import__("json").dumps(old))  # a gate stored before KG3 is run again on a GPU
    assert "streams" in ggml_engine.gate(str(tmp_path), str(tmp_path / "m.gguf"), cache=tmp_path / "cache")
    monkeypatch.setattr(ggml_engine, "gpu_layers", lambda L=None: 0)
    on_cpu = ggml_engine.gate(str(tmp_path), str(tmp_path / "m2.gguf"), cache=tmp_path / "cache2")
    assert on_cpu["streams"] is None


def test_seq1_measures_without_the_pilot(monkeypatch, tmp_path):
    """SEQ1's KV-size probe opens 256 cells; the Pilot's cost measurement decodes past that, so with it the probe
    failed on every machine and the width fell back to 8. Its children run with IRONMULE_PILOT=0."""
    import subprocess
    import types as _types

    from ironmule import ggml_engine

    seen = []

    def run(command, env=None, **kwargs):
        seen.append(env or {})
        return _types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(ggml_engine, "file_sha256", lambda path: "0" * 64)
    (tmp_path / "m.gguf").write_bytes(b"0")
    ggml_engine.choose_sequences(str(tmp_path / "m.gguf"), {}, cache=tmp_path / "c")
    assert seen and all(env.get("IRONMULE_PILOT") == "0" for env in seen)


def test_kg5_gate_key_binds_to_the_native_backend_and_probe(monkeypatch, tmp_path):
    from ironmule import ggml_engine as gg

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model")
    monkeypatch.setenv("IRONMULE_KG5", "1")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(tmp_path))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "probe.so"))
    monkeypatch.setattr(gg, "gpu_layers", lambda: 999)
    for name in ("libllama.so", "libggml-cuda.so", "probe.so"):
        (tmp_path / name).write_bytes(name.encode())
    def key():
        return gg.gate_key(str(gguf), str(gguf))

    original = key()
    assert len(gg._kg5_files()) == 3 and key() == original
    (tmp_path / "probe.so").write_bytes(b"new probe")
    probe = key()
    (tmp_path / "libggml-cuda.so").write_bytes(b"another CUDA build")
    assert len({original, probe, key()}) == 3
    monkeypatch.setenv("LD_PRELOAD", "unresolved-name.so")
    assert not gg._kg5_files()


@pytest.mark.parametrize("speculation", ["padding", "drafts"])
def test_pilot_weights_only_apply_to_actual_plain_decode_rows(speculation):
    import types

    from ironmule.ggml_engine import Job

    engine, _, _ = _streams_engine()
    switches, calls = [], []
    learning = types.SimpleNamespace(bin_size=64, max_rows=4, choose=lambda *args: 2)
    engine.pilot = types.SimpleNamespace(learning=learning, load=types.SimpleNamespace(foreign=0),
                                         last_fallback=False, observe=lambda *args: None,
                                         decide=lambda *args: {"threads": 4, "cap": 4, "budget": 4096, "draft": True})
    def set_threads(n):
        switches.append(n)
        engine.threads = n
    engine.set_threads = set_threads
    decode = engine._decode

    def traced(rows):
        calls.append((engine.threads, list(rows)))
        return decode(rows)
    engine._decode = traced
    if speculation == "padding":
        engine.pad_to, engine.padded = {3: 6}, 0
    else:
        # The engine's own n-gram source produces real drafts once the answer repeats 7s.
        engine.policy = types.SimpleNamespace(choose=lambda offers: [min(1, len(tokens)) for tokens, _ in offers],
                                              observe_step=lambda *args: None, observe_drafts=lambda *args: None)

    jobs = [Job([2] * 64, 6), Job([3] * 64, 3), Job([4] * 64, 15)]
    engine.run(jobs)
    assert all(job.tokens == [7] * job.limit for job in jobs)
    assert any(len(rows) > len({seq for _, _, seq, _ in rows}) and all(row[3] for row in rows)
               for _, rows in calls)
    assert calls[0][0] == 4
    assert any(threads == 2 for threads, rows in calls)
    for threads, rows in calls:
        if any(not logits for _, _, _, logits in rows) or len(rows) != len({seq for _, _, seq, _ in rows}):
            assert threads == 4  # prompt, padding or placeholder rows
    # Consecutive plain steps keep their committed pool.
    expected = [t for i, (t, _) in enumerate(calls) if i == 0 or t != calls[i - 1][0]]
    assert switches == expected


@pytest.mark.parametrize("agent,override,enabled", [("on", None, True), ("on", "0", False),
                                                   ("off", None, False), ("off", "1", True)])
def test_serve_wires_cpu_pilot_learning_with_default_speculation(tmp_path, monkeypatch, agent, override, enabled):
    import types

    from ironmule import ggml_engine as ge
    from ironmule_product import backend, cli
    from ironmule_product.ggml_bridge import GgmlBridge
    from ironmule_product.state import ProductStore

    state = tmp_path / "store"
    ProductStore(state).setup("server")
    model = tmp_path / "test.gguf"
    model.write_bytes(b"local model fixture")
    monkeypatch.setenv("IRONMULE_GGML_SEQS", "4")
    if override is None:
        monkeypatch.delenv("IRONMULE_PILOT_LEARN", raising=False)
    else:
        monkeypatch.setenv("IRONMULE_PILOT_LEARN", override)
    monkeypatch.setattr(ge, "available", lambda: None)
    monkeypatch.setattr(ge, "gpu_layers", lambda: 0)
    monkeypatch.setattr(ge, "gate", lambda *args, **kwargs: {"passes": True})
    class Captured(Exception):
        pass

    def capture(spec, **options):
        assert options["speculate"] is True
        bridge = GgmlBridge.__new__(GgmlBridge)
        pilot = types.SimpleNamespace(learning=None, weights_path=model, cores=4)
        bridge._engine = types.SimpleNamespace(pilot=pilot, settings={}, seqs=4, seq_ctx=512, shared=False,
                                               cap=4, gpu_layers=0)
        bridge._speculate, bridge._gguf_sha256, bridge._gate = True, "model", options["ggml_gate"]
        bridge._bind_pilot_learning()
        assert (pilot.learning is not None) == enabled
        if enabled:
            assert pilot.learning.path.parent == state / "pilot"
        raise Captured

    monkeypatch.setattr(backend, "MLXWorkerClient", capture)
    with pytest.raises(Captured):
        cli.serve(["--gguf", str(model), "--agent", agent, "--state-dir", str(state)])


def test_native_batch_refuses_oversized_quality_chunk_before_pointer_access():
    from types import SimpleNamespace

    from ironmule.ggml_engine import GgmlEngine

    engine = object.__new__(GgmlEngine)
    engine.seq_ctx = 512
    engine.batch = SimpleNamespace(n_tokens=17)
    engine.np = None
    engine._clear = lambda: None
    with pytest.raises(ValueError, match="513 decode rows"):
        engine.chunk_nll([1] * 513, [1])
    assert engine.batch.n_tokens == 17
    assert engine._batch_for(512).n_tokens == 512


@pytest.mark.parametrize("changed_fixture", [False, True])
def test_interrupted_gate_only_reuses_completed_stages_for_identical_fixtures(tmp_path, monkeypatch, changed_fixture):
    import json
    import subprocess
    from collections import Counter
    from types import SimpleNamespace

    from ironmule import ggml_engine as ge
    from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS

    monkeypatch.setattr("ironmule.ggml_binary.identity", lambda: {"sha256": {"libllama": "f" * 64}})
    _fake_gate_children(monkeypatch, [2.0] * CHUNKS)
    child = subprocess.run
    calls = Counter()
    signature = "a" * 64
    monkeypatch.setattr(ge, "fa1_version", lambda: 3)
    monkeypatch.setattr(ge, "gpu_layers", lambda: 0)

    def probe(command, **kwargs):
        code = command[2]
        if code == ge.GATE_IDS:
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "ids": [1] * (CHUNKS * CHUNK_TOKENS + 1), "bos": 1, "facts_sha256": signature}), stderr="")
        if code == ge.TUNE_CHILD:
            calls["tune"] += 1
        if code == ge.GATE_CANDIDATE:
            row = json.loads(kwargs["input"])
            label = "reference" if row["reference"] else "shared" if row["shared"] else "candidate"
            calls[label] += 1
            if label == "shared" and calls[label] == 1:
                raise KeyboardInterrupt
        return child(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", probe)
    cache = tmp_path / "cache"
    model = str(tmp_path / "m.gguf")
    with pytest.raises(KeyboardInterrupt):
        ge.gate(model, model, cache=cache)
    assert not list(cache.glob("*.json"))
    records = list(cache.glob("stages-*/*.json"))
    assert len(records) == 3
    assert all('"ids"' not in path.read_text() and '"prompt"' not in path.read_text() for path in records)
    if changed_fixture:
        signature = "b" * 64
    result = ge.gate(model, model, cache=cache)
    assert result["passes"] and result["shared"]["passes"]
    expected = 2 if changed_fixture else 1
    assert calls == {"tune": expected, "reference": expected, "candidate": expected, "shared": 2}
    assert result["checkpoint_stages"]["reused"] == (0 if changed_fixture else 3)


def test_concurrent_gate_stages_recheck_after_lock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from ironmule.ggml_gate_checkpoint import stage

    started, proceed = Event(), Event()
    executions = []

    def run():
        executions.append(1)
        started.set()
        assert proceed.wait(10)
        return {"rate": 2}, ""

    def call():
        count = {"reused": 0, "measured": 0}
        result = stage(tmp_path / "private", {"role": "tuning"}, run, count, lambda r: r == {"rate": 2})
        return result, count

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call)
        assert started.wait(10)
        second = pool.submit(call)
        proceed.set()
        a, b = first.result(), second.result()
    assert executions == [1] and a[0] == b[0]
    assert a[1]["measured"] == b[1]["reused"] == 1


def test_admission_observer_has_only_numeric_completed_group_metadata():
    from ironmule.ggml_engine import Job

    engine, _, _ = _stub_engine()
    samples = []
    engine.admission_observer = lambda *fields: samples.append(fields)
    engine.run([Job([12131, 27131, 1231], 2), Job([24131], 2)])
    started, lengths, limit, rate, ttft = samples.pop()
    assert type(started) is int and lengths == (3, 1) and limit == 2 and min(rate, ttft) > 0
    engine.run([Job([123], 1), Job([123], 2)])  # a mixed output limit has no single matched rule
    engine.run([Job([123], 2), Job([123], 2, cancelled=lambda: True)])
    assert not samples


def test_receipted_long_group_cannot_jump_a_preceding_short_prompt():
    from ironmule.ggml_engine import Job

    def schedule(long_mode):
        engine, calls, _ = _streams_engine()
        engine.prefill_batch = engine.prefill_batch_long = long_mode
        engine.prefill_long_shapes = {5: (2,)}
        jobs = [Job([14] * 2, 1), Job([15] * 2, 1), Job([12] * 5, 1), Job([13] * 5, 1)]
        engine.run(jobs)
        return [[seq for _, pos, seq in call if pos == 0] for call in calls]

    stock, grouped = schedule(False), schedule(True)
    assert [2, 3] in grouped
    assert [seq for call in stock for seq in call] == [seq for call in grouped for seq in call]


def test_gb1_after_a_burst_a_lone_arrival_waits_for_the_rest_of_the_next_one(monkeypatch):
    import time

    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job

    monkeypatch.setattr(ge, "GATHER_S", 0.02)
    monkeypatch.setattr(ge, "GATHER_QUIET_S", 0.004)
    monkeypatch.setattr(ge, "GATHER_BURST_S", 0.3)
    monkeypatch.setattr(ge, "GATHER_BURST_QUIET_S", 0.06)

    def first_call_seqs(recent_burst: bool) -> set[int]:
        engine, fed, _ = _stub_engine()
        engine.pack, engine.settings, engine.batch_cap = True, {"kv_unified": False}, 256
        if recent_burst:
            engine._burst_at = time.monotonic()
        started = time.monotonic()
        late = [Job([5, 6, 7], 2)]

        def admit(room):  # the burst's second request reaches the server 15 ms after the first
            return [late.pop()] if late and time.monotonic() - started > 0.015 else []

        engine.run([Job([1, 2, 3], 2)], admit=admit)
        if recent_burst:  # a group of two renews the memory; a lone request that waited in vain clears it
            assert engine._burst_at > started
            alone, _, _ = _stub_engine()
            alone.pack, alone.settings, alone.batch_cap, alone._burst_at = True, {"kv_unified": False}, 256, time.monotonic()
            alone.run([Job([1, 2, 3], 2)], admit=lambda room: [])
            assert alone._burst_at == float("-inf")
        return {s for _, _, s in fed[0]}

    assert len(first_call_seqs(False)) == 1  # 4 ms without an arrival: the lone request starts at once
    assert len(first_call_seqs(True)) == 2  # after a burst it waits with the burst's window and both start together


def test_pk1c_the_window_expects_the_last_waves_size_and_ends_when_it_is_there(monkeypatch):
    import time

    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job

    monkeypatch.setattr(ge, "GATHER_S", 0.02)
    monkeypatch.setattr(ge, "GATHER_QUIET_S", 0.004)
    monkeypatch.setattr(ge, "GATHER_BURST_S", 0.3)
    monkeypatch.setattr(ge, "GATHER_BURST_QUIET_S", 0.025)
    monkeypatch.setattr(ge, "GATHER_EXPECT_QUIET_S", 0.08)

    def wave(on: bool):
        monkeypatch.setattr(ge, "PK1C", on)
        engine, fed, _ = _stub_engine()
        engine.pack, engine.settings, engine.batch_cap = True, {"kv_unified": False}, 256
        engine.run([Job([1, 2, 3], 2), Job([4, 5, 6], 2), Job([7, 8, 9], 2)], admit=lambda room: [])  # the last wave
        assert engine._burst_peak == 3
        fed.clear()
        started = time.monotonic()
        late = [Job([5, 6, 7], 2), Job([6, 7, 8], 2)]
        given = []

        def admit(room):  # as the service: one request first, the other two 40 ms later as joins
            if late and time.monotonic() - started > 0.04:
                given.append(time.monotonic())
                out, late[:] = list(late), []
                return out
            return []

        engine.run([Job([1, 2, 3], 2)], admit=admit)
        return {s for _, _, s in fed[0]}, given

    seqs, given = wave(True)
    assert len(seqs) == 3  # 40 ms is longer than the 25 ms quiet gap, within the 80 ms it now waits for

    # a wave whose window caught only its first request still renews the burst memory once the rest joined
    monkeypatch.setattr(ge, "PK1C", True)
    engine, fed, _ = _stub_engine()
    engine.pack, engine.settings, engine.batch_cap = True, {"kv_unified": False}, 256
    started = time.monotonic()
    late = [Job([5, 6, 7], 2), Job([6, 7, 8], 2)]
    # the window (no memory yet) starts the first request alone; the other two join after its prefill call
    engine.run([Job([1, 2, 3], 2)], admit=lambda room: [late.pop() for _ in range(len(late))] if fed else [])
    assert len({s for _, _, s in fed[0]}) == 1
    assert engine._burst_peak == 3 and engine._burst_at > started
    seqs, _ = wave(False)
    assert len(seqs) == 1  # PK1-C off: the window closed before the joins came


def test_px1_a_packed_group_keeps_the_same_prefix_when_that_saves_passes(monkeypatch):
    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job, split_passes

    monkeypatch.setenv("IRONMULE_PS1", "0")  # PX1 decides only where the passes stay separate

    # llama.cpp's sequential split: equal lengths in consecutive streams make one pass, unequal ones several
    assert split_passes({0: 30, 1: 30, 2: 30}, 8192) == 1
    assert split_passes({0: 30, 1: 22, 2: 30}, 8192) == 3  # after 22, streams 0 and 2 are no longer consecutive
    assert split_passes({0: 33, 1: 25, 2: 33, 3: 444, 4: 436}, 8192) == 5

    engine, fed, removed = _stub_engine(seqs=4)
    engine.batch_cap = 8192
    engine.px1 = True
    monkeypatch.setattr(ge, "UBATCH", 8192)
    group = [(0, Job(list(range(33)), 2), 3), (1, Job(list(range(33)), 2), 11), (2, Job(list(range(444)), 2), 3),
             (3, Job(list(range(444)), 2), 11)]
    engine.reused_tokens = 28
    evened = engine._even_reuse(group)
    assert [keep for _, _, keep in evened] == [3, 3, 3, 3]  # 2 passes instead of 4 for 16 rows recomputed
    assert sorted(removed) == [(1, 3, -1), (3, 3, -1)] and engine.reused_tokens == 12
    assert [job.reused for _, job, _ in evened] == [0, 3, 0, 3]  # only the cut streams' records change

    engine, fed, removed = _stub_engine(seqs=4)
    engine.batch_cap = 8192
    engine.px1 = True
    far = [(0, Job(list(range(300)), 2), 0), (1, Job(list(range(300)), 2), 200)]
    assert engine._even_reuse(far) == far and removed == []  # 200 rows for one pass saved: not worth it

    engine.px1 = False
    assert engine._even_reuse(group) == group


def test_pk2_m_mix_calibration_times_half_short_half_long_prompts(monkeypatch):
    import types

    from ironmule import ggml_engine as ge

    def calibrate(mode):
        monkeypatch.setattr(ge, "PK2_CALIBRATION", mode)
        thresholds = []
        monkeypatch.setattr(ge, "pk2_set_min_rows", lambda rows: thresholds.append(rows) or 0)
        engine = ge.GgmlEngine.__new__(ge.GgmlEngine)
        engine.gpu_layers, engine.pack, engine.seqs, engine.seq_ctx, engine.vocab = 999, True, 32, 2048, None
        engine.L = types.SimpleNamespace(llama_vocab_bos=lambda vocab: 1)
        engine.tokenize = lambda text: list(range(5000))
        engine._clear = lambda: None
        seen = []
        engine._pack_prefill = lambda group: seen.append(sorted(len(job.ids) for _, job, _ in group))
        engine.chunk_nll = lambda chunk, prefix: 1.0
        out = engine.measure_pk2(repeats=1)
        return out, seen[-1], set(thresholds)

    out, lengths, tried = calibrate("equal")
    assert lengths == [444] * 8 and out["calibration"] == "equal" and 128 not in tried
    out, lengths, tried = calibrate("mix")
    assert lengths == [33] * 4 + [444] * 4 and out["rows"] == 4 * 33 + 4 * 444
    assert out["calibration"] == "mix" and 128 in tried


def _pod_pass_costs():
    """PX2: pass costs shaped as pods 32/33's 186 packed calls fit (16.5 ms + 0.98 ms a row under PK2's 128-row cuBLAS
    threshold, 91 ms + 0.287 ms a row from it on), at the sizes the engine measures."""
    from ironmule.ggml_engine import PassCosts

    small = lambda r: (16.45 + 0.978 * r) / 1000  # noqa: E731
    big = lambda r: (91.15 + 0.2869 * r) / 1000  # noqa: E731
    return PassCosts({1: small(1), 8: small(8), 16: small(16), 32: small(32), 127: small(127), 128: big(128),
                      256: big(256), 1024: big(1024)})


def test_px2_prices_the_passes_each_kept_prefix_level_would_make(monkeypatch):
    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job, split_shapes

    monkeypatch.setenv("IRONMULE_PS1", "0")

    monkeypatch.setattr(ge, "UBATCH", 8192)
    assert split_shapes({0: 33, 1: 25, 2: 33, 3: 444, 4: 436}, 8192) == [125, 8, 24, 806, 8]
    costs = _pod_pass_costs()
    assert abs(costs(6576) - (91.15 + 0.2869 * 6576) / 1000) < 1e-6  # beyond the largest size: its slope

    def engine_with_costs():
        engine, fed, removed = _stub_engine(seqs=32)
        engine.batch_cap, engine.px1, engine.pass_costs = 16384, True, costs
        return engine, removed

    # the benchmark's later waves: 16 short and 16 long prompts, template prefixes of 0-12 tokens per stream; every
    # stream is cut to the lowest level, 0 here, so that the group runs as 2 passes
    keeps = [3, 11, 12, 3, 11, 12, 0, 12, 3, 11, 12, 3, 11, 12, 3, 11] * 2
    group = [(i, Job(list(range(33 if i < 16 else 444)), 2), keeps[i]) for i in range(32)]
    engine, removed = engine_with_costs()
    assert ge.split_passes({s: len(j.ids) - k for s, j, k in group}, 8192) > 2
    evened = engine._even_reuse(group)
    assert {keep for _, _, keep in evened} == {0} and len({s for s, _, _ in removed}) == 30
    assert ge.split_passes({s: len(j.ids) - k for s, j, k in evened}, 8192) == 2

    # 80 rows for 2 passes saved: the fixed limit (64) refuses them, the measured costs take them (3 passes of 60-80
    # rows cost about 264 ms, one of 300 rows about 177 ms)
    three = [(0, Job(list(range(100)), 2), 0), (1, Job(list(range(100)), 2), 80), (2, Job(list(range(100)), 2), 0)]
    engine, removed = engine_with_costs()
    assert split_shapes({s: len(j.ids) - k for s, j, k in three}, 8192) == [60, 80, 80]
    assert [keep for _, _, keep in engine._even_reuse(three)] == [0, 0, 0] and removed == [(1, -1, -1)]
    engine, removed = engine_with_costs()
    engine.pass_costs = None  # the safe fallback: the fixed limit decides, and refuses
    assert engine._even_reuse(three) == three and removed == []

    # a large reusable prefix (a shared 1,000-token system prompt): the level that keeps it is chosen, not the lowest
    big = [(0, Job(list(range(1100)), 2), 1000), (1, Job(list(range(1100)), 2), 1012),
           (2, Job(list(range(1100)), 2), 1000), (3, Job(list(range(1100)), 2), 0)]
    engine, removed = engine_with_costs()
    evened = engine._even_reuse(big)
    assert [keep for _, _, keep in evened] == [1000, 1000, 1000, 0] and removed == [(1, 1000, -1)]

    # unequal prompt lengths whose split cutting cannot shorten: nothing is recomputed
    uneven = [(0, Job(list(range(40)), 2), 5), (1, Job(list(range(400)), 2), 0)]
    engine, removed = engine_with_costs()
    assert engine._even_reuse(uneven) == uneven and removed == []

    # 200 rows for one large pass saved: by the measured costs worth it (91 ms against 57 ms), by the fixed limit not
    far = [(0, Job(list(range(300)), 2), 0), (1, Job(list(range(300)), 2), 200)]
    engine, removed = engine_with_costs()
    assert [keep for _, _, keep in engine._even_reuse(far)] == [0, 0]
    engine, removed = engine_with_costs()
    engine.pass_costs = ge.PassCosts({1: 0.020, 8: 0.021, 256: 0.020 + 0.5, 1024: 0.020 + 2.0})  # rows are dear here
    assert engine._even_reuse(far) == far and removed == []


def test_px2_measures_passes_of_its_own_sizes_with_logits_on_the_last_row(monkeypatch):
    import types


    engine, fed, removed = _stub_engine(seqs=4)
    engine.seq_ctx = engine.batch_cap = 2048
    engine.batch = types.SimpleNamespace(token=[0] * 2048, pos=[0] * 2048, n_seq_id=[0] * 2048, logits=[0] * 2048,
                                         seq_id=[[0] for _ in range(2048)], n_tokens=0)
    engine.L.llama_vocab_bos = lambda vocab: 1
    one = engine.L.llama_get_logits_ith
    engine.L.llama_get_logits_ith = lambda ctx, i: one(ctx, 0)  # the stub holds 64 logit rows
    asked: list[list[int]] = []
    real = engine._decode

    def decode(rows, **kw):
        asked.append([i for i, row in enumerate(rows) if row[3]])
        return real(rows, **kw)
    engine._decode = decode
    engine.pk2 = {"min_rows": 128}
    costs = engine.measure_pass_costs(repeats=1)
    sizes = [r for r, _ in costs.points]
    assert sizes == [1, 8, 16, 32, 127, 128, 256, 1024]  # small, both sides of the cuBLAS threshold, two large
    measured = [rows for rows in asked if len(fed[asked.index(rows)]) in sizes]
    assert all(rows == [len(fed[asked.index(rows)]) - 1] for rows in measured)  # logits on the last row only
    engine.seq_ctx = engine.batch_cap = 64  # no room for the sizes: PX1's fixed limit decides
    assert engine.measure_pass_costs(repeats=1) is None


def test_ps1_keeps_every_kept_prefix_since_the_passes_share_one_call(monkeypatch):
    from ironmule.ggml_engine import Job

    monkeypatch.setenv("IRONMULE_PS1", "1")
    engine, fed, removed = _stub_engine(seqs=4)
    engine.batch_cap, engine.px1, engine.pass_costs = 8192, True, _pod_pass_costs()
    group = [(0, Job(list(range(33)), 2), 3), (1, Job(list(range(33)), 2), 11), (2, Job(list(range(444)), 2), 3),
             (3, Job(list(range(444)), 2), 11)]
    assert engine._even_reuse(group) == group and removed == []  # nothing recomputed
    monkeypatch.setenv("IRONMULE_PS1", "0")
    assert [keep for _, _, keep in engine._even_reuse(group)] == [3, 3, 3, 3]  # separate passes: PX2 evens them


def test_ps1_and_pk6_are_on_by_default_without_warm_up_dh2_and_sv5():
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if k not in ("IRONMULE_PS1", "IRONMULE_PS1_WARM", "IRONMULE_GGML_DH2", "IRONMULE_PK6", "IRONMULE_SV5")}
    code = ("import os; from ironmule import ggml_engine as g; from ironmule_product import worker as w; "
            "print(g.PS1, g.PS1_WARM, g.DH2, os.environ['IRONMULE_PS1'], g.PK6, os.environ['IRONMULE_PK6'], w._sv5_on())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["True", "0", "False", "1", "True", "1", "False"]  # the configuration pod 37 measured (K = PK6)


def test_dh2_delivers_every_token_in_order_and_the_last_from_the_engine_thread(monkeypatch):
    import threading

    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job

    monkeypatch.setattr(ge, "DH2", True)
    engine, fed, removed = _stub_engine(seqs=4)
    engine.gpu_layers = 1
    seen: dict[int, list[tuple[int, bool]]] = {0: [], 1: []}
    main = threading.get_ident()
    jobs = []
    for k in range(2):
        job = Job([1, 2, 3 + k], 5)
        job.emit = (lambda k: lambda token: seen[k].append((token, threading.get_ident() == main)))(k)
        jobs.append(job)
    out = engine.run(jobs)
    assert [j.tokens for j in out] == [[7] * 5, [7] * 5]
    assert [t for t, _ in seen[0]] == [7] * 5 and [t for t, _ in seen[1]] == [7] * 5  # all delivered, in order
    assert seen[0][-1][1] and seen[1][-1][1]  # the last token (and its done frame) on the engine thread
    assert not all(on_main for on_main, in [(m,) for _, m in seen[0][:-1]])  # the others on the emitter

    broken = ge._Emitter()
    broken.put(lambda token: 1 / 0, 7)
    try:
        broken.close()
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("a delivery error is raised by close()")


def test_dh2_a_failed_token_delivery_fails_the_run(monkeypatch):
    """Codex review on #57: the emitter's error was swallowed (the check of sys.exc_info() ran inside its own
    except); a run whose intermediate token delivery failed now raises it."""
    import pytest

    from ironmule import ggml_engine as ge
    from ironmule.ggml_engine import Job

    monkeypatch.setattr(ge, "DH2", True)
    engine, fed, removed = _stub_engine(seqs=4)
    engine.gpu_layers = 1
    calls = []

    def emit(token):
        calls.append(token)
        if len(calls) == 2:  # an intermediate token, delivered on the emitter thread
            raise ZeroDivisionError("delivery failed")

    job = Job([1, 2, 3], 5)
    job.emit = emit
    with pytest.raises(ZeroDivisionError):
        engine.run([job])
