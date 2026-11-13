"""GG1: the ggml engine (`ironmule/ggml_engine.py`), its policy version 3, the GGUF builder and the product wiring.

The engine itself needs llama.cpp's bindings and a GGUF; those tests run where `IRONMULE_GGML_BINDINGS` (or an
installed `llama-cpp-python`) and `IRONMULE_TEST_GGUF` are set, and are skipped elsewhere."""

from __future__ import annotations

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
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    class Backend:
        device, ready, engine_configuration, execution_variant = "cpu", True, "ggml", "current_engine"
        ggml_gate = {"ratio": 1.003, "interval": [1.001, 1.0048], "passes": True, "seconds": 70}
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


def _fake_gate_children(monkeypatch, option_nll):
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
            out = {"ids": [1, 2, 3], "bos": 1}
        else:
            settings = _json.loads(input)["settings"]
            out = {"nll": [2.0] * 16 if settings == ggml_engine.DEFAULT_SETTINGS else option_nll}
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
