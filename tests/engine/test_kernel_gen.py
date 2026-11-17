"""KG1/KG4: the generated sources and the engine's per-token-count choice of the library (no GPU needed)."""
import ctypes

from ironmule import kernel_gen as kg


def test_kg4_tune_source_lists_every_tensor_core_variant_and_its_measurement():
    src = kg.tune_source(kg.variants())
    assert src.count("launch_mma<") >= len(kg.MMA_VARIANTS)
    assert 'extern "C" int kg_measure_mma' in src and "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32" in src
    assert {v[0] for v in kg.MMA_VARIANTS} == {1, 2, 3, 4}  # 1-4 tiles of 8 tokens: up to 32


def test_kg4_library_serves_up_to_32_tokens_and_builds_tensor_core_entries():
    table = {"arch": "75", "entries": [
        {"K": 2560, "N": 9728, "B": 3, "L": 8, "R": 2, "W": 4, "us": 1.0},
        {"K": 2560, "N": 9728, "B": 20, "kind": "mma", "T": 3, "R": 2, "W": 4, "KB": 1, "SK": 1, "us": 1.0}]}
    src = kg.library_source(table)
    assert "{12, 2560, 9728, 20, nullptr, nullptr, launch_mma<3, 2, 4, 1, 1>, nullptr, 3}," in src
    assert "{12, 2560, 9728, 3, launch<8, 2, 4, 3>, nullptr, nullptr, nullptr, 0}," in src
    assert "B > 32" in src and "quantize_frag(" in src


def test_kg4_q6k_entries_are_typed_and_dispatched_with_their_own_layout():
    table = {"arch": "75", "entries": [
        {"K": 2560, "N": 151936, "B": 32, "kind": "mma6", "type": 14, "T": 4, "R": 1, "W": 8, "KB": 2, "SK": 1, "us": 1.0}]}
    src = kg.library_source(table)
    assert "{14, 2560, 151936, 32, nullptr, nullptr, nullptr, launch_mma6<4, 1, 8, 2, 1>, 4}," in src
    assert "type != 14" in src and "e.type == type" in src and "hit->mm6 != nullptr" in src
    assert {v[0] for v in kg.MMA6_VARIANTS} == {1, 2, 3, 4}


def test_kg4_the_engine_keeps_a_tile_count_where_its_step_is_faster(monkeypatch):
    from ironmule.ggml_engine import GgmlEngine

    class Lib:
        mask = None

        def __init__(self, path):
            def set_mask(mask):
                Lib.mask = mask

            self.ironmule_kg_set_mask = set_mask  # a plain function: ctypes' argtypes can be set on it
            self.ironmule_kg_table = lambda: b'{"entries": [1, 2]}'

    monkeypatch.setattr(ctypes, "CDLL", Lib)
    engine = GgmlEngine.__new__(GgmlEngine)
    faster = {2, 16, 32}  # the library is faster for 2 tokens, 9-16 and 25-32; not for 17-24 or the rest

    def prices(counts, repeats=1):
        on = Lib.mask != 0
        return {t: (0.5 if on and t in faster else 1.0) for t in counts}

    engine._step_prices = prices
    out = engine.measure_kg("lib.so", repeats=3)
    want = (1 << 1) | sum(1 << (u - 1) for u in range(9, 17)) | sum(1 << (u - 1) for u in range(25, 33))
    assert Lib.mask == want
    assert out["tokens_served"] == [2, 16, 32] and out["entries"] == 2


def test_kg4_q6k_is_kept_only_where_it_beats_the_step_without_it(monkeypatch):
    from ironmule.ggml_engine import GgmlEngine

    class Lib:
        mask = mask6 = None

        def __init__(self, path):
            def set_mask(mask):
                Lib.mask = mask

            def set_mask6(mask):
                Lib.mask6 = mask

            self.ironmule_kg_set_mask, self.ironmule_kg_set_mask6 = set_mask, set_mask6
            self.ironmule_kg_table = lambda: b'{"entries": []}'

    monkeypatch.setattr(ctypes, "CDLL", Lib)
    engine = GgmlEngine.__new__(GgmlEngine)

    def prices(counts, repeats=1):  # Q4_K helps at 8 and 32; Q6_K on top helps at 8 and costs at 32
        out = {}
        for t in counts:
            p = 1.0
            if Lib.mask and t in (8, 32):
                p = 0.8
            if Lib.mask6 and t == 8:
                p -= 0.2
            if Lib.mask6 and t == 32:
                p += 0.1
            out[t] = p
        return out

    engine._step_prices = prices
    out = engine.measure_kg("lib.so", repeats=3)
    assert out["tokens_served"] == [8, 32] and out["q6k_tokens_served"] == [8]
    assert Lib.mask6 == 1 << 7 and Lib.mask == (1 << 7) | sum(1 << (u - 1) for u in range(25, 33))


def test_kg5_the_engine_keeps_the_library_attention_only_with_kv_streams_and_where_it_is_faster(monkeypatch):
    from ironmule.ggml_engine import GgmlEngine

    class Lib:
        fa = None

        def __init__(self, path):
            def set_fa(on):
                Lib.fa = on

            self.ironmule_kg_set_mask = lambda mask: None
            self.ironmule_kg_set_fa = set_fa
            self.ironmule_kg_table = lambda: b'{"entries": []}'

    monkeypatch.setattr(ctypes, "CDLL", Lib)
    for unified, faster, want in ((False, True, True), (False, False, False), (True, True, None)):
        Lib.fa = None
        engine = GgmlEngine.__new__(GgmlEngine)
        engine.settings = {"kv_unified": unified}
        engine._step_prices = lambda counts, repeats=1: {t: 1.0 for t in counts}
        engine._streams_step_price = lambda: (0.8 if faster and Lib.fa else 1.0)
        out = engine.measure_kg("lib.so", repeats=3)
        if want is None:
            assert "attention" not in out and Lib.fa is None  # the shared pool: no KV streams, not measured
        else:
            assert out["attention"]["served"] is want and Lib.fa == int(want)


def test_kg5_the_library_source_has_the_attention_entry_and_its_switch():
    src = kg.library_source({"arch": "75", "entries": []})
    assert 'extern "C" int ironmule_kg_flash_attn(' in src and "ironmule_kg_set_fa" in src
    assert "kfa_combine" in src and "FA(128, 4)" in src
