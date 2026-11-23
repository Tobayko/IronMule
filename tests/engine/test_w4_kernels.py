"""W4R tuner table (ironmule/w4_kernels.py): which token counts the measured widths cover."""
from ironmule import w4_kernels as wk


def test_the_measured_widths_cover_every_tile_count_up_to_32_tokens():
    # a token count is served by the measurement of its tile count (8 tokens a tile); 17-24 tokens are three tiles
    assert {-(-t // 8) for t in wk.TOKENS if t > 1} == {1, 2, 3, 4}


def test_every_token_count_from_1_to_32_gets_a_table_line():
    entries = [{"type": wk.TYPE_Q4_K, "N": 25600, "K": 5120, "B": b, "variant": 7, "ksplit": 1} for b in wk.TOKENS]
    lines = wk.table_lines({"entries": entries}).split()
    served = sorted(int(lines[i + 3]) for i in range(0, len(lines), 6))
    assert served == list(range(1, 33))
    without_24 = [e for e in entries if e["B"] != 24]  # the tuner before the fix
    lines = wk.table_lines({"entries": without_24}).split()
    assert not {int(lines[i + 3]) for i in range(0, len(lines), 6)} & set(range(17, 25))
