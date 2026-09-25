"""Every headline number in the status documents must still exist in evidence.

This project's whole value is that a claim can be traced to a sealed
measurement. Documentation drift - a number edited, rounded differently, or
carried over from a superseded run - is therefore its worst defect class, and
nothing checked for it. This ledger does.

Each entry pins the German-formatted string as it appears in the document and
the raw value as it appears in the evidence. Both forms are written out on
purpose: a test that recomputed the formatting would pass while the document
said something else.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (label, documented text, document, evidence file, raw value in that file)
JSON_CLAIMS = [
    ("persistent process ratio", "0,346968", "docs/PROJECT_STATUS.md",
     "experiments/persistent_process/results.json", "0.34696789209993684"),
    ("persistent process effect", "65,3032", "docs/PROJECT_STATUS.md",
     "experiments/persistent_process/results.json", "-65.30321079000632"),
    ("cycle 16 ratio", "0,9295921887", "docs/PROJECT_STATUS.md",
     "experiments/matmul_compile_ab/results.json", "0.9295921887"),
    ("cycle 17 ratio", "0,9581074518", "docs/PROJECT_STATUS.md",
     "experiments/batched_readback_compile/results.json", "0.9581074518"),
    ("cycle 21 ratio", "1,000510010", "docs/PROJECT_STATUS.md",
     "experiments/fused_greedy_compile_v4/results.json", "1.000510009822041"),
]

_R11 = "experiments/kaggle_compat/results/perf1-run11-7b29bb97"
_R12 = "experiments/kaggle_compat/results/perf1-run12-c4c35978"
_R13 = "experiments/kaggle_compat/results/perf1-run13-93ae1f80"
JSON_CLAIMS += [
    ("24B stock decode", "instead of 2.140", "README.md", f"{_R11}/e2e-mistral-24b-stock.json", '"decode_tps_median": 2.14'),
    ("24B kernel decode", "decodes at 11.431", "README.md", f"{_R11}/e2e-mistral-24b-kernel.json", '"decode_tps_median": 11.431'),
    ("24B mma width 8", "gives 15.168 against 8.802", "README.md",
     f"{_R11}/server-mistral-24b-kernel+mma.json", '"aggregate_tps": 15.168'),
    ("24B row width 8", "gives 15.168 against 8.802", "README.md",
     f"{_R11}/server-mistral-24b-kernel.json", '"aggregate_tps": 8.802'),
    ("32B stock decode", "1.566 tokens per second stock", "README.md",
     f"{_R12}/e2e-qwen3-32b-stock.json", '"decode_tps_median": 1.566'),
    ("32B kernel decode", "8.54 with the native kernels", "README.md",
     f"{_R12}/e2e-qwen3-32b-kernel+p16.json", '"decode_tps_median": 8.54'),
    ("32B per-card memory", "9551 MiB per", "README.md", f"{_R12}/e2e-qwen3-32b-kernel+p16.json", "9551 MiB"),
    ("32B mma width 8", "reach 20.403 tokens", "README.md",
     f"{_R12}/server-qwen3-32b-kernel+mma+p16.json", '"aggregate_tps": 20.403'),
    ("24B p16 TTFT", "after 1.68 s instead of 77.6 s", "README.md",
     f"{_R13}/e2e-mistral-24b-kernel+p16.json", '"ttft_ms_median": 1684.18'),
    ("24B kernel TTFT", "after 1.68 s instead of 77.6 s", "README.md",
     f"{_R13}/e2e-mistral-24b-kernel.json", '"ttft_ms_median": 77605.63'),
    ("24B mma+p16 width 8", "reach 31.05 tokens", "README.md",
     f"{_R13}/server-mistral-24b-kernel+mma+p16.json", '"aggregate_tps": 31.05'),
]

#: Decisions must match the wording the status table reports.
DECISION_CLAIMS = [
    ("experiments/persistent_process/results.json", "engineering_gain_confirmed_exact_scope"),
    ("experiments/matmul_compile_ab/results.json", "runtime_compile_wins_exact_scope"),
    ("experiments/batched_readback_compile/results.json", "no_clear_speedup_baseline_retained"),
    ("experiments/fused_greedy_compile_v4/results.json", "fused_greedy_compile_inconclusive"),
]

#: The README speed table, one cell per entry: the text it prints and the run it
#: came from. A speed-up is 1/ratio, so a re-measured run moves both halves.
README_SPEEDUPS = [
    ("1B on the M1 Max", "1.61× · +61%",
     "experiments/kaggle_compat/results/apple-abcd/abcd-1b.json",
     ("wall_ratios", "D/A", "median_ratio")),
    ("4B on the M1 Max", "1.27× · +27%",
     "experiments/kaggle_compat/results/apple-abcd/abcd-4b.json",
     ("wall_ratios", "D/A", "median_ratio")),
    ("12B on the M1 Max", "1.11× · +11%",
     "experiments/kaggle_compat/results/apple-abcd/abcd-12b.json",
     ("wall_ratios", "D/A", "median_ratio")),
    ("1B on the T4", "1.82× · +82%",
     "experiments/kaggle_compat/results/port1-run6-c3af42bd/cross-1b.json",
     ("summary", "ironmule", "median_ratio")),
    ("4B on the T4", "1.05× · +5%",
     "experiments/kaggle_compat/results/port1-run6-c3af42bd/cross-4b.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("4B on the T4, float32", "1.91× · +91%",
     "experiments/kaggle_compat/results/port1-run6-c3af42bd/cross-4b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("12B on the T4", "1.03× · +3%",
     "experiments/kaggle_compat/results/port1-run7-1f40ad2b/cross-12b.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("12B on the T4, float32", "2.04× · +104%",
     "experiments/kaggle_compat/results/port1-run7-1f40ad2b/cross-12b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Gemma 3 4B on the T4", "1.07× · +7%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-gemma3-4b.json",
     ("summary", "ironmule_fused", "median_ratio")),
    ("Gemma 3 4B on the T4, float32", "1.91× · +91%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-gemma3-4b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Gemma 3 4B on the T4, float16", "3.21× · +221%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-gemma3-4b.json",
     ("summary", "ironmule_fp16", "median_ratio")),
    ("Llama 3.1 8B on the T4", "1.03× · +3%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-llama31-8b.json",
     ("summary", "ironmule_fused", "median_ratio")),
    ("Qwen 3 8B on the T4", "1.04× · +4%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-qwen3-8b.json",
     ("summary", "ironmule_fused", "median_ratio")),
    ("Qwen 3 8B on the T4, float32", "1.87× · +87%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-qwen3-8b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Qwen 3 8B on the T4, float16", "3.23× · +223%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-qwen3-8b.json",
     ("summary", "ironmule_fp16", "median_ratio")),
    ("Qwen 3 14B on the T4", "1.03× · +3%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-qwen3-14b.json",
     ("summary", "ironmule_fused", "median_ratio")),
    ("Qwen 3 14B on the T4, float32", "1.94× · +94%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-qwen3-14b.json",
     ("summary", "ironmule_fused_fp32", "median_ratio")),
    ("gpt-oss 20B on the T4", "1.03× · +3%",
     "experiments/kaggle_compat/results/port2-run2-74fe1a6d/cross-gptoss-20b.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("gpt-oss 20B on the T4, float32", "3.55× · +255%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-gptoss-20b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("gpt-oss 20B on the T4, float16", "5.02× · +402%",
     "experiments/kaggle_compat/results/port2-run6-59ce8efc/cross-fp16-gptoss-20b.json",
     ("summary", "ironmule_fp16", "median_ratio")),
    ("Mistral 24B on the T4", "1.01× · +1%",
     "experiments/kaggle_compat/results/port2-run3-281b971a/cross-mistral-lean.json",
     ("summary", "ironmule_lean", "median_ratio")),
    ("Mistral 24B on the T4, float32", "1.82× · +82%",
     "experiments/kaggle_compat/results/port2-run3-281b971a/cross-mistral-lean.json",
     ("summary", "ironmule_lean_fp32", "median_ratio")),
    ("Gemma 4 E2B on the T4", "1.11× · +11%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e2b.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("Gemma 4 E2B on the T4, float32", "2.43× · +143%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e2b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Gemma 4 E2B on the T4, float16", "3.94× · +294%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e2b.json",
     ("summary", "ironmule_fp16", "median_ratio")),
    ("Gemma 4 E4B on the T4", "1.08× · +8%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("Gemma 4 E4B on the T4, float32", "2.23× · +123%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Gemma 4 E4B on the T4, float16", "3.69× · +269%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b.json",
     ("summary", "ironmule_fp16", "median_ratio")),
    ("Gemma 4 E4B qat on the T4", "1.09× · +9%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b-qat.json",
     ("summary", "ironmule_exact", "median_ratio")),
    ("Gemma 4 E4B qat on the T4, float32", "1.73× · +73%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b-qat.json",
     ("summary", "ironmule_fp32", "median_ratio")),
    ("Gemma 4 E4B qat on the T4, float16", "3.18× · +218%",
     "experiments/kaggle_compat/results/port2-run9b-e8751c84/cross-gemma4-e4b-qat.json",
     ("summary", "ironmule_fp16", "median_ratio")),
]

README_SPEEDUPS += [
    ("Qwen 3 8B on the T4, native", "4.97× · +397%",
     "experiments/kaggle_compat/results/perf1-run7-080bfab7/cross-native-qwen3-8b.json",
     ("summary", "ironmule_native", "median_ratio")),
    ("Qwen 3 14B on the T4, native", "4.81× · +381%",
     "experiments/kaggle_compat/results/perf1-run7-080bfab7/cross-native-qwen3-14b.json",
     ("summary", "ironmule_native", "median_ratio")),
    ("Gemma 3 12B on the T4, native", "5.00× · +400%",
     "experiments/kaggle_compat/results/perf1-run18-863237d6/cross-gemma3-12b.json",
     ("summary", "ironmule_native", "median_ratio")),
]

#: The one documented cell where a plan is *slower* than stock. It cannot share the
#: formatter above, which prints a leading "+", and leaving it unpinned would make the
#: single unflattering number in the README the only one that could drift.
README_REGRESSIONS = [
    ("Llama 3.1 8B on the T4, float32", "0.66× · \u221234%",
     "experiments/kaggle_compat/results/port2-run4-c86664a3/cross-fused-llama31-8b.json",
     ("summary", "ironmule_fused_fp32", "median_ratio")),
]

#: The same table's per-optimisation rows, which print a ratio and a percentage.
README_MECHANISMS = [
    ("head-skip prefill", "0.846", "+18%",
     "experiments/head_skip_formal/results.json",
     ("calculated_decision", "intervals", "all", "ratio")),
    ("prefix cache, warm", "0.622", "+61%",
     "research/raw/E10-prefix-cache-session-ab.json", ("ratio_warm", "median_ratio")),
    ("prefix cache, cold", "0.621", "+61%",
     "research/raw/E10-prefix-cache-session-ab.json", ("ratio_cold", "median_ratio")),
]

#: The head-skip study lives in a sealed database rather than a JSON file.
HEAD_SKIP_DATABASE = ROOT / ".friday-data" / "head-skip-v1.sqlite3"
HEAD_SKIP_CLAIMS = [
    ("ratio", "0,846385", 0.8463845562069244),
    ("ci_low", "0,843147", 0.8431470041496976),
    ("ci_high", "0,851284", 0.8512844842159696),
]


def document(name: str) -> str:
    return (ROOT / name).read_text()


@pytest.mark.parametrize(
    "label,text,doc,evidence,raw",
    JSON_CLAIMS,
    ids=[entry[0].replace(" ", "-") for entry in JSON_CLAIMS],
)
def test_a_documented_number_is_still_in_its_evidence(label, text, doc, evidence, raw):
    assert text in document(doc), f"{label}: {text} is no longer in {doc}"
    body = (ROOT / evidence).read_text()
    assert raw in body, f"{label}: {raw} is no longer in {evidence}"


@pytest.mark.parametrize("evidence,decision", DECISION_CLAIMS, ids=[entry[1] for entry in DECISION_CLAIMS])
def test_each_study_still_records_the_decision_the_status_reports(evidence, decision):
    payload = json.loads((ROOT / evidence).read_text())
    assert payload.get("decision") == decision
    assert payload.get("formal_claim") is False


def test_head_skip_numbers_match_the_sealed_database():
    if not HEAD_SKIP_DATABASE.is_file():
        pytest.skip("sealed head-skip database is not present in this checkout")
    connection = sqlite3.connect(f"file:{HEAD_SKIP_DATABASE}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM records WHERE kind='study_decision' ORDER BY seq"
        ).fetchall()
    finally:
        connection.close()
    assert rows, "the sealed study carries no decision record"
    intervals = json.loads(rows[0][0])["intervals"]["all"]
    status = document("docs/PROJECT_STATUS.md")
    for field, text, expected in HEAD_SKIP_CLAIMS:
        assert intervals[field] == pytest.approx(expected, abs=1e-12), field
        assert text in status, f"{field}: {text} is no longer in PROJECT_STATUS.md"
    # Six confirmation sessions, all token-identical, is what the table claims.
    assert intervals["sessions"] == 6


def _measured(evidence: str, path: tuple[str, ...]) -> float:
    payload = json.loads((ROOT / evidence).read_text())
    for key in path:
        payload = payload[key]
    return payload


@pytest.mark.parametrize("label,text,evidence,path", README_SPEEDUPS,
                         ids=[entry[0].replace(" ", "-") for entry in README_SPEEDUPS])
def test_the_readme_speed_table_matches_the_run_behind_each_cell(label, text, evidence, path):
    """The README is the first thing a stranger reads, so it is pinned hardest.

    Both directions are checked: the printed cell must be what the run says,
    and the run's number must still be the one the README prints.
    """

    speedup = 1 / _measured(evidence, path)
    assert text == f"{speedup:.2f}× · +{(speedup - 1) * 100:.0f}%", label
    assert text in document("README.md"), f"{label}: {text} is no longer in the README"


@pytest.mark.parametrize("label,text,evidence,path", README_REGRESSIONS,
                         ids=[entry[0].replace(" ", "-") for entry in README_REGRESSIONS])
def test_the_readme_prints_its_one_slowdown_exactly(label, text, evidence, path):
    speedup = 1 / _measured(evidence, path)
    assert speedup < 1, f"{label}: no longer a slowdown, move it to README_SPEEDUPS"
    assert text == f"{speedup:.2f}\u00d7 \u00b7 \u2212{(1 - speedup) * 100:.0f}%", label
    assert text in document("README.md"), f"{label}: {text} is no longer in the README"


@pytest.mark.parametrize("label,ratio,percent,evidence,path", README_MECHANISMS,
                         ids=[entry[0].replace(" ", "-").replace(",", "") for entry in README_MECHANISMS])
def test_the_readme_mechanism_table_matches_its_study(label, ratio, percent, evidence, path):
    measured = _measured(evidence, path)
    assert ratio == f"{measured:.3f}", label
    assert percent == f"+{(1 / measured - 1) * 100:.0f}%", label
    readme = document("README.md")
    assert ratio in readme and percent in readme, f"{label}: no longer in the README"


def test_the_ledger_would_notice_a_changed_number():
    """A guard nobody has seen fail is not a guard."""

    body = (ROOT / "experiments/persistent_process/results.json").read_text()
    assert "0.34696789209993684" in body
    assert "0.34696789209993685" not in body


# --- Constants in code that encode a measured value ---------------------------
#
# The ledger above binds the documents. A constant in code is the same claim in
# a place nobody proofreads, so it gets the same treatment.


def test_confirmed_ratios_match_the_studies_they_name():
    from friday_optimizer.integration import CONFIRMED_RATIOS

    persistent = json.loads((ROOT / "experiments/persistent_process/results.json").read_text())
    assert CONFIRMED_RATIOS["persistent_process"] == pytest.approx(
        persistent["metrics"]["all_pair_median_ratio"], abs=5e-7
    )

    matmul = (ROOT / "experiments/matmul_compile_ab/results.json").read_text()
    assert str(CONFIRMED_RATIOS["fixed_compiled_cache"]) in matmul

    if not HEAD_SKIP_DATABASE.is_file():
        pytest.skip("sealed head-skip database is not present in this checkout")
    connection = sqlite3.connect(f"file:{HEAD_SKIP_DATABASE}?mode=ro", uri=True)
    try:
        payload = connection.execute(
            "SELECT payload_json FROM records WHERE kind='study_decision'"
        ).fetchone()[0]
    finally:
        connection.close()
    ratio = json.loads(payload)["intervals"]["all"]["ratio"]
    assert CONFIRMED_RATIOS["head_skip_prefill"] == pytest.approx(ratio, abs=5e-7)


def test_the_measured_point_cost_is_derived_and_rounded_the_safe_way():
    """167 s per point comes from a sealed budget, not from a guess.

    Rounding must go up: a longer estimate yields fewer points per approved
    block, so the plan under-promises rather than over-promises.
    """

    from friday_optimizer.campaign import BLOCK_SECONDS, MEASURED_POINT_SECONDS

    matmul = json.loads((ROOT / "experiments/matmul_compile_ab/results.json").read_text())
    measured = matmul["budget"]["parent_wall_seconds"] / matmul["metrics"]["runs_completed"]
    assert measured == pytest.approx(166.7, abs=0.5)
    assert MEASURED_POINT_SECONDS >= measured, "rounding must not promise more points"
    assert MEASURED_POINT_SECONDS - measured < 1.0, "the constant has drifted from its evidence"
    # The standing rule is a 30-minute block; the plan may not quietly extend it.
    assert BLOCK_SECONDS == 30 * 60


def test_chosen_constants_are_labelled_as_chosen():
    """A reader must be able to tell a measurement from a judgement.

    These four are judgement calls. Their modules must say so, or a later
    reader will treat a threshold somebody picked as a number somebody
    measured - which is how a project like this loses its footing.
    """

    import sys as _sys

    sources = {
        "DEFAULT_MIN_SAMPLES": ROOT / "research/friday_optimizer/replay.py",
        "TIE_MARGIN": ROOT / "experiments/identity_forensics/gap_analysis.py",
        "RATE_TOLERANCE": ROOT / "experiments/w1_regime/regime_analysis.py",
        "PROMPT_TOLERANCE": ROOT / "experiments/w1_regime/measure_long_answer.py",
    }
    missing = []
    for name, path in sources.items():
        text = path.read_text()
        index = text.index(name)
        # Two-sided: a label may follow the name as easily as precede it.
        context = text[max(0, index - 400):index + 400]
        if not any(word in context.lower() for word in
                   ("preregistered", "chosen", "judgement", "judgment", "stated", "sealed baseline")):
            missing.append(name)
    assert not missing, f"these constants read as measurements: {missing}"


def test_the_readback_gain_belongs_to_a_stacked_arm_not_a_standalone_one():
    """4.19 % was measured on top of fixed_compiled, not against baseline.

    Both arms of the cycle-17 study carry fixed_compiled; they differ only in
    readback_every. Reading the ratio as a standalone decode gain would count
    the two knobs as alternatives when they are in fact stacked.
    """

    payload = json.loads((ROOT / "experiments/batched_readback_compile/results.json").read_text())
    paired = payload["metrics"]["paired"]
    assert paired["baseline_arm"] == "fixed_compiled_readback_1"
    assert paired["candidate_arm"] == "fixed_compiled_readback_8"
    assert paired["primary"]["median"] == pytest.approx(0.9581074518316217, abs=1e-12)
    assert set(payload["metrics"]["arms"]) == {
        "fixed_compiled_readback_1", "fixed_compiled_readback_8"
    }


def test_the_registry_readback_candidate_is_not_the_measured_one():
    """readback_every_2 has no measurement; cycle 17 measured N = 8.

    The candidate's name asserts a parameter that its own parameters mapping
    does not carry and that no sealed study measured. Pinned here so the
    4.19 % is never cited as evidence for N = 2.
    """

    from friday_optimizer.candidates import CandidateRegistry

    specification = CandidateRegistry().get("readback_every_2")
    assert dict(specification.parameters) == {}, "the name carries the only claim"
    payload = json.loads((ROOT / "experiments/batched_readback_compile/results.json").read_text())
    measured = set(payload["metrics"]["arms"])
    assert not any("readback_2" in arm for arm in measured), (
        "if N=2 is ever measured, this discrepancy is resolved and the test should say so"
    )
