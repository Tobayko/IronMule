"""QG1. Independent, short-answer checks for a reference, its repeat, and a candidate.

The prompt and grading protocol are fixed before a comparison. Vary the seed for fresh
arithmetic and extraction tasks when selecting a new adapter. Facts test retention. Never
provide the expected answers to the generation callback.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Mapping

VERSION = "qg1-v3"
KG5_SHAPE = "1024,32,8"
MAX_TOKENS = 32
# Complete, affirmative answers. Match the subject as well as the predicate: e.g. Tokyo is
# correct for Japan, but a sentence identifying Tokyo as Italy's capital must fail.
_FACTS = (
    ("Which planet is known as the Red Planet?", "Mars",
     ("Mars is known as the Red Planet", "The Red Planet is Mars", "The Red Planet is known as Mars")),
    ("What is the capital city of Italy?", "Rome",
     ("The capital city of Italy is Rome", "The capital of Italy is Rome", "Rome is the capital of Italy")),
    ("Which is the largest ocean on Earth? Answer with the ocean's one-word name.", "Pacific",
     ("The Pacific Ocean", "Pacific Ocean", "The largest ocean on Earth is the Pacific Ocean")),
    ("What is the chemical symbol for gold?", "Au",
     ("The chemical symbol for gold is Au", "Gold's chemical symbol is Au")),
    ("How many sides does a hexagon have? Use digits.", "6",
     ("six", "six sides", "6 sides", "A hexagon has 6 sides", "A hexagon has six sides")),
    ("What is the capital city of Japan?", "Tokyo",
     ("The capital city of Japan is Tokyo", "The capital of Japan is Tokyo", "Tokyo is the capital of Japan")),
)


@dataclass(frozen=True)
class FactCase:
    key: str
    category: str
    prompt: str
    expected: str
    accepted_forms: tuple[str, ...] = ()


def cases(seed: int = 0) -> tuple[FactCase, ...]:
    """Six questions per category. Seeded values are reproducible without modifying any global RNG."""
    rng = random.Random(seed)
    result = []
    for index, (fact, answer, fact_forms) in enumerate(_FACTS):
        a, b = rng.randint(2, 89), rng.randint(2, 19)
        sign, value = (("+", a + b), ("-", a - b), ("*", a * b))[index % 3]
        records = rng.sample(range(10000, 99999), 3)
        slots = ("A", "B", "C")
        slot = rng.randrange(3)
        context = ", ".join(f"{label} = {number}" for label, number in zip(slots, records))
        label = slots[slot]
        record = str(records[slot])
        extraction_forms = (f"The value for {label} is {record}", f"The value of {label} is {record}",
                            f"{label} is {record}", f"{label} = {record}")
        equation = f"{a} {sign} {b} = {value}"
        for category, question, expected, forms in (
            ("arithmetic", f"What is {a} {sign} {b}? Use digits.", str(value), (equation,)),
            ("extraction", f"Given the records {context}, what is the value for {label}?", record, extraction_forms),
            ("facts", fact, answer, fact_forms),
        ):
            result.append(FactCase(f"{seed}:{category}-{index}", category,
                                   question + " Reply with the answer only.", expected, forms))
    return tuple(result)


def _protocol(seed: int, kg5: bool) -> tuple[FactCase, ...]:
    base = cases(seed)
    if not kg5:
        return base
    # Fill all remaining streams with distinct graded problems, without over-weighting repeated facts.
    extra = [case for other in (seed + 1, seed + 2) for case in cases(other) if case.category != "facts"]
    return base + tuple(extra[:32 - len(base)])


def _chat_ids(tokenizer, case: FactCase, kg5: bool) -> list[int]:
    def render(content: str) -> list[int]:
        messages = [{"role": "user", "content": content}]
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False))
    if not kg5:
        return render(case.prompt)
    # Size the document with the model's own tokenizer, including its chat framing. The answer still fits in
    # 1024 cells and the decode step sees 1024 padded KV rows. Each record starts with its own identifier.
    def document(n: int):
        return render(f"Record {case.key}. Background notes:" + " note" * n + "\nQuestion: " + case.prompt)

    low, high = 0, 2048
    best = []
    while low <= high:
        mid = (low + high) // 2
        ids = document(mid)
        if len(ids) <= 970:
            best = ids
            low = mid + 1
        else:
            high = mid - 1
    if not 960 <= len(best) <= 970:
        raise ValueError("tokenizer cannot form the KG5 qualification context")
    return best


def correct(case: FactCase, answer: str) -> bool:
    """Grade the complete final answer; an expected substring in reasoning or in a wrong number cannot pass."""
    if not isinstance(answer, str):
        raise TypeError("the generator must return the answer text")
    # Some templates open a thinking block outside the returned tokens. An unclosed block has no final answer.
    final = answer.rsplit("</think>", 1)[-1]
    if "<think>" in final:
        return False
    def normalise(text: str) -> str:
        text = " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!").strip()
        # Normalize valid thousands groups only, never the decimal point or a comma-delimited list.
        text = re.sub(r"(?<![\d,.])-?\d{1,3}(?:,\d{3})+(?![\d,.])", lambda m: m[0].replace(",", ""), text)
        # Tokenization can change whitespace or render standard arithmetic symbols differently.
        return re.sub(r"\s*([=*+\-])\s*", r"\1", text.replace("×", "*").replace("−", "-"))

    final = normalise(final)
    forms = (case.expected, f"The answer is {case.expected}", f"Answer: {case.expected}", *case.accepted_forms)
    return any(final == normalise(form) for form in forms)


def evaluate(generate: Callable[[str], str], *, seed: int = 0) -> dict[str, bool]:
    return {case.key: correct(case, generate(case.prompt)) for case in cases(seed)}


def compare(reference: Mapping[str, bool], repeat: Mapping[str, bool], candidate: Mapping[str, bool],
            *, seed: int = 0, kg5: bool = False) -> dict:
    """Accept when each category is informative and its loss is within the reference's own score variation."""
    protocol = _protocol(seed, kg5)
    keys = {case.key for case in protocol}
    for scores in (reference, repeat, candidate):
        if set(scores) != keys or any(type(value) is not bool for value in scores.values()):
            raise ValueError("QG1 requires one boolean score per case in every arm")
    by_category = {}
    for category in ("arithmetic", "extraction", "facts"):
        ids = [case.key for case in protocol if case.category == category]
        a, aa, b = (sum(scores[key] for key in ids) for scores in (reference, repeat, candidate))
        by_category[category] = {
            "total": len(ids), "reference_correct": a, "repeat_correct": aa, "candidate_correct": b,
            "aa_difference": abs(a - aa), "passes": min(a, aa) > 0 and b >= min(a, aa),
        }
    return {
        "version": VERSION, "seed": seed, "shape": KG5_SHAPE if kg5 else None,
        "total": len(keys), "reference_correct": sum(reference.values()),
        "repeat_correct": sum(repeat.values()), "candidate_correct": sum(candidate.values()),
        "by_category": by_category, "passes": all(row["passes"] for row in by_category.values()),
    }


def ggml_scores(engine, tokenizer, *, seed: int = 0, kg5: bool = False) -> dict[str, bool]:
    """Score one cold greedy pass through the real multi-request engine and its chat template."""
    from .ggml_engine import Job

    protocol = _protocol(seed, kg5)
    jobs = [Job(_chat_ids(tokenizer, case, kg5), MAX_TOKENS) for case in protocol]
    engine._clear()
    engine.run(jobs)
    scores = {}
    for case, job in zip(protocol, jobs):
        if job.reason not in ("eos", "length"):
            raise RuntimeError(f"QG1 generation failed for {case.key}: {job.reason}")
        answer = tokenizer.decode([token for token in job.tokens if not engine.eog(token)])
        scores[case.key] = correct(case, answer)
    return scores



def kg5_counter(engine):
    """Read the launch counter from the active probe or the already loaded patched CUDA backend."""
    import ctypes
    from pathlib import Path

    try:
        count = ctypes.CDLL(None).ironmule_kg5_launches
    except AttributeError:
        # ctypes loads llama.cpp locally by default. Its CUDA symbols are not always visible through RTLD_DEFAULT.
        folder = Path(engine.L.llama_cpp._lib._name).parent
        lib = next(folder.glob("libggml-cuda.so*"), None)
        if lib is None:
            raise RuntimeError("KG5 launch counter unavailable for QG1")
        count = ctypes.CDLL(str(lib)).ironmule_kg5_launches
    count.restype = ctypes.c_uint64
    count.argtypes = []
    return count
