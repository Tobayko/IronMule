"""Qualify opted-in 2- and 4-stream GGUF prefill at exactly 444 chat tokens per stream."""
from __future__ import annotations

import hashlib
import json
import random

from .factual_gate import FactCase, correct
from .ggml_engine import Job

PROTOCOL = "ironmule.long-prefill.v1"
LENGTH = 444
LIMIT = 32
WIDTHS = (2, 4)
_CATEGORIES = ("arithmetic", "extraction", "facts")
_ARMS = ("reference", "repeat", "candidate")
_NOTES = (
    "The garden path follows the wall, then bends around the old shed before returning to the entrance.",
    "A visitor can leave a folded map on the table by the windows, along with the other pamphlets.",
    "The office keeps a paper copy of the public guide for readers who prefer to work away from a screen.",
    "An open window at the end of the corridor looks over the lawn and a stand of tall trees.",
    "When the weather allows, volunteers meet outside for a short walk before the building opens.",
    "The reading room has a quiet corner with chairs facing the courtyard and lamps beside the desks.",
    "Staff return borrowed materials to their usual places so the next group can find them.",
    "The neighborhood newsletter sometimes features an interview about the history of the building.",
    "A notice near the front door describes the public entrance and the route to the meeting room.",
    "People working in the archive keep the curtains partly closed to protect the old documents.",
    "The workshop schedule is available at the desk for visitors who would like to return on another day.",
    "At closing time the volunteer collects loose papers and puts the cushions back on the chairs.",
)
_FACTS = (
    ("Which planet is closest to the Sun?", "Mercury",
     ("Mercury is the closest planet to the Sun", "The planet closest to the Sun is Mercury")),
    ("What is the chemical symbol for oxygen?", "O", ("The chemical symbol for oxygen is O",)),
    ("What is the capital city of France?", "Paris",
     ("The capital city of France is Paris", "The capital of France is Paris")),
    ("Which organ pumps blood around the human body?", "heart",
     ("The heart", "The heart pumps blood around the human body")),
)


def _documents(width):
    """Separate deterministic problems per shape; extraction labels come from the supplied records."""
    rng = random.Random(f"{PROTOCOL}/{width}")
    for i, category in enumerate(("extraction", "arithmetic", "facts", "arithmetic", "facts", "extraction",
                                  "arithmetic", "extraction")):
        source = ""
        if category == "facts":
            fact_index = (0 if width == 2 else 2) + int(i == 4)
            question, expected, forms = _FACTS[fact_index]
        elif category == "arithmetic":
            a, b = rng.randint(21, 50), rng.randint(2, 7)
            op, result = {1: ("+", a + b), 3: ("-", a - b), 6: ("*", a * b)}[i]
            question = f"What is {a} {op} {b}?"
            expected = str(result)
            forms = (f"{a} {op} {b} = {result}", f"{a} {op} {b} is {result}")
        else:
            labels = ["Aster", "Linden", "Cedar"]
            rng.shuffle(labels)
            values = dict(zip(labels, rng.sample(range(100, 999), 3), strict=True))
            requested = rng.choice(labels)
            source = "Shelf assignments: " + ", ".join(f"{label} = {n}" for label, n in values.items()) + "."
            question = f"Which shelf number is assigned to {requested} in the shelf assignments?"
            expected = str(values[requested])
            forms = (f"{requested} = {expected}", f"The shelf number assigned to {requested} is {expected}",
                     f"{requested} is assigned to shelf {expected}")
        case = FactCase(f"long-{width}-{category}-{i}", category, question, expected, tuple(forms))
        yield case, source, i


def _render(tokenizer, case, source, offset, count, fill=""):
    notes = [_NOTES[(offset + j) % len(_NOTES)] for j in range(count)]
    if source:
        notes.insert((count * (1 + offset % 3)) // 4, source)
    content = "Archive notes:\n" + "\n".join(notes) + fill + "\n\nQuestion: " + case.prompt + " Reply with the answer only."
    return list(tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False))


def fixtures(tokenizer, width):
    """Build complete, independently rendered chat requests. Never truncate framing or answer instructions."""
    if width not in WIDTHS:
        raise ValueError("unsupported long prefill shape")
    result = []
    for case, source, offset in _documents(width):
        low, high = 0, 64
        while low < high:
            mid = (low + high + 1) // 2
            if len(_render(tokenizer, case, source, offset, mid)) <= LENGTH:
                low = mid
            else:
                high = mid - 1
        ids = _render(tokenizer, case, source, offset, low)
        for atom in (" page", " note", " x", "."):
            if len(ids) == LENGTH:
                break
            # A common BPE token often occupies one position; other tokenizers may split or merge repeats.
            for count in range(1, min(128, max(8, 2 * (LENGTH - len(ids)) + 8))):
                attempt = _render(tokenizer, case, source, offset, low, atom * count)
                if len(attempt) == LENGTH:
                    ids = attempt
                    break
        if len(ids) != LENGTH:
            raise ValueError("tokenizer could not render an exact 444-token chat request")
        result.append((case, ids))
    if len({tuple(ids) for _, ids in result}) != len(result):
        raise ValueError("long prefill requests have duplicate token IDs")
    return result


def _counters(engine):
    return (engine.prefill_batch_calls, engine.prefill_batch_prompts, engine.prefill_batch_rows)


def _shape(engine, tokenizer, width):
    if engine.seqs < width or engine.seq_ctx < width * LENGTH:
        return {"matches": False, "calls": 0, "reasons": ["insufficient_context_or_streams"]}
    rows = fixtures(tokenizer, width)
    scores = [{"key": case.key, "category": case.category, "expected": case.expected, "prompt_tokens": len(ids),
               "prompt_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
              for case, ids in rows]
    hits = {name: [] for name in _ARMS}
    reasons = set()
    for arm in _ARMS:
        engine.prefill_batch = arm == "candidate"
        for start in range(0, len(rows), width):
            engine._clear()
            before = _counters(engine)
            cohort = rows[start:start + width]
            jobs = [Job(list(ids), LIMIT) for _, ids in cohort]
            engine.run(jobs)
            delta = [now - was for now, was in zip(_counters(engine), before, strict=True)]
            hits[arm].append(dict(zip(("calls", "prompts", "rows"), delta, strict=True)))
            if delta != ([1, width, width * LENGTH] if arm == "candidate" else [0, 0, 0]):
                reasons.add("grouped_shape_not_proven")
            for i, ((case, _), job) in enumerate(zip(cohort, jobs, strict=True)):
                tokens = list(job.tokens)
                if not tokens or job.reason not in ("eos", "length") or len(tokens) > LIMIT:
                    reasons.add("incomplete_generation")
                answer = tokenizer.decode([t for t in tokens if not engine.eog(t)])
                scores[start + i][arm] = {"tokens": tokens, "stop": job.reason, "correct": correct(case, answer)}
    by_category = {}
    for category in _CATEGORIES:
        members = [r for r in scores if r["category"] == category]
        tallies = {arm: sum(row[arm]["correct"] for row in members) for arm in _ARMS}
        by_category[category] = {"total": len(members), **tallies}
        if not tallies["reference"] or not tallies["repeat"]:
            reasons.add(f"uninformative_{category}")
    for row in scores:
        if row["reference"] != row["repeat"]:
            reasons.add("unstable_reference")
        if row["reference"] != row["candidate"]:
            reasons.add("candidate_differs")
    calls = sum(w["calls"] for w in hits["candidate"])
    return {"matches": not reasons, "calls": calls, "reasons": sorted(reasons), "witnesses": hits,
            "by_category": by_category, "cases": scores}


def check(engine, tokenizer):
    """Run cold A/A'/B on each target shape. Restore grouping mode and clear the test's KV state."""
    enabled = engine.prefill_batch
    if not (enabled and getattr(engine, "prefill_batch_long", False)):
        return {"protocol": PROTOCOL, "matches": False, "calls": 0, "shapes": {}, "reasons": ["long_mode_inactive"]}
    results = {}
    previous_shapes = getattr(engine, "prefill_long_shapes", None)
    engine.prefill_long_shapes = {LENGTH: WIDTHS}  # scoped only to this cold target-shape holdout
    try:
        for width in WIDTHS:
            results[f"{width}x{LENGTH}"] = _shape(engine, tokenizer, width)
    finally:
        engine.prefill_batch = enabled
        engine.prefill_long_shapes = previous_shapes
        engine._clear()
    return {"protocol": PROTOCOL, "matches": all(s["matches"] for s in results.values()),
            "calls": sum(s["calls"] for s in results.values()), "shapes": results}
