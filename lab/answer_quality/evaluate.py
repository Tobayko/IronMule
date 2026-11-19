"""Measure document grounding, abstention and two-step tool use on an unchanged local GGUF."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROTOCOL = "ironmule.answer-quality.v2"
FIELDS = ("dock", "seal")
SYSTEM = ("Use only the supplied ledger. Return exactly the requested code, or UNKNOWN if there "
          "is no entry for that shipment and field. Never fill in a missing field from another record.")
ABSTENTIONS = {"unknown", "not specified", "not provided", "not listed", "insufficient information"}
CODE = re.compile(r"[A-Z]-\d{5}", re.IGNORECASE)


def tool(name, description, argument):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {argument: {"type": "string"}}, "required": [argument],
        "additionalProperties": False}}}


TOOLS = [tool("shipment_depot", "Find the depot identifier for the given shipment.", "shipment"),
         tool("depot_dock", "Find the dock code for the given depot identifier.", "depot")]
TOOL_SYSTEM = ("Use the provided lookup tools to find the requested dock. Call one function at a time. "
               "After both lookups, return only the dock code. If the data cannot be found, reply UNKNOWN.")


@dataclass(frozen=True)
class Turn:
    text: str
    finish: str
    tokens: int


def cases(seed, split, count):
    """Generate independent manifests and tool tables. Derive labels via lookup in those tables."""
    rng = random.Random(f"{PROTOCOL}/{split}/{seed}")
    ids = rng.sample(range(10000, 100000), 12 * count)
    batches = []
    for index in range(count):
        n = ids[12 * index:12 * (index + 1)]
        shipments = [f"S-{v}" for v in n[:3]]
        rows = [{field: f"{prefix}-{n[3 + 2*j + i]}" for i, (field, prefix) in
                 enumerate(zip(FIELDS, ("D", "Z")))} for j in range(3)]
        manifest = dict(zip(shipments, rows))
        field = FIELDS[index % 2]
        # Half the unsupported questions refer to a present shipment with a missing field.
        # The other half refer to an absent shipment, with similarly named distractors present.
        if index % 4 in (0, 3):
            del manifest[shipments[1]][field]
            missing = shipments[1]
        else:
            missing = f"S-{n[11]}"
        order = list(manifest)
        rng.shuffle(order)
        ledger = "\n".join(shipment + " | " + " | ".join(f"{key}={value}" for key, value in manifest[shipment].items())
                           for shipment in order)
        record = {"id": f"{split}-{index}", "cases": []}
        for shipment in (shipments[0], missing):
            expected = manifest.get(shipment, {}).get(field)
            record["cases"].append({"shipment": shipment, "field": field,
                                    "prompt": f"Ledger:\n{ledger}\n\nWhat is the {field} code for shipment {shipment}?",
                                    "expected": expected})
        depots = dict(zip(shipments, (f"P-{v}" for v in n[9:12])))
        docks = {depots[s]: manifest[s].get("dock", f"D-{n[2]}") for s in shipments}
        record["tool"] = {"shipment": shipments[2], "depots": depots, "docks": docks}
        batches.append(record)
    return batches


def format_compliant(text):
    return re.fullmatch(r"[A-Z]-\d{5}|UNKNOWN", text.strip()) is not None


def answer_kind(text, expected, *, field="dock", shipment=None):
    """Accept terse answers and affirmative answers about the requested field and shipment."""
    value = text.strip().removesuffix(".").strip()
    # Strip balanced inline markup around a single code or abstention literal only.
    value = re.sub(r"(\*\*|`)([A-Z]-\d{5}|UNKNOWN)\1", r"\2", value, flags=re.IGNORECASE)
    first, separator, explanation = value.partition(". ")
    statement = re.fullmatch(r"(?:the )?(dock|seal)(?: code)? for shipment (S-\d{5}) is (.+)", first, re.IGNORECASE)
    if statement:
        if statement[1].casefold() != field or statement[2].upper() != shipment:
            return "wrong_reference"
        first = statement[3]
    is_abstention = first.casefold() in ABSTENTIONS
    if separator:
        ledger = r"(?:the|the provided) ledger"
        no_entry = rf"there is no entry (?:for this shipment in {ledger}|in {ledger} for this shipment)"
        if not is_abstention or not re.fullmatch(no_entry, explanation, re.IGNORECASE):
            return "other_text"
    if is_abstention:
        return "abstained"
    if CODE.fullmatch(first):
        return "correct" if expected is not None and first.upper() == expected else "unsupported_value"
    return "other_text"


def tool_call(text):
    """Parse one complete Qwen-style call or a plain tool-call JSON object."""
    content = text.strip()
    tagged = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if tagged:
        content = tagged[1]
    try:
        obj = json.loads(content)
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None
    if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        return None
    return obj


def execute(call, table):
    """Look up the supplied arguments. Never substitute the expected call or its result."""
    functions = {"shipment_depot": ("shipment", "depots", "depot"), "depot_dock": ("depot", "docks", "dock")}
    entry = functions.get(call["name"])
    if entry is None:
        return {"error": "unknown function"}
    arg, mapping, field = entry
    args = call["arguments"]
    if set(args) != {arg} or type(args[arg]) is not str:
        return {"error": "invalid arguments"}
    result = table[mapping].get(args[arg])
    return {field: result} if result is not None else {"error": "record not found"}


def checked_turns(generate, conversations, tools):
    turns = generate(conversations, tools)
    if len(turns) != len(conversations):
        raise ValueError("missing model responses")
    return turns


EVIDENCE_LEDGER = (" Check the exact shipment identifier first and look for the requested field on the same row. "
                   "Other rows do not establish its value. If the shipment or field is missing, answer UNKNOWN.")
EVIDENCE_TOOLS = (" Resolve each dependency before answering. An intermediate identifier enables another lookup. "
                  "Give the requested field only when supported by a result from a tool you have called.")
TURN_LIMIT = 4


def evaluate(generate, fixtures, condition="reference"):
    if condition not in ("reference", "evidence_first"):
        raise ValueError("unknown evaluation condition")
    verify_evidence = condition == "evidence_first"
    grounded = []
    conversations = []
    for record in fixtures:
        for item in record["cases"]:
            grounded.append({"id": record["id"], **item})
            conversations.append([{"role": "system", "content": SYSTEM + (EVIDENCE_LEDGER if verify_evidence else "")}, {"role": "user", "content": item["prompt"]}])
    for row, turn in zip(grounded, checked_turns(generate, conversations, None)):
        row.update(response=turn.text, finish=turn.finish, tokens=turn.tokens, format_compliant=format_compliant(turn.text),
                   grade=answer_kind(turn.text, row["expected"], field=row["field"], shipment=row["shipment"]))

    trajectories, pending = [], list(range(len(fixtures)))
    for record in fixtures:
        table = record["tool"]
        depot = table["depots"][table["shipment"]]
        expected_calls = [{"name": "shipment_depot", "arguments": {"shipment": table["shipment"]}},
                          {"name": "depot_dock", "arguments": {"depot": depot}}]
        trajectories.append({"id": record["id"], "shipment": table["shipment"], "expected_calls": expected_calls,
                             "expected": table["docks"][depot], "turns": [], "calls": []})
    histories = [[{"role": "system", "content": TOOL_SYSTEM + (EVIDENCE_TOOLS if verify_evidence else "")}, {"role": "user",
                  "content": f"Which dock is assigned to shipment {r['shipment']}?"}] for r in trajectories]
    for step in range(TURN_LIMIT):
        if not pending:
            break
        next_pending = []
        turns = checked_turns(generate, [histories[i] for i in pending], TOOLS)
        for i, turn in zip(pending, turns):
            row = trajectories[i]
            trace = {"response": turn.text, "finish": turn.finish, "tokens": turn.tokens}
            row["turns"].append(trace)
            call = tool_call(turn.text)
            if call is None:
                # This workflow may inspect public tool returns, never the solution or hidden lookup table.
                observed = {t["tool_return"]["dock"] for t in row["turns"] if "dock" in t.get("tool_return", {})}
                abstained = answer_kind(turn.text, None, shipment=row["shipment"]) == "abstained"
                supported_output = any(answer_kind(turn.text, value, shipment=row["shipment"]) == "correct"
                                       for value in observed)
                if verify_evidence and not (abstained or supported_output):
                    trace["unverified_final"] = True
                    if step + 1 < TURN_LIMIT:
                        notice = ("The proposed final answer is not supported by a dock field in the observed tool "
                                  "results. Resolve outstanding lookups using the available functions. If you cannot "
                                  "obtain the requested value, reply UNKNOWN.")
                        histories[i].extend([{"role": "assistant", "content": turn.text}, {"role": "user", "content": notice}])
                        next_pending.append(i)
                        continue
                    row["withheld"] = True
                    row["final_response"] = "UNKNOWN"
                else:
                    row["final_response"] = turn.text
                continue
            if step + 1 == TURN_LIMIT:
                # No turn remains for the model to use a third/fourth tool result.
                row["withheld"] = verify_evidence
                row["final_response"] = "UNKNOWN" if verify_evidence else None
                continue
            row["calls"].append(call)
            reply = execute(call, fixtures[i]["tool"])
            trace["tool_return"] = reply
            histories[i].extend([{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
                                 {"role": "tool", "name": call["name"], "content": json.dumps(reply, sort_keys=True)}])
            next_pending.append(i)
        pending = next_pending
    for row in trajectories:
        final = row["final_response"]
        row["final_grade"] = answer_kind(final, row["expected"], shipment=row["shipment"]) if final else "no_final_answer"
        calls = iter(row["calls"])
        row["path_correct"] = all(any(actual == wanted for actual in calls) for wanted in row["expected_calls"])
        row["success"] = row["path_correct"] and row["final_grade"] == "correct"

    supported = Counter(r["grade"] for r in grounded if r["expected"] is not None)
    unsupported = Counter(r["grade"] for r in grounded if r["expected"] is None)
    return {"condition": condition, "supported": {"total": sum(supported.values()), "correct": supported["correct"],
                          "false_abstentions": supported["abstained"], "wrong_codes": supported["unsupported_value"],
                          "wrong_reference": supported["wrong_reference"], "other_text": supported["other_text"],
                          "format_compliant": sum(r["format_compliant"] for r in grounded if r["expected"] is not None)},
            "unsupported": {"total": sum(unsupported.values()), "correct_abstentions": unsupported["abstained"],
                            "unsupported_codes": unsupported["unsupported_value"], "wrong_reference": unsupported["wrong_reference"],
                            "other_text": unsupported["other_text"],
                            "format_compliant": sum(r["format_compliant"] for r in grounded if r["expected"] is None)},
            "tools": {"total": len(trajectories), "first_call_correct": sum(r["calls"][:1] == r["expected_calls"][:1]
                     for r in trajectories), "both_calls_correct": sum(r["path_correct"] for r in trajectories),
                      "correct_final": sum(r["final_grade"] == "correct" for r in trajectories),
                      "end_to_end_correct": sum(r["success"] for r in trajectories),
                      "model_turns": sum(len(r["turns"]) for r in trajectories),
                      "model_tokens": sum(t["tokens"] for r in trajectories for t in r["turns"]),
                      "tool_executions": sum(len(r["calls"]) for r in trajectories),
                      "withheld_after_budget": sum(bool(r.get("withheld")) for r in trajectories),
                      "wrong_final_codes": sum(r["final_grade"] == "unsupported_value" for r in trajectories),
                      "false_abstentions": sum(r["final_grade"] == "abstained" for r in trajectories)},
            "ledger_cases": grounded, "tool_traces": trajectories}


def run(args, *, conditions=None):
    from ironmule import ggml_binary
    from ironmule.ggml_engine import GgmlEngine, Job, file_sha256
    from ironmule.gguf_tokenizer import GgufTokenizer

    tokenizer = GgufTokenizer(str(args.gguf))
    engine = GgmlEngine(str(args.gguf), threads=args.threads, seqs=args.seqs, seq_ctx=768, speculate=False)
    started = time.monotonic()
    def generate(conversations, tools):
        out = []
        for pos in range(0, len(conversations), args.seqs):
            ids = [tokenizer.apply_chat_template(messages, enable_thinking=False, **({"tools": tools} if tools else {}))
                   for messages in conversations[pos:pos + args.seqs]]
            if any(len(row) + args.max_tokens > engine.seq_ctx for row in ids):
                raise ValueError("prompt exceeds the native context")
            if tools and not all("shipment_depot" in tokenizer.decode(row) and "depot_dock" in tokenizer.decode(row) for row in ids):
                raise ValueError("this template did not render the tool definitions")
            engine._clear()
            jobs = [Job(row, args.max_tokens) for row in ids]
            engine.run(jobs)
            for job in jobs:
                text = tokenizer.decode([t for t in job.tokens if not engine.eog(t)])
                out.append(Turn(text, job.reason, len(job.tokens)))
        return out
    try:
        dataset = cases(args.seed, args.split, args.count)
        if conditions is None:
            results = {name: evaluate(generate, dataset, name) for name in ("reference", "evidence_first")}
        else:
            results = {name: measure(generate, dataset) for name, measure in conditions.items()}
        return {"protocol": PROTOCOL, "seed": args.seed, "split": args.split, "groups": args.count,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
                "model_file": args.gguf.name, "model_sha256": file_sha256(str(args.gguf)),
                "native_runtime": ggml_binary.identity(), "threads": engine.threads, "sequences": engine.seqs,
                "context": engine.seq_ctx, "gpu_layers": engine.gpu_layers, "max_tokens": args.max_tokens,
                "decoding": "greedy", "thinking": False, "elapsed_seconds": round(time.monotonic() - started, 3),
                "conditions": results}
    finally:
        engine.close()
        tokenizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), default="holdout")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seqs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if not (1 <= args.count <= 256 and 1 <= args.threads <= 256 and 1 <= args.seqs <= 64 and 1 <= args.max_tokens <= 128):
        parser.error("choose counts within the supported bounds")
    if args.output.exists():
        parser.error("output already exists")
    result = run(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    for name, scores in result["conditions"].items():
        print(name, json.dumps({k: scores[k] for k in ("supported", "unsupported", "tools")}))


if __name__ == "__main__":
    main()
