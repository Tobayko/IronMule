"""Compare the reference decoder with a structured lookup application on the same fresh cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from . import evaluate as base

STUDY = "ironmule.structured-answer.v1"
FORMAT = "Return the requested code or UNKNOWN. Use the observed lookup result."


def read_shipment_field(prompt):
    """Resolve this public ledger format and typed question. This function cannot see labels."""
    ledger, divider, question = prompt.partition("\n\n")
    target = re.fullmatch(r"What is the (dock|seal) code for shipment (S-\d{5})\?", question)
    if not divider or not ledger.startswith("Ledger:\n") or not target:
        return {"status": "unsupported_input"}
    field, shipment = target.groups()
    query = {"shipment": shipment, "field": field}
    matches = []
    for line in ledger.splitlines()[1:]:
        key, *fields = line.split(" | ")
        if key == shipment:
            matches.append(fields)
    if not matches:
        return {**query, "status": "missing_shipment"}
    if len(matches) > 1:
        return {**query, "status": "ambiguous_shipment"}
    values = [p.partition("=")[2] for p in matches[0] if p.partition("=")[0] == field]
    if not values:
        return {**query, "status": "missing_field"}
    if len(values) != 1 or not base.CODE.fullmatch(values[0]):
        return {**query, "status": "ambiguous_field"}
    return {**query, "status": "found", "value": values[0]}


def present(turn, observed):
    """Preserve a supported model answer; otherwise format the actual tool result."""
    canonical = observed["value"] if observed["status"] == "found" else "UNKNOWN"
    if turn.text.strip() == canonical:
        return turn.text, False
    return canonical, True


def ledger(generate, prompts):
    """Run the model with observations from the public input only."""
    observations = [read_shipment_field(p) for p in prompts]
    chats = [[{"role": "system", "content": FORMAT}, {"role": "user", "content": p},
              {"role": "user", "content": "Application lookup result: " + json.dumps(r, sort_keys=True)}]
             for p, r in zip(prompts, observations, strict=True)]
    records = []
    for obs, turn in zip(observations, base.checked_turns(generate, chats, None), strict=True):
        value, fallback = present(turn, obs)
        records.append({"observation": obs, "response": turn.text, "finish": turn.finish, "tokens": turn.tokens,
                        "delivered": value, "source_renderer_used": fallback})
    return records


def dock_workflow(generate, shipments, dispatchers):
    """Call the second tool only after a valid first call has actually supplied its argument."""
    chats = [[{"role": "system", "content": base.TOOL_SYSTEM}, {"role": "user",
             "content": f"Which dock is assigned to shipment {shipment}?"}] for shipment in shipments]
    rows, remaining = [], []
    for i, turn in enumerate(base.checked_turns(generate, chats, base.TOOLS)):
        row = {"shipment": shipments[i], "calls": [], "turns": [{"response": turn.text, "tokens": turn.tokens,
               "finish": turn.finish}], "delivered": "UNKNOWN", "source_renderer_used": False}
        rows.append(row)
        call = base.tool_call(turn.text)
        if call is None:
            row["source_renderer_used"] = True
            continue
        first = dispatchers[i](call)
        row["calls"].append({"actor": "model", "call": call, "return": first})
        if (call["name"] != "shipment_depot" or call["arguments"] != {"shipment": shipments[i]}
                or not isinstance(first.get("depot"), str)):
            row["source_renderer_used"] = True
            continue
        second = {"name": "depot_dock", "arguments": {"depot": first["depot"]}}
        reply = dispatchers[i](second)
        row["calls"].append({"actor": "application", "call": second, "return": reply})
        chats[i].extend([{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
                         {"role": "tool", "name": call["name"], "content": json.dumps(first, sort_keys=True)},
                         {"role": "user", "content": "The application executed this dependent lookup. " + FORMAT +
                          "\n" + json.dumps({"call": second, "result": reply}, sort_keys=True)}])
        remaining.append(i)
    if remaining:
        turns = base.checked_turns(generate, [chats[i] for i in remaining], base.TOOLS)
        for i, turn in zip(remaining, turns, strict=True):
            rows[i]["turns"].append({"response": turn.text, "tokens": turn.tokens, "finish": turn.finish})
            observed = rows[i]["calls"][-1]["return"]
            result = {"shipment": shipments[i], "field": "dock",
                      "status": "found" if "dock" in observed else "missing_field", "value": observed.get("dock")}
            rows[i]["delivered"], rows[i]["source_renderer_used"] = present(turn, result)
    return rows


def assess(generate, fixtures):
    items = [(group["id"], row) for group in fixtures for row in group["cases"]]
    ledger_results = ledger(generate, [row["prompt"] for _, row in items])
    for result, (name, row) in zip(ledger_results, items, strict=True):
        result.update(id=name, **row)
        for source, dest in (("response", "model_grade"), ("delivered", "grade")):
            result[dest] = base.answer_kind(result[source], row["expected"], field=row["field"], shipment=row["shipment"])
    # Only the tool executor closes over a lookup table. The application receives opaque callables.
    from functools import partial
    dispatchers = [partial(base.execute, table=group["tool"]) for group in fixtures]
    tool_results = dock_workflow(generate, [r["tool"]["shipment"] for r in fixtures], dispatchers)
    for result, group in zip(tool_results, fixtures, strict=True):
        table = group["tool"]
        shipment = result["shipment"]
        depot = table["depots"][shipment]
        dock = table["docks"][depot]
        actual = [r["call"] for r in result["calls"]]
        expected = [{"name": "shipment_depot", "arguments": {"shipment": shipment}},
                    {"name": "depot_dock", "arguments": {"depot": depot}}]
        final = result["turns"][-1]["response"]
        result.update(id=group["id"], expected=dock, expected_calls=expected, path_correct=actual == expected,
                      model_final_grade=base.answer_kind(final, dock, shipment=shipment),
                      final_grade=base.answer_kind(result["delivered"], dock, shipment=shipment))
        result["success"] = result["path_correct"] and result["final_grade"] == "correct"
    supported, missing = ([r for r in ledger_results if (r["expected"] is None) == val] for val in (False, True))
    return {"condition": "structured", "ledger_cases": ledger_results, "tool_traces": tool_results,
            "supported": {"total": len(supported), "delivered": dict(Counter(r["grade"] for r in supported)),
                          "model": dict(Counter(r["model_grade"] for r in supported))},
            "unsupported": {"total": len(missing), "delivered": dict(Counter(r["grade"] for r in missing)),
                            "model": dict(Counter(r["model_grade"] for r in missing))},
            "ledger_source_renderer_used": sum(r["source_renderer_used"] for r in ledger_results),
            "tools": {"total": len(tool_results), "end_to_end_correct": sum(r["success"] for r in tool_results),
                      "both_calls_correct": sum(r["path_correct"] for r in tool_results),
                      "model_final_correct": sum(r["model_final_grade"] == "correct" for r in tool_results),
                      "false_abstentions": sum(r["final_grade"] == "abstained" for r in tool_results),
                      "model_turns": sum(len(r["turns"]) for r in tool_results),
                      "model_tokens": sum(t["tokens"] for r in tool_results for t in r["turns"]),
                      "executed_by": dict(Counter(call["actor"] for r in tool_results for call in r["calls"])),
                      "source_renderer_used": sum(r["source_renderer_used"] for r in tool_results)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.seed, args.split, args.count = 20261120, "holdout", 12
    args.threads, args.seqs, args.max_tokens = 1, 4, 64
    if args.output.exists():
        parser.error("choose a fresh output path")
    report = base.run(args, conditions={"reference": base.evaluate, "structured": assess})
    report["study"] = STUDY
    report["structured_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
