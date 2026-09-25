"""PORT1-F: what does the tune confirmation child write before it exits with status 1?

Usage: python confirm_child_probe.py MODEL_ID OUT.json

Runs `tune.confirm` for the screening winner BACKLOG1 and BACKLOG2 both found on Gemma 3 4B
(`head_skip_prefill`) and records each failed child's stderr through the hook `ab.run` calls
on a failure. A Kaggle diagnostic only: the prompt is tune's built-in `DEFAULT_PROMPT`, so the
stderr carries no user text; the product keeps stderr out of every error.
"""
import importlib
import json
import sys

from ironmule.runtime import BASELINE, Knobs

ab = importlib.import_module("ironmule.ab")
tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` is also a function

captured = []
original = ab._child_exception_type


def spy(stderr):
    captured.append(stderr[-20000:])
    return original(stderr)


ab._child_exception_type = spy
model_id, out = sys.argv[1:3]
result = {"model_id": model_id, "candidate": {"head_skip_prefill": True}}
try:
    tune.confirm(model_id, BASELINE, Knobs(head_skip_prefill=True), tune.DEFAULT_PROMPT, 32)
    result["outcome"] = "completed"
except Exception as exc:  # noqa: BLE001 - the failure is what is being recorded
    result["outcome"] = f"{type(exc).__name__}: {exc}"
result["child_stderr"] = captured
with open(out, "w") as stream:
    json.dump(result, stream, indent=1)
print(json.dumps({key: value for key, value in result.items() if key != "child_stderr"}))
for stderr in captured:
    print(stderr[-4000:])
