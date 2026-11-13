"""FA1 gate: perplexity on the numeric gate's 16 chunks (multi-position decode, as ggml_engine.gate scores a setting):
llama.cpp's default (no flash attention), ggml's flash attention (FA1=0) and FA1, in child processes.
Usage: gate.py GGUF SNAPSHOT"""
import json, os, subprocess, sys
sys.path.insert(0, "/home/user/IronMule")
from pathlib import Path
from ironmule.ggml_engine import GATE_CANDIDATE, GATE_IDS
from ironmule.numeric_choice import ratio_interval
gguf, snap = sys.argv[1], sys.argv[2]
env = {**os.environ, "PYTHONPATH": "/home/user/IronMule"}
ids = json.loads(subprocess.run([sys.executable, "-c", GATE_IDS, snap], env=env, capture_output=True, text=True, check=True).stdout.splitlines()[-1])
def nll(settings, fa1):
    out = subprocess.run([sys.executable, "-c", GATE_CANDIDATE, gguf], env={**env, "FA1": fa1}, input=json.dumps({**ids, "settings": settings}),
                         capture_output=True, text=True, check=True).stdout.splitlines()[-1]
    return json.loads(out)["nll"]
ref = nll({"flash_attn": False}, "1")
for name, st, fa1 in (("ggml_fa", {"flash_attn": True}, "0"), ("fa1", {"flash_attn": True}, "1")):
    print(name, json.dumps(ratio_interval(ref, nll(st, fa1))), flush=True)
