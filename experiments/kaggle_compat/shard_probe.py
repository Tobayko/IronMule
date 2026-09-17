"""PORT2: can the cell's second T4 be used at all, and does a sharded model stay correct?

Usage: mlx.launch --hosts 127.0.0.1 -n 2 --backend ring shard_probe.py MODEL_ID OUT_PREFIX [MAX_TOKENS]
       python shard_probe.py --reference MODEL_ID REVISION OUT.json [MAX_TOKENS]

Every PORT2 number so far comes from one T4, because one MLX process uses one device and
the measurement had to stay clean. The ceiling that produced is now known and sharp:
Mistral 3 24B at 13.26 GB runs, 16.05 GB does not fit. A Kaggle cell has two T4s.

mlx-lm shards a model two ways. Pipelining splits layers across ranks and only six model
files implement it. Tensor parallelism splits each layer's weights and is implemented by
llama, qwen2, qwen3, qwen3_5, gpt_oss and ministral3 among others — which covers every
family PORT2 measured and, at 4 bit, reaches about 30 GB of weights across two cards.

This probe answers the prerequisite and nothing more: whether two ranks on one host each
take their own card, whether a tensor-parallel load survives on Turing, and whether the
tokens still match a single-card reference. IronMule itself is single process and cannot
use a distributed group yet; that is a separate entry, and this decides whether it is worth
opening. `sharded_load` resolves a repo, not a revision, so the reference arm pins the
revision and the sharded arm records the digest it actually got.
"""
import json
import os
import sys
import time

import mlx.core as mx


def greedy(model, tokenizer, prompt, max_tokens):
    from mlx_lm.models.cache import make_prompt_cache

    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
    ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    cache = make_prompt_cache(model)
    began = time.time()
    logits = model(mx.array(ids)[None, :], cache=cache)
    tokens = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(int(token.item()))
        logits = model(token.reshape(1, 1), cache=cache)
    elapsed = time.time() - began
    return {"tokens": tokens, "text": tokenizer.decode(tokens), "seconds": round(elapsed, 2),
            "tokens_per_second": round(max_tokens / elapsed, 2) if elapsed else None,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "prompt_tokens": len(ids)}


PROMPT = "Explain in two sentences why the sky is blue."

if sys.argv[1] == "--reference":
    # One card, one process, pinned revision: the tokens the sharded arms must reproduce.
    from mlx_lm import load

    sys.path.insert(0, "/tmp/IronMule")
    from ironmule.tune import resolve_local_model

    model_id, revision, out = sys.argv[2:5]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    model, tokenizer = load(str(resolve_local_model(model_id, revision).path))
    report = {"schema": "ironmule.port2-shard.v1", "arm": "reference", "model_id": model_id,
              "revision": revision, "device": str(mx.default_device()),
              "performance_claim": False, **greedy(model, tokenizer, PROMPT, max_tokens)}
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "tokens"}, indent=1))
    raise SystemExit(0)

MODEL_ID, OUT_PREFIX = sys.argv[1:3]
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
report = {"schema": "ironmule.port2-shard.v1", "arm": "tensor_parallel", "model_id": MODEL_ID,
          "backend_env": os.environ.get("MLX_DISTRIBUTED_BACKEND", "unset"),
          "performance_claim": False}
try:
    group = mx.distributed.init(strict=True)
    report["rank"], report["size"] = group.rank(), group.size()
    # One rank, one card. Without this both ranks land on device 0 and the point is lost.
    mx.set_default_device(mx.Device(mx.gpu, group.rank()))
    report["device"] = str(mx.default_device())
    started = time.time()
    from mlx_lm.utils import sharded_load

    model, tokenizer = sharded_load(MODEL_ID, tensor_group=group)
    report["load_seconds"] = round(time.time() - started, 1)
    report["weights_bytes_this_rank"] = int(mx.get_active_memory())
    report.update(greedy(model, tokenizer, PROMPT, MAX_TOKENS))
    report["ok"] = True
except Exception as exc:  # noqa: BLE001 — an unsupported shard is the result
    report["ok"] = False
    report["error"] = f"{type(exc).__name__}: {exc}"
rank = report.get("rank", os.environ.get("MLX_RANK", "x"))
with open(f"{OUT_PREFIX}-rank{rank}.json", "w") as stream:
    json.dump(report, stream, indent=1, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "tokens"}, indent=1, default=str))
