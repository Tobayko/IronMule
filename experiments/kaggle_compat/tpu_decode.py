# PORT2/DATA1 TPU decode: a real model decoding on a free Kaggle TPU, as a stock reference.
# Private notebook, internet on. TPU quota. No IronMule, and no comparison it cannot carry.
#
# The smoke established the ground: 8 x TPU v5 lite, 16.9 GB HBM each, jax 0.10.2 with
# libtpu, flax, torch_xla 2.8 and transformers 5.12 present, MLX absent. MLX has no TPU
# backend, so IronMule cannot run here at all and nothing below is an IronMule measurement.
#
# What this adds is the execution DATA1's TPU gate asked for and never had: a real causal LM
# decoding greedily on the chip. Two things it must not be read as. It is not comparable to
# the T4 numbers as a speed-up — a TPU has no 4-bit path, so this is dense bfloat16 against
# the T4's 4-bit weights, a different computation on a different memory system. And bf16
# here is native, which is the exact opposite of Turing, where bf16 is emulated and the
# whole `compute_dtype="float32"` plan exists to get away from it.
import json
import os
import time

WORK = "/kaggle/working"
MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"]
PROMPT = "Explain in two sentences why the sky is blue."
MAX_TOKENS = 32
report = {"schema": "ironmule.tpu-decode.v1", "performance_claim": False, "models": {}}


def save():
    with open(f"{WORK}/tpu-decode-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def decode(model_id):
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = xm.xla_device()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device).eval()
    xm.mark_step()
    load_seconds = round(time.time() - started, 1)

    rendered = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # Two passes: the first compiles the XLA graph, only the second is a rate.
    rows = []
    for pass_index in range(2):
        began = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        xm.mark_step()
        generated = out[0, ids.shape[1]:].cpu().tolist()
        elapsed = time.time() - began
        rows.append({"pass": pass_index, "seconds": round(elapsed, 2),
                     "tokens_per_second": round(len(generated) / elapsed, 2) if elapsed else None,
                     "tokens": generated, "text": tokenizer.decode(generated)})
    return {"load_seconds": load_seconds, "dtype": "bfloat16", "device": str(device),
            "prompt_tokens": int(ids.shape[1]), "passes": rows,
            "passes_agree": rows[0]["tokens"] == rows[1]["tokens"]}


os.makedirs(WORK, exist_ok=True)
for model_id in MODELS:
    started = time.time()
    try:
        report["models"][model_id] = decode(model_id)
    except BaseException as exc:  # noqa: BLE001 — an OOM on one chip is the result
        report["models"][model_id] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}"}
    report["models"][model_id]["total_seconds"] = round(time.time() - started, 1)
    save()
    print(f"== {model_id}: {json.dumps({k: v for k, v in report['models'][model_id].items() if k != 'passes'}, default=str)[:600]}", flush=True)
report["finished"] = True
save()
