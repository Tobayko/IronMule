"""Serve the submitted ordered batch once in the selected immutable engine process."""
import ctypes
import json
from contextlib import closing
from pathlib import Path
import sys

from ironmule import ggml_engine as ge
from ironmule.ggml_binary import identity
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule.kernel_gen import fingerprint
from native_trial import TEMPLATE
from restart_arm import mapped_vendor
from serving_trial import run_wave


def main(request_file):
    request = json.loads(request_file.read_bytes())
    binding = request["binding"]
    runtime = binding["runtime"]
    model = request["model"]
    with closing(ge.GgmlEngine(model, seqs=32, seq_ctx=2048, speculate=True, flash_attn=True)) as engine:
        if not engine.pilot or engine.gpu_layers != runtime["gpu_layers"] or engine.settings != runtime["settings"]:
            raise ValueError("native engine did not restore its qualified profile")
        plugin = ctypes.CDLL(request["kernel"])
        masks = []
        for fn in (plugin.ironmule_kg_get_mask, plugin.ironmule_kg_get_mask6):
            fn.restype = ctypes.c_uint32
            masks.append(fn())
        selected = engine.pk2
        if (masks != binding["mask_bits"] or selected["min_rows"] != binding["pk2_min_rows"]
                or float(selected["reason"].split("likelihood ratio ")[1]) > 1.005
                or mapped_vendor() != binding["vendor"] or identity() != runtime["native"]
                or fingerprint() != runtime["fp"] or engine.threads != runtime["threads"]):
            raise ValueError("native libraries or freshly measured execution plan changed")
        with closing(GgufTokenizer(model, engine.L)) as tok:
            prompts = [(TEMPLATE.format(prompt), "") for prompt in request["prompts"]]
            if any(len(tok.encode(prompt)) + request["max_tokens"] > engine.seq_ctx for prompt, _ in prompts):
                raise ValueError("request exceeds the configured per-sequence context")
            wave = run_wave(engine, tok, prompts, request["max_tokens"], {})
    result = {"answers": [{k:a[k] for k in ("answer","prompt_digest","output_ids","stop")} for a in wave["answers"]],
              "ttfts_s": wave["ttfts_s"], "generated_tokens": wave["tokens"], "pilot_decisions": engine.pilot.decisions}
    Path("response.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
