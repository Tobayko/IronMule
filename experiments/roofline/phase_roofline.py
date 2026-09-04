"""How far is each phase from the physical limit of this device?

Offline analysis over sealed evidence plus the model's own safetensors header.
No model start, no hardware probe, no writes outside stdout.

The device peaks are datasheet values for the Apple M1 Max with the 32-core
GPU, not measurements on this machine. Everything derived from them is a
bound, not a promise: no kernel reaches 100 % of a roofline, so the table also
carries a realistic engineering target.
"""

from __future__ import annotations

import json
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from friday_optimizer.integration import prefill_share, project_request_ratio  # noqa: E402

# Datasheet, Apple M1 Max, 32-core GPU. Stated, never measured here.
BANDWIDTH_BYTES_PER_S = 400e9
PEAK_FP16_FLOPS = 10.4e12
#: No production kernel sustains a roofline; this is the honest target a good
#: implementation can reach on unified memory.
REALISTIC_UTILISATION = 0.85

SNAPSHOT = (ROOT / ".friday-data" / "models" / "hub"
            / "models--mlx-community--gemma-3-4b-it-4bit" / "snapshots")
PERSISTENT = ROOT / "experiments" / "persistent_process" / "results.json"


def weight_bytes() -> tuple[int, float]:
    """Exact weight bytes and quantised parameter count from the header alone."""

    revision = next(SNAPSHOT.iterdir())
    total = 0
    packed_bytes = 0
    for path in sorted(revision.glob("*.safetensors")):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            total += end - start
            if meta["dtype"] == "U32":
                packed_bytes += end - start
    # Four bits per weight, so two weights per byte of packed storage.
    return total, packed_bytes * 2.0


def model_shape() -> tuple[int, int]:
    """Layer count and hidden size, read from the snapshot's own config."""

    revision = next(SNAPSHOT.iterdir())
    config = json.loads((revision / "config.json").read_text())
    text = config.get("text_config", config)
    return int(text["num_hidden_layers"]), int(text["hidden_size"])


def warm_profile() -> dict[str, float]:
    data = json.loads(PERSISTENT.read_text())
    pairs = data["characterization"]["pairs"] + data["validation"]["pairs"]
    warm = [pair["warm"] for pair in pairs]
    tokens = len(warm[0]["tokens"])
    ttft = statistics.median(item["ttft_ns"] for item in warm) / 1e9
    total = statistics.median(item["total_wall_ns"] for item in warm) / 1e9
    return {
        "ttft_seconds": ttft, "tokens": tokens,
        "decode_tps": (tokens - 1) / (total - ttft),
        "prompt_tokens": warm[0]["prompt_tokens"],
    }


def main() -> int:
    total_bytes, quantised_params = weight_bytes()
    profile = warm_profile()
    prompt = profile["prompt_tokens"]

    decode_bytes_per_s = profile["decode_tps"] * total_bytes
    decode_utilisation = decode_bytes_per_s / BANDWIDTH_BYTES_PER_S
    prefill_tps = prompt / profile["ttft_seconds"]
    layers, hidden = model_shape()
    # Attention is quadratic in prompt length and is not covered by 2*N*T.
    # Counting it can only raise utilisation, so leaving it out would flatter
    # the headroom rather than the device.
    attention_flops = 4.0 * layers * prompt * prompt * hidden
    matmul_flops = 2.0 * quantised_params * prompt
    prefill_flops = (matmul_flops + attention_flops) / profile["ttft_seconds"]
    prefill_utilisation = prefill_flops / PEAK_FP16_FLOPS

    print(f"weights           {total_bytes/1e9:.3f} GB, {quantised_params/1e9:.2f} G quantised parameters")
    print(f"prefill work      {matmul_flops/1e12:.3f} TFLOP matmul + {attention_flops/1e12:.3f} TFLOP attention"
          f" ({100*attention_flops/(matmul_flops+attention_flops):.2f} % attention)")
    print(f"warm baseline     prompt {prompt} tok, ttft {profile['ttft_seconds']:.4f} s,"
          f" {profile['tokens']} generated at {profile['decode_tps']:.2f} tok/s")
    print()
    print("phase      measured            device peak         utilisation   headroom")
    print(f"decode     {decode_bytes_per_s/1e9:7.1f} GB/s        {BANDWIDTH_BYTES_PER_S/1e9:5.0f} GB/s"
          f"          {decode_utilisation*100:5.1f} %      {1/decode_utilisation:.2f}x")
    print(f"prefill    {prefill_flops/1e12:7.2f} TFLOP/s      {PEAK_FP16_FLOPS/1e12:5.1f} TFLOP/s"
          f"        {prefill_utilisation*100:5.1f} %      {1/prefill_utilisation:.2f}x")
    print(f"           ({prefill_tps:.1f} prefill tok/s vs {profile['decode_tps']:.1f} decode tok/s"
          f" = {prefill_tps/profile['decode_tps']:.1f}x per token)")
    print()

    shared = {"ttft_seconds": profile["ttft_seconds"], "tokens": profile["tokens"],
              "decode_tps": profile["decode_tps"]}
    share = prefill_share(**shared)
    print(f"prefill share of the warm request: {share*100:.2f} %")
    print()
    print("what each phase is worth end to end, at 32 generated tokens")
    print(f"{'scenario':38s} {'ratio':>9s} {'gain':>8s}")
    scenarios = []
    for label, utilisation in (("roofline (100 %)", 1.0), (f"realistic ({REALISTIC_UTILISATION*100:.0f} %)", REALISTIC_UTILISATION)):
        decode_ratio = utilisation / decode_utilisation
        prefill_ratio = prefill_utilisation / utilisation
        scenarios.extend([
            (f"decode at {label}", {"decode_tps_ratio": decode_ratio}),
            (f"prefill at {label}", {"ttft_ratio": prefill_ratio}),
            (f"both at {label}", {"decode_tps_ratio": decode_ratio, "ttft_ratio": prefill_ratio}),
        ])
    for label, kwargs in scenarios:
        ratio = project_request_ratio(**shared, **kwargs)
        print(f"{label:38s} {ratio:9.6f} {100*(1-ratio):7.2f} %")
    print()
    # Where does the leading lever change? Everything above depends on the
    # registered answer length, which was chosen for measurement budget, not
    # because it is what a request looks like.
    decode_ratio = REALISTIC_UTILISATION / decode_utilisation
    prefill_ratio = prefill_utilisation / REALISTIC_UTILISATION

    def ceilings(count: int) -> tuple[float, float]:
        base = {"ttft_seconds": profile["ttft_seconds"], "tokens": count,
                "decode_tps": profile["decode_tps"]}
        return (1 - project_request_ratio(**base, decode_tps_ratio=decode_ratio),
                1 - project_request_ratio(**base, ttft_ratio=prefill_ratio))

    low, high = 2, 8192
    while high - low > 1:
        middle = (low + high) // 2
        decode_gain, prefill_gain = ceilings(middle)
        if prefill_gain > decode_gain:
            low = middle
        else:
            high = middle
    crossover = low
    decode_gain, prefill_gain = ceilings(crossover)
    print(f"leading lever changes at {crossover} generated tokens"
          f" (both worth {decode_gain*100:.2f} % there,"
          f" prefill share {prefill_share(ttft_seconds=profile['ttft_seconds'], tokens=crossover, decode_tps=profile['decode_tps'])*100:.2f} %)")
    print(f"the registered workload generates {profile['tokens']} tokens, well below it")
    print()
    print(f"{'tokens':>7} {'decode ceiling':>15} {'prefill ceiling':>16}  leads")
    for count in (16, 32, 64, 128, 256, 512, 1024):
        decode_gain, prefill_gain = ceilings(count)
        print(f"{count:7d} {decode_gain*100:14.2f} % {prefill_gain*100:15.2f} %"
              f"  {'prefill' if prefill_gain > decode_gain else 'decode'}")
    print()
    print("measured to date, for comparison")
    for label, kwargs in (
        ("head_skip_prefill (candidate 19)", {"ttft_ratio": 0.846385}),
        ("fixed_compiled (cycle 16)", {"decode_tps_ratio": 1 / 0.9295921887}),
        ("both together (F1 warm arm)", {"ttft_ratio": 0.846385, "decode_tps_ratio": 1 / 0.9295921887}),
    ):
        ratio = project_request_ratio(**shared, **kwargs)
        print(f"{label:38s} {ratio:9.6f} {100*(1-ratio):7.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
