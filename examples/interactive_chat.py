"""Interactive mode: one caller, lowest latency, sequential batch-1.

    python examples/interactive_chat.py "Explain unified memory in two sentences."
"""

import sys

import ironmule


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "Explain unified memory in two sentences."
    rt = ironmule.Runtime.load(mode=ironmule.InteractiveMode())
    result = rt.generate(prompt, max_tokens=96)

    print(result.text.strip())
    snap = rt.telemetry.snapshot()
    print(f"\n[{snap['mode']}] {result.metrics['generated_tokens']} tokens, "
          f"stop={result.stop_reason}")
    print(f"  service TTFT {result.metrics['service_ttft_ms']:.1f} ms   "
          f"engine TTFT {result.metrics['engine_ttft_ms']:.1f} ms   "
          f"latency {result.metrics['latency_ms']:.1f} ms")
    print(f"  inter-token p50 {snap['inter_token_p50_ms']:.2f} ms   "
          f"{snap['aggregate_tokens_per_second']:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
