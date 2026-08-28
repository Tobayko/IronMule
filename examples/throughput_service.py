"""Throughput mode: several concurrent requests, grouped batch-1 at width <= 4.

Prints both modes so the trade is visible rather than asserted: grouping raises
aggregate throughput and cuts service TTFT sharply, and costs median latency.

    python examples/throughput_service.py
"""

import ironmule

QUESTIONS = [
    "What is unified memory?",
    "What does quantisation change about a model?",
    "Why is decoding memory bound?",
    "What is a KV cache?",
    "What does a sliding attention window do?",
    "Why does batch size affect throughput?",
]


def run(rt, mode, requests):
    rt.mode = mode
    results = rt.serve(requests)
    snap = rt.telemetry.snapshot()
    print(f"\n{mode.name}")
    print(f"  wall {snap['wall_ms']:.0f} ms   {snap['aggregate_tokens_per_second']:.1f} tok/s"
          f"   mean realised width {snap['mean_realised_width']:.2f}")
    print(f"  service TTFT  p50 {snap['service_ttft_p50_ms']:7.1f} ms   "
          f"p95 {snap['service_ttft_p95_ms']:7.1f} ms")
    print(f"  full latency  p50 {snap['latency_p50_ms']:7.1f} ms   "
          f"p95 {snap['latency_p95_ms']:7.1f} ms")
    if snap["correctness_check_performed"]:
        correctness = f"correctness errors {snap['correctness_errors']}"
    else:
        correctness = "correctness not checked"
    print(f"  fallbacks {snap['fallbacks']}   {correctness}")
    return results


def main() -> int:
    rt = ironmule.Runtime.load()
    plan = ironmule.StrictOneShotPlan()
    build = lambda: [ironmule.Request(prompt_ids=rt.encode(q), max_tokens=48, plan=plan,
                                   rid=f"q{i}") for i, q in enumerate(QUESTIONS)]

    sequential = run(rt, ironmule.InteractiveMode(), build())
    grouped = run(rt, ironmule.ThroughputMode(), build())

    same = all(a.tokens == b.tokens for a, b in zip(sequential, grouped))
    print(f"\nidentical answers in both modes: {same}")
    print("throughput and per-request latency are reported separately on purpose;")
    print("group time is never divided by group width.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
