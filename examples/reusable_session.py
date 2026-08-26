"""Reusable session: one document, many questions, prefill computed once.

Within this plan a cache hit is bit exact. It does not agree with
StrictOneShotPlan, which is why the plan is chosen here explicitly.

    python examples/reusable_session.py
"""

import ironmule

DOCUMENT = (
    "Apple silicon uses a unified memory architecture in which the CPU and GPU "
    "address the same physical memory, so a tensor does not have to be copied "
    "between separate device pools. The M1 Max provides 32 GPU cores and a wide "
    "memory interface. MLX evaluates arrays lazily and executes them on the GPU, "
    "which means a timer must synchronise before it can be trusted. Quantised "
    "weights are stored in four bits with a scale and a bias per group of 64, so "
    "the effective cost of a weight is about 4.5 bits."
) * 4

QUESTIONS = [
    "How many GPU cores does the M1 Max provide?",
    "How many bits are quantised weights stored in?",
    "What must a timer do before it can be trusted?",
    "What does unified memory avoid?",
]


def main() -> int:
    rt = ironmule.Runtime.load(mode=ironmule.InteractiveMode())
    plan = rt.session_plan(DOCUMENT, name="apple-silicon")
    print(f"session prefix: {len(plan.prefix_ids)} tokens\n")

    for question in QUESTIONS:
        result = rt.generate(DOCUMENT + "\n\nQuestion: " + question,
                             plan=plan, max_tokens=48)
        print(f"Q {question}")
        print(f"A {result.text.strip()}")
        print(f"  service TTFT {result.metrics['service_ttft_ms']:7.1f} ms   "
              f"cache {plan.describe()['hits']} hits / {plan.describe()['misses']} misses\n")

    print("The first request pays the prefix; every later one reuses it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
