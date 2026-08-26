"""E13: does the execution plan change measurable answer quality?

Preregistered at commit ad8815f, research/raw/E13_preregistration.md
(SHA-256 0fa9621c7ea1f4d14980fb9955ddc9a48a0982ee8201dd099361aabf0bbd73d3).

StrictOneShotPlan vs ReusableSessionPlan on SQuAD v1.1 dev. Ground truth is human
written, so neither the model nor the experimenter decides what is correct. Nothing
in ironmule/ is modified; both plans are called exactly as the runtime exposes them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import string
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx

from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, PrefixCache, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data" / "squad-dev-v1.1.json"

# --- frozen constants --------------------------------------------------------
INSTRUCTION = ("Read the document and answer the question using the shortest exact span from "
               "the document. Give only that span, with no explanation and no full sentence.\n\n"
               "Document:\n")
BANDS = {"SHORT": (512, 899), "NEAR": (900, 1150), "LONG": (1151, 2048)}
BAND_ORDER = ["SHORT", "NEAR", "LONG"]
PILOT_ARTICLE_INDICES = (0, 1, 2)
QUESTIONS_PER_CONTEXT = 8
MAX_NEW_TOKENS = 24
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260825
MARGIN = 0.05
KNOBS = Knobs(compiled_fixed_cache=True, fused_argmax=False, head_skip_prefill=True,
              fuse_projections=True)


# --- scoring -----------------------------------------------------------------

_PUNCT = set(string.punctuation)


def normalise(text: str) -> list[str]:
    """SQuAD official normalisation, returned as tokens rather than a string.

    Token sequences rather than raw substrings, so a gold answer cannot be matched
    inside a longer word.
    """
    lowered = text.lower()
    stripped = "".join(" " if ch in _PUNCT else ch for ch in lowered)
    return [w for w in re.split(r"\s+", stripped) if w and w not in ("a", "an", "the")]


def contains(prediction: str, golds: list[str]) -> bool:
    """Primary metric: any gold token sequence occurs contiguously in the prediction."""
    pred = normalise(prediction)
    for gold in golds:
        want = normalise(gold)
        if not want:
            continue
        for start in range(len(pred) - len(want) + 1):
            if pred[start:start + len(want)] == want:
                return True
    return False


def exact_match(prediction: str, golds: list[str]) -> bool:
    pred = normalise(prediction)
    return any(pred == normalise(gold) for gold in golds)


def token_f1(prediction: str, golds: list[str]) -> float:
    pred = normalise(prediction)
    best = 0.0
    for gold in golds:
        want = normalise(gold)
        if not pred or not want:
            best = max(best, float(pred == want))
            continue
        common = 0
        pool = list(want)
        for token in pred:
            if token in pool:
                pool.remove(token)
                common += 1
        if common:
            precision, recall = common / len(pred), common / len(want)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


SCORER_CONTROLS = [
    # name, gold answers, prediction, expected containment
    ("deliberately_wrong", ["Denver Broncos"], "Carolina Panthers", False),
    ("different_case", ["Denver Broncos"], "denver broncos", True),
    ("trailing_punctuation", ["Denver Broncos"], "Denver Broncos.", True),
    ("wrapped_in_prose", ["Denver Broncos"], "The answer is the Denver Broncos team.", True),
    ("leading_article", ["Denver Broncos"], "the Denver Broncos", True),
    ("markdown_emphasis", ["Denver Broncos"], "**Denver Broncos**", True),
    ("empty_prediction", ["Denver Broncos"], "", False),
    ("gold_inside_longer_word", ["art"], "started restarting", False),
    ("second_gold_variant", ["gold", "Gold Rush"], "the Gold Rush of 1849", True),
    ("shotgun_listing", ["Denver Broncos"], "Denver Broncos, Carolina Panthers, Seahawks, Patriots", True),
]


def run_scorer_controls() -> dict:
    rows = []
    for name, golds, prediction, expected in SCORER_CONTROLS:
        got = contains(prediction, golds)
        rows.append({"control": name, "golds": golds, "prediction": prediction,
                     "expected": expected, "got": got, "passed": got == expected,
                     "exact_match": exact_match(prediction, golds),
                     "token_f1": round(token_f1(prediction, golds), 4)})
    return {"controls": rows, "all_passed": all(r["passed"] for r in rows),
            "shotgun_scores_correct": next(r["got"] for r in rows if r["control"] == "shotgun_listing")}


# --- evaluation set ----------------------------------------------------------

def render_prefix(tok, document: str) -> str:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + document + "\n\n@@CUT@@"}],
        tokenize=False, add_generation_prompt=True)
    return rendered.split("@@CUT@@")[0]


def render_full(tok, document: str, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + document + "\n\nQuestion: " + question}],
        tokenize=False, add_generation_prompt=True)


def build_eval_set(tok, pilot: bool = False) -> tuple[list[dict], list[dict]]:
    """The frozen selection rule. Content blind: band assignment is by index only."""
    data = json.loads(DATA.read_text())
    articles = sorted(data["data"], key=lambda a: a["title"])
    encode = lambda t: list(tok.encode(t, add_special_tokens=False))

    if pilot:
        chosen = [(articles[i], BAND_ORDER[i % 3]) for i in PILOT_ARTICLE_INDICES]
    else:
        rest = [a for i, a in enumerate(articles) if i not in PILOT_ARTICLE_INDICES]
        chosen = [(a, BAND_ORDER[i % 3]) for i, a in enumerate(rest)]

    contexts, excluded = [], []
    for article, band in chosen:
        low, high = BANDS[band]
        document, qas, landed = "", [], None
        for paragraph in article["paragraphs"]:
            document = (document + "\n\n" + paragraph["context"]).strip()
            qas = qas + paragraph["qas"]
            count = len(encode(render_prefix(tok, document)))
            if count > high:
                break
            if count >= low:
                landed = count
                break
        if landed is None:
            excluded.append({"title": article["title"], "band": band, "reason": "band_overshoot"})
            continue
        if len(qas) < QUESTIONS_PER_CONTEXT:
            excluded.append({"title": article["title"], "band": band, "reason": "too_few_questions",
                             "questions": len(qas)})
            continue
        prefix_ids = encode(render_prefix(tok, document))
        questions = []
        for qa in qas[:QUESTIONS_PER_CONTEXT]:
            full_ids = encode(render_full(tok, document, qa["question"]))
            if full_ids[:len(prefix_ids)] != prefix_ids:
                excluded.append({"title": article["title"], "band": band,
                                 "reason": "tokenisation_gate", "qid": qa["id"]})
                questions = []
                break
            questions.append({"id": qa["id"], "question": qa["question"],
                              "golds": sorted({x["text"] for x in qa["answers"]}),
                              "full_ids": full_ids, "prompt_tokens": len(full_ids)})
        if not questions:
            continue
        contexts.append({"title": article["title"], "band": band, "document": document,
                         "prefix_ids": prefix_ids, "prefix_tokens": landed,
                         "paragraphs": document.count("\n\n") + 1, "questions": questions})
    return contexts, excluded


# --- running one context -----------------------------------------------------

def copy_state(state):
    """Fresh dict wrapping the same arrays. FixedKVCache rebinds dict entries, so a
    caller that wants to keep a state must not hand over its own dict."""
    return {"position": {"offset": state["position"]["offset"]},
            "layers": [{"keys": l["keys"], "values": l["values"]} for l in state["layers"]]}


def top_gap(logits) -> float:
    f = logits.astype(mx.float32).reshape(-1)
    first = mx.max(f)
    second = mx.max(mx.where(f == first, mx.array(-1e30, mx.float32), f))
    return float((first - second).item())


class Runner:
    def __init__(self, model_id: str):
        self.engine, self.tok = load_engine(model_id, KNOBS)
        self.eos = _eos_ids(self.tok)
        self.trunk = _trunk(self.engine.model)

    def prefill(self, full_ids, capacity):
        started = time.perf_counter_ns()
        state, token = self.engine._prefill(list(full_ids), capacity)
        return state, token, time.perf_counter_ns() - started

    def answer_nll(self, state, capacity, gold: str) -> float | None:
        """Per-token NLL of the gold answer under teacher forcing, continuation only.

        The first answer token is excluded. Both plans expose only `(state, token)`
        from their prefill, not the logits behind it, and re-running the prefill to
        recover them would measure a different execution than the one under test.
        Positions 2..n are scored, which keeps the metric identical in construction
        between the two plans and therefore still comparable; it simply covers only
        gold answers of two tokens or more. Coverage is reported.
        """
        gold_ids = list(self.tok.encode(gold, add_special_tokens=False))
        if len(gold_ids) < 2:
            return None
        work = copy_state(state)
        caches = [FixedKVCache(layer, work["position"], capacity) for layer in work["layers"]]
        hidden = self.trunk(mx.array(gold_ids[:-1])[None, :], cache=caches)
        logits = _project(self.engine.model, hidden)
        mx.eval(logits)
        logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
        total = 0.0
        for position, target in enumerate(gold_ids[1:]):
            total += float(logprobs[0, position, target].item())
        return -total / (len(gold_ids) - 1)

    def generate(self, state, first_token, capacity):
        body = self.engine._body(capacity, 1)
        tokens = [int(first_token.reshape((-1,)).item())]
        gaps = []
        token = first_token
        started = time.perf_counter_ns()
        while len(tokens) < MAX_NEW_TOKENS and tokens[-1] not in self.eos:
            out = body(token, state)
            logits = out[0][:, -1, :]
            state = out[1]
            mx.eval(logits, *_leaves(state))
            mx.synchronize()
            gaps.append(top_gap(logits))
            token = mx.argmax(logits.astype(mx.float32), axis=-1).reshape((1, 1))
            tokens.append(int(token.reshape((-1,)).item()))
        elapsed = time.perf_counter_ns() - started
        visible = [t for t in tokens if t not in self.eos]
        return {"tokens": tokens, "text": self.tok.decode(visible), "decode_ns": elapsed,
                "gaps": gaps, "stopped": "eos" if tokens[-1] in self.eos else "length"}

    def run_context(self, context, plan: str) -> dict:
        capacity = ((max(q["prompt_tokens"] for q in context["questions"])
                     + MAX_NEW_TOKENS + 63) // 64) * 64
        self.engine.prefix_cache = (PrefixCache(context["prefix_ids"])
                                    if plan == "reusable" else None)
        self.engine._compiled = None
        session_started = time.perf_counter_ns()
        rows = []
        for question in context["questions"]:
            state, token, ttft_ns = self.prefill(question["full_ids"], capacity)
            nll = self.answer_nll(copy_state(state), capacity, question["golds"][0])
            out = self.generate(copy_state(state), token, capacity)
            rows.append({
                "id": question["id"], "question": question["question"],
                "golds": question["golds"], "prediction": out["text"],
                "tokens": out["tokens"], "stopped": out["stopped"],
                "contains": contains(out["text"], question["golds"]),
                "exact_match": exact_match(out["text"], question["golds"]),
                "token_f1": token_f1(out["text"], question["golds"]),
                "answer_nll": nll, "ttft_ns": ttft_ns, "decode_ns": out["decode_ns"],
                "top_gaps": out["gaps"], "prompt_tokens": question["prompt_tokens"],
            })
        session_ns = time.perf_counter_ns() - session_started
        hits = self.engine.prefix_cache.hits if self.engine.prefix_cache else 0
        self.engine.prefix_cache = None
        return {"plan": plan, "title": context["title"], "band": context["band"],
                "prefix_tokens": context["prefix_tokens"], "capacity": capacity,
                "cache_hits": hits, "session_ns": session_ns, "questions": rows}


# --- analysis ----------------------------------------------------------------

def paired_cluster_bootstrap(diffs: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                             seed: int = BOOTSTRAP_SEED) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"mean": sum(diffs) / n, "ci_low": means[int(0.025 * resamples)],
            "ci_high": means[int(0.975 * resamples)], "clusters": n}


def analyse(strict: list[dict], reusable: list[dict]) -> dict:
    by_plan = {"strict": {c["title"]: c for c in strict},
               "reusable": {c["title"]: c for c in reusable}}
    titles = [c["title"] for c in strict]

    per_context, discord = [], {"strict_only": [], "reusable_only": []}
    divergences = []
    divergent = 0
    total_questions = 0
    for title in titles:
        s, r = by_plan["strict"][title], by_plan["reusable"][title]
        s_acc = sum(q["contains"] for q in s["questions"]) / len(s["questions"])
        r_acc = sum(q["contains"] for q in r["questions"]) / len(r["questions"])
        for sq, rq in zip(s["questions"], r["questions"]):
            total_questions += 1
            if sq["tokens"] != rq["tokens"]:
                divergent += 1
                index = next((i for i in range(min(len(sq["tokens"]), len(rq["tokens"])))
                              if sq["tokens"][i] != rq["tokens"][i]),
                             min(len(sq["tokens"]), len(rq["tokens"])))
                divergences.append({
                    "title": title, "band": s["band"], "id": sq["id"],
                    "first_differing_index": index,
                    "strict_gap": sq["top_gaps"][index - 1] if 0 < index <= len(sq["top_gaps"]) else None,
                    "reusable_gap": rq["top_gaps"][index - 1] if 0 < index <= len(rq["top_gaps"]) else None,
                    "both_correct": sq["contains"] and rq["contains"],
                    "changed_correctness": sq["contains"] != rq["contains"]})
            if sq["contains"] and not rq["contains"]:
                discord["strict_only"].append({"title": title, "band": s["band"], "id": sq["id"],
                                               "question": sq["question"], "golds": sq["golds"],
                                               "strict": sq["prediction"], "reusable": rq["prediction"]})
            if rq["contains"] and not sq["contains"]:
                discord["reusable_only"].append({"title": title, "band": s["band"], "id": sq["id"],
                                                 "question": sq["question"], "golds": sq["golds"],
                                                 "strict": sq["prediction"], "reusable": rq["prediction"]})
        per_context.append({
            "title": title, "band": s["band"], "prefix_tokens": s["prefix_tokens"],
            "strict_acc": s_acc, "reusable_acc": r_acc, "diff": r_acc - s_acc,
            "strict_em": sum(q["exact_match"] for q in s["questions"]) / len(s["questions"]),
            "reusable_em": sum(q["exact_match"] for q in r["questions"]) / len(r["questions"]),
            "strict_f1": sum(q["token_f1"] for q in s["questions"]) / len(s["questions"]),
            "reusable_f1": sum(q["token_f1"] for q in r["questions"]) / len(r["questions"]),
            "strict_nll": [q["answer_nll"] for q in s["questions"] if q["answer_nll"] is not None],
            "reusable_nll": [q["answer_nll"] for q in r["questions"] if q["answer_nll"] is not None],
            "strict_ttft_ms": statistics.median(q["ttft_ns"] for q in s["questions"]) / 1e6,
            "reusable_ttft_ms": statistics.median(q["ttft_ns"] for q in r["questions"]) / 1e6,
            "strict_session_ms": s["session_ns"] / 1e6, "reusable_session_ms": r["session_ns"] / 1e6,
            "cache_hits": r["cache_hits"],
        })

    primary = paired_cluster_bootstrap([c["diff"] for c in per_context])
    bands = {}
    for band in BAND_ORDER:
        subset = [c for c in per_context if c["band"] == band]
        if subset:
            bands[band] = {**paired_cluster_bootstrap([c["diff"] for c in subset]),
                           "strict_acc": sum(c["strict_acc"] for c in subset) / len(subset),
                           "reusable_acc": sum(c["reusable_acc"] for c in subset) / len(subset),
                           "prefix_tokens": [c["prefix_tokens"] for c in subset]}

    below = [b for b in bands.values() if b["ci_high"] < 0]
    above = [b for b in bands.values() if b["ci_low"] > 0]
    if below and above:
        verdict = "LENGTH_OR_TASK_DEPENDENT"
    elif primary["ci_low"] > 0:
        verdict = "REUSABLE_BETTER"
    elif primary["ci_high"] < 0:
        verdict = "REUSABLE_WORSE"
    elif primary["ci_low"] > -MARGIN:
        verdict = "REUSABLE_NONINFERIOR"
    else:
        verdict = "INCONCLUSIVE"

    flat = lambda key: [v for c in per_context for v in c[key]]
    return {
        "experiment": "E13", "verdict": verdict, "margin": MARGIN,
        "contexts": len(per_context), "questions": total_questions,
        "strict_accuracy": sum(c["strict_acc"] for c in per_context) / len(per_context),
        "reusable_accuracy": sum(c["reusable_acc"] for c in per_context) / len(per_context),
        "paired_difference": primary,
        "strict_em": sum(c["strict_em"] for c in per_context) / len(per_context),
        "reusable_em": sum(c["reusable_em"] for c in per_context) / len(per_context),
        "em_difference": paired_cluster_bootstrap([c["reusable_em"] - c["strict_em"] for c in per_context]),
        "strict_f1": sum(c["strict_f1"] for c in per_context) / len(per_context),
        "reusable_f1": sum(c["reusable_f1"] for c in per_context) / len(per_context),
        "f1_difference": paired_cluster_bootstrap([c["reusable_f1"] - c["strict_f1"] for c in per_context]),
        "strict_nll_mean": statistics.mean(flat("strict_nll")) if flat("strict_nll") else None,
        "reusable_nll_mean": statistics.mean(flat("reusable_nll")) if flat("reusable_nll") else None,
        "answer_divergence_rate": divergent / total_questions if total_questions else 0.0,
        "divergent_questions": divergent,
        "discordance": {"strict_only": len(discord["strict_only"]),
                        "reusable_only": len(discord["reusable_only"])},
        "nll_coverage": len(flat("strict_nll")) / total_questions if total_questions else 0.0,
        "divergences": divergences,
        "divergences_changing_correctness": sum(d["changed_correctness"] for d in divergences),
        "bands": bands, "per_context": per_context, "discordant_cases": discord,
    }


# --- driver ------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["controls", "pilot", "main"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if args.stage == "controls":
        result = run_scorer_controls()
        for row in result["controls"]:
            print(f"{row['control']:26s} expected={str(row['expected']):5s} got={str(row['got']):5s} "
                  f"{'PASS' if row['passed'] else 'FAIL'}   em={str(row['exact_match']):5s} f1={row['token_f1']:.3f}")
        print(f"\nall scorer controls passed: {result['all_passed']}")
        print(f"shotgun listing scores correct (known containment bound): {result['shotgun_scores_correct']}")
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / "E13_scorer_controls.json").write_text(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result["all_passed"] else 1

    busy = gpu_busy()
    if busy:
        print(f"ABORT gpu_busy: {busy}")
        return 2
    env = bench.environment()
    if env["power_source"] != "AC":
        print(f"ABORT power_source={env['power_source']}")
        return 2

    runner = Runner(args.model)
    pilot = args.stage == "pilot"
    contexts, excluded = build_eval_set(runner.tok, pilot=pilot)
    print(f"{args.stage}: {len(contexts)} contexts, "
          f"{sum(len(c['questions']) for c in contexts)} questions, excluded {len(excluded)}")
    for entry in excluded:
        print(f"  excluded {entry['title']} ({entry['band']}): {entry['reason']}")

    started = time.perf_counter()
    strict, reusable = [], []
    for index, context in enumerate(contexts):
        strict.append(runner.run_context(context, "strict"))
        reusable.append(runner.run_context(context, "reusable"))
        s_acc = sum(q["contains"] for q in strict[-1]["questions"]) / QUESTIONS_PER_CONTEXT
        r_acc = sum(q["contains"] for q in reusable[-1]["questions"]) / QUESTIONS_PER_CONTEXT
        print(f"  [{index+1:2d}/{len(contexts)}] {context['title'][:30]:30s} {context['band']:5s} "
              f"tok={context['prefix_tokens']:5d} strict={s_acc:.3f} reuse={r_acc:.3f} "
              f"hits={reusable[-1]['cache_hits']}", flush=True)
        mx.clear_cache()

    payload = {"experiment": "E13", "stage": args.stage,
               "preregistration_sha256": "0fa9621c7ea1f4d14980fb9955ddc9a48a0982ee8201dd099361aabf0bbd73d3",
               "prereg_commit": "ad8815f6729b8b2645ff88eb667547e961eac158",
               "excluded": excluded, "wall_seconds": time.perf_counter() - started,
               "strict": strict, "reusable": reusable,
               "mlx_peak_bytes": mx.get_peak_memory(), "environment": env}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E13_results_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path} ({payload['wall_seconds']:.0f}s, peak {payload['mlx_peak_bytes']/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
