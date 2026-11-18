# Grounded answers and tool execution

This local lab compares two answer workflows on the same GGUF and the same generated shipments. It uses the IronMule GGML engine, the model's embedded chat template, greedy decoding, disabled thinking and a fresh native context state for each batch. The tool protocol requires a chat template that renders function schemas.

Each group has three independently checkable tasks.

- For a supported lookup, the answer is the field on the requested ledger row.
- For an unsupported lookup, either the shipment or the requested field is absent. Other shipments contain plausible distractor values.
- A tool task starts with a shipment identifier. The model can look up its depot and then use that result to look up the dock. The harness executes the actual requested arguments against a private in-memory table. Function errors and intermediate values go back to the model. The evaluator derives its labels from the tables and checks the actual ordered calls and final answer.

Both workflows have at most four model turns per tool task and the same per-turn output limit. The reference ends on the first plain answer. The evidence-first condition adds a general instruction to verify the source row and follow all outstanding tool dependencies. If a proposed dock lacks a matching field in the tool returns, the harness requests another turn. At the turn limit it withholds such an answer and returns `UNKNOWN`. The report distinguishes an exhausted budget, an explicit model abstention, a wrong code, a wrong reference and other text. Each tool trajectory contains the raw proposed answers, actual results and delivered final answer. The workflow is opt-in to this evaluation.

The grader reports strict formatting separately from accuracy. It accepts a bare code or an affirmative statement naming the requested field and shipment. It accepts a fixed set of complete abstentions. Prose outside the recognized grammar stays in `other_text`. A successful tool task requires both calls in the proper order and the correct final code. A guessed final code alone earns only the separate final-answer metric.

## Reproduce the measured pair

Use a matching llama-cpp-python 0.3.16 native build as described in the portable GPU lab. Fetch the public pinned GGUF if needed. The command writes both conditions, raw answers, tool traces, per-file model and runtime fingerprints, and aggregate counts to a fresh output path.

```sh
python lab/portable_gpu/fetch.py qwen3-06b --dir /models
IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_PILOT=0 IRONMULE_KG5=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m lab.answer_quality.evaluate --gguf /models/Qwen_Qwen3-0.6B-Q4_K_M.gguf \
  --seed 20261119 --split holdout --count 12 --seqs 4 --threads 1 --max-tokens 64 \
  --output /tmp/qwen3-06b-answer-quality.json
```

The development pilot exposed a difference between exact output formatting and answer correctness. Its earlier seed 20261118 and first grading protocol informed the semantic rubric and evidence-first instructions. The final pair uses a new seed, a versioned generator and the frozen rubric. No examples from these generated ledgers were supplied for weight training. The report records observations from one small pretrained decoder on this bounded English task family.

## Qwen3-0.6B paired result

The unchanged Q4_K_M decoder saw the same 12 groups in both conditions. The frozen grader awarded full credit on the following cases.

| Held-out task | Reference | Evidence-first |
|---|---:|---:|
| Correct value on a supported ledger field | 12/12 | 1/12 |
| Recognized abstention on an unsupported field | 1/12 | 1/12 |
| Explicit unsupported code on an unsupported field | 10/12 | 0/12 |
| First tool called with the correct shipment | 12/12 | 12/12 |
| Both required tool calls completed in order | 0/12 | 0/12 |
| Correct final dock from the completed workflow | 0/12 | 0/12 |
| Final depot identifier returned as the dock | 12/12 | 0/12 |
| Final refusal on an answerable tool task | 0/12 | 12/12 |
| Model turns and generated tokens on the tool tasks | 24 turns, 420 tokens | 36 turns, 452 tokens |

The automated abstention counts leave out complete answers that fall outside the response grammar. Inspection of the paired raw ledger answers gives this additional breakdown. It was performed after scoring and did not change the rubric or saved scores.

- On the 12 unsupported questions the reference gave an unsupported code ten times. It correctly declined twice, including the longer answer in group 10 that the grader placed in `other_text`. The candidate declined all 12. In groups 3, 4, 7 and 8 its explanation said there was “no entry for shipment” even though that shipment had a row and only the requested field was missing.
- On the 12 supported questions the candidate falsely declined groups 0 through 6 and 8 through 10, for ten false abstentions. Group 7 was correct. Group 11 contained the correct seal code in a complete sentence but hit the 64-token cap halfway through a repeated final answer. The reference answered all 12 without abstaining.
- On all 12 tool tasks the unguarded decoder returned the depot identifier as its final dock code. The candidate also proposed the depot identifier, received a request to obtain evidence and then explicitly answered `UNKNOWN`. No forced abstentions at the turn limit occurred. Both conditions executed exactly 12 lookups. Their 12 missing second lookups account for all 12 unsuccessful tasks in each condition.

This paired observation supports leaving the workflow experimental. Its reduction in unsupported codes came with a loss of coverage, incorrect explanations and no improvement in completed tool tasks. These are counts for one synthetic domain, checkpoint and seed. Both conditions ran entirely on one local CPU thread for the saved comparison.

[Raw paired responses and tool traces](../../evidence/answer-quality/qwen3-06b-grounding-tools-20261118.json)

To reproduce the frozen scorecards from the saved responses without a model, run the following command.

```sh
python -m lab.answer_quality.check_report evidence/answer-quality/qwen3-06b-grounding-tools-20261118.json
```

## Structured application workflow

A third workflow tests a narrower application with an explicit shipment-lookup contract. It pairs the unchanged reference prompt and decoder with a freshly generated set of shipments (seed 20261120). Both conditions use the same questions, records, limits and model.

For a ledger request, the application parses the supplied question and checks the exact shipment and field in the visible input. It reports duplicate or missing data explicitly. For a dock task, the model chooses the first tool and supplies the shipment. When a successful lookup for the requested shipment returns a depot, the application passes that actual depot to the second function. The model sees the authentic returns. The final renderer preserves a model response matching the canonical source value, otherwise it delivers the returned value or `UNKNOWN` when a value could not be obtained. This renderer cannot reconstruct a missing field. A failed or off-target first tool call cannot receive credit for the full workflow.

The execution functions receive public prompts, query identifiers and callable tools. The grading function reads the separately derived labels after execution. Reports identify the author of each tool call, each return value, the model response, and the answer actually delivered. Model-level and application-level results can therefore be checked separately.

```sh
IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_PILOT=0 IRONMULE_KG5=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m lab.answer_quality.structured --gguf /models/Qwen_Qwen3-0.6B-Q4_K_M.gguf \
  --output /tmp/qwen3-06b-structured.json
```
