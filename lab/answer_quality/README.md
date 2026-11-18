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
  --seed 20261119 --split holdout --count 12 --seqs 4 --threads 2 --max-tokens 64 \
  --output /tmp/qwen3-06b-answer-quality.json
```

The development pilot exposed a difference between exact output formatting and answer correctness. Its earlier seed 20261118 and first grading protocol informed the semantic rubric and evidence-first instructions. The final pair uses a new seed, a versioned generator and the frozen rubric. No examples from these generated ledgers were supplied for weight training. The report records observations from one small pretrained decoder on this bounded English task family.
