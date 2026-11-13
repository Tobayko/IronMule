# CMP: IronMule against llama.cpp, Ollama, LM Studio, vLLM and SGLang

The owner's bar (2026-11-13): clearly faster than llama.cpp, Ollama, LM Studio, vLLM and SGLang on the same hardware and
model, with the same answer quality. MLX is left out for now. Results go to `docs/LAB.md` (CMP1, ...).

## The harness

`bench.py URL MODEL LABEL OUT.json` drives any OpenAI-compatible server; `run_all.sh` starts each engine with the same
GGUF on this machine and runs the workloads one engine at a time. Tokens are counted by re-tokenizing every answer with
one tokenizer, so engines that report usage differently count alike. Answers are checked on the 50 facts.

| Workload | What it stands for | Measured |
| --- | --- | --- |
| `burst-sys` | a chat app or RAG front end: N users at once, one long system prompt, own questions | tokens/s, first token mean and max |
| `burst` | N users at once, short prompts | same |
| `poisson` | a server with open-loop arrivals | tokens/s, first token p50/p90/p99, end to end p50/p90 |
| `single` | one person on a desktop (what Ollama and LM Studio users feel) | decode tokens/s, first token |
| `facts` | answer quality | correct of 50 |

## The engines and how each is set up (its own best documented way to serve many requests)

| Engine | Version | Setup | Where |
| --- | --- | --- | --- |
| IronMule | this branch | `ironmule serve --agent on --gguf`, FA1/FA1u and ROW1 B16 in the library, SEQ1 width, shared pool | CPU here |
| llama.cpp | 4227c9b (stock: ROW1 and FA1 off) | `llama-server -np 32 -c 32768 --jinja`, with and without `-fa` | CPU here |
| Ollama | 0.40.16 (its own llama.cpp) | `OLLAMA_NUM_PARALLEL=32 OLLAMA_CONTEXT_LENGTH=1024` | CPU here |
| LM Studio | llmster 0.0.27, runtime llama.cpp avx2 2.66.0 | `lms load --parallel 32 --context-length 32768 --gpu off` | CPU here |
| vLLM | 0.33.0 CPU build from source (no CPU wheel on PyPI; triton-cpu unreachable) | `vllm serve` bf16 | CPU here, T4 x2 on Kaggle |
| SGLang | 0.5.23 | GPU first (`--attention-backend triton` on a T4) | Kaggle T4 x2 |

## More tests worth running (not all run yet)

1. *Multi-turn chats*: each user's second and third turn repeat the whole conversation as prompt; prefix reuse across
   turns and users decides the first token.
2. *Long context*: 4k-16k prompts (documents), one and several at once; attention dominates (KV1, FA1).
3. *Mixed lengths*: answers from 16 to 1,024 tokens in one burst; tests continuous batching and joining.
4. *Overload*: arrivals above capacity for a minute, then quiet; queueing, admission, no failed request (STAB2).
5. *Many distinct system prompts*: 4-8 tenants, each with its own prompt; tests how the shared pool keeps and evicts.
6. *Structured output and tool calls*: JSON answers; speculation from the context should help most here.
7. *Other models*: Gemma 3 4B, Llama 3.1 8B, Mistral 7B, Qwen3 8B/14B Q4_K_M (most used open weights that fit a
   desktop or a T4).
8. *GPUs*: one T4, two T4s (split and one engine per GPU) against llama.cpp, Ollama, vLLM and SGLang (MG2 harness).
9. *Energy*: tokens per joule where RAPL or nvidia-smi power is readable.

## Ideas that do not exist yet (or only on paper), and their state

| Idea | State |
| --- | --- |
| FA1: grouped-query, blocked flash attention with f32 accumulation for llama.cpp's CPU path | built, 2.69x a step at 16 x 1,024 (LAB FA1) |
| FA1u + SP1: one KV pool for all sequences, prefixes shared across running and finished requests (RadixAttention's idea on llama.cpp's CPU path) | built, 5.1x a cold burst, 1.54x warm (LAB SP1) |
| A learning agent that picks width and attention from the traffic it dreams (AG10) | built, simulated, live run kept the best configuration |
| Collective speculation: draft tokens from other users' answers, not only from the request's own context | next |
| AMX-BF16 tiles for attention scores and V (bf16 KV cache), AMX for prefill attention | open; needs a quality gate for a bf16 KV cache |
| Kept prefixes compressed (q8) or moved to disk, restored on a hit | open |
| Prefix-aware admission: start requests that share a prefix together (one prefill) | partly (SP1 waits a step for a sharer) |
| One engine per NUMA node or per GPU with prefix-aware routing between them | open (MG2 routes by queue length) |
