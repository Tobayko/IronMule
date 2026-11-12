"""Round 5 part 7: where a single request's first token goes. Usage: ttft_probe.py GGUF MODEL_ID GATE_JSON.
(1) the engine alone in process: prompt prefill + first token, 5 runs; (2) the product (service + worker + engine):
first streamed token, 5 runs, and the service's schedule policy."""
import json, os, sys, tempfile, time
from pathlib import Path
ROOT = Path("/home/user/IronMule"); sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "training")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gg1_measure
gguf, model, gate = sys.argv[1], sys.argv[2], json.loads(open(sys.argv[3]).read())
spec = gg1_measure.model_spec(model)
text = "Write a short story about a lighthouse keeper. /no_think"
from mlx_lm.tokenizer_utils import load
tok = load(Path(spec.snapshot_path))
ids = tok._tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_dict=False)
print("prompt tokens", len(ids), flush=True)
mode = os.environ.get("PROBE", "both")
if mode in ("both", "engine"):
    from ironmule.ggml_engine import GgmlEngine, Job
    eng = GgmlEngine(gguf, speculate=True, **gate.get("settings", {}))
    for _ in range(6):
        t0 = time.perf_counter(); first = []
        j = Job(list(ids), 8, emit=lambda t: first.append(time.perf_counter() - t0) if not first else None)
        eng.run([j]); print("engine first %.3f s, 8 tokens %.3f s" % (first[0], time.perf_counter() - t0), flush=True)
    eng.close()
if mode in ("both", "product"):
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest
    store = ProductStore(Path(tempfile.mkdtemp(prefix="ttft-")) / "state"); store.setup("server"); store.register_model(spec)
    backend = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml", ggml_gguf=gguf,
                              ggml_gate=gate, speculate=True, startup_timeout=1200)
    backend.start(); service = ProductService(store, backend=backend, spec=spec)
    print("schedule", json.dumps(service.health().get("schedule")), flush=True)
    try:
        for i in range(6):
            req = GenerationRequest(spec.model_id, (("user", text),), max_tokens=8, request_id=f"p{i}", stream=True)
            t0 = time.perf_counter(); first = None; kinds = []
            for ev in service.stream(req):
                if isinstance(ev, dict):
                    kinds.append(ev.get("type"))
                    if ev.get("type") == "token" and first is None:
                        first = time.perf_counter() - t0
            print("product first %.3f s, 8 tokens %.3f s" % (first, time.perf_counter() - t0), kinds[:3], flush=True)
    finally:
        service.close()
