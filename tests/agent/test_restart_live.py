"""The passive endpoint can stay idle, or accept an independent first client before choosing a worker."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lab.pk1 import restart_http as startup, restart_live as live
from training import scoped_release as release


@pytest.mark.parametrize("arrive", [False, True])
def test_listener_generates_no_traffic_and_external_process_can_drive_selection(tmp_path, monkeypatch, arrive):
    receipt = {"qualification": {"candidate_sha256": "new", "reference_sha256": "old", "binding": {"runtime": {"settings": {}}}}}
    args = SimpleNamespace(state=tmp_path, gguf=tmp_path/"test.gguf", port=0,
                           deadline=time.monotonic()+20, lifetime=2 if arrive else .1,
                           receipt_sha256="pin", geometry_sha256="geom",
                           geometry_record={"count": 18, "minimum": [6,7,7], "maximum": [6,7,7]})
    case = {"passed": True, "reference_bounds": {}}
    release.save(tmp_path/"decision.json",case)
    for arm in ("reference","candidate"):
        release.save(tmp_path/f"gate-{arm}.json",{})
    monkeypatch.setattr(startup,"assess",lambda *_:case)
    monkeypatch.setattr(release,"selected",lambda *_:"candidate")
    answers = dict(live.orders())
    seen = []
    class Existing:
        def note(self, *a, **kw): return None
        def ready(self): return True
        def health(self): return {"completed_requests":len(seen)}
        def outstanding(self): return 0
        def live_samples(self): return []
        def stream(self, req):
            seen.append(req.request_id)
            return iter([{"type":"token","text":answers[req.messages[0][1]]},
                         {"type":"done","finish_reason":"stop","prompt_tokens":34,"completion_tokens":2}])
    @contextmanager
    def worker(*_,**kw):
        yield {},Existing(),SimpleNamespace(_process=SimpleNamespace(pid=123))
    monkeypatch.setattr(startup.http,"server",worker)
    monkeypatch.setattr(startup,"choose", lambda *_, previous=False,family, **kw:
                        {"profile":"candidate" if not previous and family=="factual-first-18" else "reference"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(live.passive,args,receipt,tmp_path,tmp_path/"inactive")
        end=time.monotonic()+5
        ready=tmp_path/"live-ready.json"
        while not ready.exists() and time.monotonic()<end:
            time.sleep(.01)
        assert ready.is_file() and not seen
        if arrive:
            dest=tmp_path/"external.json"
            subprocess.run([sys.executable,"-m","lab.pk1.restart_live","--ready",str(ready),
                            "--receipt-sha256","pin","--out",str(dest),"--timeout","10"],check=True,timeout=12)
        listener=future.result(timeout=10)
    assert listener["self_generated_requests"]==0
    if arrive:
        witness=live.read(dest)
        assert live.verify(listener,witness)["completions"][0]["graded_correct"]==18
        witness["pid"]=witness["listener_pid"]
        with pytest.raises(ValueError):
            live.verify(listener,witness)
    else:
        assert listener["arm"]=="idle" and listener["requests_served"]==0 and not listener["selections"]


def test_new_product_attempt_requires_old_failure_and_successful_original_tokenizer(tmp_path):
    import hashlib

    from lab.rp.restart_campaign import tokenizer_retry

    code = "print('authenticated tokenizer')"
    source = tmp_path / "ironmule/ggml_engine.py"
    source.parent.mkdir()
    source.write_text(f"GATE_IDS = {code!r}\n")
    digest = release.checksum(source)
    prior = tmp_path / "product/gate-reference.json"
    prior.parent.mkdir()
    release.save(prior, {"passes": False, "source_sha256": digest,
                         "diagnostic": {"stage": "tokenize", "code": "missing_dependency"}})
    verified = {"order": {"order_id": "order"}, "qualification": {"binding": {"runtime": {"model_sha256":"model"}}}}
    preflight = tmp_path / "preflight.json"
    good = {"schema": "ironmule.product-tokenizer-preflight.v1", "order_id": "order", "source_sha256":digest,
            "gguf_sha256": "model", "returncode": 0, "gate_ids_sha256": hashlib.sha256(code.encode()).hexdigest(), "ids_count": 8193,
            "facts_sha256":"a"*64, "diagnosed_dependency":"package", "correction":"bound installation"}
    release.save(preflight, good)
    args = SimpleNamespace(checkout=tmp_path)
    assert tokenizer_retry(tmp_path, args, verified, preflight) == good
    release.save(preflight, {**good, "ids_count":0})
    with pytest.raises(ValueError):
        tokenizer_retry(tmp_path, args, verified, preflight)
    release.save(preflight, good)
    release.save(prior, {"passes":False, "diagnostic":{"stage":"candidate","code":"quality_rejected"}})
    with pytest.raises(ValueError):
        tokenizer_retry(tmp_path, args, verified, preflight)
