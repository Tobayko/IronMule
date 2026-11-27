import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ironmule_product import compare
from ironmule_product.errors import BackendUnavailable
from ironmule_product.http_server import create_server
from lab.pk1.restart_transport import Trace
from training.code_trial import read


def test_original_client_loses_post_header_error_and_observer_retains_it(tmp_path):
    class Failing:
        def ready(self):
            return True
        def reference_for(self, req):
            return None
        def note(self, *args, **kwargs):
            pass
        def cancel(self, *_):
            pass
        def stream(self, req):
            yield {"type":"token", "text":"first"}
            raise BackendUnavailable("local diagnostic fixture")

    server = create_server(Failing(), host="127.0.0.1", port=0)
    t = threading.Thread(target=server.serve_forever,daemon=True)
    t.start()
    trace = Trace(tmp_path)
    address = {"base_url":f"http://127.0.0.1:{server.server_address[1]}", "model_id":"local"}
    try:
        with trace.observe(), ThreadPoolExecutor(max_workers=1) as one:
            with pytest.raises(RuntimeError,match="returned a stream error"):
                one.submit(compare.batch,"ironmule",address,["real localhost post"]*18,18,32,time.monotonic()+10).result()
            limit=time.monotonic()+3
            while time.monotonic()<limit and not any("sse_error" in r for r in trace.requests):
                time.sleep(.01)
    finally:
        server.shutdown()
        server.server_close()
        t.join()
    rows = read(tmp_path/"requests.json")["requests"]
    failed = next(row for row in rows if row.get("sse_error"))
    assert failed["status"] == 200 and failed["text_prefix"] == "first"
    assert failed["sse_error"]["code"] == "backend_unavailable"
    assert failed["phase"] == "cold" and failed["chunks"] == 1
