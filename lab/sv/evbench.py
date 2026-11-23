# Local: the service's grouped streaming path with a fake worker that emits 32 requests x 32 tokens, one step every
# 31 ms; client threads read each session's events. Delivery latency per event and the tail after the last frame.
import statistics as st, sys, threading, time
sys.path.insert(0, '/home/user/IronMule/tests/engine')
from test_product_group_scheduler import _service, _Transport, _Session, _request
from ironmule_product.serving_policy import ServingPolicy
N, STEPS, STEP_S = int(sys.argv[1]) if len(sys.argv) > 1 else 32, 32, 0.031
sent = {}
class Fake(_Transport):
    streams_batches = True
    batch_capacity = 64
    def complete_batch(self, requests, *, cancels, timeout, on_event=None, joins=None):
        if joins is not None:
            joins()
        for k in range(STEPS):
            time.sleep(STEP_S)
            for r in requests:
                sent[(r.request_id, k)] = time.perf_counter()
                on_event(r.request_id, {"type": "token", "request_id": r.request_id, "text": "ab", "token_id": 7,
                                        "prompt_tokens": 5, "completion_tokens": k + 1})
                if joins is not None:
                    joins()
        for r in requests:
            sent[(r.request_id, 'done')] = time.perf_counter()
            on_event(r.request_id, {"type": "done", "request_id": r.request_id, "finish_reason": "length",
                                    "prompt_tokens": 5, "completion_tokens": STEPS})
        self.last = time.perf_counter()
        return [{"request_id": r.request_id, "events": [], "metadata": {}} for r in requests]
backend = Fake()
service = _service(backend)
service.policy = ServingPolicy(join=True)
sessions = []
for i in range(N):
    req = _request(f"r{i}")
    req = type(req)(req.model, req.messages, max_tokens=STEPS, request_id=f"r{i}")
    sessions.append(_Session(req, None))
got = {}
def client(s):
    k = 0
    while True:
        ev = s.events.get()
        t = time.perf_counter()
        if isinstance(ev, dict) and ev.get("type") == "token":
            got[(s.request.request_id, k)] = t; k += 1
        else:
            got[(s.request.request_id, 'done')] = t
            break
threads = [threading.Thread(target=client, args=(s,), daemon=True) for s in sessions]
for t in threads: t.start()
t0 = time.perf_counter()
service._perform_batch(sessions)
for t in threads: t.join(5)
lat = [1e3 * (got[k] - sent[k]) for k in sent if k in got]
tail = 1e3 * (max(got[(s.request.request_id, 'done')] for s in sessions) - backend.last)
print(f"N {N}: events {len(lat)}, delivery ms median {st.median(lat):.3f} p95 {sorted(lat)[int(.95*len(lat))]:.3f} max {max(lat):.2f}; "
      f"tail after last frame {tail:.1f} ms; wall {time.perf_counter() - t0:.3f} s (ideal {STEPS * STEP_S:.3f})")
