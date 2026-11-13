"""The 50 facts of lab/row1/facts.py through llama-server's chat API (greedy, 48 tokens, /no_think, 8 at a time), with
the system prompt in ROW1_SYSTEM if set. Usage: facts_server.py SERVER GGUF [extra flags]"""
import json, os, sys, threading
sys.path.insert(0, "/home/user/IronMule/lab/row1")
src = open("/home/user/IronMule/lab/row1/facts.py").read()
Q = eval(src[src.index("Q = [") + 4:src.index("]\nge.") + 1])
s = open("/home/user/IronMule/lab/row1/serve.py").read()
import http.client, socket, subprocess, time
ns = {**globals(), "server": sys.argv[1], "gguf": sys.argv[2], "extra": sys.argv[3:]}
exec(s[s.index("def free_port"):s.index("def block")], ns)
proc, port = ns["start"](8, 8 * 2048)
ok = 0
try:
    for k in range(0, len(Q), 8):
        grp, out = Q[k:k + 8], [None] * 8
        th = [threading.Thread(target=lambda i=i: out.__setitem__(i, ns["ask"](port, grp[i][0] + " /no_think", 48, 0)))
              for i in range(len(grp))]
        [t.start() for t in th]; [t.join() for t in th]
        for (q, exp), o in zip(grp, out):
            a = o["text"].replace("<think>", "").replace("</think>", "").strip().lower()
            ok += any(e in a for e in exp)
finally:
    ns["stop"](proc)
print(json.dumps({"facts": ok, "system": bool(os.environ.get("ROW1_SYSTEM")), "flags": sys.argv[3:]}))
