"""Exercise the gate transport using real local subprocesses and a deterministic device fixture."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lab.pk1 import restart_gate_broker as broker
from lab.rp import restricted_worker


def test_original_child_program_and_status_cross_a_separate_os_process(tmp_path, monkeypatch):
    roots = str(Path(broker.__file__).resolve().parents[2])
    model = tmp_path / "fixed.gguf"
    env = {"PYTHONPATH": roots}
    layout = SimpleNamespace(env=env, args=SimpleNamespace(gguf=model,deadline=time.monotonic()+15,worker_uid=60000,
                             gate_driver_record=tmp_path/"registry.json"), command=[sys.executable], profile=tmp_path/"config",
                             out=tmp_path,readonly=[], gate_scripts={"GATE_IDS":broker.fingerprint("original-tokenizer")})
    @contextmanager
    def fresh(argv, **kw):
        assert kw["env"]["IRONMULE_CUDA_GRAPHS_ROWS"]=="1"
        code = "import sys; print('local-subprocess-output'); print("+repr(broker.MARKER+json.dumps(
                {"stage":"GATE_IDS","provider":"cuda","gpu_layers":999,"device":"test-fixture"}))+",file=sys.stderr)"
        with subprocess.Popen([sys.executable,"-c",code],stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
            yield child
    monkeypatch.setattr(restricted_worker,"open_worker",fresh)
    driver = """
import os,sys,subprocess
from types import SimpleNamespace
from lab.pk1 import restart_gate_broker as b
ge = SimpleNamespace(GATE_IDS='original-tokenizer', TUNE_CHILD='tune', GATE_CANDIDATE='evaluate')
b.install_driver(ge,sys.stdout)
env = {'PYTHONPATH':os.environ['PYTHONPATH']+os.pathsep+os.environ['PYTHONPATH'],'HF_HUB_OFFLINE':'1','IRONMULE_CUDA_GRAPHS_ROWS':'1'}
r = subprocess.run([sys.executable,'-c',ge.GATE_IDS,sys.argv[1]],env=env,input=None,capture_output=True,text=True,timeout=3,check=False)
assert r.returncode == 0 and r.stdout.strip() == 'local-subprocess-output'
"""
    with subprocess.Popen([sys.executable,"-c",driver,str(model)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as original:
        receipt = broker.exchange(original,layout)
        assert original.wait(timeout=5)==0
    assert receipt[0]["code_sha256"]==layout.gate_scripts["GATE_IDS"] and receipt[0]["child_status"]==0
    forged={"stage":"GATE_IDS","code":layout.gate_scripts["GATE_IDS"],"gguf":str(model),"input":None,
            "env":{"PYTHONPATH":"/unreviewed"}}
    with pytest.raises(ValueError):
        broker.checked_request(layout,forged)
