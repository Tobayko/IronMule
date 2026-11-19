import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ironmule.factual_gate import cases
from ironmule_product import compare, compare_servers
from lab.portable_gpu import awq_server
from lab.portable_gpu import production_compare as production


def test_cross_format_deployments_share_requests_and_keep_separate_identities(tmp_path, monkeypatch):
    calls = []
    ids = {}
    altered = set()
    model, tok, awq, out = (tmp_path / n for n in ('m.gguf', 'tok', 'awq', 'result.json'))
    argv = ['--gguf',str(model),'--awq-dir',str(awq),'--tokenizer',str(tok),'--vllm','/bin/vllm',
            '--output',str(out),'--seed','7','--repeats','3','--concurrency','1,2']

    def inputs(args):
        args.gguf_sha256, args.tokenizer_sha256 = 'a'*64, 'b'*64
        args.awq_manifest = {'verified': True, 'manifest_sha256': 'c'*64}
        return {'gguf': {'sha256': args.gguf_sha256}, 'awq': args.awq_manifest}

    @contextmanager
    def server(engine, args, rep, deadline):
        calls.append((engine, rep))
        common = {'device': 'GPU-test', 'version': '0.11.0', 'placement_verified': True,
                  'tokenizer_sha256': 'b'*64, 'autonomous': engine == 'ironmule'}
        if engine == 'ironmule':
            common.update(gguf_sha256='a'*64, argv=['ironmule','--gguf',str(model),'--no-thinking'])
        else:
            common.update(awq_manifest_sha256='c'*64, weight_format='awq', loaded_quantization='awq_marlin',
                          argv=['vllm','serve',str(awq),'--tokenizer',str(tok)])
        yield common
        if (engine,rep) in altered:
            raise ValueError('file changed')

    def response(query):
        return {'prompt_tokens': len(query), 'text': 'Mars', 'tokens': 4, 'ttft_ms': 2, 'finish_reason': 'stop'}

    def batch(engine, info, prompts, width, limit, deadline, *, request_id_prefix):
        ids.setdefault(engine, []).extend((f'{request_id_prefix}-{i}', q) for i,q in enumerate(prompts))
        return [response(q) for q in prompts], 1

    monkeypatch.setattr(production, 'bind_inputs', inputs)
    monkeypatch.setattr(compare_servers, 'open_server', server)
    monkeypatch.setattr(awq_server, 'open_awq_server', lambda args, rep, deadline: server('vllm',args,rep,deadline))
    monkeypatch.setattr(compare_servers, 'refresh_identity', lambda *args: None)
    monkeypatch.setattr(compare, 'workloads', lambda args, rep: {w: (['abc']*w, cases(rep)[:1]) for w in args.concurrency})
    monkeypatch.setattr(compare, 'stream', lambda engine, info, q, *a, **kw: response(q))
    monkeypatch.setattr(compare, 'batch', batch)
    monkeypatch.setattr(production, 'proof', lambda engine, server, args, texts, deadline: {
        'prompt_tokens': list(map(len, texts)), 'digest': 'd'*64, 'prompt_ids_sha256': ['d'*64]*len(texts)})
    assert production.main(argv) == 0
    result = json.loads(out.read_text())
    assert result['schema'] == production.SCHEMA and result['status'] == 'measured'
    assert 'verdict' not in result and 'learning' not in result and 'runs' not in result
    assert [n for n,_ in calls] == ['ironmule','vllm','vllm','ironmule','ironmule','vllm']
    assert ids['ironmule'] == ids['vllm'] and len({i for i,_ in ids['ironmule']}) == len(ids['ironmule'])
    assert len(result['pairs']) == 6 and result['by_concurrency']['1']['paired_repeats'] == 3
    awq_identity = result['deployments'][1]['identity']
    assert 'gguf_sha256' not in awq_identity and awq_identity['loaded_quantization'] == 'awq_marlin'
    assert production.main(argv) == 1  # no overwriting of evidence
    from lab.portable_gpu import mixed_traffic

    heldout = []
    def mixed(engine, server, seed, deadline, *, gguf):
        heldout.append((engine, seed, gguf))
        return {'seed': seed, 'fixture': engine}
    monkeypatch.setattr(mixed_traffic, 'measure', mixed)
    monkeypatch.setattr(mixed_traffic, 'pair', lambda left, right: {'status': 'measured'} if
                        (left['fixture'], right['fixture']) == ('ironmule', 'vllm') else None)
    opted = argv.copy()
    opted[opted.index(str(out))] = str(tmp_path/'mixed.json')
    assert production.main(opted + ['--mixed-trace-seconds', '20']) == 0
    extra = json.loads((tmp_path/'mixed.json').read_text())
    assert [engine for engine, _, _ in heldout] == ['ironmule', 'vllm']
    assert heldout[0][1:] == heldout[1][1:]
    assert extra['heldout_mixed_pair']['status'] == 'measured' and len(extra['pairs']) == 6
    altered.add(('vllm',0))
    argv[argv.index(str(out))] = str(tmp_path/'changed.json')
    assert production.main(argv) == 1
    partial = json.loads((tmp_path/'changed.json').read_text())
    assert partial['status'] == 'incomplete' and partial['measurements'] and not partial['pairs']


def test_confounded_artifacts_never_enter_equal_gguf_check(tmp_path):
    args = SimpleNamespace(tokenizer_sha256='b'*64, awq_dir=tmp_path, tokenizer=tmp_path,
                           awq_manifest={'manifest_sha256':'c'*64})
    server = {'device':'GPU-1', 'placement_verified':True, 'version':'0.11.0', 'tokenizer_sha256':'b'*64,
              'gguf_sha256':'a'*64, 'weight_format':'awq', 'awq_manifest_sha256':'c'*64, 'loaded_quantization':'awq'}
    with pytest.raises(compare.Unverified, match='AWQ'):
        production.check_server('vllm_awq', server, args, 'GPU-1')


def test_failed_prefill_stream_preserves_other_rounds_and_response_privacy(tmp_path, monkeypatch):
    from lab.portable_gpu import prefill_http as trial

    gguf = tmp_path / 'qwen.gguf'
    gguf.write_bytes(b'x')
    args = SimpleNamespace(gguf=gguf, gguf_sha256='checked', output=tmp_path/'result.json',
                           state_dir=tmp_path/'state', arms=['stock', 'shared'], repeats=2,
                           concurrency=[1, 8], seed=9, context=2048, requests=8, max_tokens=32, budget_seconds=100)
    files = {**trial.compare_tokenizer._QWEN_FILES, 'config.json': trial.compare_tokenizer.TOKENIZERS['qwen3-32b'][2]}
    monkeypatch.setattr(trial.compare_servers, 'prepare_artifacts', lambda _: {'tokenizer_files': files})
    monkeypatch.setitem(trial.fetch.MODELS, 'qwen3-32b', ('repo', 'rev', 'file', 1, 'checked'))
    monkeypatch.setattr(trial.compare, 'workloads', lambda *_: {})
    for key in ('IRONMULE_GGML_STREAMS', 'IRONMULE_GGML_PREFILL_BATCH'):
        monkeypatch.setenv(key, '0')
    visits = []

    def measure(_, __, rep, data, report, save, deadline, *, profile, label):
        visits.append((label, rep))
        if (label, rep) == ('shared', 0):
            raise RuntimeError('private response content')
    monkeypatch.setattr(trial.production_compare, 'measure', measure)
    assert trial.run(args) == 1
    result = json.loads(args.output.read_text())
    assert result['status'] == 'partial' and len(result['errors']) == 1
    assert visits == [('stock', 0), ('shared', 0), ('shared', 1), ('stock', 1)]
    assert 'private response content' not in args.output.read_text()
