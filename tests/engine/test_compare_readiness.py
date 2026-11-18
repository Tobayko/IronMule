from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from lab.portable_gpu import compare_readiness as smoke
from ironmule_product import compare, compare_servers


@pytest.mark.parametrize('failure,modes', [('cuda_graph_or_compiler', ['default', 'enforce_eager']),
                                           ('gpu_out_of_memory', ['default'])])
def test_only_a_compile_failure_retries_and_all_other_servers_are_kept(monkeypatch, failure, modes):
    args = SimpleNamespace(engines=('vllm', 'llama_cpp', 'ironmule'), budget_seconds=20)
    prompts = []

    @contextmanager
    def server(engine, args, rep, deadline):
        if engine == 'vllm' and not args.vllm_enforce_eager:
            raise compare_servers.StartupError('/secret/private', 'model_loading', failure)
        yield {'version': 'x', 'placement_verified': True, 'baseline_stock': True, 'autonomous': True}

    def reply(engine, server, prompt, limit, deadline):
        prompts.append((engine, limit))
        return {'text': 'private output', 'tokens': 4, 'prompt_tokens': 9, 'finish_reason': 'length'}

    monkeypatch.setattr(compare_servers, 'open_server', server)
    monkeypatch.setattr(compare_servers, 'refresh_identity', lambda *unused: None)
    monkeypatch.setattr(compare, 'stream', reply)
    result = smoke.run(args, lambda result: None)
    assert [r['mode'] for r in result['attempts'] if r['engine'] == 'vllm'] == modes
    assert result['ready_engines'][-2:] == ['llama_cpp', 'ironmule']
    assert len(prompts) == len(result['ready_engines']) and all(limit == 4 for _, limit in prompts)
    assert result['vllm_mode'] == ('enforce_eager' if len(modes) == 2 else None)
    assert result['status'] == ('ready' if len(modes) == 2 else 'partial')
    assert result['attempts'][0]['startup_stage'] == 'model_loading'
    assert 'private' not in str(result)
