from types import SimpleNamespace

from lab.pk1 import restart_release, restart_trial
from training import pilot_restart_profiles as learner
from training.code_author import write_once


def test_grouped_training_does_not_fit_heldout_factual_pairs():
    train = [(32 + i / 10, 15) for i in range(6)]
    _, first, _, a = learner.make_heads(train, [(32, 15)] * 2, 81)
    _, second, _, b = learner.make_heads(train, [(15, 32)] * 2, 81)
    assert all((x == y).all() for h, k in zip(first, second) for x, y in zip(h, k))
    assert a['holdout_correct_profile'] and not b['holdout_correct_profile']
    assert a['holdout_seconds']['fixed_use_cache'] == a['holdout_seconds']['scratch_pilot']


def test_release_head_uses_evidence_and_returns_to_reference_on_rollback(tmp_path, monkeypatch):
    order = {'order_id': 'a' * 64, 'base_commit': '2' * 40}
    q = {'passed': True, 'reference_sha256': '1'*64, 'candidate_sha256': '2'*64,
         'evaluator_sha256': '3'*64, 'binding': {'uuid': 'bound gpu'}}
    args = SimpleNamespace(order=tmp_path / 'order.json', trial=tmp_path, report=tmp_path/'fit.json',
                           weights=tmp_path/'head.npz', seed=15)
    monkeypatch.setattr(restart_release, 'approved', lambda unused: q)
    write_once(args.order, order)
    write_once(tmp_path/'result.json', {'passed': True})
    for i in range(restart_trial.PAIRS):
        for tag, value in [('ref', 32), ('cand', 15)]:
            write_once(tmp_path/f'{tag}-{i}.json', {'cold_s': value})
            if i < 2:
                write_once(tmp_path/f'facts-{tag}-{i}.json', {'factual_first': [{'completed_s': value}]})
    report = learner.fit(args)
    sha = report['weights_sha256']
    assert learner.fit(args)['weights_sha256'] == sha
    control = {'qualification': q, 'order': order, 'adapter': 'qwen-restart'}
    def choose(**kwargs):
        params = dict(active=True, cache_ready=True, family='factual-first-18', expected_sha256=sha)
        return learner.select(args.weights, control, **{**params, **kwargs})
    assert choose()['profile'] == 'candidate'
    assert choose(previous=True)['profile'] == choose(active=False)['profile'] == 'reference'
    assert choose(cache_ready=False)['profile'] == choose(family='unobserved')['profile'] == 'reference'
    assert choose(expected_sha256='0'*64)['profile'] == 'reference'
    q['binding'] = {'uuid': 'foreign gpu'}
    assert choose()['profile'] == 'reference'
