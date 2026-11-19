"""Compare real Qwen3 (32B by default, or 8B) deployments using IronMule GGUF and vLLM AWQ on a single GPU."""
from __future__ import annotations

import argparse
import os
import secrets
import statistics
import sys
import time
from pathlib import Path

from ironmule.factual_gate import MAX_TOKENS, correct
from ironmule_product import compare, compare_servers, compare_tokenizer
from ironmule_product.state import _atomic_write
from lab.portable_gpu import fetch, fetch_awq

SCHEMA = 'ironmule.deployment-comparison.v1'
DEPLOYMENTS = ('ironmule_gguf', 'vllm_awq')


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, help_text in [('gguf', 'official pinned Qwen3 Q4_K_M GGUF of --model'),
                            ('awq-dir', 'verified local Qwen3 AWQ snapshot of --model'),
                            ('tokenizer', 'verified local base tokenizer and config of --model'),
                            ('vllm', 'vLLM executable'), ('output', 'new, separate deployment report path')]:
        parser.add_argument('--' + flag, type=Path, required=True, help=help_text)
    for flag in ('ironmule', 'agent-dir', 'log-dir'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--model', choices=sorted(fetch_awq.CHECKPOINTS), default=fetch_awq.DEFAULT)
    parser.add_argument('--concurrency', default='1,8,32')
    for flag, default in [('seed', secrets.randbits(32)), ('context', 2048), ('requests', 8), ('repeats', 3),
                          ('max-tokens', 32), ('startup-timeout', 1200), ('budget-seconds', 2700)]:
        parser.add_argument('--' + flag, type=int, default=default)
    parser.add_argument('--mixed-trace-seconds', type=int, default=0,
                        help='optional per-deployment budget for a paired held-out trace, after final standard timings')
    parser.add_argument('--gpu-memory-utilization', type=float, default=.88)
    args = parser.parse_args(argv)
    fetch_awq.select(args.model)
    try:
        args.concurrency = [int(w) for w in args.concurrency.split(',')]
    except ValueError:
        parser.error('concurrency must be a comma-separated list of integers')
    if (not 1 <= len(args.concurrency) <= 8 or len(set(args.concurrency)) != len(args.concurrency)
            or any(not 1 <= w <= 64 for w in args.concurrency) or not 1 <= args.repeats <= 9
            or not 1 <= args.requests <= 128 or not 1 <= args.max_tokens <= 2048 or args.context <= args.max_tokens
            or args.budget_seconds < 1 or args.startup_timeout < 1 or not 0 < args.gpu_memory_utilization <= 1
            or not 0 <= args.mixed_trace_seconds <= 120 or not 0 <= args.seed < 2**32):
        parser.error('invalid workload or budget bounds')
    return args


def bind_inputs(args):
    artifacts = compare_servers.prepare_artifacts(args)
    repo, revision, _, size, sha = fetch.MODELS[args.model]
    if args.gguf.stat().st_size != size or args.gguf_sha256 != sha:
        raise compare.Unverified('GGUF differs from the pinned official Q4_K_M release')
    tokenizer_repo, tokenizer_commit, config = compare_tokenizer.TOKENIZERS[args.model]
    if artifacts['tokenizer_files'] != {**compare_tokenizer._QWEN_FILES, 'config.json': config}:
        raise compare.Unverified(f'tokenizer differs from the official {args.model} base release')
    args.awq_manifest = fetch_awq.verify(args.awq_dir)
    if not args.awq_manifest.get('verified') or not args.awq_manifest.get('tokenizer_files_match'):
        raise compare.Unverified('AWQ checkpoint or tokenizer could not be confirmed')
    return {'gguf': {'repo': repo, 'commit': revision, 'bytes': size, 'sha256': sha, 'quantization': 'Q4_K_M'},
            'awq': {k: v for k, v in args.awq_manifest.items() if k != 'file_stamps'},
            'tokenizer': {'repo': tokenizer_repo, 'commit': tokenizer_commit,
            'sha256': args.tokenizer_sha256, 'files': artifacts['tokenizer_files']}}


def check_server(name, server, args, device):
    if not server.get('placement_verified') or not server.get('version') or not str(server.get('device')).startswith('GPU-'):
        raise compare.Unverified('NVIDIA device placement or engine version is unverified')
    if device is not None and device != server['device']:
        raise compare.Unverified('GPU UUID changed between deployments')
    if server.get('tokenizer_sha256') != args.tokenizer_sha256:
        raise compare.Unverified('base tokenizer identity changed between deployments')
    if name == 'ironmule_gguf':
        if server.get('gguf_sha256') != args.gguf_sha256:
            raise compare.Unverified('IronMule did not bind the expected GGUF')
    else:
        argv = server.get('argv') or []
        if ('gguf_sha256' in server or server.get('weight_format') != 'awq'
                or server.get('awq_manifest_sha256') != args.awq_manifest['manifest_sha256']
                or server.get('loaded_quantization') not in ('awq', 'awq_marlin')
                or str(args.awq_dir.resolve()) not in argv or '--tokenizer' not in argv
                or argv[argv.index('--tokenizer') + 1:argv.index('--tokenizer') + 2] != [str(args.tokenizer.resolve())]):
            raise compare.Unverified('vLLM AWQ weights, base tokenizer or effective quantization are unverified')


def proof(engine, server, args, prompts, deadline):
    messages = [compare.conversation(prompt) for prompt in prompts]
    if engine == 'ironmule':
        result = compare_tokenizer.preflight(engine, server, args, messages, deadline)
    else:
        result = compare_tokenizer.rendered_prompt_ids(engine, server, messages, deadline)
    if result.get('verified') is not True or len(result.get('prompt_ids_sha256', [])) != len(prompts):
        raise compare.Unverified('unable to verify all prompt token IDs')
    return result


def measure(args, name, rep, data, report, save, deadline, *, profile=None, label=None):
    from lab.portable_gpu.awq_server import open_awq_server

    engine = 'ironmule' if name == 'ironmule_gguf' else 'vllm'
    launched = (compare_servers.open_server(engine, args, rep, deadline) if engine == 'ironmule'
                else open_awq_server(args, rep, deadline))
    prefix = f'qwen3-{args.seed}-{rep}'
    with launched as server:
        check_server(name, server, args, report.get('device'))
        report['device'] = server['device']
        arm = {'deployment': label or name, 'repeat': rep, 'identity': server, 'prompt_ids_verified': False}
        report['deployments'].append(arm)
        save('warmup')
        warm = compare.stream(engine, server, compare.WARMUP, min(8, args.max_tokens), deadline, request_id=prefix + '-warm')
        compare_servers.refresh_identity(engine, server, deadline)
        if engine == 'ironmule' and server.get('autonomous') is not True:
            raise compare.Unverified('IronMule agent did not handle the warmup')
        texts, lengths, cells = [compare.WARMUP], [warm['prompt_tokens']], []
        for width, (prompts, cases) in data.items():
            report['current']['concurrency'] = width
            ids = f'{prefix}-{width}'
            save('benchmark')
            if profile is not None:
                checkpoint = profile.before(server, deadline)
            rows, elapsed = compare.batch(engine, server, prompts, width, args.max_tokens, deadline,
                                          request_id_prefix=ids + '-perf')
            start = len(texts)
            result = {'deployment': label or name, 'repeat': rep, 'concurrency': width, 'request_id_prefix': ids,
                      **compare.performance_metrics(rows, elapsed), 'quality': None}
            if profile is not None:
                result['profile'] = profile.after(server, deadline, checkpoint, rows)
            report['measurements'].append(result)
            texts.extend(prompts)
            lengths.extend(result['performance_prompt_tokens'])
            save('quality')
            questions = [case.prompt for case in cases]
            rows, _ = compare.batch(engine, server, questions, width, MAX_TOKENS, deadline, request_id_prefix=ids + '-quality')
            answers = {case.key: correct(case, row['text']) for case, row in zip(cases, rows, strict=True)}
            result['quality'] = {'answers': answers, 'passed': sum(answers.values()), 'total': len(answers)}
            texts.extend(questions)
            lengths.extend(row['prompt_tokens'] for row in rows)
            cells.append((result, start, len(texts)))
        save('verifying_prompts')
        checked = proof(engine, server, args, texts, deadline)
        if lengths != checked['prompt_tokens']:
            raise compare.Unverified('served prompt lengths disagree with verified token IDs')
        for row, start, end in cells:
            row['prompt_ids_sha256'] = checked['prompt_ids_sha256'][start:end]
        seconds = getattr(args, 'mixed_trace_seconds', 0)
        if seconds and rep == args.repeats - 1 and label is None and profile is None:
            from lab.portable_gpu import mixed_traffic

            save('heldout_mixed')
            try:
                arm['heldout_mixed'] = mixed_traffic.measure(engine, server, args.seed ^ 0xE02D9B17,
                    min(deadline, time.monotonic() + seconds), gguf=args.gguf)
            except Exception as exc:  # optional trace failure preserves the standard paired measurements
                arm['heldout_mixed_error'] = type(exc).__name__
    # An adapter can still reject modified weights during process cleanup. Qualify this evidence afterwards.
    arm.update(prompt_ids_verified=True, prompt_token_digest=checked['digest'])
    save('verified')


def summarize(report):
    rows = {(r['deployment'], r['repeat'], r['concurrency']): r for r in report['measurements']}
    verified = {(a['deployment'], a['repeat']) for a in report['deployments'] if a['prompt_ids_verified']}
    pairs = []
    for rep in range(report['protocol']['repeats']):
        if any((name, rep) not in verified for name in DEPLOYMENTS):
            continue
        for width in report['protocol']['concurrency']:
            a, b = (rows.get((name, rep, width)) for name in DEPLOYMENTS)
            if not a or not b or not a.get('prompt_ids_sha256') or a['prompt_ids_sha256'] != b.get('prompt_ids_sha256'):
                continue
            if a['request_id_prefix'] != b['request_id_prefix'] or not a.get('quality') or not b.get('quality'):
                continue
            ours, theirs = a['quality']['answers'], b['quality']['answers']
            if not ours or ours.keys() != theirs.keys():
                continue
            pairs.append({'repeat': rep, 'concurrency': width, 'output_tokens_and_stops_match': a['generation'] == b['generation'],
                          'tokens_per_second_ratio': a['tokens_per_second'] / b['tokens_per_second'],
                          'requests_per_second_ratio': a['requests_per_second'] / b['requests_per_second'],
                          'ttft_p95_ratio': a['ttft_p95_ms'] / b['ttft_p95_ms'],
                          'quality_correct': {'ironmule_gguf': sum(ours.values()), 'vllm_awq': sum(theirs.values())},
                          'quality_total': len(ours), 'lost_answers': [k for k in ours if theirs[k] and not ours[k]],
                          'gained_answers': [k for k in ours if ours[k] and not theirs[k]]})
    return pairs


def run(args, report, save, deadline):
    report['artifacts'] = bind_inputs(args)
    for rep in range(args.repeats):
        data = compare.workloads(args, rep)
        order = DEPLOYMENTS if rep % 2 == 0 else reversed(DEPLOYMENTS)
        for name in order:
            report['current'] = {'deployment': name, 'repeat': rep, 'concurrency': None}
            save('starting')
            compare.remaining(deadline)
            measure(args, name, rep, data, report, save, deadline)
    report['pairs'] = summarize(report)
    if len(report['pairs']) != args.repeats * len(args.concurrency):
        raise compare.Unverified('some deployment pairs have mismatched requests, prompt IDs or quality cases')
    if args.mixed_trace_seconds:
        from lab.portable_gpu import mixed_traffic

        final = {arm['deployment']: arm for arm in report['deployments'] if arm['prompt_ids_verified']
                 and arm['repeat'] == args.repeats - 1}
        traces = [final[name].get('heldout_mixed') for name in DEPLOYMENTS]
        if all(traces):
            try:
                report['heldout_mixed_pair'] = mixed_traffic.pair(*traces)
            except compare.Unverified as exc:
                report['heldout_mixed_pair'] = {'status': 'incomplete', 'reason': str(exc)}
        else:
            report['heldout_mixed_pair'] = {'status': 'incomplete', 'missing': [
                name for name, trace in zip(DEPLOYMENTS, traces) if not trace]}


def main(argv=None):
    args = arguments(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Preserve older reports and prevent overlapping writers, including the equal-GGUF benchmark monitor.
    try:
        descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(f'Deployment report already exists: {args.output}', file=sys.stderr)
        return 1
    os.close(descriptor)
    started = time.monotonic()
    report = {'schema': SCHEMA, 'status': 'running', 'started_at': compare.utc_now(),
              'protocol': {'seed': args.seed, 'repeats': args.repeats, 'concurrency': args.concurrency,
                           'requests': args.requests, 'max_tokens': args.max_tokens, 'context': args.context,
                           'budget_seconds': args.budget_seconds, 'ratio_order': list(DEPLOYMENTS),
                           'claim_scope': 'deployment configurations with different weight files'},
              'current': {}, 'deployments': [], 'measurements': [], 'pairs': [], 'device': None}

    if args.mixed_trace_seconds:
        report['protocol']['heldout_mixed_budget_per_arm_seconds'] = args.mixed_trace_seconds

    def save(stage):
        report.update(stage=stage, updated_at=compare.utc_now(), elapsed_seconds=round(time.monotonic()-started, 3))
        _atomic_write(args.output, report)

    save('hashing')
    try:
        run(args, report, save, started + args.budget_seconds)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - finish the evidence file and stop all remaining arms
        report['status'] = 'incomplete'
        report['pairs'] = summarize(report)
        report['failure'] = {'stage': report['stage'], 'code': type(exc).__name__}
        if isinstance(exc, compare_servers.StartupError):
            report['failure'].update(exc.report_fields())
        elif isinstance(exc, compare.Unverified):
            report['failure']['reason'] = str(exc)[:300]
        save('finished')
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    report['status'] = 'measured'
    report['by_concurrency'] = {}
    for width in args.concurrency:
        pairs = [p for p in report['pairs'] if p['concurrency'] == width]
        report['by_concurrency'][str(width)] = {
            'paired_repeats': len(pairs), 'equal_output_work_repeats': sum(p['output_tokens_and_stops_match'] for p in pairs),
            'median_pair_ratios': {m: statistics.median(p[m] for p in pairs) for m in
                                  ('tokens_per_second_ratio', 'requests_per_second_ratio', 'ttft_p95_ratio')},
            'quality_correct': {name: sum(p['quality_correct'][name] for p in pairs) for name in DEPLOYMENTS},
            'quality_total': sum(p['quality_total'] for p in pairs)}
    save('finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
