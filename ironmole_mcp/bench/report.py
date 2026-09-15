"""Offline local-MCP report; use Friday Evidence rather than copying statistics."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from friday_evidence.statistics import paired_ratio, summarise


def report(raw):
    if raw['status'] != 'completed':
        raise ValueError('Only complete, correctness-checked runs can be summarized')
    by_arm = {}
    by_key = {}
    cold = {}
    first = {}
    for block in raw['blocks']:
        arm = block['arm']
        by_arm.setdefault(arm, []).extend(block['samples'])
        cold.setdefault(arm, []).extend(s['wallMs'] for s in block['cold'])
        first.setdefault(arm, []).extend(s['firstCaseSinceProcessStartMs'] for s in block['cold']
                                          if s['firstCaseSinceProcessStartMs'] is not None)
        for s in block['samples']:
            if not s['quality'] or s['deoptimization']:
                raise ValueError('Invalid run: failed quality or deoptimization')
            key = (block['block'], s['repeat'], s['caseId'])
            if (arm, key) in by_key:
                raise ValueError('Duplicate measurement key')
            by_key[arm, key] = s
    summaries = {}
    ratios = {}
    for arm, samples in by_arm.items():
        summaries[arm] = {
            'repeated_ms': summarise([s['wallMs'] for s in samples]),
            'cold_request_ms': summarise(cold[arm]),
            'process_start_to_first_result_ms': summarise(first[arm]),
            'tool_calls': summarise([s['toolCalls'] for s in samples]),
            'routing_ms': summarise([s['routingMs'] for s in samples]),
            'runtime_overhead_ms': summarise([s['runtimeOverheadMs'] for s in samples]),
            'preparation_ms': summarise([s['preparationMs'] for s in samples]),
            'quality_passes': len(samples), 'deoptimizations': 0,
            'model_calls': None, 'input_tokens': None, 'output_tokens': None, 'cost_usd': None,
            'runtime_model_calls': 0, 'final_response_included': False,
            'cases': {c['id']: summarise([s['wallMs'] for s in samples if s['caseId'] == c['id']])
                      for c in raw['cases']},
        }
        block_ratios = []
        for block in range(6):
            rs = []
            for repeat in range(4):
                for case in raw['cases']:
                    key = (block, repeat, case['id'])
                    rs.append(by_key[arm, key]['wallMs'] / by_key['C_WORKFLOW', key]['wallMs'])
            block_ratios.append(statistics.median(rs))
        ratios[arm] = paired_ratio(block_ratios, [1.0] * 6, resamples=10000, seed=20260912)
    aa = ratios['C_AA']
    control_ok = 0.95 <= aa['median_ratio'] <= 1.05 and aa['ci_low'] <= 1 <= aa['ci_high']
    return {
        'run_id': raw['runId'], 'seal': raw['seal'], 'environment': raw['environment'],
        'status': 'completed', 'scope': 'local runtime only; no final agent response',
        'unavailable': raw['unavailable'], 'selected_width': raw['selection']['width'],
        'learning_ms': raw['selection']['learningMs'], 'validation_selection_ms': raw['selection']['validationMs'],
        'arms': summaries, 'ratios_to_C_clustered_by_process_block': ratios,
        'aa_control_acceptable': control_ok, 'thresholds': raw['criteria'],
        'candidate_activated': False,
        'adaptive_added_value': 'not demonstrated; complete agent latency and cost comparisons unavailable',
        'end_to_end_break_even_uses': None,
        'uncertainty': '95% paired bootstrap of six independent block medians; 96 repeated samples per arm are clustered, not 96 independent processes.',
    }


def main():
    raw = json.loads(Path(sys.argv[1]).read_text())
    summary = report(raw)
    with Path(sys.argv[2]).open('x') as out:
        json.dump(summary, out, indent=2)
        out.write('\n')
    for arm, values in summary['arms'].items():
        s = values['repeated_ms']
        print(f"{arm}: n={s['n']} median={s['median']:.3f} ms p95={s['p95']:.3f} ms")
    print('Adaptive added value: not demonstrated. A/B model arms not executed.')


if __name__ == '__main__':
    main()
