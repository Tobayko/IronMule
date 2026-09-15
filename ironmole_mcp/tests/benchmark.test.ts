import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setup, measure, planFor, shuffle, ARMS } from '../bench/harness.js';
import { cases, comparable } from '../bench/fixtures.js';
import { canonical } from '../src/util.js';
import { seedPlan } from '../src/ir.js';
import { validateAgentMetrics } from '../src/agent-adapter.js';

test('benchmark oracle verifies all frozen cases across widths before measurement',async()=>{
  const directory=mkdtempSync(join(realpathSync(tmpdir()),'ironmole-bench-test-'));
  const env=await setup(directory,['holdout-a','holdout-b']);
  try {
    for(const c of cases) {
      const baseline=await measure(env,c,'REFERENCE',1,seedPlan());
      for(const width of [1,2,4] as const) {
        const candidate=await measure(env,c,'C_WORKFLOW',width,planFor('C_WORKFLOW',width));
        assert.equal(canonical(candidate.comparable),canonical(baseline.comparable));
      }
    }
    assert.deepEqual(shuffle(ARMS,42),shuffle(ARMS,42));assert.notDeepEqual(shuffle(ARMS,42),shuffle(ARMS,43));
    assert.equal(comparable({bundle:null} as never),null);
  } finally {await env.close();rmSync(directory,{recursive:true,force:true});}
});
test('agent usage stays unknown without observation and never substitutes inter-tool delays for reasoning',()=>{
  const metrics={source:'trusted_agent_adapter',provider:'test',model:'fixture',modelCalls:1,inputTokens:null,outputTokens:null,
    costUsd:null,priceBasis:null,totalTaskMs:null,finalResponseIncluded:false};
  assert.equal(validateAgentMetrics(metrics).costUsd,null);
  assert.throws(()=>validateAgentMetrics({...metrics,costUsd:0.1}),/price basis/);
  assert.throws(()=>validateAgentMetrics({...metrics,totalTaskMs:1}),/final response/);
  assert.throws(()=>validateAgentMetrics({...metrics,inputTokens:-1}));
  assert.throws(()=>validateAgentMetrics({...metrics,totalTaskMs:NaN}));
});
