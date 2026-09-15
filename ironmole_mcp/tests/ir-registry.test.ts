import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { z } from 'zod';
import { Ajv } from 'ajv';
import { planSchema, taskSchema, resultSchema, TASK } from '../src/contracts.js';
import { adapterFingerprint } from '../src/adapter.js';
import { compile, seedPlan } from '../src/ir.js';
import { Registry, environmentFingerprint, type Evaluation } from '../src/registry.js';
import { graphSignature } from '../src/trace.js';
import { fixture } from './helpers.js';

const clone=()=>structuredClone(seedPlan());
test('MoleIR JSON examples compile and published JSON schemas match executable contracts',()=>{
  const ajv=new Ajv({strict:false});
  for(const [name,schema] of [['task',taskSchema],['moleir',planSchema],['result',resultSchema]] as const) {
    const json=JSON.parse(readFileSync('schemas/'+name+'.schema.json','utf8')) as object;
    assert.deepEqual(json,z.toJSONSchema(schema,{target:'draft-7'}));ajv.compile(json);
  }
  const schema=ajv.compile(JSON.parse(readFileSync('schemas/moleir.schema.json','utf8')));
  for(const name of ['reference','optimized']) {
    const plan=JSON.parse(readFileSync('examples/'+name+'.moleir.json','utf8'));
    assert.ok(schema(plan));assert.equal(compile(plan).order.length,10);
  }
});
test('compiler rejects cycles, missing bindings, effects, schema changes and graph mutations',()=>{
  const malformed:unknown[]=[];
  let p=clone();p.nodes[0]!.dependsOn=['return'];malformed.push(p);
  p=clone();p.nodes[0]!.dependsOn=['missing'];malformed.push(p);
  p=clone();p.nodes.push(p.nodes[0]!);malformed.push(p);
  p=clone();p.nodes.splice(2,1);malformed.push(p);
  p=clone();p.limits.maxNodes=2;malformed.push(p);
  p=clone();const call=p.nodes[1]!;if(call.op==='CALL')call.args['query']={source:'constant',value:'changed'};malformed.push(p);
  p=clone();const projection=p.nodes[3]!;if(projection.op==='PROJECT')projection.input={source:'node',node:'authorize'};malformed.push(p);
  malformed.push({...clone(),adapterVersion:'9.9.9'});
  malformed.push({...clone(),irVersion:'moleir.v2'});
  p=clone();malformed.push({...p,nodes:[...p.nodes,{id:'evil',op:'CALL',tool:'shell',args:{code:'eval(...)'},dependsOn:[]}]});
  p=clone();malformed.push({...p,nodes:p.nodes.map(n=>n.op==='BOUNDED_MAP'?{...n,concurrency:100}:n)});
  for(const bad of malformed)assert.throws(()=>compile(bad));
});
test('provenance normalization permits node renaming, not just similar call strings',()=>{
  const p=seedPlan();const c=compile(p);const json=JSON.stringify(p).replaceAll('authorize','permission_check');
  assert.equal(graphSignature(c),graphSignature(compile(JSON.parse(json))));
  assert.notEqual(graphSignature(c),graphSignature(compile(seedPlan({concurrency:2}))));
});
test('miner requires repeated dependency-rich same-task traces and varied proven bindings',async()=>{
  const f=await fixture();try {
    await f.runtime.run(f.task);await f.runtime.run(f.task);assert.equal(f.runtime.registry.mine().candidates.length,0);
    await f.runtime.run({...f.task,query:'Needle'});
    const mined=f.runtime.registry.mine();assert.equal(mined.candidates.length,3);
    assert.ok(mined.candidates.every(id=>f.runtime.registry.state(id)==='candidate'));
    assert.ok(mined.candidates.every(id=>f.store.plan(id).origin==='mined'));
    f.store.append('trace','passive',{...f.runtime.lastTrace,dependencyEvidence:'observation_only'});
    assert.equal(f.runtime.registry.mine().ignored,1);
    assert.equal(f.runtime.registry.mine().candidates.length,3);
    assert.equal((await f.runtime.run(f.task)).strategy,'REFERENCE_PLAN');
  } finally {await f.close();}
});
async function candidateFixture() {
  const f=await fixture();
  for(const query of ['needle','Needle','header'])await f.runtime.run({...f.task,query});
  const registry=new Registry(f.store),id=registry.mine().candidates[0]!;
  const repo=f.runtime.lastTrace!.repositoryDigest;
  // Synthetic gate inputs, ONLY for control-flow tests; never reported as measurements.
  const evaluation:Evaluation={version:'evaluation.v1',planId:id,environment:environmentFingerprint(),adapterFingerprint,
    objective:'latency',measurementScope:'task_including_final_response',trainingRepositories:[repo],selectionRepositories:['selection'],testRepositories:['holdout'],
    pairs:20,independentExpectedResults:true,correct:true,permissionsPreserved:true,aaAcceptable:true,preRegistered:true,
    medianRatio:0.5,ciHigh:0.6,p95Ratio:0.8,baselineMs:200,candidateMs:100,candidateCostUsd:0,baselineCostUsd:0,
    learningMs:10,validationMs:20,learningCostUsd:0,routingMs:1,guardMs:1,deoptProbability:0.1,
    spentBeforeAbortMs:10,handoffMs:1,continuationAgentMs:200,handoffCostUsd:0,expectedUses:100,workloadClass:TASK,
    createdAt:Date.now(),expiresAt:Date.now()+60_000};
  return {...f,registry,id,evaluation};
}
test('activation requires end-to-end evidence, strong comparator gain, known costs, holdouts and amortization',async()=>{
  const f=await candidateFixture();try {
    const objective=f.host.config().objective;
    for(const change of [
      {measurementScope:'runtime_only'}, {correct:false}, {permissionsPreserved:false}, {candidateCostUsd:null},
      {handoffMs:null}, {continuationAgentMs:null}, {ciHigh:0.99}, {aaAcceptable:false}, {preRegistered:false},
      {testRepositories:f.evaluation.trainingRepositories}, {learningMs:1e9}, {candidateCostUsd:1}, {environment:'foreign'},
      {expiresAt:0}, {pairs:5}
    ]) {
      const eid=f.registry.recordEvaluation({...f.evaluation,...change});
      assert.equal(f.registry.qualify(eid,objective).active,false,JSON.stringify(change));assert.equal(f.registry.state(f.id),'candidate');
    }
    assert.throws(()=>f.registry.transition(f.id,'active','skip-validation'),/ILLEGAL/);
    const eid=f.registry.recordEvaluation(f.evaluation);assert.equal(f.registry.qualify(eid,objective).active,true);
    assert.equal(f.registry.select(objective,true).strategy,'OPTIMIZED_PLAN');
    assert.equal(f.registry.select(objective,false).strategy,'REFERENCE_PLAN');
    f.registry.suspend(f.id,'test drift');assert.equal(f.registry.select(objective,true).strategy,'REFERENCE_PLAN');
    assert.equal(f.registry.state(f.id),'suspended');
    assert.throws(()=>f.registry.transition(f.id,'active','passive speed'),/ILLEGAL/);
    assert.equal(f.registry.qualify(eid,objective).active,true);
    const changed=structuredClone(objective);changed.expectedUses=1;
    assert.equal(f.registry.select(changed,true).strategy,'REFERENCE_PLAN');assert.equal(f.registry.state(f.id),'suspended');
  } finally {await f.close();}
});
test('cost objective keeps money and time separate and rejects unknown unit costs',async()=>{
  const f=await candidateFixture();try {
    const e={...f.evaluation,objective:'cost',baselineCostUsd:0.1,candidateCostUsd:0.01};
    const id=f.registry.recordEvaluation(e);
    assert.equal(f.registry.qualify(id,{primary:'cost',latencyLimitMs:1,expectedUses:100}).active,false);
    assert.equal(f.registry.qualify(id,{primary:'cost',latencyLimitMs:250,expectedUses:100}).active,true);
  } finally {await f.close();}
});
