import { cpus, platform, arch } from 'node:os';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { checked, fail, TASK, type Objective, type Plan, type Strategy } from './contracts.js';
import { compile, optimize, seedPlan } from './ir.js';
import { Store } from './store.js';
import { type Trace, graphSignature } from './trace.js';
import { digest } from './util.js';
import { adapterFingerprint } from './adapter.js';

export type Lifecycle='observed'|'candidate'|'validated'|'active'|'suspended';
export function implementationFingerprint():string {
  const dir=dirname(fileURLToPath(import.meta.url));
  return digest(readdirSync(dir).filter(p=>p.endsWith('.js')).sort().map(p=>[p,readFileSync(join(dir,p),'utf8')]));
}
export function environmentFingerprint():string {
  return digest({node:process.version,platform:platform(),arch:arch(),cpu:cpus()[0]?.model??'unknown',adapterFingerprint,
    implementation:implementationFingerprint()});
}
const evaluationSchema=z.strictObject({ version:z.literal('evaluation.v1'), planId:z.string(), environment:z.string(),
  adapterFingerprint:z.string(), objective:z.enum(['latency','cost']),
  measurementScope:z.enum(['runtime_only','task_including_final_response']),
  trainingRepositories:z.array(z.string()),selectionRepositories:z.array(z.string()),testRepositories:z.array(z.string()),
  pairs:z.int().min(1), independentExpectedResults:z.boolean(), correct:z.boolean(), permissionsPreserved:z.boolean(),
  aaAcceptable:z.boolean(), preRegistered:z.boolean(), medianRatio:z.number().positive(),ciHigh:z.number().positive(),
  p95Ratio:z.number().positive(), baselineMs:z.number().positive(),candidateMs:z.number().positive(),
  candidateCostUsd:z.number().nonnegative().nullable(),baselineCostUsd:z.number().nonnegative().nullable(),
  learningMs:z.number().nonnegative(),validationMs:z.number().nonnegative(),learningCostUsd:z.number().nonnegative().nullable(),
  routingMs:z.number().nonnegative(),guardMs:z.number().nonnegative(),deoptProbability:z.number().min(0).max(1),
  spentBeforeAbortMs:z.number().nonnegative(),handoffMs:z.number().nonnegative().nullable(),
  continuationAgentMs:z.number().nonnegative().nullable(),handoffCostUsd:z.number().nonnegative().nullable(),
  expectedUses:z.int().positive(),workloadClass:z.literal(TASK),createdAt:z.int(),expiresAt:z.int() });
export type Evaluation=z.infer<typeof evaluationSchema>;
export class Registry {
  constructor(readonly store:Store) {}
  state(id:string):Lifecycle|null {
    return (this.store.events('lifecycle').filter(e=>e.entity===id).at(-1)?.payload as {state:Lifecycle}|undefined)?.state??null;
  }
  transition(id:string,next:Lifecycle,reason:string):void {
    const previous=this.state(id);
    const legal:Record<string,Lifecycle[]>={none:['observed'],observed:['candidate'],candidate:['validated'],validated:['active','suspended'],active:['suspended'],suspended:['validated']};
    if(!legal[previous??'none']!.includes(next)) fail('GUARD_FAILED','ILLEGAL_PLAN_LIFECYCLE');
    this.store.append('lifecycle',id,{state:next,reason,at:Date.now()});
  }
  mine():{observations:number,candidates:string[],ignored:number} {
    const groups=new Map<string,Trace[]>();let ignored=0;
    for(const row of this.store.events('trace')) {
      const t=row.payload as Trace;
      if(t.version!=='trace.v1'||t.taskType!==TASK||!t.graph||t.dependencyEvidence!=='runtime_verified'||
        t.resultEvaluation!=='contract_verified'||!['COMPLETE','PARTIAL'].includes(t.status)||
        t.calls.some(c=>c.outcome!=='completed'||Object.keys(c.origins).length===0)) {ignored++;continue;}
      try {
        const compiled=compile(t.graph.plan);
        if(compiled.id!==t.planId||t.adapterFingerprint!==adapterFingerprint) {ignored++;continue;}
        const sig=graphSignature(compiled);groups.set(sig,[...(groups.get(sig)??[]),t]);
      } catch {ignored++;}
    }
    const candidates:string[]=[];
    for(const [signature,group] of groups) {
      if(this.state(signature)===null) this.transition(signature,'observed','verified same-task graph');
      if(group.length<3||new Set(group.map(t=>t.taskBindingDigest)).size<2) continue;
      for(const concurrency of [1,2,4] as const) {
        const plan=seedPlan({origin:'mined',concurrency,dedup:true,merge:true,preparation:'indexed'});
        const id=this.store.savePlan(compile(plan).plan);
        if(this.state(id)===null) {
          this.transition(id,'observed',signature);this.transition(id,'candidate','three verified graphs and two parameter bindings');
          this.store.append('candidate',id,{sourceGraph:signature,trainingRepositories:[...new Set(group.map(t=>t.repositoryDigest))],
            trainingSnapshots:[...new Set(group.map(t=>t.snapshotId))],runs:group.map(t=>t.runId),parameterization:'typed task/result bindings only'});
        }
        candidates.push(id);
      }
    }
    return {observations:[...groups.values()].reduce((a,b)=>a+b.length,0),candidates:[...new Set(candidates)],ignored};
  }
  recordEvaluation(value:unknown):string {
    const evaluation=checked(evaluationSchema,value,'EVALUATION_SCHEMA');
    if(!['candidate','suspended'].includes(this.state(evaluation.planId)??'')) fail('GUARD_FAILED','EVALUATION_REQUIRES_CANDIDATE');
    const id=digest(evaluation);this.store.append('evaluation',id,evaluation);return id;
  }
  qualify(evaluationId:string,objective:Objective):{active:boolean,reason:string} {
    const row=this.store.events('evaluation').find(e=>e.entity===evaluationId);
    if(!row) return {active:false,reason:'missing comparative evidence'};
    const e=checked(evaluationSchema,row.payload);
    const reject=(reason:string)=>({active:false,reason});
    if(e.environment!==environmentFingerprint()||e.adapterFingerprint!==adapterFingerprint||Date.now()>=e.expiresAt)
      return reject('identity mismatch or expired evidence');
    const sources=this.store.events('candidate').find(x=>x.entity===e.planId)?.payload as {trainingRepositories:string[]}|undefined;
    if(!sources||sources.trainingRepositories.some(r=>!e.trainingRepositories.includes(r))) return reject('training provenance missing');
    const partitions=[e.trainingRepositories,e.selectionRepositories,e.testRepositories];
    if(partitions.some(p=>!p.length)||new Set(partitions.flat()).size!==partitions.flat().length) return reject('repository holdout overlap');
    if(e.measurementScope!=='task_including_final_response') return reject('final agent response not measured');
    if(e.pairs<20||!e.correct||!e.independentExpectedResults||!e.permissionsPreserved||!e.aaAcceptable||!e.preRegistered)
      return reject('quality, permission, control or sample gate');
    if(e.objective!==objective.primary||e.ciHigh>0.95||e.p95Ratio>1.05) return reject('no qualifying benefit against strong workflow');
    if(e.handoffMs===null||e.continuationAgentMs===null||e.candidateCostUsd===null||e.baselineCostUsd===null||e.handoffCostUsd===null||e.learningCostUsd===null)
      return reject('unknown costs or continuation measurements');
    const uses=Math.min(objective.expectedUses,e.expectedUses);
    const expectedMs=e.routingMs+e.guardMs+(1-e.deoptProbability)*e.candidateMs+
      e.deoptProbability*(e.spentBeforeAbortMs+e.handoffMs+e.continuationAgentMs)+(e.learningMs+e.validationMs)/uses;
    const expectedCost=(1-e.deoptProbability)*e.candidateCostUsd+
      e.deoptProbability*(e.candidateCostUsd+e.handoffCostUsd+e.baselineCostUsd)+e.learningCostUsd/uses;
    if(objective.primary==='latency'&&(expectedMs>=e.baselineMs||expectedCost>objective.costLimitUsd)) return reject('amortized latency/cost limit');
    if(objective.primary==='cost'&&(expectedCost>=e.baselineCostUsd||expectedMs>objective.latencyLimitMs)) return reject('amortized cost/latency limit');
    this.transition(e.planId,'validated',evaluationId);this.transition(e.planId,'active',evaluationId);
    this.store.append('activation',e.planId,{evaluationId,environment:e.environment,objective,expiresAt:e.expiresAt});
    return {active:true,reason:'held-out comparison passed'};
  }
  suspend(id:string,reason:string):void {if(this.state(id)==='active')this.transition(id,'suspended',reason);}
  select(objective:Objective,adaptive:boolean):{strategy:Strategy,plan:Plan,reason:string} {
    if(adaptive) {
      for(const row of this.store.events('activation').reverse()) {
        const a=row.payload as {environment:string,objective:Objective,expiresAt:number};
        if(this.state(row.entity)!=='active')continue;
        if(a.environment!==environmentFingerprint()||Date.now()>=a.expiresAt||digest(a.objective)!==digest(objective)) {
          this.suspend(row.entity,'identity, objective or evidence expired');continue;
        }
        return {strategy:'OPTIMIZED_PLAN',plan:this.store.plan(row.entity),reason:'qualified held-out comparison'};
      }
    }
    const reference=compile(seedPlan());
    this.store.savePlan(reference.plan);
    return {strategy:'REFERENCE_PLAN',plan:this.store.plan(reference.id),reason:'insufficient comparative evidence'};
  }
  static storedWorkflow():Plan {return optimize(seedPlan({preparation:'indexed'}),{concurrency:4,dedup:true,merge:true}).plan;}
}
