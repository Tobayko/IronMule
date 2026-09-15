import { createHmac } from 'node:crypto';
import { type Compiled } from './ir.js';
import { type Status, type Strategy, type Task, type ToolName } from './contracts.js';
import { canonical, digest } from './util.js';

export interface AgentMetrics {
  source:'trusted_agent_adapter'; provider:string; model:string; modelCalls:number;
  inputTokens:number|null; outputTokens:number|null; costUsd:number|null;
  priceBasis:string|null; totalTaskMs:number|null; finalResponseIncluded:boolean;
}
export interface CallTrace {
  id:string; node:string; tool:ToolName; argsDigest:string;
  arguments:Record<string,unknown>; origins:Record<string,string>;
  dependencies:string[]; startedMs:number; durationMs:number|null; resultBytes:number|null;
  outcome:'running'|'completed'|'failed'|'unknown'; error:string|null;
}
export interface Trace {
  version:'trace.v1'; runId:string; taskType:string; contractVersion:'1.0.0';
  planId:string|null; planOrigin:'manual'|'mined'|null; adapterFingerprint:string;
  repositoryDigest:string; snapshotId:string; inputShape:Record<string,number|string>;
  taskBindingDigest:string; graph:Compiled|null; dependencyEvidence:'runtime_verified'|'observation_only';
  calls:CallTrace[]; steps:{id:string,op:string,outcome:'completed'|'failed',durationMs:number}[];
  guards:{node:string,ok:boolean,reason:string|null}[];
  status:Status; strategy:Strategy; routeReason:string; resultEvaluation:'contract_verified'|'not_completed';
  metrics:{runtimeMs:number,routingMs:number,toolElapsedSumMs:number,toolCriticalMs:number,
    runtimeOverheadMs:number,outputBytes:number,model:AgentMetrics|null};
}
export function privateDigest(value:unknown,key:Buffer):string {
  return createHmac('sha256',key).update(canonical(value)).digest('hex');
}
export function inputShape(task:Task):Trace['inputShape'] {
  return {queryBytes:Buffer.byteLength(task.query),pathFilterCount:task.pathFilters.length,contextLines:task.contextLines,
    maxMatches:task.maxMatches,outputBudgetBytes:task.outputBudgetBytes,continuation:task.continuation?'present':'absent'};
}
export function graphSignature(compiled:Compiled):string {
  const names=new Map(compiled.order.map((n,i)=>[n.id,'n'+i]));
  const value=JSON.parse(JSON.stringify(compiled.plan)) as Record<string,unknown>;
  value['origin']='normalized';
  function normalize(v:unknown):unknown {
    if(Array.isArray(v))return v.map(normalize);
    if(v&&typeof v==='object')return Object.fromEntries(Object.entries(v).map(([k,c])=>
      [k,k==='id'||k==='node'?names.get(c as string)??c:k==='dependsOn'?(c as string[]).map(n=>names.get(n)):normalize(c)]));
    return v;
  }
  return digest(normalize(value));
}
