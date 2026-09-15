import { mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { HostContext } from '../src/host.js';
import { hostSchema, checked, type Plan } from '../src/contracts.js';
import { McpAdapter } from '../src/adapter.js';
import { cliPath } from '../src/cli.js';
import { Runtime } from '../src/runtime.js';
import { Store } from '../src/store.js';
import { importSnapshot } from '../src/snapshot.js';
import { compile, seedPlan } from '../src/ir.js';
import { writePrivate } from '../src/util.js';
import { filesFor, taskFor, oracle, comparable, type Case } from './fixtures.js';

export const ARMS=['REBUILD','REFERENCE','PARALLEL','DEDUP','MERGE','PREPARATION','C_WORKFLOW','C_AA','D_ADAPTIVE'] as const;
export type Arm=typeof ARMS[number];
export function shuffle<T>(items:readonly T[],seed:number):T[] {
  const out=[...items];let state=seed>>>0;
  const rand=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296;};
  for(let i=out.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[out[i],out[j]]=[out[j]!,out[i]!];}return out;
}
export async function setup(directory:string,repositories:string[]) {
  mkdirSync(directory,{recursive:true,mode:0o700});const state=join(directory,'state');const store=new Store(state);
  const snapshots:Record<string,string>={},resources=[];
  for(const id of repositories) {
    const root=join(directory,id);mkdirSync(root,{mode:0o700});
    for(const [path,text] of Object.entries(filesFor(id))){mkdirSync(dirname(join(root,path)),{recursive:true});writeFileSync(join(root,path),text);}
    snapshots[id]=store.saveSnapshot(importSnapshot(id,root,['.']));resources.push({id,root,allowedPaths:['.'],concurrency:4});
  }
  const config=checked(hostSchema,{version:'ironmole-host.v1',principal:'benchmark',policyVersion:'sealed',enabled:true,
    expiresAt:Date.now()+3600000,allowedTools:['repo.search','repo.read'],repositories:resources,
    objective:{primary:'latency',costLimitUsd:0,expectedUses:100},adaptive:true});
  const hostPath=join(state,'host.json');writePrivate(hostPath,config);
  const adapter=await McpAdapter.connect(cliPath,hostPath,state);const runtime=new Runtime(new HostContext(hostPath),store,adapter);
  return {store,runtime,snapshots,close:async()=>{await adapter.close();store.close();}};
}
export function planFor(arm:Arm,width:1|2|4):Plan|undefined {
  if(arm==='D_ADAPTIVE')return undefined;
  if(arm==='C_WORKFLOW'||arm==='C_AA')return seedPlan({concurrency:width,dedup:true,merge:true,preparation:'indexed'});
  if(arm==='PREPARATION')return seedPlan({preparation:'indexed'});
  if(arm==='PARALLEL')return seedPlan({concurrency:width});
  if(arm==='DEDUP')return seedPlan({dedup:true});
  if(arm==='MERGE')return seedPlan({merge:true});
  return seedPlan();
}
export async function measure(env:Awaited<ReturnType<typeof setup>>,c:Case,arm:Arm,width:1|2|4,stored:Plan|undefined) {
  const task=taskFor(c,env.snapshots[c.repository]!);
  const start=performance.now();
  let plan=stored;
  if(arm==='REBUILD')plan=compile(seedPlan()).plan;
  if(arm==='REFERENCE'&&stored)plan=env.store.plan(env.store.savePlan(stored));
  const result=await env.runtime.run(task,plan?{plan,strategy:arm==='D_ADAPTIVE'?'OPTIMIZED_PLAN':'REFERENCE_PLAN'}:{});
  const serialized=JSON.stringify(result);const wallMs=performance.now()-start;
  oracle(result,task,filesFor(c.repository));
  const trace=env.runtime.lastTrace!;
  return {caseId:c.id,arm,width,wallMs,status:result.status,toolCalls:trace.calls.length,
    routingMs:trace.metrics.routingMs,runtimeOverheadMs:trace.metrics.runtimeOverheadMs,
    preparationMs:trace.steps.filter(s=>s.op==='RETURN').reduce((n,s)=>n+s.durationMs,0),
    outputBytes:Buffer.byteLength(serialized),quality:true,completeness:result.bundle!.completeness,
    deoptimization:result.handoff!==null,modelCalls:null,inputTokens:null,outputTokens:null,costUsd:null,
    runtimeModelCalls:0,finalAgentResponseIncluded:false,comparable:comparable(result)};
}
