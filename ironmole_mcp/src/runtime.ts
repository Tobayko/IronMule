import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { type ToolAdapter, adapterFingerprint } from './adapter.js';
import { buildBundle, indexReads, mergeRanges, rangesFor, verifyRead } from './bundle.js';
import { LIMITS, MoleError, TASK, checked, fail, readInputSchema, readOutputSchema, resultSchema,
  searchInputSchema, searchOutputSchema, taskSchema, type Expr, type Handoff, type Plan, type Range,
  type ReadOutput, type Result, type SearchOutput, type Strategy, type Task, type ToolName } from './contracts.js';
import { HostContext } from './host.js';
import { compile, type Compiled } from './ir.js';
import { validateAgentMetrics } from './agent-adapter.js';
import { Registry } from './registry.js';
import { fileAt, type Snapshot } from './snapshot.js';
import { Store } from './store.js';
import { type AgentMetrics, type CallTrace, type Trace, inputShape, privateDigest } from './trace.js';
import { bytes, canonical, decodeCursor, within } from './util.js';

export interface RunOptions {
  // Host-only controls. The MCP task schema does not expose these fields.
  plan?:Plan; strategy?:Strategy; agentMetrics?:AgentMetrics; signal?:AbortSignal;
}
export class Runtime {
  readonly registry:Registry;
  lastTrace:Trace|null=null;
  private busy=false;
  constructor(readonly host:HostContext,readonly store:Store,readonly adapter:ToolAdapter) {this.registry=new Registry(store);}
  async run(input:unknown,options:RunOptions={}):Promise<Result> {
    if(this.busy) fail('GUARD_FAILED','RUNTIME_BUSY');
    this.busy=true;
    try {return await this.execute(input,options);} finally {this.busy=false;}
  }
  private async execute(input:unknown,options:RunOptions):Promise<Result> {
    const start=performance.now(),runId=randomUUID(),key=this.store.key();
    const trace:Trace={version:'trace.v1',runId,taskType:TASK,contractVersion:'1.0.0',planId:null,planOrigin:null,
      adapterFingerprint:this.adapter.fingerprint,repositoryDigest:'',snapshotId:'',inputShape:{},taskBindingDigest:'',graph:null,
      dependencyEvidence:'observation_only',calls:[],steps:[],guards:[],status:'GUARD_FAILED',strategy:'REFERENCE_PLAN',
      routeReason:'not routed',resultEvaluation:'not_completed',
      metrics:{runtimeMs:0,routingMs:0,toolElapsedSumMs:0,toolCriticalMs:0,runtimeOverheadMs:0,outputBytes:0,model:options.agentMetrics?validateAgentMetrics(options.agentMetrics):null}};
    let task:Task|null=null,compiled:Compiled|null=null,snapshot:Snapshot|null=null,policyHash:string|undefined;
    let result:Result={version:'ironmole-result.v1',runId,status:'GUARD_FAILED',strategy:'REFERENCE_PLAN',planId:null,bundle:null,handoff:null};
    const values=new Map<string,unknown>();const reads:ReadOutput[]=[];const invalidResults:string[]=[];
    let receivedBytes=0;
    let stopping:MoleError|null=null;const controller=new AbortController();
    const stop=(error:MoleError)=>{if(stopping===null)stopping=error;controller.abort();};
    const onCancel=()=>stop(new MoleError('INFRASTRUCTURE_ERROR','HOST_CANCELLED',true));
    options.signal?.addEventListener('abort',onCancel,{once:true});
    if(options.signal?.aborted)onCancel();
    const elapsed=()=>performance.now()-start;
    const checkGate=()=>{
      if(stopping)throw stopping;
      if(elapsed()>(compiled?.plan.limits.maxDurationMs??LIMITS.taskMs)) {
        const error=new MoleError('GUARD_FAILED','TASK_TIME_BUDGET');stop(error);throw error;
      }
      if(task) this.host.task(task,policyHash);
      if(this.adapter.fingerprint!==adapterFingerprint)fail('GUARD_FAILED','ADAPTER_CONTRACT_CHANGED');
    };
    const value=(expr:Expr):unknown=>{
      if(expr.source==='task')return task?.[expr.field];
      if(expr.source==='constant')return expr.value;
      if(!values.has(expr.node))fail('GUARD_FAILED','MISSING_NODE_RESULT');return values.get(expr.node);
    };
    const call=async(tool:ToolName,args:unknown,node:string,dependencies:string[],origins:Record<string,string>):Promise<unknown>=>{
      checkGate();
      if(!task||!snapshot||!compiled)fail('GUARD_FAILED','UNBOUND_CALL');
      if(trace.calls.length>=compiled.plan.limits.maxCalls)fail('GUARD_FAILED','TOOL_CALL_BUDGET');
      if(bytes(args)>LIMITS.inputBytes)fail('GUARD_FAILED','TOOL_INPUT_BUDGET');
      const a=tool==='repo.search'?checked(searchInputSchema,args):checked(readInputSchema,args);
      this.host.authorize(tool,a.repositoryId,'path' in a?[a.path]:a.pathFilters,policyHash);
      if(a.repositoryId!==task.repositoryId||a.snapshotId!==task.snapshotId)fail('GUARD_FAILED','INNER_RESOURCE_BINDING');
      const safeArgs=Object.fromEntries(Object.entries(a).map(([k,v])=>[k,
        ['query','path','pathFilters','repositoryId'].includes(k)?{hmac:privateDigest(v,key)}:v]));
      const event:CallTrace={id:'call-'+trace.calls.length,node,tool,argsDigest:privateDigest(a,key),arguments:safeArgs,
        origins,dependencies,startedMs:elapsed(),durationMs:null,resultBytes:null,outcome:'running',error:null};
      trace.calls.push(event);
      let timer:ReturnType<typeof setTimeout>|undefined;
      const timeout=Math.max(1,Math.min(LIMITS.toolMs,compiled.plan.limits.maxDurationMs-elapsed()));
      try {
        const raw=await Promise.race([this.adapter.call(tool,a,controller.signal),new Promise<never>((_,reject)=>{
          timer=setTimeout(()=>reject(new MoleError('INFRASTRUCTURE_ERROR','TOOL_TIMEOUT',true)),timeout);
        })]);
        event.durationMs=elapsed()-event.startedMs;
        event.resultBytes=bytes(raw);
        if(event.resultBytes>LIMITS.toolOutputBytes)fail('GUARD_FAILED','TOOL_OUTPUT_BUDGET');
        receivedBytes+=event.resultBytes;
        if(receivedBytes>compiled.plan.limits.maxIntermediateBytes)fail('GUARD_FAILED','CUMULATIVE_TOOL_OUTPUT_BUDGET');
        const parsed=tool==='repo.search'?checked(searchOutputSchema,raw):checked(readOutputSchema,raw);
        this.host.authorize(tool,a.repositoryId,'path' in a?[a.path]:a.pathFilters,policyHash);
        checkGate();
        if(tool==='repo.read') {
          const r=checked(readInputSchema,a);
          verifyRead(checked(readOutputSchema,parsed),r,task.snapshotId,fileAt(snapshot,r.path).text);
        }
        event.outcome='completed';return parsed;
      } catch(e) {
        const error=e instanceof MoleError?e:new MoleError('INFRASTRUCTURE_ERROR','MCP_TRANSPORT_FAILURE',true);
        event.durationMs??=elapsed()-event.startedMs;
        event.outcome=error.unknownOutcome?'unknown':'failed';event.error=error.code;
        if(event.resultBytes!==null)invalidResults.push(event.id);
        stop(error);throw error;
      } finally {if(timer)clearTimeout(timer);}
    };
    try {
      if(input&&typeof input==='object'&&'taskType' in input&&input.taskType!==TASK) {
        result.status='BYPASS';result.strategy='BYPASS';trace.routeReason='unsupported explicit task class';
        fail('GUARD_FAILED','UNSUPPORTED_TASK_CLASS');
      }
      if(bytes(input)>LIMITS.inputBytes)fail('GUARD_FAILED','TASK_INPUT_BUDGET');
      task=checked(taskSchema,input,'TASK_INPUT_SCHEMA');
      trace.repositoryDigest=privateDigest(task.repositoryId,key);trace.snapshotId=task.snapshotId;
      trace.inputShape=inputShape(task);trace.taskBindingDigest=privateDigest({...task,continuation:null},key);
      policyHash=this.host.task(task);snapshot=this.store.snapshot(task.snapshotId);
      if(snapshot.repositoryId!==task.repositoryId)fail('GUARD_FAILED','SNAPSHOT_REPOSITORY_MISMATCH');
      const routingStart=performance.now();const bound=this.host.bind(task.repositoryId);
      const selection=options.plan?{plan:options.plan,strategy:options.strategy??'REFERENCE_PLAN',reason:'explicit host evaluation plan'}:
        this.registry.select(bound.host.objective,bound.host.adaptive);
      compiled=compile(selection.plan);this.store.savePlan(compiled.plan);
      trace.metrics.routingMs=performance.now()-routingStart;
      trace.strategy=selection.strategy;trace.routeReason=selection.reason;trace.planId=compiled.id;
      trace.planOrigin=compiled.plan.origin;trace.graph=compiled;trace.dependencyEvidence='runtime_verified';
      result.strategy=selection.strategy;result.planId=compiled.id;
      const offset=decodeCursor(task,key);
      for(const node of compiled.order) {
        checkGate();const stepStart=performance.now();
        try {
          let output:unknown;
          switch(node.op) {
            case 'GUARD':
              if(node.check==='search_valid') {
                const search=checked(searchOutputSchema,value(node.input!));
                const expected:SearchOutput['matches']=[];let total=0,scanned=0;
                for(const f of snapshot.files) if(task.pathFilters.some(p=>within(f.path,p))) {
                  scanned++;const ls=fileAt(snapshot,f.path).lines;
                  for(let i=0;i<ls.length;i++)if(ls[i]!.includes(task.query)) {
                    if(total>=offset&&expected.length<task.maxMatches)expected.push({path:f.path,line:i+1,fileSha256:f.sha256});total++;
                  }
                }
                if(search.snapshotId!==task.snapshotId||search.totalMatches!==total||search.offset!==offset||
                  search.nextOffset!==(offset+expected.length<total?offset+expected.length:null)||search.scannedFiles!==scanned||
                  canonical(search.matches)!==canonical(expected)) fail('GUARD_FAILED','SEARCH_CONTRACT_VIOLATION');
              }
              if(node.check==='reads_valid') {
                const rs=value(node.input!) as ReadOutput[];
                if(rs.length!==reads.length)fail('GUARD_FAILED','READS_INCOMPLETE');
              }
              trace.guards.push({node:node.id,ok:true,reason:null});output=null;break;
            case 'CALL': {
              const args=Object.fromEntries(Object.entries(node.args).map(([k,e])=>[k,value(e)]));args['offset']=offset;
              output=await call('repo.search',args,node.id,compiled.dependencies[node.id]!,
                {...Object.fromEntries(Object.entries(node.args).map(([k,e])=>[k,'task.'+(e.source==='task'?e.field:'invalid')])),offset:'validated task continuation'});break;
            }
            case 'PROJECT':
              output=rangesFor((value(node.input) as SearchOutput).matches,value(node.context) as number,p=>fileAt(snapshot!,p).lines.length);break;
            case 'FILTER':output=(value(node.input) as Range[]).filter(r=>r.startLine<=r.endLine);break;
            case 'DEDUP': {
              const rs=value(node.input) as Range[];output=node.enabled?[...new Map(rs.map(r=>[canonical(r),r])).values()]:rs;break;
            }
            case 'MERGE':output=node.mode==='overlapping_intervals'?mergeRanges(value(node.input) as Range[]):value(node.input);break;
            case 'BOUNDED_MAP': {
              const ranges=value(node.input) as Range[];
              if(ranges.length>node.maxItems)fail('GUARD_FAILED','MAP_ITEM_BUDGET');
              const resource=this.host.bind(task.repositoryId).repository;
              // Resource and adapter contracts both independently authorize this width.
              const width=Math.min(node.concurrency,this.adapter.maxConcurrency,resource.concurrency);
              if(width<1)fail('GUARD_FAILED','PARALLELISM_NOT_AUTHORIZED');
              const results=new Array<ReadOutput>(ranges.length);let index=0;
              const worker=async()=>{
                while(!stopping) {
                  const current=index++;if(current>=ranges.length)return;const range=ranges[current]!;
                  try {
                    const response=await call('repo.read',{...range,repositoryId:task!.repositoryId,snapshotId:task!.snapshotId},
                      node.id,compiled!.dependencies[node.id]!,{repositoryId:'task.repositoryId',snapshotId:'task.snapshotId',
                        path:'search.match.path',startLine:'PROJECT(contextLines)+MERGE',endLine:'PROJECT(contextLines)+MERGE',fileSha256:'search.match.fileSha256'});
                    results[current]=checked(readOutputSchema,response);reads.push(results[current]!);
                  } catch(e) {const error=e instanceof MoleError?e:new MoleError('INFRASTRUCTURE_ERROR','MAP_FAILURE');stop(error);return;}
                }
              };
              await Promise.all(Array.from({length:Math.min(width,ranges.length)},()=>worker()));
              if(stopping)throw stopping;output=results;break;
            }
            case 'RETURN': {
              const search=value(node.matches) as SearchOutput;
              const rs=value(node.reads) as ReadOutput[];
              const prepared=node.preparation==='indexed'?indexReads(rs):undefined;
              let count=search.matches.length;
              while(true) {
                result.bundle=buildBundle(task,search,rs,count,key,prepared);
                result.status=result.bundle.completeness==='complete'?'COMPLETE':'PARTIAL';
                if(bytes(result)<=task.outputBudgetBytes)break;
                if(count===0)fail('GUARD_FAILED','OUTPUT_ENVELOPE_BUDGET');count--;
                checkGate();
              }
              output=result.bundle;break;
            }
          }
          if(bytes(output)>compiled.plan.limits.maxIntermediateBytes)fail('GUARD_FAILED','INTERMEDIATE_BUDGET');
          values.set(node.id,output);
          if(bytes([...values.values()])>compiled.plan.limits.maxIntermediateBytes)fail('GUARD_FAILED','TOTAL_INTERMEDIATE_BUDGET');
          trace.steps.push({id:node.id,op:node.op,outcome:'completed',durationMs:performance.now()-stepStart});
        } catch(e) {
          trace.steps.push({id:node.id,op:node.op,outcome:'failed',durationMs:performance.now()-stepStart});
          if(node.op==='GUARD') {
            trace.guards.push({node:node.id,ok:false,reason:e instanceof MoleError?e.code:'GUARD_EXCEPTION'});
            if(node.input?.source==='node') {
              const producer=node.input.node;
              invalidResults.push(...trace.calls.filter(c=>c.node===producer).map(c=>c.id));
            }
          }
          throw e;
        }
      }
      checkGate();trace.resultEvaluation='contract_verified';
    } catch(e) {
      const error=stopping??(e instanceof MoleError?e:new MoleError('INFRASTRUCTURE_ERROR','RUNTIME_FAILURE'));
      stop(error);
      if(result.status!=='BYPASS')result.status=error.status;
      result.bundle=null;
      if(result.status!=='POLICY_BLOCKED'&&result.status!=='BYPASS')result.strategy='RETURN_TO_AGENT';
      const validIds=trace.calls.filter(c=>c.outcome==='completed'&&!invalidResults.includes(c.id)).map(c=>c.id);
      // A revoked policy cannot be worked around by exposing already-read content.
      const searchNode=compiled?.order.find(n=>n.op==='CALL');
      const searchGuard=compiled?.order.find(n=>n.op==='GUARD'&&n.check==='search_valid');
      const searchConfirmed=searchGuard&&trace.guards.some(g=>g.node===searchGuard.id&&g.ok);
      const confirmedSearch=searchConfirmed&&searchNode?values.get(searchNode.id):null;
      const retain=(reads.length||confirmedSearch)&&error.status!=='POLICY_BLOCKED'?
        this.store.retain(runId,{search:confirmedSearch,reads}):null;
      const handoff:Handoff={taskType:TASK,planId:compiled?.id??null,planVersion:compiled?.plan.planVersion??null,
        reason:error.code,violatedAssumption:error.code,
        completedSteps:trace.steps.filter(s=>s.outcome==='completed').map(s=>s.id),
        failedSteps:[...trace.steps.filter(s=>s.outcome==='failed').map(s=>s.id),...trace.calls.filter(c=>c.outcome==='failed').map(c=>c.id)],
        unknownOutcomes:trace.calls.filter(c=>c.outcome==='unknown'||c.outcome==='running').map(c=>c.id),
        validResults:error.status==='POLICY_BLOCKED'?[]:validIds,
        invalidResults:error.status==='POLICY_BLOCKED'?[...new Set([...invalidResults,...validIds])]:invalidResults,
        remainingGoal:'Complete unconfirmed search/read work and produce a contract-complete context bundle.',
        continuationConstraints:['No automatic whole-task restart.','Rebind current policy, adapter and snapshot before any continuation.',
          'Timeout/cancellation does not prove non-execution or rollback.',
          ...(error.status==='POLICY_BLOCKED'?['Do not execute the blocked operation through a fallback.']:[])],
        retainedResultsRef:retain,detailsTruncated:false};
      result.handoff=handoff;
      const budget=task?.outputBudgetBytes??4096;
      while(bytes(result)>budget) {
        const arrays=[handoff.completedSteps,handoff.failedSteps,handoff.unknownOutcomes,handoff.validResults,handoff.invalidResults];
        const longest=arrays.sort((a,b)=>b.length-a.length)[0]!;
        if(!longest.length)break;longest.pop();handoff.detailsTruncated=true;
      }
      if(compiled&&trace.strategy==='OPTIMIZED_PLAN')this.registry.suspend(compiled.id,error.code);
    } finally {options.signal?.removeEventListener('abort',onCancel);}
    result=checked(resultSchema,result,'RESULT_SCHEMA');
    trace.status=result.status;trace.strategy=result.strategy;
    trace.metrics.runtimeMs=elapsed();trace.metrics.outputBytes=bytes(result);
    trace.metrics.toolElapsedSumMs=trace.calls.reduce((a,c)=>a+(c.durationMs??0),0);
    let coveredUntil=0,critical=0;
    for(const c of trace.calls) {
      const end=c.startedMs+(c.durationMs??0);critical+=Math.max(0,end-Math.max(coveredUntil,c.startedMs));coveredUntil=Math.max(coveredUntil,end);
    }
    trace.metrics.toolCriticalMs=critical;trace.metrics.runtimeOverheadMs=Math.max(0,trace.metrics.runtimeMs-critical);
    this.lastTrace=trace;this.store.saveTrace(runId,trace);
    return result;
  }
}
