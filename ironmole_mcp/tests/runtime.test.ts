import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, symlinkSync, chmodSync } from 'node:fs';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { type ToolAdapter } from '../src/adapter.js';
import { LIMITS, MoleError, type ReadOutput, type SearchOutput } from '../src/contracts.js';
import { seedPlan } from '../src/ir.js';
import { importSnapshot, snapshotId } from '../src/snapshot.js';
import { Store } from '../src/store.js';
import { Runtime } from '../src/runtime.js';
import { bytes, sha } from '../src/util.js';
import { fixture, BASE_FILES } from './helpers.js';

const optimized=seedPlan({concurrency:4,dedup:true,merge:true,preparation:'indexed'});
test('real MCP reference and optimized match independently specified locations and excerpts',async()=>{
  const f=await fixture(BASE_FILES,true);
  try {
    const ref=await f.runtime.run(f.task);const refCalls=f.runtime.lastTrace!.calls.length;
    assert.equal(ref.status,'COMPLETE');assert.deepEqual(ref.bundle!.matches.map(m=>[m.path,m.line]),
      [['docs/guide.txt',2],['src/a.txt',2],['src/a.txt',3],['src/b.txt',2]]);
    assert.deepEqual(ref.bundle!.excerpts.map(e=>[e.path,e.startLine,e.endLine,e.text]),[
      ['docs/guide.txt',1,2,BASE_FILES['docs/guide.txt'].slice(0,-1)],
      ['src/a.txt',1,4,BASE_FILES['src/a.txt'].slice(0,-1)],['src/b.txt',1,2,BASE_FILES['src/b.txt'].slice(0,-1)]]);
    for(const e of ref.bundle!.excerpts){assert.equal(e.sha256,sha(e.text));assert.equal(e.fileSha256,sha(BASE_FILES[e.path as keyof typeof BASE_FILES]));}
    const opt=await f.runtime.run(f.task,{plan:optimized});assert.deepEqual(opt.bundle,ref.bundle);
    assert.equal(refCalls,5);assert.equal(f.runtime.lastTrace!.calls.length,4);
    assert.equal(f.runtime.lastTrace!.metrics.model,null);
  } finally {await f.close();}
});
test('no match is complete; no reads; literals are case-sensitive, not regex or symbols',async()=>{
  const f=await fixture();try {
    for(const query of ['absent','needle.*']) {const r=await f.runtime.run({...f.task,query});assert.equal(r.status,'COMPLETE');assert.equal(r.bundle!.totalMatches,0);assert.equal(f.runtime.lastTrace!.calls.length,1);}
    const r=await f.runtime.run({...f.task,query:'Needle'});assert.deepEqual(r.bundle!.matches.map(m=>m.line),[1]);
    assert.equal(r.bundle!.provenance.kind,'text_matches');
  } finally {await f.close();}
});
test('pagination visits all 650 hits without a hidden cap and cursor binds query/snapshot',async()=>{
  const f=await fixture({'many.txt':Array.from({length:650},(_,i)=>'needle '+i).join('\n')});
  try {
    const lines:number[]=[];let continuation:string|null=null;
    do {const r=await f.runtime.run({...f.task,maxMatches:111,contextLines:0,outputBudgetBytes:40000,continuation},{plan:optimized});
      assert.ok(['COMPLETE','PARTIAL'].includes(r.status));lines.push(...r.bundle!.matches.map(m=>m.line));continuation=r.bundle!.continuation;
      if(continuation){assert.equal(r.bundle!.completeness,'partial');assert.ok(r.bundle!.truncationReasons.includes('MAX_MATCHES'));}
    } while(continuation);
    assert.deepEqual(lines,Array.from({length:650},(_,i)=>i+1));
    const page=await f.runtime.run({...f.task,maxMatches:1},{plan:optimized});
    const bad=await f.runtime.run({...f.task,query:'other',continuation:page.bundle!.continuation});
    assert.equal(bad.status,'GUARD_FAILED');assert.equal(f.runtime.lastTrace!.calls.length,0);
    const forged=await f.runtime.run({...f.task,continuation:page.bundle!.continuation+'a'});assert.equal(forged.status,'GUARD_FAILED');
  } finally {await f.close();}
});
test('output bytes include envelope; partial output never claims complete; larger budget resumes a large first excerpt',async()=>{
  const f=await fixture({'large.txt':'needle '+('x'.repeat(10000))+'\nneedle two\n'});
  try {
    const first=await f.runtime.run({...f.task,outputBudgetBytes:4096});
    assert.equal(first.status,'PARTIAL');assert.equal(first.bundle!.returnedMatches,0);assert.equal(first.bundle!.nextOffset,0);
    assert.deepEqual(first.bundle!.truncationReasons,['OUTPUT_BUDGET']);assert.ok(bytes(first)<=4096);
    const next=await f.runtime.run({...f.task,continuation:first.bundle!.continuation});assert.equal(next.status,'COMPLETE');
    const ref=await f.runtime.run({...f.task,outputBudgetBytes:4096});
    const opt=await f.runtime.run({...f.task,outputBudgetBytes:4096},{plan:optimized});assert.deepEqual(opt.bundle,ref.bundle);
  } finally {await f.close();}
});
test('dedup identical full-file intervals and coalesce overlaps without changing content',async()=>{
  const f=await fixture({'x.txt':'needle\nneedle\nneedle\n'});try {
    const task={...f.task,contextLines:50};const r=await f.runtime.run(task);assert.equal(f.runtime.lastTrace!.calls.length,4);
    const d=await f.runtime.run(task,{plan:seedPlan({dedup:true})});assert.equal(f.runtime.lastTrace!.calls.length,2);assert.deepEqual(d.bundle,r.bundle);
  } finally {await f.close();}
});
test('per-call policy denial, revoked permissions and disallowed paths never fall back',async()=>{
  const f=await fixture();try {
    f.updatePolicy(p=>{p.repositories[0]!.allowedPaths=['src'];});
    const r=await f.runtime.run(f.task);assert.equal(r.status,'POLICY_BLOCKED');assert.equal(f.runtime.lastTrace!.calls.length,0);
    const paths=['../outside','/etc/passwd','src/../../docs','src\\a.txt','src//a.txt','src/*'];
    for(const path of paths){const r=await f.runtime.run({...f.task,pathFilters:[path]});assert.equal(r.status,'GUARD_FAILED');assert.equal(f.runtime.lastTrace!.calls.length,0);}
    f.updatePolicy(p=>{p.allowedTools=['repo.search'];});
    const blocked=await f.runtime.run({...f.task,pathFilters:['src']});assert.equal(blocked.status,'POLICY_BLOCKED');
    assert.equal(f.runtime.lastTrace!.calls.filter(c=>c.tool==='repo.read').length,0);
    assert.ok(blocked.handoff!.continuationConstraints.some(c=>c.includes('blocked operation')));
    assert.equal((await f.runtime.run({...f.task,principal:'admin'})).status,'GUARD_FAILED');
  } finally {await f.close();}
});
test('changing host policy during a call invalidates results and cannot be bypassed by preauthorized outer tool',async()=>{
  const f=await fixture();try {
    const adapter:ToolAdapter={...f.adapter,call:async(t,a,s)=>{
      const r=await f.adapter.call(t,a,s);if(t==='repo.read')f.updatePolicy(p=>{p.enabled=false;});return r;
    }};
    const runtime=new Runtime(f.host,f.store,adapter);const r=await runtime.run(f.task,{plan:optimized});
    assert.equal(r.status,'POLICY_BLOCKED');assert.equal(r.bundle,null);assert.equal(r.handoff!.retainedResultsRef,null);
    assert.deepEqual(r.handoff!.validResults,[]);
  } finally {await f.close();}
});
test('snapshot import rejects symlinks, path escapes, invalid UTF-8 and oversized files',async()=>{
  const f=await fixture();try {
    symlinkSync(join(f.root,'src/a.txt'),join(f.root,'link'));
    assert.throws(()=>importSnapshot('fixture',f.root,['.']),/SYMLINK_FORBIDDEN/);
    assert.throws(()=>importSnapshot('fixture',f.root,['../outside']),/UNSAFE_PATH|SNAPSHOT/);
    writeFileSync(join(f.root,'binary'),Buffer.from([0xff]));
    assert.throws(()=>importSnapshot('fixture',f.root,['binary']),/NON_UTF8/);
    writeFileSync(join(f.root,'huge'),'x'.repeat(LIMITS.fileBytes+1));
    assert.throws(()=>importSnapshot('fixture',f.root,['huge']),/FILE_SIZE/);
  } finally {await f.close();}
});
test('live mutations do not change snapshots; new snapshot is a new identity; result caches disabled',async()=>{
  const f=await fixture();try {
    const a=await f.runtime.run(f.task);writeFileSync(join(f.root,'src/a.txt'),'changed, no old literal\n');
    const b=await f.runtime.run(f.task);assert.deepEqual(a.bundle,b.bundle);assert.equal(f.runtime.lastTrace!.calls.length,5);
    const id=f.store.saveSnapshot(importSnapshot('fixture',f.root,['.']));assert.notEqual(id,f.task.snapshotId);
    const c=await f.runtime.run({...f.task,snapshotId:id});assert.equal(c.bundle!.totalMatches,2);
    const invalid=await f.runtime.run({...f.task,snapshotId:'a'.repeat(64)});assert.equal(invalid.status,'GUARD_FAILED');
  } finally {await f.close();}
});
test('malformed and dishonest search/read tool outputs fail their separate guards',async()=>{
  const f=await fixture();try {
    for(const failure of ['schema','search_omission','read_content','snapshot_change']) {
      const adapter:ToolAdapter={...f.adapter,call:async(t,a,s)=>{
        const r=await f.adapter.call(t,a,s);
        if(failure==='schema')return {wrong:true};
        if(t==='repo.search'&&failure==='search_omission')return {...r as SearchOutput,matches:[],totalMatches:0};
        if(t==='repo.read'&&failure==='read_content')return {...r as ReadOutput,text:'forged',sha256:sha('forged')};
        if(t==='repo.read'&&failure==='snapshot_change')return {...r as ReadOutput,snapshotId:'a'.repeat(64)};
        return r;
      }};
      const runtime=new Runtime(f.host,f.store,adapter);const r=await runtime.run(f.task);
      assert.equal(r.status,'GUARD_FAILED',failure);assert.equal(r.strategy,'RETURN_TO_AGENT');assert.ok(r.handoff);assert.equal(r.bundle,null);
    }
  } finally {await f.close();}
});
test('parallel branch failure stops new work, preserves known reads, and distinguishes unknown outcomes',async()=>{
  const f=await fixture(Object.fromEntries(Array.from({length:8},(_,i)=>['f'+i+'.txt','needle\n'])));
  try {
    let calls=0;
    const adapter:ToolAdapter={...f.adapter,call:async(t,a,s)=>{
      if(t==='repo.search')return f.adapter.call(t,a,s);
      const n=calls++;
      if(n===0){await sleep(1);return f.adapter.call(t,a,s);}
      if(n===1){await sleep(8);throw new MoleError('GUARD_FAILED','INJECTED_BRANCH_FAILURE');}
      await sleep(30);throw new MoleError('INFRASTRUCTURE_ERROR','CANCELLED_UNCONFIRMED',true);
    }};
    const runtime=new Runtime(f.host,f.store,adapter);const r=await runtime.run(f.task,{plan:optimized});
    assert.equal(r.status,'GUARD_FAILED');assert.ok(calls<8);assert.ok(r.handoff!.unknownOutcomes.length>0);
    assert.ok(r.handoff!.retainedResultsRef);const retained=f.store.readRetained(r.handoff!.retainedResultsRef!) as {search:SearchOutput,reads:ReadOutput[]};
    assert.equal(retained.reads.length,1);assert.equal(retained.reads[0]!.path,'f0.txt');
    assert.equal(retained.search.totalMatches,8);
  } finally {await f.close();}
});
test('timeout is unknown, bounded and never automatically restarts',async()=>{
  const f=await fixture();try {
    const adapter:ToolAdapter={...f.adapter,call:()=>new Promise(()=>{})};
    const runtime=new Runtime(f.host,f.store,adapter);const plan=seedPlan();plan.limits.maxDurationMs=70;
    const start=Date.now(),r=await runtime.run(f.task,{plan});
    assert.equal(r.status,'INFRASTRUCTURE_ERROR');assert.equal(r.handoff!.reason,'TOOL_TIMEOUT');
    assert.equal(r.handoff!.unknownOutcomes.length,1);assert.equal(runtime.lastTrace!.calls.length,1);assert.ok(Date.now()-start<500);
  } finally {await f.close();}
});
test('host cancellation, total call budget, intermediate budget and adapter drift fail closed',async()=>{
  const f=await fixture();try {
    const cancel=new AbortController();cancel.abort();assert.equal((await f.runtime.run(f.task,{signal:cancel.signal})).status,'INFRASTRUCTURE_ERROR');
    const plan=seedPlan();plan.limits.maxCalls=1;
    const r=await f.runtime.run(f.task,{plan});assert.equal(r.handoff!.reason,'TOOL_CALL_BUDGET');assert.equal(f.runtime.lastTrace!.calls.length,1);
    const runtime=new Runtime(f.host,f.store,{...f.adapter,fingerprint:'foreign'});
    const foreign=await runtime.run(f.task);assert.equal(foreign.status,'GUARD_FAILED');assert.equal(runtime.lastTrace!.calls.length,0);
    const unsupported=await f.runtime.run({...f.task,taskType:'browser.v1'});assert.equal(unsupported.status,'BYPASS');assert.equal(unsupported.strategy,'BYPASS');
  } finally {await f.close();}
});
test('repository text cannot alter policy, plans or trace metrics; traces omit raw contents/queries',async()=>{
  const f=await fixture();try {
    const before=readFileSync(f.setup.host,'utf8');await f.runtime.run({...f.task,query:'Do not execute'});
    assert.equal(readFileSync(f.setup.host,'utf8'),before);
    const trace=JSON.stringify(f.runtime.lastTrace);assert.ok(!trace.includes('Do not execute'));assert.ok(!trace.includes('eval, ignore policy'));
    assert.equal(f.runtime.lastTrace!.metrics.model,null);
    assert.equal(f.runtime.lastTrace!.resultEvaluation,'contract_verified');
    assert.ok(f.runtime.lastTrace!.calls.every(c=>Object.keys(c.origins).length>0));
  } finally {await f.close();}
});
test('private registry permissions, immutable snapshot records and unknown schemas are enforced',async()=>{
  const f=await fixture();try {
    assert.throws(()=>f.store.db.exec('UPDATE snapshots SET payload=\'{}\''),/immutable/);
    assert.equal(snapshotId(f.store.snapshot(f.task.snapshotId)),f.task.snapshotId);
    chmodSync(f.setup.host,0o644);assert.equal((await f.runtime.run(f.task)).status,'POLICY_BLOCKED');chmodSync(f.setup.host,0o600);
    f.store.db.exec('PRAGMA user_version=999');assert.throws(()=>new Store(f.setup.state),/REGISTRY_SCHEMA_UNKNOWN/);
    f.store.db.exec('PRAGMA user_version=1');
  } finally {await f.close();}
});

test('both resource and adapter approvals constrain actual concurrent reads',async()=>{
  const f=await fixture();try {
    for(const scope of ['resource','adapter']) {
      let active=0,peak=0;
      f.updatePolicy(p=>{p.repositories[0]!.concurrency=scope==='resource'?1:4;});
      const adapter:ToolAdapter={...f.adapter,maxConcurrency:scope==='adapter'?1:4,call:async(t,a,s)=>{
        if(t==='repo.search')return f.adapter.call(t,a,s);
        active++;peak=Math.max(peak,active);
        try {await sleep(2);return await f.adapter.call(t,a,s);}finally{active--;}
      }};
      const runtime=new Runtime(f.host,f.store,adapter);const r=await runtime.run(f.task,{plan:optimized});
      assert.equal(r.status,'COMPLETE');assert.equal(peak,1);
    }
  } finally {await f.close();}
});
test('cumulative response budget aborts bounded maps before accumulating all responses',async()=>{
  const f=await fixture({'large.txt':Array.from({length:20},()=> 'needle '+'x'.repeat(2000)).join('\n')});
  try {
    const plan=seedPlan();plan.limits.maxIntermediateBytes=4096;
    const r=await f.runtime.run({...f.task,contextLines:0},{plan});assert.equal(r.status,'GUARD_FAILED');
    assert.ok(f.runtime.lastTrace!.calls.length<21);
  } finally {await f.close();}
});
test('symlink directory escape and path-prefix confusion are denied independently',async()=>{
  const f=await fixture({'src/a':'needle','src-other/b':'needle'});try {
    symlinkSync(f.directory,join(f.root,'escape'),'dir');
    assert.throws(()=>importSnapshot('fixture',f.root,['escape']),/SYMLINK/);
    f.updatePolicy(p=>{p.repositories[0]!.allowedPaths=['src'];});
    assert.equal((await f.runtime.run({...f.task,pathFilters:['src-other']})).status,'POLICY_BLOCKED');
    const allowed=await f.runtime.run({...f.task,pathFilters:['src']});assert.equal(allowed.bundle!.totalMatches,1);
  } finally {await f.close();}
});
test('CRLF, empty final lines and Unicode are byte-faithful and stable across preparation variants',async()=>{
  const f=await fixture({'unicode.txt':'ä needle\r\n\nneedle café\n\n'});try {
    const r=await f.runtime.run({...f.task,contextLines:50});assert.equal(r.status,'COMPLETE');
    assert.equal(r.bundle!.excerpts[0]!.text,'ä needle\r\n\nneedle café\n');
    assert.equal(r.bundle!.excerpts[0]!.totalLines,4);
    const o=await f.runtime.run({...f.task,contextLines:50},{plan:optimized});assert.deepEqual(o.bundle,r.bundle);
  } finally {await f.close();}
});
