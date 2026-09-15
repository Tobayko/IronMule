import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { execFile, execFileSync } from 'node:child_process';
import { promisify } from 'node:util';
import { performance } from 'node:perf_hooks';
import { arch, platform, cpus } from 'node:os';
import { cases, comparable, oracle, filesFor, taskFor } from './fixtures.js';
import { setup, measure, planFor, shuffle, ARMS } from './harness.js';
import { sourceSeal, sealManifest } from './seal.js';
import { seedPlan } from '../src/ir.js';
import { canonical } from '../src/util.js';
const runId=process.argv[2]??'MOLE1-LOCAL-2-'+new Date().toISOString().replace(/[:.]/g,'-');
if(!/^[a-zA-Z0-9_-]+$/.test(runId))throw new Error('use a path-free run ID');
const directory=resolve('bench/runs',runId);
if(existsSync(directory))throw new Error('run already exists; never overwrite evidence');
process.umask(0o077);mkdirSync(directory,{recursive:true,mode:0o700});
const seal=sourceSeal();const sealed={runId,seal,inputs:sealManifest(),createdAt:new Date().toISOString(),
  environment:{node:process.version,platform:platform(),arch:arch(),cpu:cpus()[0]?.model??'unknown'},
  criteria:{ratioUpper:0.95,p95Ratio:1.05,expectedUses:100},
  unavailable:{A_DIRECT_AGENT:'No configured model agent adapter; not executed.',B_PTC:'No actual PTC adapter; not executed.'}};
writeFileSync(join(directory,'seal.json'),JSON.stringify(sealed,null,2)+'\n',{flag:'wx'});
const exec=promisify(execFile);const blocks:unknown[]=[];
try {
  const learningStart=performance.now();
  const train=await setup(join(directory,'training'),['train']);
  let mining:unknown;
  try {
    for(const query of ['needle','rare','dense'])await measure(train,{id:query,repository:'train',query,context:2,maxMatches:100,budget:65536},'REFERENCE',1,seedPlan());
    mining=train.runtime.registry.mine();
  } finally {await train.close();}
  const learningMs=performance.now()-learningStart;
  const validationStart=performance.now();
  const selection=await setup(join(directory,'selection'),['selection']);
  const selections:{width:1|2|4,wallMs:number}[]=[];
  try {
    const c={id:'select-overlap',repository:'selection',query:'needle',context:3,maxMatches:100,budget:65536};
    for(let repeat=0;repeat<6;repeat++)for(const width of shuffle([1,2,4] as const,1100+repeat)) {
      const m=await measure(selection,c,'C_WORKFLOW',width,planFor('C_WORKFLOW',width));selections.push({width,wallMs:m.wallMs});
    }
    // Separate preparation correctness cases; these cannot qualify a production route.
    const task=taskFor(c,selection.snapshots['selection']!);const ref=await selection.runtime.run(task,{plan:seedPlan()});
    for(const width of [1,2,4] as const) {
      const result=await selection.runtime.run(task,{plan:seedPlan({concurrency:width,dedup:true,merge:true})});
      oracle(result,task,filesFor('selection'));
      if(canonical(comparable(ref))!==canonical(comparable(result)))throw new Error('preparation changed reference result');
    }
  } finally {await selection.close();}
  const median=(xs:number[])=>{const s=xs.slice().sort((a,b)=>a-b);return (s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
  const widths=([1,2,4] as const).map(width=>({width,median:median(selections.filter(s=>s.width===width).map(s=>s.wallMs))})).sort((a,b)=>a.median-b.median||a.width-b.width);
  const width=widths[0]!.width;const validationMs=performance.now()-validationStart;
  const selectionRecord={width,raw:selections,summary:widths,learningMs,validationMs,mining,activation:false,reason:'runtime-only evidence lacks final response/costs'};
  writeFileSync(join(directory,'selection.json'),JSON.stringify(selectionRecord,null,2)+'\n',{flag:'wx'});
  // Do not inspect final repositories before the selector has been frozen above.
  const baseOrder=shuffle(ARMS,20260912);
  for(let block=0;block<6;block++) {
    const order=[...baseOrder.slice(block%ARMS.length),...baseOrder.slice(0,block%ARMS.length)];
    for(const arm of order) {
      const childDir=join(directory,`block-${block}-${arm}`);
      const started=performance.now();
      const child=await exec(process.execPath,['dist/bench/worker.js',childDir,arm,String(width),String(block),seal],
        {maxBuffer:16_000_000,timeout:60_000,env:{...process.env,NODE_NO_WARNINGS:'1'}});
      const raw=JSON.parse(child.stdout) as {seal:string,samples:{caseId:string,repeat:number,comparable:unknown}[]};
      if(raw.seal!==seal)throw new Error('worker identity mismatch');
      writeFileSync(join(directory,`block-${block}-${arm}.json`),JSON.stringify({...raw,childWallMs:performance.now()-started},null,2)+'\n',{flag:'wx'});
      blocks.push(raw);process.stderr.write(`block ${block+1}/6 ${arm} completed\n`);
    }
  }
  // Compare independently checked arms again against the exact reference in each block/case/repetition.
  const rawBlocks=blocks as {arm:string,block:number,samples:{caseId:string,repeat:number,comparable:unknown}[]}[];
  for(const block of rawBlocks)for(const sample of block.samples) {
    const ref=rawBlocks.find(r=>r.arm==='REFERENCE'&&r.block===block.block)!.samples.find(s=>s.caseId===sample.caseId&&s.repeat===sample.repeat)!;
    if(canonical(ref.comparable)!==canonical(sample.comparable))throw new Error('cross-arm result mismatch');
  }
  if(sourceSeal()!==seal)throw new Error('source changed during benchmark');
  const raw={...sealed,status:'completed',selection:selectionRecord,cases,blocks};
  writeFileSync(join(directory,'raw.json'),JSON.stringify(raw)+'\n',{flag:'wx'});
  execFileSync('python3',['bench/report.py',join(directory,'raw.json'),join(directory,'summary.json')],{stdio:'inherit'});
  process.stdout.write(JSON.stringify({runId,status:'completed',summary:join(directory,'summary.json'),seal})+'\n');
} catch(e) {
  writeFileSync(join(directory,'aborted.json'),JSON.stringify({runId,status:'aborted',seal,completedBlocks:blocks,
    reason:e instanceof Error?e.message:'unknown',sourceStillSealed:sourceSeal()===seal})+'\n',{flag:'wx'});
  process.stderr.write('Benchmark aborted; evidence retained.\n');process.exitCode=1;
}
// Make accidental use of a partially written raw artifact impossible for downstream callers.
if(existsSync(join(directory,'raw.json')))JSON.parse(readFileSync(join(directory,'raw.json'),'utf8'));
