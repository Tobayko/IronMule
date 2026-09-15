import { mkdirSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { cases } from './fixtures.js';
import { setup, shuffle, measure, planFor, ARMS, type Arm } from './harness.js';
import { digest } from '../src/util.js';
import { sourceSeal } from './seal.js';

const [directory,armArg,widthArg,blockArg,sealArg]=process.argv.slice(2);
if(!directory||!ARMS.includes(armArg as Arm)||!widthArg||!blockArg||!sealArg)throw new Error('invalid worker arguments');
if(sourceSeal()!==sealArg)throw new Error('source changed before worker');
const width=Number(widthArg) as 1|2|4,block=Number(blockArg),arm=armArg as Arm;
process.umask(0o077);mkdirSync(directory,{mode:0o700,recursive:true});
const env=await setup(directory,['holdout-a','holdout-b']);
const stored=planFor(arm,width);if(stored)env.store.savePlan(stored);
const samples:unknown[]=[],cold:unknown[]=[];
try {
  let first=true;
  for(const c of shuffle(cases,7000+block)) {
    const m=await measure(env,c,arm,width,stored);cold.push({...m,firstCaseSinceProcessStartMs:first?performance.now():null});first=false;
  }
  for(let warmup=0;warmup<2;warmup++)for(const c of cases)await measure(env,c,arm,width,stored);
  for(let repeat=0;repeat<4;repeat++)for(const c of shuffle(cases,9000+block*100+repeat)) {
    const m=await measure(env,c,arm,width,stored);samples.push({...m,repeat,block});
  }
  if(sourceSeal()!==sealArg)throw new Error('source changed during worker');
  process.stdout.write(JSON.stringify({version:'local-worker.v1',arm,width,block,seal:sealArg,
    snapshots:env.snapshots,fixtureIdentity:digest(env.snapshots),cold,samples})+'\n');
} finally {await env.close();}
