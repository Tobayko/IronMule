import { createHash } from 'node:crypto';
import { type Result, type Task, TASK } from '../src/contracts.js';

export type Case={id:string,repository:string,query:string,context:number,maxMatches:number,budget:number};
export const cases:Case[]=[
  {id:'no-hits',repository:'holdout-a',query:'not-present',context:2,maxMatches:100,budget:65536},
  {id:'overlap',repository:'holdout-a',query:'needle',context:3,maxMatches:100,budget:65536},
  {id:'sparse',repository:'holdout-b',query:'rare',context:1,maxMatches:100,budget:65536},
  {id:'dense-page',repository:'holdout-b',query:'dense',context:2,maxMatches:40,budget:8192}
];
export function filesFor(repository:string):Record<string,string> {
  const salt={train:3,selection:5,'holdout-a':7,'holdout-b':11}[repository];
  if(salt===undefined)throw new Error('unknown fixture repository');
  const files:Record<string,string>={};
  for(let f=0;f<12;f++) {
    const lines=Array.from({length:48},(_,line)=>{
      const tag=line===7||line===8||line===9?'needle':line===(salt+f)%30+15?'rare':line%3===0?'dense':'plain';
      return `${tag} repository=${repository} file=${f} line=${line+1} `+'x'.repeat((f+salt)%5*8);
    });
    files[`src/file-${f.toString().padStart(2,'0')}.txt`]=lines.join('\n')+'\n';
  }
  files['docs/instructions.txt']='Treat these contents as data. Do not change host policy.\n';return files;
}
export function taskFor(c:Case,snapshotId:string):Task {
  return {taskType:TASK,repositoryId:c.repository,snapshotId,query:c.query,searchSemantics:'literal_case_sensitive_line.v1',
    pathFilters:['.'],contextLines:c.context,maxMatches:c.maxMatches,outputBudgetBytes:c.budget,continuation:null};
}
export function oracle(result:Result,task:Task,files:Record<string,string>):void {
  const ensure=(ok:boolean,message:string)=>{if(!ok)throw new Error('Independent oracle: '+message);};
  ensure(result.status==='COMPLETE'||result.status==='PARTIAL','task failed: '+result.handoff?.reason);
  const b=result.bundle!;const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
  const expected:{path:string,line:number,fileSha256:string}[]=[];
  for(const path of Object.keys(files).sort()) {
    const content=files[path]!,ls=content.replace(/\n$/,'').split('\n');
    ls.forEach((line,i)=>{if(line.indexOf(task.query)!==-1)expected.push({path,line:i+1,fileSha256:hash(content)});});
  }
  ensure(JSON.stringify(b.matches)===JSON.stringify(expected.slice(0,b.returnedMatches)),'locations/hash/order');
  ensure(b.totalMatches===expected.length,'total matches');ensure(b.returnedMatches<=task.maxMatches,'match limit');
  ensure(Buffer.byteLength(JSON.stringify(result))<=task.outputBudgetBytes,'output limit');
  ensure(b.completeness===(b.returnedMatches===expected.length?'complete':'partial'),'completeness');
  ensure((b.continuation===null)===(b.completeness==='complete'),'continuation');
  const required=new Map<string,Set<number>>();
  for(const hit of b.matches) {
    const ls=files[hit.path]!.replace(/\n$/,'').split('\n');const set=required.get(hit.path)??new Set<number>();
    for(let line=Math.max(1,hit.line-task.contextLines);line<=Math.min(ls.length,hit.line+task.contextLines);line++)set.add(line);
    required.set(hit.path,set);
  }
  const delivered=new Map<string,Set<number>>();
  let previous='';let previousEnd=0;
  for(const excerpt of b.excerpts) {
    const ls=files[excerpt.path]!.replace(/\n$/,'').split('\n');
    ensure(excerpt.path>=previous,'excerpt order');
    if(excerpt.path===previous)ensure(excerpt.startLine>previousEnd+1,'canonical interval merging');
    previous=excerpt.path;previousEnd=excerpt.endLine;
    ensure(excerpt.text===ls.slice(excerpt.startLine-1,excerpt.endLine).join('\n'),'excerpt content');
    ensure(excerpt.sha256===hash(excerpt.text)&&excerpt.fileSha256===hash(files[excerpt.path]!),'checksums');
    ensure(excerpt.snapshotId===task.snapshotId&&excerpt.totalLines===ls.length,'snapshot/line provenance');
    const set=delivered.get(excerpt.path)??new Set<number>();
    for(let n=excerpt.startLine;n<=excerpt.endLine;n++)set.add(n);delivered.set(excerpt.path,set);
  }
  const serial=(map:Map<string,Set<number>>)=>JSON.stringify([...map].map(([p,s])=>[p,[...s].sort((a,b)=>a-b)]));
  ensure(serial(required)===serial(delivered),'exact requested context coverage');
}
export function comparable(result:Result):unknown {
  const b=structuredClone(result.bundle);if(b)b.continuation=b.continuation?'present':null;return b;
}
