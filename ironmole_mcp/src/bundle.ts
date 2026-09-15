import { ADAPTER, ADAPTER_VERSION, CONTRACT, checked, bundleSchema, fail,
  type Bundle, type Match, type Range, type ReadOutput, type SearchOutput, type Task } from './contracts.js';
import { comparePath, encodeCursor, lines, sha } from './util.js';

export function mergeRanges(ranges:Range[]):Range[] {
  const ordered=ranges.map(r=>({...r})).sort((a,b)=>comparePath(a.path,b.path)||a.startLine-b.startLine||a.endLine-b.endLine);
  const out:Range[]=[];
  for(const r of ordered) {
    const prev=out.at(-1);
    if(prev&&prev.path===r.path&&prev.fileSha256===r.fileSha256&&r.startLine<=prev.endLine+1)
      prev.endLine=Math.max(prev.endLine,r.endLine);
    else out.push(r);
  }
  return out;
}
export function rangesFor(matches:Match[],context:number,totalLines:(path:string)=>number):Range[] {
  return matches.map(m=>({path:m.path,startLine:Math.max(1,m.line-context),
    endLine:Math.min(totalLines(m.path),m.line+context),fileSha256:m.fileSha256}));
}
export type ReadIndex=Map<string,{data:Map<number,string>,totalLines:number}>;
export function indexReads(reads:ReadOutput[]):ReadIndex {
  const index:ReadIndex=new Map();
  for(const r of reads) {
    const entry=index.get(r.path)??{data:new Map<number,string>(),totalLines:r.totalLines};
    if(entry.totalLines!==r.totalLines)fail('GUARD_FAILED','READ_LINE_COUNT_DISAGREEMENT');
    r.text.split('\n').forEach((text,i)=>{
      const line=r.startLine+i;
      if(entry.data.has(line)&&entry.data.get(line)!==text)fail('GUARD_FAILED','OVERLAPPING_READ_DISAGREEMENT');
      entry.data.set(line,text);
    });
    index.set(r.path,entry);
  }
  return index;
}
export function buildBundle(task:Task,search:SearchOutput,reads:ReadOutput[],count:number,key:Buffer,index?:ReadIndex):Bundle {
  const matches=search.matches.slice(0,count);
  const totalLines=(path:string)=>{
    if(index?.has(path))return index.get(path)!.totalLines;
    const read=reads.find(r=>r.path===path); if(!read) fail('GUARD_FAILED','MISSING_READ_RESULT');return read.totalLines;
  };
  const excerpts:ReadOutput[]=mergeRanges(rangesFor(matches,task.contextLines,totalLines)).map(range=>{
    const indexed=index?.get(range.path);
    const data=indexed?.data??new Map<number,string>();let n=indexed?.totalLines??0;
    if(!indexed)for(const r of reads) if(r.path===range.path) {
      // Excerpt text omits only the terminal file delimiter; split preserves a final empty line.
      const ls=r.text.split('\n');n=r.totalLines;
      for(let i=0;i<ls.length;i++) {
        const line=r.startLine+i;
        if(data.has(line)&&data.get(line)!==ls[i]) fail('GUARD_FAILED','OVERLAPPING_READ_DISAGREEMENT');
        data.set(line,ls[i]!);
      }
    }
    const selected:string[]=[];
    for(let i=range.startLine;i<=range.endLine;i++) {
      const text=data.get(i);if(text===undefined) fail('GUARD_FAILED','INCOMPLETE_EXCERPT');selected.push(text);
    }
    const text=selected.join('\n');return {...range,text,sha256:sha(text),snapshotId:task.snapshotId,totalLines:n};
  });
  const next=search.offset+count<search.totalMatches?search.offset+count:null;
  const reasons:Bundle['truncationReasons']=[];
  if(search.nextOffset!==null) reasons.push('MAX_MATCHES');
  if(count<search.matches.length) reasons.push('OUTPUT_BUDGET');
  return checked(bundleSchema,{snapshotId:task.snapshotId,repositoryId:task.repositoryId,
    searchSemantics:task.searchSemantics,matches,excerpts,
    provenance:{kind:'text_matches',server:ADAPTER,adapterVersion:ADAPTER_VERSION,contract:CONTRACT,snapshotVerified:true},
    completeness:next===null?'complete':'partial',truncationReasons:reasons,
    continuation:next===null?null:encodeCursor(task,next,key),totalMatches:search.totalMatches,
    returnedMatches:count,nextOffset:next});
}
export function verifyRead(read:ReadOutput,expected:Range,snapshotId:string,fullText:string):void {
  const ls=lines(fullText), end=Math.min(expected.endLine,ls.length);
  const text=ls.slice(expected.startLine-1,end).join('\n');
  if(read.path!==expected.path||read.startLine!==expected.startLine||read.endLine!==end||
    read.snapshotId!==snapshotId||read.fileSha256!==expected.fileSha256||read.totalLines!==ls.length||
    read.text!==text||read.sha256!==sha(text)) fail('GUARD_FAILED','READ_CONTRACT_VIOLATION');
}
