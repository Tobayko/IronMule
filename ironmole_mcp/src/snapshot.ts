import { lstatSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { z } from 'zod';
import { checked, fail, hashSchema, idSchema, LIMITS, pathSchema } from './contracts.js';
import { checkedPath, comparePath, digest, lines, rootBound, safeFile, sha, within } from './util.js';

export const snapshotSchema=z.strictObject({ version:z.literal('snapshot.v1'), repositoryId:idSchema,
  files:z.array(z.strictObject({path:pathSchema,text:z.string(),sha256:hashSchema})).max(LIMITS.files) });
export type Snapshot=z.infer<typeof snapshotSchema>;
export function snapshotId(snapshot:Snapshot):string { return digest(snapshot); }
export function validateSnapshot(value:unknown,id:string):Snapshot {
  const snap=checked(snapshotSchema,value,'SNAPSHOT_SCHEMA');
  if(snapshotId(snap)!==id) fail('GUARD_FAILED','SNAPSHOT_DIGEST_MISMATCH');
  let total=0, last:string|null=null;
  for(const f of snap.files) {
    total+=Buffer.byteLength(f.text);
    if((last!==null&&comparePath(last,f.path)>=0)||f.path==='.'||sha(f.text)!==f.sha256||
      f.text.includes('\0')||Buffer.byteLength(f.text)>LIMITS.fileBytes) fail('GUARD_FAILED','SNAPSHOT_FILE_INVALID');
    last=f.path;
  }
  if(total>LIMITS.snapshotBytes) fail('GUARD_FAILED','SNAPSHOT_SIZE');
  return snap;
}
export function importSnapshot(repositoryId:string,root:string,allowedPaths:string[]):Snapshot {
  checked(idSchema,repositoryId); allowedPaths.forEach(checkedPath);
  if(allowedPaths.length===0) fail('POLICY_BLOCKED','EMPTY_SNAPSHOT_SCOPE');
  const absolute=resolve(root);
  rootBound(absolute,'.');
  const files:Snapshot['files']=[]; let total=0; let visited=0;
  function visit(path:string):void {
    if(++visited>10_000) fail('GUARD_FAILED','IMPORT_ENTRY_LIMIT');
    if(path!=='.'&&!allowedPaths.some(p=>within(path,p)||within(p,path))) return;
    const full=rootBound(absolute,path); const info=lstatSync(full);
    if(info.isDirectory()) {
      for(const entry of readdirSync(full).sort(comparePath)) visit(path==='.'?entry:path+'/'+entry);
    } else {
      if(!info.isFile()) fail('POLICY_BLOCKED','SPECIAL_FILE_FORBIDDEN');
      if(!allowedPaths.some(p=>within(path,p))) return;
      const data=safeFile(full,LIMITS.fileBytes);
      let text:string;
      try { text=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(data); }
      catch { return fail('GUARD_FAILED','NON_UTF8_SNAPSHOT_FILE'); }
      if(text.includes('\0')) fail('GUARD_FAILED','BINARY_SNAPSHOT_FILE');
      total+=data.length;
      if(total>LIMITS.snapshotBytes||files.length>=LIMITS.files) fail('GUARD_FAILED','SNAPSHOT_IMPORT_LIMIT');
      files.push({path,text,sha256:sha(data)});
    }
  }
  visit('.');
  files.sort((a,b)=>comparePath(a.path,b.path));
  const snap:Snapshot={version:'snapshot.v1',repositoryId,files};
  return validateSnapshot(snap,snapshotId(snap));
}
export function fileAt(snapshot:Snapshot,path:string) {
  const file=snapshot.files.find(f=>f.path===path);
  if(!file) fail('GUARD_FAILED','SNAPSHOT_PATH_MISSING');
  return {...file,lines:lines(file.text)};
}
