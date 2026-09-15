import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { constants, closeSync, fstatSync, lstatSync, openSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { resolve, dirname, isAbsolute, relative, sep } from 'node:path';
import { fail, pathSchema, type Task } from './contracts.js';

export function canonical(value:unknown):string {
  if(value===null || typeof value==='boolean' || typeof value==='string') return JSON.stringify(value);
  if(typeof value==='number' && Number.isFinite(value)) return JSON.stringify(value);
  if(Array.isArray(value)) return '['+value.map(canonical).join(',')+']';
  if(typeof value==='object' && value && (Object.getPrototypeOf(value)===Object.prototype||Object.getPrototypeOf(value)===null))
    return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical((value as Record<string,unknown>)[k])).join(',')+'}';
  throw new TypeError('non-JSON value');
}
export function sha(value:string|Buffer):string { return createHash('sha256').update(value).digest('hex'); }
export function digest(value:unknown):string { return sha(canonical(value)); }
export function bytes(value:unknown):number { return Buffer.byteLength(JSON.stringify(value)); }
export function within(path:string,prefix:string):boolean { return prefix==='.' || path===prefix || path.startsWith(prefix+'/'); }
export function checkedPath(path:string):string {
  if(!pathSchema.safeParse(path).success) fail('POLICY_BLOCKED','UNSAFE_PATH');
  return path;
}
export function comparePath(a:string,b:string):number { return a<b?-1:a>b?1:0; }
export function lines(text:string):string[] { return text===''?[]:text.endsWith('\n')?text.slice(0,-1).split('\n'):text.split('\n'); }
export function safeFile(path:string, maxBytes:number, privateFile=false):Buffer {
  const absolute=resolve(path);
  assertNoSymlinks(absolute);
  const fd=openSync(absolute,constants.O_RDONLY|constants.O_NOFOLLOW);
  try {
    const before=fstatSync(fd);
    if(!before.isFile() || before.size>maxBytes) fail('GUARD_FAILED','FILE_SIZE_OR_TYPE');
    if(privateFile && ((before.mode&0o077)!==0 || before.uid!==process.getuid?.())) fail('POLICY_BLOCKED','PRIVATE_FILE_PERMISSIONS');
    const data=readFileSync(fd); const after=fstatSync(fd);
    assertNoSymlinks(absolute);
    const current=lstatSync(absolute);
    if(before.dev!==current.dev || before.ino!==current.ino || before.size!==after.size ||
      before.mtimeMs!==after.mtimeMs || data.length!==before.size || data.length>maxBytes)
      fail('GUARD_FAILED','FILE_CHANGED_DURING_IMPORT');
    return data;
  } finally { closeSync(fd); }
}
export function assertNoSymlinks(path:string):void {
  let p=resolve(path);
  while(true) {
    if(lstatSync(p).isSymbolicLink()) fail('POLICY_BLOCKED','SYMLINK_FORBIDDEN');
    const parent=dirname(p); if(parent===p) break; p=parent;
  }
}
export function rootBound(root:string,path:string):string {
  if(!isAbsolute(root)) fail('POLICY_BLOCKED','ROOT_MUST_BE_ABSOLUTE');
  const full=resolve(root,checkedPath(path));
  const rel=relative(root,full);
  if(rel.startsWith('..'+sep)||rel==='..'||isAbsolute(rel)) fail('POLICY_BLOCKED','PATH_ESCAPE');
  assertNoSymlinks(full);
  if(realpathSync(full)!==full) fail('POLICY_BLOCKED','NONCANONICAL_PATH');
  return full;
}
export function writePrivate(path:string,value:unknown):void {
  writeFileSync(path,JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
}
export function cursorBinding(task:Task):string {
  return digest({repositoryId:task.repositoryId,snapshotId:task.snapshotId,query:task.query,
    searchSemantics:task.searchSemantics,pathFilters:task.pathFilters,contextLines:task.contextLines});
}
export function encodeCursor(task:Task,offset:number,key:Buffer):string {
  const payload=Buffer.from(JSON.stringify({v:1,b:cursorBinding(task),offset})).toString('base64url');
  return payload+'.'+createHmac('sha256',key).update(payload).digest('hex');
}
export function decodeCursor(task:Task,key:Buffer):number {
  if(task.continuation===null) return 0;
  const [payload,signature,...extra]=task.continuation.split('.');
  if(!payload||!signature||extra.length||!/^[a-f0-9]{64}$/.test(signature)) fail('GUARD_FAILED','CURSOR_INVALID');
  const expected=createHmac('sha256',key).update(payload).digest();
  if(!timingSafeEqual(expected,Buffer.from(signature,'hex'))) fail('GUARD_FAILED','CURSOR_INVALID');
  try {
    const v:unknown=JSON.parse(Buffer.from(payload,'base64url').toString('utf8'));
    const c=v as {v:unknown,b:unknown,offset:unknown};
    if(c.v!==1||c.b!==cursorBinding(task)||typeof c.offset!=='number'||!Number.isSafeInteger(c.offset)||c.offset<0)
      fail('GUARD_FAILED','CURSOR_BINDING_CHANGED');
    return c.offset;
  } catch { return fail('GUARD_FAILED','CURSOR_INVALID'); }
}
