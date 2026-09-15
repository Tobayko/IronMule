import { DatabaseSync } from 'node:sqlite';
import { chmodSync, existsSync, lstatSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { randomBytes } from 'node:crypto';
import { type Plan, checked, fail, planSchema } from './contracts.js';
import { type Snapshot, snapshotId, validateSnapshot } from './snapshot.js';
import { assertNoSymlinks, canonical, digest, safeFile } from './util.js';

const SCHEMA=`
CREATE TABLE metadata (version INTEGER NOT NULL CHECK(version=1));
INSERT INTO metadata VALUES(1);
CREATE TABLE snapshots (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE plans (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, entity TEXT NOT NULL,
  payload TEXT NOT NULL, checksum TEXT NOT NULL);
CREATE TRIGGER snapshots_immutable BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER snapshots_no_delete BEFORE DELETE ON snapshots BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER plans_immutable BEFORE UPDATE ON plans BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER plans_no_delete BEFORE DELETE ON plans BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER events_immutable BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'immutable'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'immutable'); END;
PRAGMA user_version=1;`;
export class Store {
  readonly db:DatabaseSync;
  readonly directory:string;
  constructor(directory:string,readonly readOnly=false) {
    this.directory=resolve(directory);
    if(!existsSync(this.directory)&&!readOnly) mkdirSync(this.directory,{recursive:true,mode:0o700});
    assertNoSymlinks(this.directory);
    const info=lstatSync(this.directory);
    if(!info.isDirectory()||(info.mode&0o077)!==0||info.uid!==process.getuid?.()) fail('POLICY_BLOCKED','STATE_DIRECTORY_PERMISSIONS');
    const path=join(this.directory,'registry.sqlite'); const present=existsSync(path);
    if(present) safeFile(path,64_000_000,true);
    else if(!readOnly) writeFileSync(path,'',{mode:0o600,flag:'wx'});
    this.db=new DatabaseSync(path,{readOnly});
    this.db.exec('PRAGMA busy_timeout=1000;');
    const version=this.db.prepare('PRAGMA user_version').get() as {user_version:number};
    if(!present&&!readOnly) { this.db.exec(SCHEMA); this.db.exec('PRAGMA journal_mode=WAL;'); }
    else if(version.user_version!==1) { this.db.close(); fail('GUARD_FAILED','REGISTRY_SCHEMA_UNKNOWN'); }
    if(!readOnly) chmodSync(path,0o600);
    const keyPath=join(this.directory,'cursor.key');
    if(!existsSync(keyPath)&&!readOnly) writeFileSync(keyPath,randomBytes(32),{mode:0o600,flag:'wx'});
  }
  key():Buffer { return safeFile(join(this.directory,'cursor.key'),32,true); }
  close():void { this.db.close(); }
  saveSnapshot(snapshot:Snapshot):string {
    const id=snapshotId(snapshot); validateSnapshot(snapshot,id);
    this.db.prepare('INSERT OR IGNORE INTO snapshots VALUES(?,?)').run(id,canonical(snapshot)); return id;
  }
  snapshot(id:string):Snapshot {
    const row=this.db.prepare('SELECT payload FROM snapshots WHERE id=?').get(id) as {payload:string}|undefined;
    if(!row) fail('GUARD_FAILED','SNAPSHOT_UNAVAILABLE');
    try { return validateSnapshot(JSON.parse(row.payload),id); }
    catch(e) { if(e instanceof Error&&e.name==='MoleError') throw e; return fail('GUARD_FAILED','SNAPSHOT_CORRUPT'); }
  }
  savePlan(plan:Plan):string {
    const parsed=checked(planSchema,plan),id=digest(parsed);
    this.db.prepare('INSERT OR IGNORE INTO plans VALUES(?,?)').run(id,canonical(parsed)); return id;
  }
  plan(id:string):Plan {
    const row=this.db.prepare('SELECT payload FROM plans WHERE id=?').get(id) as {payload:string}|undefined;
    if(!row) fail('GUARD_FAILED','PLAN_UNAVAILABLE');
    try {
      const plan=checked(planSchema,JSON.parse(row.payload));
      if(digest(plan)!==id) fail('GUARD_FAILED','PLAN_DIGEST_MISMATCH'); return plan;
    } catch { return fail('GUARD_FAILED','PLAN_CORRUPT'); }
  }
  append(kind:string,entity:string,payload:unknown):void {
    this.db.prepare('INSERT INTO events(kind,entity,payload,checksum) VALUES(?,?,?,?)').run(kind,entity,canonical(payload),digest(payload));
  }
  events(kind:string):{entity:string,payload:unknown}[] {
    const rows=this.db.prepare('SELECT entity,payload,checksum FROM events WHERE kind=? ORDER BY seq').all(kind) as
      {entity:string,payload:string,checksum:string}[];
    return rows.map(r=>{
      let payload:unknown; try { payload=JSON.parse(r.payload); } catch { return fail('GUARD_FAILED','EVENT_CORRUPT'); }
      if(digest(payload)!==r.checksum) fail('GUARD_FAILED','EVENT_DIGEST_MISMATCH');
      return {entity:r.entity,payload};
    });
  }
  saveTrace(runId:string,trace:unknown):void {
    this.append('trace',runId,trace);
    const path=join(this.directory,'trace-'+runId+'.json');
    writeFileSync(path,JSON.stringify(trace)+'\n',{flag:'wx',mode:0o600});
  }
  retain(runId:string,results:unknown):string {
    const filename='retained-'+runId+'.json';
    writeFileSync(join(this.directory,filename),JSON.stringify(results),{flag:'wx',mode:0o600}); return filename;
  }
  readRetained(name:string):unknown {
    if(!/^retained-[a-f0-9-]+\.json$/.test(name)) fail('POLICY_BLOCKED','INVALID_RETAINED_REFERENCE');
    return JSON.parse(safeFile(join(this.directory,name),16_000_000,true).toString('utf8')) as unknown;
  }
}
