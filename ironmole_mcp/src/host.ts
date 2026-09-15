import { type HostConfig, checked, fail, hostSchema, type Task, type ToolName } from './contracts.js';
import { digest, safeFile, within } from './util.js';

export class HostContext {
  constructor(readonly configPath:string) {}
  config():HostConfig {
    try { return checked(hostSchema,JSON.parse(safeFile(this.configPath,65536,true).toString('utf8')),'HOST_SCHEMA_CHANGED'); }
    catch(e) { if(e instanceof Error&&e.name==='MoleError') throw e; return fail('POLICY_BLOCKED','HOST_POLICY_UNAVAILABLE'); }
  }
  bind(repositoryId:string) {
    const host=this.config();
    if(!host.enabled||Date.now()>=host.expiresAt) fail('POLICY_BLOCKED','POLICY_DISABLED_OR_EXPIRED');
    const matches=host.repositories.filter(r=>r.id===repositoryId);
    if(matches.length!==1) fail('POLICY_BLOCKED','REPOSITORY_NOT_REGISTERED');
    return {host,repository:matches[0]!,policyHash:digest(host)};
  }
  authorize(tool:ToolName,repositoryId:string,paths:string[],policyHash?:string):string {
    const bound=this.bind(repositoryId);
    if(!bound.host.allowedTools.includes(tool)) fail('POLICY_BLOCKED','INNER_TOOL_DENIED');
    if(paths.length===0||paths.some(p=>!bound.repository.allowedPaths.some(prefix=>within(p,prefix))))
      fail('POLICY_BLOCKED','PATH_NOT_AUTHORIZED');
    if(policyHash!==undefined&&policyHash!==bound.policyHash) fail('POLICY_BLOCKED','POLICY_CHANGED');
    return bound.policyHash;
  }
  task(task:Task,policyHash?:string):string {
    return this.authorize('repo.search',task.repositoryId,task.pathFilters,policyHash);
  }
}
