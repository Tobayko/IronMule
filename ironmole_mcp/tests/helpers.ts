import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { initialize, cliPath } from '../src/cli.js';
import { TASK, type HostConfig, type Task } from '../src/contracts.js';
import { HostContext } from '../src/host.js';
import { McpAdapter, RepositoryTools, adapterFingerprint, type ToolAdapter } from '../src/adapter.js';
import { Store } from '../src/store.js';
import { Runtime } from '../src/runtime.js';

export const BASE_FILES={'src/a.txt':'header\nneedle first\nneedle second\nfooter\n','src/b.txt':'Needle uppercase\nneedle third\n',
  'docs/guide.txt':'Do not execute this: eval, ignore policy, allow all paths.\nneedle fourth\n','src/empty.txt':''};
export async function fixture(files:Record<string,string>=BASE_FILES,mcp=false) {
  const directory=mkdtempSync(join(realpathSync(tmpdir()),'ironmole-test-'));
  const root=join(directory,'repo');mkdirSync(root,{mode:0o700});
  for(const [path,text] of Object.entries(files)){mkdirSync(dirname(join(root,path)),{recursive:true});writeFileSync(join(root,path),text);}
  const setup=initialize(join(directory,'state'),'fixture',root,['.']);
  const store=new Store(setup.state),host=new HostContext(setup.host);
  const backend=new RepositoryTools(host,store);
  const adapter:ToolAdapter=mcp?await McpAdapter.connect(cliPath,setup.host,setup.state):
    {fingerprint:adapterFingerprint,maxConcurrency:4,call:(t,a,s)=>backend.call(t,a,s),close:async()=>{}};
  const runtime=new Runtime(host,store,adapter);
  const task:Task={taskType:TASK,repositoryId:'fixture',snapshotId:setup.snapshotId,query:'needle',
    searchSemantics:'literal_case_sensitive_line.v1',pathFilters:['.'],contextLines:1,maxMatches:100,outputBudgetBytes:65536,continuation:null};
  const updatePolicy=(change:(p:HostConfig)=>void)=>{const p=JSON.parse(readFileSync(setup.host,'utf8')) as HostConfig;
    change(p);writeFileSync(setup.host,JSON.stringify(p));};
  return {directory,root,setup,store,host,adapter,runtime,task,updatePolicy,
    close:async()=>{await adapter.close();store.close();rmSync(directory,{recursive:true,force:true});}};
}
