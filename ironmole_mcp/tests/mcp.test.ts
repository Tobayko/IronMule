import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { join } from 'node:path';
import { writeFileSync } from 'node:fs';
import { configToml, cliPath } from '../src/cli.js';
import { checked, resultSchema, PROTOCOL } from '../src/contracts.js';
import { fixture } from './helpers.js';

async function connect(args:string[]) {
  const client=new Client({name:'ironmole-test',version:'1.0.0'},{capabilities:{}});
  const transport=new StdioClientTransport({command:process.execPath,args,stderr:'pipe',env:{PATH:process.env.PATH??'',NODE_NO_WARNINGS:'1'}});
  let protocol:string|null=null;Object.assign(transport,{setProtocolVersion:(v:string)=>{protocol=v;}});
  await client.connect(transport,{timeout:5000});return {client,transport,protocol};
}
test('outer MCP tool uses explicit class and strict schemas; structured result roundtrip',async()=>{
  const f=await fixture();const {client,protocol}=await connect([cliPath,'serve','--state',f.setup.state,'--host',f.setup.host]);
  try {
    assert.equal(protocol,PROTOCOL);const listing=await client.listTools();assert.deepEqual(listing.tools.map(t=>t.name),['ironmole.execute']);
    assert.equal(listing.tools[0]!.inputSchema['additionalProperties'],false);
    const r=await client.callTool({name:'ironmole.execute',arguments:f.task});
    const result=checked(resultSchema,r.structuredContent);assert.equal(result.status,'COMPLETE');assert.equal(result.bundle!.totalMatches,4);
    const forged=await client.callTool({name:'ironmole.execute',arguments:{...f.task,permissions:['all']}});
    assert.equal(checked(resultSchema,forged.structuredContent).status,'GUARD_FAILED');
    await assert.rejects(()=>client.callTool({name:'shell',arguments:{command:'true'}}));
  } finally {await client.close();await f.close();}
});
test('controlled MCP server independently denies direct unapproved inner operations and unknown interactions',async()=>{
  const f=await fixture();f.updatePolicy(p=>{p.allowedTools=['repo.search'];});
  const {client}=await connect([cliPath,'adapter','--state',f.setup.state,'--host',f.setup.host]);
  try {
    const read=await client.callTool({name:'repo.read',arguments:{repositoryId:'fixture',snapshotId:f.task.snapshotId,
      path:'src/a.txt',startLine:1,endLine:2,fileSha256:'a'.repeat(64)}});
    assert.equal(read.isError,true);assert.ok(JSON.stringify(read.content).includes('INNER_TOOL_DENIED'));
    const unknown=await client.callTool({name:'unknown',arguments:{}});assert.equal(unknown.isError,true);
    await assert.rejects(()=>client.listResources());
  } finally {await client.close();await f.close();}
});
test('generated Codex TOML parses and its exact command completes MCP initialization and task',async()=>{
  const f=await fixture();try {
    const text=configToml(f.setup.state,f.setup.host);const path=join(f.directory,'codex-example.toml');writeFileSync(path,text);
    const parsed=JSON.parse(execFileSync('python3',['-c','import json,sys,tomllib; print(json.dumps(tomllib.load(open(sys.argv[1],"rb"))["mcp_servers"]["ironmole"]))',path],{encoding:'utf8'})) as {command:string,args:string[]};
    assert.equal(parsed.command,process.execPath);
    const {client}=await connect(parsed.args);
    try {const r=await client.callTool({name:'ironmole.execute',arguments:f.task});assert.equal(checked(resultSchema,r.structuredContent).status,'COMPLETE');}
    finally {await client.close();}
  } finally {await f.close();}
});
test('installed Codex accepts generated server configuration (no model execution or global write)',async(t)=>{
  const probe=spawnSync('codex',['--version'],{encoding:'utf8'});
  if(probe.error||probe.status!==0){t.skip('Codex CLI not installed; SDK handshake still tested');return;}
  const f=await fixture();try {
    const toml=configToml(f.setup.state,f.setup.host);
    const fields=toml.split('\n').slice(1).filter(Boolean).join(', ');
    const raw=execFileSync('codex',['-c','mcp_servers.ironmole={'+fields+'}','mcp','get','ironmole','--json'],
      {encoding:'utf8',cwd:f.directory,stdio:['ignore','pipe','pipe']});
    const output=JSON.parse(raw) as {name:string,transport:{command:string,args:string[]}};
    assert.equal(output.name,'ironmole');assert.equal(output.transport.command,process.execPath);assert.ok(output.transport.args.includes('serve'));
  } finally {await f.close();}
});
