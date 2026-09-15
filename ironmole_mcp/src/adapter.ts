import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { ADAPTER, ADAPTER_VERSION, CONTRACT, LIMITS, PROTOCOL, MoleError, VERSION, adapterContract,
  checked, fail, readInputSchema, readOutputSchema, searchInputSchema, searchOutputSchema, toolSchemas,
  type ToolName, type SearchOutput } from './contracts.js';
import { HostContext } from './host.js';
import { fileAt } from './snapshot.js';
import { Store } from './store.js';
import { bytes, canonical, digest, sha, within } from './util.js';

export function jsonSchema(schema:z.ZodType):Record<string,unknown> {
  const value=z.toJSONSchema(schema,{target:'draft-7'}); delete value.$schema; return value;
}
export const toolDefinitions=Object.entries(toolSchemas).map(([name,schema])=>({name,
  description:name==='repo.search'?'Literal case-sensitive matching lines in an authorized immutable snapshot.':'Read an authorized line interval in an immutable snapshot.',
  inputSchema:jsonSchema(schema.input) as {type:'object',properties:Record<string,unknown>},
  outputSchema:jsonSchema(schema.output) as {type:'object',properties:Record<string,unknown>},
  annotations:{readOnlyHint:true,destructiveHint:false,idempotentHint:true,openWorldHint:false}}));
export const adapterFingerprint=digest({contract:adapterContract,tools:toolDefinitions});
export interface ToolAdapter {
  readonly fingerprint:string;
  readonly maxConcurrency:number;
  call(tool:ToolName,args:unknown,signal:AbortSignal):Promise<unknown>;
  close():Promise<void>;
}
// This is the only filesystem adapter implementation. Tests may decorate its MCP
// client to inject transport faults; runtime permission checks still surround it.
export class RepositoryTools {
  constructor(readonly host:HostContext,readonly store:Store) {}
  async call(tool:ToolName,args:unknown,signal:AbortSignal):Promise<unknown> {
    if(signal.aborted) throw new MoleError('INFRASTRUCTURE_ERROR','CANCELLED',true);
    if(bytes(args)>LIMITS.inputBytes) fail('GUARD_FAILED','TOOL_INPUT_BUDGET');
    if(tool==='repo.search') {
      const a=checked(searchInputSchema,args);
      const policy=this.host.authorize(tool,a.repositoryId,a.pathFilters);
      const snapshot=this.store.snapshot(a.snapshotId);
      if(snapshot.repositoryId!==a.repositoryId) fail('GUARD_FAILED','SNAPSHOT_REPOSITORY_MISMATCH');
      const matches:SearchOutput['matches']=[];let total=0,scannedFiles=0;
      for(const file of snapshot.files) {
        if(!a.pathFilters.some(p=>within(file.path,p))) continue;
        this.host.authorize(tool,a.repositoryId,[file.path],policy);
        const ls=fileAt(snapshot,file.path).lines;scannedFiles++;
        for(let i=0;i<ls.length;i++) if(ls[i]!.includes(a.query)) {
          if(total>=a.offset&&matches.length<a.limit) matches.push({path:file.path,line:i+1,fileSha256:file.sha256});total++;
        }
        if(scannedFiles%16===0) await new Promise<void>(resolve=>setImmediate(resolve));
        if(signal.aborted) throw new MoleError('INFRASTRUCTURE_ERROR','CANCELLED',true);
      }
      this.host.authorize(tool,a.repositoryId,a.pathFilters,policy);
      if(a.offset>total) fail('GUARD_FAILED','CURSOR_OFFSET_OUT_OF_RANGE');
      return checked(searchOutputSchema,{snapshotId:a.snapshotId,matches,totalMatches:total,offset:a.offset,
        nextOffset:a.offset+matches.length<total?a.offset+matches.length:null,scannedFiles});
    }
    const a=checked(readInputSchema,args);
    const policy=this.host.authorize(tool,a.repositoryId,[a.path]);
    const snapshot=this.store.snapshot(a.snapshotId);
    if(snapshot.repositoryId!==a.repositoryId) fail('GUARD_FAILED','SNAPSHOT_REPOSITORY_MISMATCH');
    const file=fileAt(snapshot,a.path);
    if(file.sha256!==a.fileSha256||a.startLine>a.endLine||a.startLine>file.lines.length)
      fail('GUARD_FAILED','READ_RANGE_OR_HASH');
    const endLine=Math.min(a.endLine,file.lines.length);
    const text=file.lines.slice(a.startLine-1,endLine).join('\n');
    this.host.authorize(tool,a.repositoryId,[a.path],policy);
    return checked(readOutputSchema,{path:a.path,startLine:a.startLine,endLine,fileSha256:file.sha256,
      snapshotId:a.snapshotId,text,sha256:sha(text),totalLines:file.lines.length});
  }
}
export async function serveRepository(host:HostContext,store:Store):Promise<Server> {
  const backend=new RepositoryTools(host,store);
  const server=new Server({name:ADAPTER,version:ADAPTER_VERSION},{capabilities:{tools:{}}});
  server.setRequestHandler(ListToolsRequestSchema,async()=>({tools:toolDefinitions}));
  let active=0;
  server.setRequestHandler(CallToolRequestSchema,async(request,extra)=>{
    try {
      const name=request.params.name;
      if(name!=='repo.search'&&name!=='repo.read') fail('POLICY_BLOCKED','UNKNOWN_TOOL');
      if(active>=LIMITS.concurrency) fail('GUARD_FAILED','ADAPTER_CONCURRENCY_LIMIT');
      active++;
      try {
        const value=await backend.call(name,request.params.arguments,extra.signal);
        if(bytes(value)>LIMITS.toolOutputBytes) fail('GUARD_FAILED','TOOL_OUTPUT_BUDGET');
        return {content:[{type:'text' as const,text:JSON.stringify(value)}],structuredContent:value as Record<string,unknown>};
      } finally {active--;}
    } catch(e) {
      const error=e instanceof MoleError?e:new MoleError('INFRASTRUCTURE_ERROR','ADAPTER_FAILURE');
      return {isError:true,content:[{type:'text' as const,text:JSON.stringify({status:error.status,code:error.code})}]};
    }
  });
  await server.connect(new StdioServerTransport());return server;
}
export class McpAdapter implements ToolAdapter {
  readonly fingerprint=adapterFingerprint;
  readonly maxConcurrency=4;
  private constructor(readonly client:Client,readonly transport:StdioClientTransport) {}
  static async connect(cliPath:string,hostPath:string,statePath:string):Promise<McpAdapter> {
    const client=new Client({name:'ironmole',version:VERSION},{capabilities:{}});
    const transport=new StdioClientTransport({command:process.execPath,
      args:[cliPath,'adapter','--host',hostPath,'--state',statePath],stderr:'pipe',
      env:{PATH:process.env.PATH??'',NODE_NO_WARNINGS:'1'}});
    Object.assign(transport,{setProtocolVersion:(version:string)=>{
      if(version!==PROTOCOL)fail('GUARD_FAILED','MCP_PROTOCOL_UNSUPPORTED');
    }});
    try {
      await client.connect(transport,{timeout:LIMITS.toolMs});
      const identity=client.getServerVersion();
      if(identity?.name!==ADAPTER||identity.version!==ADAPTER_VERSION) fail('GUARD_FAILED','ADAPTER_IDENTITY_CHANGED');
      const listing=await client.listTools(undefined,{timeout:LIMITS.toolMs});
      if(canonical(listing.tools)!==canonical(toolDefinitions)) fail('GUARD_FAILED','ADAPTER_SCHEMA_CHANGED');
      return new McpAdapter(client,transport);
    } catch(e) {await client.close();throw e;}
  }
  async call(tool:ToolName,args:unknown,signal:AbortSignal):Promise<unknown> {
    const result=await this.client.callTool({name:tool,arguments:args as Record<string,unknown>},undefined,
      {signal,timeout:LIMITS.toolMs,resetTimeoutOnProgress:false});
    if(result.isError) {
      const content=result.content;
      if(Array.isArray(content)&&content[0]?.type==='text') {
        try {
          const e:unknown=JSON.parse(content[0].text as string);
          const parsed=z.strictObject({status:z.enum(['GUARD_FAILED','POLICY_BLOCKED','INFRASTRUCTURE_ERROR']),code:z.string().max(160)}).safeParse(e);
          if(parsed.success) throw new MoleError(parsed.data.status,parsed.data.code);
        } catch(e) {if(e instanceof MoleError) throw e;}
      }
      throw new MoleError('INFRASTRUCTURE_ERROR','MCP_TOOL_ERROR');
    }
    if(!result.structuredContent) fail('GUARD_FAILED','STRUCTURED_TOOL_RESULT_REQUIRED');
    return result.structuredContent;
  }
  async close():Promise<void> { await this.client.close(); }
}
export const adapterIdentity={server:ADAPTER,version:ADAPTER_VERSION,contract:CONTRACT,fingerprint:adapterFingerprint};
