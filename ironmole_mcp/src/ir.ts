import { ADAPTER, ADAPTER_VERSION, CONTRACT, IR_VERSION, TASK, LIMITS, checked, fail, planSchema,
  type Expr, type IRNode, type Plan } from './contracts.js';
import { digest } from './util.js';

export type ValueType='unit'|'search'|'ranges'|'reads'|'bundle';
export type Compiled={id:string,plan:Plan,order:IRNode[],dependencies:Record<string,string[]>,outputTypes:Record<string,ValueType>};
const ref=(node:string):Expr=>({source:'node',node});
const task=(field:Extract<Expr,{source:'task'}>['field']):Expr=>({source:'task',field});
export function seedPlan(options:{concurrency?:1|2|4,dedup?:boolean,merge?:boolean,origin?:Plan['origin'],preparation?:'scan'|'indexed'}={}):Plan {
  return {irVersion:IR_VERSION,taskType:TASK,taskContractVersion:'1.0.0',planVersion:2,origin:options.origin??'manual',
    adapter:ADAPTER,adapterVersion:ADAPTER_VERSION,adapterContract:CONTRACT,
    limits:{maxNodes:32,maxCalls:501,maxDurationMs:15000,maxIntermediateBytes:LIMITS.intermediateBytes},
    nodes:[
      {id:'authorize',op:'GUARD',dependsOn:[],check:'host_ready',input:null},
      {id:'search',op:'CALL',dependsOn:['authorize'],tool:'repo.search',args:{repositoryId:task('repositoryId'),snapshotId:task('snapshotId'),
        query:task('query'),searchSemantics:task('searchSemantics'),pathFilters:task('pathFilters'),limit:task('maxMatches')}},
      {id:'check_search',op:'GUARD',dependsOn:['search'],check:'search_valid',input:ref('search')},
      {id:'ranges',op:'PROJECT',dependsOn:['check_search'],input:ref('search'),context:task('contextLines'),projection:'context_ranges'},
      {id:'filter',op:'FILTER',dependsOn:[],input:ref('ranges'),predicate:'nonempty_range'},
      {id:'dedup',op:'DEDUP',dependsOn:[],input:ref('filter'),key:'path_start_end_hash',enabled:options.dedup??false},
      {id:'merge',op:'MERGE',dependsOn:[],input:ref('dedup'),mode:options.merge?'overlapping_intervals':'identity'},
      {id:'read',op:'BOUNDED_MAP',dependsOn:[],input:ref('merge'),tool:'repo.read',binding:'range_and_task_identity',maxItems:500,concurrency:options.concurrency??1},
      {id:'check_reads',op:'GUARD',dependsOn:[],input:ref('read'),check:'reads_valid'},
      {id:'return',op:'RETURN',dependsOn:['check_reads'],matches:ref('search'),reads:ref('read'),format:'context_bundle.v1',preparation:options.preparation??'scan'}]};
}
function expressions(n:IRNode):Expr[] {
  switch(n.op) {
    case 'CALL':return Object.values(n.args);
    case 'RETURN':return [n.matches,n.reads];
    case 'PROJECT':return [n.input,n.context];
    case 'GUARD':return n.input?[n.input]:[];
    default:return [n.input];
  }
}
export function compile(value:unknown):Compiled {
  const plan=checked(planSchema,value,'IR_SCHEMA_INVALID');
  if(plan.nodes.length>plan.limits.maxNodes) fail('GUARD_FAILED','IR_NODE_BUDGET');
  const byId=new Map(plan.nodes.map(n=>[n.id,n]));
  if(byId.size!==plan.nodes.length) fail('GUARD_FAILED','IR_DUPLICATE_ID');
  const dependencies:Record<string,string[]>=Object.create(null) as Record<string,string[]>;
  for(const n of plan.nodes) {
    dependencies[n.id]=[...new Set([...n.dependsOn,...expressions(n).filter(e=>e.source==='node').map(e=>e.node)])];
    if(dependencies[n.id]!.some(id=>!byId.has(id))) fail('GUARD_FAILED','IR_MISSING_DEPENDENCY');
  }
  const order:IRNode[]=[];const seen=new Set<string>(),visiting=new Set<string>();
  function visit(id:string):void {
    if(visiting.has(id)) fail('GUARD_FAILED','IR_CYCLE'); if(seen.has(id)) return;
    visiting.add(id); for(const dependency of dependencies[id]!) visit(dependency);
    visiting.delete(id); seen.add(id); order.push(byId.get(id)!);
  }
  for(const n of plan.nodes) visit(n.id);
  const types:Record<string,ValueType>=Object.create(null) as Record<string,ValueType>;
  const expect=(expr:Expr,type:ValueType)=>{
    if(expr.source!=='node'||types[expr.node]!==type) fail('GUARD_FAILED','IR_TYPE_MISMATCH');
  };
  const ancestry=(id:string):Set<string>=>{
    const out=new Set<string>();
    const add=(n:string)=>{for(const d of dependencies[n]!) if(!out.has(d)){out.add(d);add(d);}};add(id);return out;
  };
  let calls=0,maps=0,returns=0;
  for(const n of order) {
    switch(n.op) {
      case 'GUARD':
        if(n.check==='host_ready') {if(n.input!==null) fail('GUARD_FAILED','IR_GUARD_INPUT');}
        else {if(n.input===null) fail('GUARD_FAILED','IR_GUARD_INPUT');expect(n.input,n.check==='search_valid'?'search':'reads');}
        types[n.id]='unit';break;
      case 'CALL': {
        calls++;
        const expected={repositoryId:'repositoryId',snapshotId:'snapshotId',query:'query',searchSemantics:'searchSemantics',pathFilters:'pathFilters',limit:'maxMatches'};
        if(Object.keys(n.args).sort().join()!==Object.keys(expected).sort().join()) fail('GUARD_FAILED','IR_CALL_BINDINGS');
        for(const [key,field] of Object.entries(expected)) {
          const e=n.args[key];if(e?.source!=='task'||e.field!==field) fail('GUARD_FAILED','IR_ARGUMENT_ORIGIN');
        }
        if(![...ancestry(n.id)].some(id=>{const p=byId.get(id)!;return p.op==='GUARD'&&p.check==='host_ready';}))
          fail('GUARD_FAILED','IR_MISSING_AUTH_GUARD');
        types[n.id]='search';break;
      }
      case 'PROJECT':
        expect(n.input,'search');
        if(n.context.source!=='task'||n.context.field!=='contextLines') fail('GUARD_FAILED','IR_CONTEXT_ORIGIN');
        if(![...ancestry(n.id)].some(id=>{const p=byId.get(id)!;return p.op==='GUARD'&&p.check==='search_valid';}))
          fail('GUARD_FAILED','IR_MISSING_SEARCH_GUARD');
        types[n.id]='ranges';break;
      case 'FILTER':case 'MERGE':case 'DEDUP':expect(n.input,'ranges');types[n.id]='ranges';break;
      case 'BOUNDED_MAP':expect(n.input,'ranges');maps++;types[n.id]='reads';break;
      case 'RETURN':expect(n.matches,'search');expect(n.reads,'reads');returns++;types[n.id]='bundle';
        if(![...ancestry(n.id)].some(id=>{const p=byId.get(id)!;return p.op==='GUARD'&&p.check==='reads_valid';}))
          fail('GUARD_FAILED','IR_MISSING_READ_GUARD');break;
    }
  }
  if(calls!==1||maps!==1||returns!==1||order.at(-1)?.op!=='RETURN') fail('GUARD_FAILED','IR_TASK_CONTROL_FLOW');
  const final=order.at(-1)!;
  if(ancestry(final.id).size!==order.length-1) fail('GUARD_FAILED','IR_UNREACHABLE_NODE');
  // One known task, one semantic spine. A candidate cannot remove/reorder transformations
  // or add a constant that silently changes search semantics. Node IDs may be renamed.
  const spine=order.map(n=>n.op).join(',');
  if(spine!=='GUARD,CALL,GUARD,PROJECT,FILTER,DEDUP,MERGE,BOUNDED_MAP,GUARD,RETURN')
    fail('GUARD_FAILED','IR_UNSUPPORTED_TASK_GRAPH');
  return {id:digest(plan),plan,order,dependencies,outputTypes:types};
}
export function optimize(plan:Plan,options:{concurrency:1|2|4,dedup:boolean,merge:boolean}):Compiled {
  const cloned=structuredClone(plan);
  for(const node of cloned.nodes) {
    if(node.op==='BOUNDED_MAP') node.concurrency=options.concurrency;
    if(node.op==='DEDUP') node.enabled=options.dedup;
    if(node.op==='MERGE') node.mode=options.merge?'overlapping_intervals':'identity';
  }
  return compile(cloned);
}
