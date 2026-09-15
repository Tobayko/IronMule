import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { z } from 'zod';
import { adapterContract, hostSchema, taskSchema, planSchema, resultSchema, searchInputSchema, searchOutputSchema,
  readInputSchema, readOutputSchema } from './contracts.js';
import { snapshotSchema } from './snapshot.js';
import { agentMetricsSchema } from './agent-adapter.js';
import { seedPlan } from './ir.js';
export const schemas={snapshot:snapshotSchema,'agent-metrics':agentMetricsSchema,task:taskSchema,host:hostSchema,moleir:planSchema,result:resultSchema,
  'search-input':searchInputSchema,'search-output':searchOutputSchema,'read-input':readInputSchema,'read-output':readOutputSchema};
for(const [name,schema] of Object.entries(schemas))writeFileSync(resolve('schemas',name+'.schema.json'),
  JSON.stringify(z.toJSONSchema(schema,{target:'draft-7'}),null,2)+'\n');
writeFileSync('examples/reference.moleir.json',JSON.stringify(seedPlan(),null,2)+'\n');
writeFileSync('examples/optimized.moleir.json',JSON.stringify(seedPlan({concurrency:4,dedup:true,merge:true,preparation:'indexed'}),null,2)+'\n');
writeFileSync('schemas/adapter-contract.json',JSON.stringify(adapterContract,null,2)+'\n');
