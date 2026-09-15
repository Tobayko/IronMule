import { z } from 'zod';
import { type Task, checked } from './contracts.js';
import { type AgentMetrics } from './trace.js';

export const agentMetricsSchema=z.strictObject({source:z.literal('trusted_agent_adapter'),provider:z.string().min(1),model:z.string().min(1),
  modelCalls:z.int().nonnegative(),inputTokens:z.int().nonnegative().nullable(),outputTokens:z.int().nonnegative().nullable(),
  costUsd:z.number().finite().nonnegative().nullable(),priceBasis:z.string().min(1).nullable(),
  totalTaskMs:z.number().finite().nonnegative().nullable(),finalResponseIncluded:z.boolean()});
export function validateAgentMetrics(value:unknown):AgentMetrics {
  const m=checked(agentMetricsSchema,value,'AGENT_METRICS_SCHEMA');
  if(m.costUsd!==null&&m.priceBasis===null)throw new Error('Observed money requires a documented price basis');
  if(m.totalTaskMs!==null&&!m.finalResponseIncluded)throw new Error('Total task time requires the final response');
  return m;
}
/** A host implements this interface using a real model/harness, never a passive MCP
 * proxy's inferred inter-tool gaps. No model adapter is bundled or silently chosen. */
export interface AgentBenchmarkAdapter {
  readonly provider:string;
  readonly model:string;
  readonly mode:'direct_tools'|'actual_programmatic_tool_calling';
  execute(task:Task,signal:AbortSignal):Promise<{finalResponse:unknown,metrics:AgentMetrics}>;
}
export const availableModelBenchmarks={A_DIRECT_AGENT:null,B_PTC:null} as const;
