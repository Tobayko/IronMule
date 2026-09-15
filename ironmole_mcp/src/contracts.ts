import { z } from 'zod';

export const VERSION = '0.1.0';
export const TASK = 'repository_context_bundle.v1';
export const IR_VERSION = 'moleir.v1';
export const PROTOCOL = '2025-11-25';
export const ADAPTER = 'ironmole.repository';
export const ADAPTER_VERSION = '1.0.0';
export const CONTRACT = 'repository-tools.v1';
export const LIMITS = Object.freeze({ nodes: 32, matches: 500, calls: 501, concurrency: 4,
  taskMs: 15_000, toolMs: 3_000, inputBytes: 16_384, toolOutputBytes: 2_000_000,
  intermediateBytes: 16_000_000, outputBytes: 1_048_576, fileBytes: 262_144,
  snapshotBytes: 8_388_608, files: 1_000 });
export const idSchema = z.string().regex(/^[a-z][a-z0-9_-]{0,63}$/);
export const hashSchema = z.string().regex(/^[a-f0-9]{64}$/);
export const pathSchema = z.string().min(1).max(512).refine(p => p === '.' ||
  (!p.startsWith('/') && !/[\\\x00-\x1f\x7f:*?\[\]]/.test(p) &&
    p.split('/').every(s => s !== '' && s !== '.' && s !== '..')), 'canonical relative path required');
export const taskSchema = z.strictObject({
  taskType: z.literal(TASK), repositoryId: idSchema, snapshotId: hashSchema,
  query: z.string().min(1).max(512).refine(q => !/[\r\n\0]/.test(q), 'single-line literal required'),
  searchSemantics: z.literal('literal_case_sensitive_line.v1'),
  pathFilters: z.array(pathSchema).min(1).max(32), contextLines: z.int().min(0).max(50),
  maxMatches: z.int().min(1).max(LIMITS.matches),
  outputBudgetBytes: z.int().min(4096).max(LIMITS.outputBytes),
  continuation: z.string().max(1024).nullable().default(null)
});
export type Task = z.infer<typeof taskSchema>;
export const objectiveSchema = z.discriminatedUnion('primary', [
  z.strictObject({ primary: z.literal('latency'), costLimitUsd: z.number().nonnegative(),
    expectedUses: z.int().min(1).max(1_000_000) }),
  z.strictObject({ primary: z.literal('cost'), latencyLimitMs: z.number().positive(),
    expectedUses: z.int().min(1).max(1_000_000) })
]);
export type Objective = z.infer<typeof objectiveSchema>;
export const hostSchema = z.strictObject({
  version: z.literal('ironmole-host.v1'), principal: idSchema, policyVersion: idSchema,
  enabled: z.boolean(), expiresAt: z.int().positive(),
  allowedTools: z.array(z.enum(['repo.search', 'repo.read'])).max(2),
  repositories: z.array(z.strictObject({ id: idSchema, root: z.string().min(1),
    allowedPaths: z.array(pathSchema).min(1).max(32), concurrency: z.union([z.literal(1),z.literal(2),z.literal(4)]) })).max(32),
  objective: objectiveSchema,
  adaptive: z.boolean()
});
export type HostConfig = z.infer<typeof hostSchema>;
export const rangeSchema = z.strictObject({ path: pathSchema, startLine: z.int().positive(),
  endLine: z.int().positive(), fileSha256: hashSchema });
export type Range = z.infer<typeof rangeSchema>;
export const matchSchema = z.strictObject({ path: pathSchema, line: z.int().positive(), fileSha256: hashSchema });
export type Match = z.infer<typeof matchSchema>;
export const searchInputSchema = taskSchema.pick({repositoryId:true,snapshotId:true,query:true,searchSemantics:true,pathFilters:true})
  .extend({ offset: z.int().min(0), limit: z.int().min(1).max(LIMITS.matches) });
export type SearchInput = z.infer<typeof searchInputSchema>;
export const searchOutputSchema = z.strictObject({ snapshotId: hashSchema,
  matches: z.array(matchSchema).max(LIMITS.matches), totalMatches: z.int().min(0),
  offset: z.int().min(0), nextOffset: z.int().min(0).nullable(), scannedFiles: z.int().min(0).max(LIMITS.files) });
export type SearchOutput = z.infer<typeof searchOutputSchema>;
export const readInputSchema = rangeSchema.extend({ repositoryId: idSchema, snapshotId: hashSchema });
export type ReadInput = z.infer<typeof readInputSchema>;
export const readOutputSchema = rangeSchema.extend({ snapshotId: hashSchema,
  text: z.string().max(LIMITS.toolOutputBytes), sha256: hashSchema, totalLines: z.int().min(0) });
export type ReadOutput = z.infer<typeof readOutputSchema>;
export const toolSchemas = { 'repo.search': { input: searchInputSchema, output: searchOutputSchema },
  'repo.read': { input: readInputSchema, output: readOutputSchema } } as const;
export type ToolName = keyof typeof toolSchemas;
export const adapterContract = Object.freeze({ server: ADAPTER, adapterVersion: ADAPTER_VERSION,
  contractVersion: CONTRACT, protocol: PROTOCOL, tools: ['repo.search','repo.read'],
  resources: 'host-authorized registered immutable text snapshots', effects: ['snapshot_read'],
  consistency: 'content-addressed; hash checked per binding and read',
  concurrency: { max: 4, sameSnapshotReads: true, resourceApprovalRequired: true },
  timeoutMs: LIMITS.toolMs, retries: 0, limits: LIMITS,
  cancellation: 'Stop scheduling; cancel requests; no rollback guarantee; timed-out outcomes unknown.',
  cache: 'Tool result caches disabled, including cross-run reuse.',
  unsupported: ['sampling','elicitation','resource_links','prompts','unknown tools','server instructions as policy'] });

const exprSchema = z.discriminatedUnion('source', [
  z.strictObject({ source:z.literal('task'), field:z.enum(['repositoryId','snapshotId','query','searchSemantics','pathFilters','contextLines','maxMatches']) }),
  z.strictObject({ source:z.literal('node'), node:idSchema }),
  z.strictObject({ source:z.literal('constant'), value:z.union([z.string().max(512),z.number().finite(),z.boolean(),z.null()]) })
]);
export type Expr = z.infer<typeof exprSchema>;
const base = { id:idSchema, dependsOn:z.array(idSchema).max(LIMITS.nodes) };
export const nodeSchema = z.discriminatedUnion('op', [
  z.strictObject({...base, op:z.literal('GUARD'), check:z.enum(['host_ready','search_valid','reads_valid']), input:exprSchema.nullable()}),
  z.strictObject({...base, op:z.literal('CALL'), tool:z.literal('repo.search'), args:z.record(z.string(),exprSchema)}),
  z.strictObject({...base, op:z.literal('PROJECT'), input:exprSchema, context:exprSchema, projection:z.literal('context_ranges')}),
  z.strictObject({...base, op:z.literal('FILTER'), input:exprSchema, predicate:z.literal('nonempty_range')}),
  z.strictObject({...base, op:z.literal('MERGE'), input:exprSchema, mode:z.enum(['identity','overlapping_intervals'])}),
  z.strictObject({...base, op:z.literal('DEDUP'), input:exprSchema, key:z.literal('path_start_end_hash'), enabled:z.boolean()}),
  z.strictObject({...base, op:z.literal('BOUNDED_MAP'), input:exprSchema, tool:z.literal('repo.read'),
    binding:z.literal('range_and_task_identity'), maxItems:z.int().min(1).max(LIMITS.matches),
    concurrency:z.union([z.literal(1),z.literal(2),z.literal(4)])}),
  z.strictObject({...base, op:z.literal('RETURN'), matches:exprSchema, reads:exprSchema, format:z.literal('context_bundle.v1'), preparation:z.enum(['scan','indexed'])})
]);
export type IRNode = z.infer<typeof nodeSchema>;
export const planSchema = z.strictObject({ irVersion:z.literal(IR_VERSION), taskType:z.literal(TASK),
  taskContractVersion:z.literal('1.0.0'), planVersion:z.int().positive(), origin:z.enum(['manual','mined']),
  adapter:z.literal(ADAPTER), adapterVersion:z.literal(ADAPTER_VERSION), adapterContract:z.literal(CONTRACT),
  limits:z.strictObject({ maxNodes:z.int().min(1).max(LIMITS.nodes), maxCalls:z.int().min(1).max(LIMITS.calls),
    maxDurationMs:z.int().min(1).max(LIMITS.taskMs), maxIntermediateBytes:z.int().min(4096).max(LIMITS.intermediateBytes) }),
  nodes:z.array(nodeSchema).min(1).max(LIMITS.nodes)
});
export type Plan = z.infer<typeof planSchema>;
export const bundleSchema = z.strictObject({ snapshotId:hashSchema, repositoryId:idSchema,
  searchSemantics:z.literal('literal_case_sensitive_line.v1'),
  matches:z.array(matchSchema).max(LIMITS.matches), excerpts:z.array(readOutputSchema).max(LIMITS.matches),
  provenance:z.strictObject({ kind:z.literal('text_matches'), server:z.literal(ADAPTER),
    adapterVersion:z.literal(ADAPTER_VERSION), contract:z.literal(CONTRACT), snapshotVerified:z.literal(true) }),
  completeness:z.enum(['complete','partial']), truncationReasons:z.array(z.enum(['MAX_MATCHES','OUTPUT_BUDGET'])),
  continuation:z.string().max(1024).nullable(), totalMatches:z.int().nonnegative(),
  returnedMatches:z.int().nonnegative(), nextOffset:z.int().nonnegative().nullable()
});
export type Bundle = z.infer<typeof bundleSchema>;
export const statusSchema = z.enum(['COMPLETE','PARTIAL','GUARD_FAILED','POLICY_BLOCKED','INFRASTRUCTURE_ERROR','BYPASS','RETURN_TO_AGENT']);
export type Status = z.infer<typeof statusSchema>;
export type Strategy = 'BYPASS'|'REFERENCE_PLAN'|'OPTIMIZED_PLAN'|'RETURN_TO_AGENT';
export const handoffSchema = z.strictObject({ taskType:z.literal(TASK), planId:hashSchema.nullable(), planVersion:z.int().nullable(),
  reason:z.string().max(160), violatedAssumption:z.string().max(160),
  completedSteps:z.array(z.string()).max(600), failedSteps:z.array(z.string()).max(600),
  unknownOutcomes:z.array(z.string()).max(600), validResults:z.array(z.string()).max(600),
  invalidResults:z.array(z.string()).max(600), remainingGoal:z.string().max(300),
  continuationConstraints:z.array(z.string().max(200)).max(10),
  retainedResultsRef:z.string().max(80).nullable(), detailsTruncated:z.boolean()
});
export type Handoff = z.infer<typeof handoffSchema>;
export const resultSchema = z.strictObject({ version:z.literal('ironmole-result.v1'), runId:z.string().max(64),
  status:statusSchema, strategy:z.enum(['BYPASS','REFERENCE_PLAN','OPTIMIZED_PLAN','RETURN_TO_AGENT']),
  planId:hashSchema.nullable(), bundle:bundleSchema.nullable(), handoff:handoffSchema.nullable() });
export type Result = z.infer<typeof resultSchema>;

export class MoleError extends Error {
  constructor(readonly status:'GUARD_FAILED'|'POLICY_BLOCKED'|'INFRASTRUCTURE_ERROR', readonly code:string,
    readonly unknownOutcome = false) { super(code); this.name='MoleError'; }
}
export function fail(status:MoleError['status'], code:string):never { throw new MoleError(status,code); }
export function checked<T>(schema:z.ZodType<T>, value:unknown, code='SCHEMA_MISMATCH'):T {
  const parsed=schema.safeParse(value);
  if(!parsed.success) fail('GUARD_FAILED',code);
  return parsed.data;
}
