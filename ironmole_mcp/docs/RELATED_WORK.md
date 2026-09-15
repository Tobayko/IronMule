# Related work and version choices

Sources checked on 2026-09-12. These are relevant precedents, not claims of novelty
or independently reproduced performance. This MVP does not implement the papers.

## Programmatic Tool Calling

Anthropic's [official Programmatic Tool Calling documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
describes model-authored code orchestrating tools, including parallel calls. Its
`allowed_callers` presentation flag is not a hard authorization boundary. The current
page also excludes MCP-connector tools from direct programmatic use; a compatible
client-side tool bridge and actual supported model execution must be established
before a B benchmark can be called PTC. Eliminating model roundtrips or writing a
local loop does not distinguish IronMole from this prior approach. This project has
not executed such a model/provider integration.

## ReUseIt

[ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation](https://arxiv.org/abs/2510.14308)
learns reusable workflows from successful and failed attempts and adds execution
guards, recovery actions and user-facing failure handling. This is close prior work
for guarded task reuse. Its web tasks and measured outcomes do not transfer to this
repository-search contract. IronMole uses deterministic closed transformations and
returns a bounded handoff; it has not replicated ReUseIt's synthesis or web evaluation.

## SkillRT and later SkVM naming

[SkillRT v1](https://arxiv.org/abs/2604.03088v1) describes capability-oriented skill
compilation, environment binding, concurrency extraction and runtime code solidification.
The [later version of the same paper](https://arxiv.org/abs/2604.03088) is titled
*SkVM: Revisiting Language VM for Skills across Heterogenous LLMs and Harnesses*.
The version distinction prevents conflating the requested SkillRT paper with an
unrelated Rust project using a similar name. This MVP makes no compiler/runtime
novelty or transferred speedup claim. It only mines a predeclared typed task template.

## Official MCP and Codex compatibility

The [official TypeScript SDK repository](https://github.com/modelcontextprotocol/typescript-sdk)
separates v2 packages from the v1 branch and documentation. This prototype pins
`@modelcontextprotocol/sdk` **1.29.0**, the complete version locally available for
reproducible installation; the inspected v1 branch's
[package metadata](https://github.com/modelcontextprotocol/typescript-sdk/blob/v1.x/package.json)
reports 1.30.0. The pin is not represented as the latest SDK. TypeScript 6.0.2, Zod
4.4.3, Ajv 8.20.0 and Node types 25.5.0 are pinned as well. `jose` 6.2.4 and
`range-parser` 1.2.1 are explicit compatible cached transitive pins; the lockfile
records the complete installation. No install scripts or native SQLite addon run.

The installed SDK advertises 2025-11-25 as its latest supported protocol. The inner
client verifies the negotiated version and exact tool catalogue. Older/newer protocol
variants and v2 cache features are not mixed in. See the official
[lifecycle specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
[SDK server guide](https://ts.sdk.modelcontextprotocol.io/server) and
[client guide](https://ts.sdk.modelcontextprotocol.io/client).

[Official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
documents stdio commands, argument arrays, startup/tool deadlines and tool allowlists.
The generated configuration uses these supported fields. Its actual CLI parsing and
SDK initialize/list/call behavior are tested; model-mediated use in an interactive
Codex conversation has not been measured. A successful SDK handshake establishes
protocol interoperability, not improved agent reasoning or total task cost.
