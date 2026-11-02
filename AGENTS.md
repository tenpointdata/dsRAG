# Agent instructions

<!-- tenpointdata-engineering-standards:v1 -->
## Engineering standards

- Keep every TypeScript and web-language code file at or below **600 physical lines**. Split by cohesive responsibility; do not minify, compress, or remove useful context to meet the limit. Fix generators when their checked-in output exceeds the limit.
- Keep code in other languages concise and well organized, with cohesive modules and clear responsibilities.
- Avoid TypeScript `any`, including assertions, generic defaults and declaration files. Reuse canonical domain types and schemas; accept `unknown` at untrusted boundaries and validate/narrow it. Keep strict type checking; do not bypass checks with `@ts-ignore`, `@ts-nocheck` or disabled lint rules.
- Start from the problem, constraints and invariants. Inspect existing libraries, shared types and related project capabilities before building. Extend sound implementations instead of creating parallel ones.
- Keep reusable mechanisms cohesive and independently usable. Keep product rules, UI and external integrations in configuration or thin adapters. Use explicit typed contracts, declarative configuration and composable primitives.
- Preserve the useful capability of shared mechanisms for other consumers. Document assumptions, limits, extension points and compatibility. Use the smallest sound abstraction supported by actual requirements; do not add speculative frameworks or forced cross-project coupling.
- Validate shared contracts, meaningful variations and the affected workflow. For substantial changes, explain what is reusable and what remains project-specific.

## Interaction design

- **Fewer clicks = better design.** Make frequent actions directly available, remove redundant navigation and nonconsequential confirmation steps, apply reversible filters immediately, use sensible defaults and preserve context. Evaluate the complete workflow by interactions required to reach the result.
- Retain confirmations for destructive or consequential actions. Preserve accessibility, clarity and user control.
- Prefer **tables over cards** when users scan, compare or act on multiple items. Choose another layout when it better serves the content, accessibility or task.

## Evidence and documentation

- Preserve existing repository-specific safety, data-access, ownership and workflow rules. Read instructions in the affected directories before changing code.
- Record the source revision, exact checks and results. Distinguish proposed, implemented, locally tested, merged, deployed and accepted behavior; never infer one from another.
- Keep setup, architecture and operational instructions near their owning code. Link the canonical issue for next work instead of maintaining contradictory task lists.
- For a new defect or unresolved design decision, provide evidence, impact, smallest next action and acceptance criteria. Identify hypotheses and untested boundaries explicitly; keep credentials and sensitive data out of issues and logs.
- Before publishing, run applicable source-standards, type, contract and workflow checks using the repository's documented commands. Existing debt remains visible and does not waive the rules for changed code.
<!-- /tenpointdata-engineering-standards:v1 -->

## Repository context

This repository is a public fork: preserve upstream attribution, licenses and fixture provenance. Read `pyproject.toml`, `tests/README.md` and `docs/` before editing. Keep provider/database integrations behind the existing interfaces and prefer offline contract tests. Live model and database suites require explicit test configuration; never infer their success from offline tests. Treat persisted snapshots and retrieved text as untrusted at external boundaries and keep private vulnerability details out of public issues.
