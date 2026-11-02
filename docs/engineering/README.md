# Engineering guide

This public fork provides retrieval, parsing, model and storage interfaces. Keep upstream attribution and the MIT license. [docs/](../) describes the reusable library; [tests/README.md](../../tests/README.md) owns fixture provenance and test-corpus rules.

The root [AGENTS.md](../../AGENTS.md) records the shared 600-line web-source rule, explicit-any prohibition, reusable architecture principles and interaction preferences. Existing repository-specific rules remain authoritative.

## Setup and verification

| Workflow | Command | Evidence and requirements |
| --- | --- | --- |
| Core library | `python -m pip install -e .` | Python 3.9+ and declared dependencies; not installed by this review |
| Optional providers/stores | Install only the needed extras from `pyproject.toml` | Avoid all-provider setup for offline checks |
| Full tests | `pytest` | Includes provider/database tests; requires explicit environment and can incur external calls |
| Offline corpus fixture suite | `python -m unittest tests.unit.test_fetch_corpora` | PASS: 20 tests using a loopback server |
| Offline hardware/filter sample | `python -m unittest tests.unit.test_hardware tests.unit.test_metadata_filter` | 25 tests; two macOS fixture errors, tracked in #12 |

## Architecture and ownership

| Capability | Owning code | Invariant |
| --- | --- | --- |
| Retrieval pipeline | `dsrag/knowledge_base.py` (`query`), `dsrag/rse.py` | Keep query and ingest representation compatible |
| Parsing and sectioning | `dsrag/dsparse/` | Preserve document and citation provenance |
| Provider interfaces | `dsrag/embedding.py`, `dsrag/reranker.py`, `dsrag/llm.py` | Model-specific behavior belongs in adapters |
| Storage adapters | `dsrag/database/`, `dsrag/metadata.py` | One metadata/filter contract across backends |
| Fixtures and corpora | `tests/README.md`, `tests/fetch_corpora.py` | Preserve fixture bytes; optional downloads stay local |

## Security and trust boundaries

The review inspected storage decoding, metadata filters, provider boundaries and CI configuration. A persisted snapshot is a trust boundary: do not load untrusted or externally modified stores until the decoding contract is validated. Storage migration work is tracked in #11. Sensitive implementation findings are retained privately rather than expanded in this public document. No malicious payload or production probe was run.

## Operations and release evidence

Consumers select a library revision and optional backend/provider extras. Before an upgrade, record the source SHA, dependency environment, metadata schema, embedding model and output width; preserve a restorable store snapshot. Dimension changes require coordinated re-embedding, not merely a package update. PR #10 is proposed Matryoshka support and was not treated as merged.

## Roadmap and maintenance

Use [the dated review](REVIEW-2026-11-02.md) for next work and evidence. GitHub issues remain authoritative for status. Update setup, contracts and release evidence when behavior changes; distinguish prepared, locally tested, merged, deployed and accepted behavior. Record design decisions with the problem, invariants, alternatives, decision and consequences.
