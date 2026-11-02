# Chunk metadata compatibility

Chunk adapters share `decode_chunk_metadata` in
`dsrag/database/chunk/metadata_codec.py`. It keeps the read contract consistent
without changing the persisted format or rewriting existing knowledge bases.

| Stored form | Read result |
| --- | --- |
| A native dictionary, including DynamoDB maps | The dictionary, preserving native scalar types |
| JSON object text | A dictionary |
| Legacy Python dictionary-literal text | A dictionary, preserving literal container values |
| Missing, empty text, or an encoded null value | `None` |
| Non-object data or malformed text | A `ValueError` from the decoder |

SQLite and PostgreSQL surface decoder errors. DynamoDB retains its existing
adapter-level error handling and returns `None` when a document cannot be read.
Error messages do not repeat stored metadata content.

This change preserves the current write formats. It does not migrate local
pickle snapshots, add serialization versions, enforce metadata size/depth
budgets, or validate a caller's domain-specific field schema. Those remain
separate work under [the storage contract issue](https://github.com/tenpointdata/dsRAG/issues/11).
Keep untrusted imports behind an explicit validation boundary; a data-only
decoder is not a general resource sandbox. Python documents the resource limits
of literal parsing in its [AST reference](https://docs.python.org/3/library/ast.html#ast.literal_eval).

The focused offline regression suite uses temporary SQLite storage and fake
PostgreSQL/DynamoDB clients:

```sh
python -m unittest tests.unit.test_chunk_metadata_contract
```

It exercises JSON/legacy compatibility, nulls, wrong shapes, native decimal
values and the absence of general expression evaluation. No cloud account,
model provider, database service or execution payload is required.
