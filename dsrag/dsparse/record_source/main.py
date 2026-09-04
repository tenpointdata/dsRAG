"""
The record source's entrypoint — the sibling of ``parse_and_chunk``.

A file goes: parse → SECTION (generative) → chunk. A record stream goes:
project → section (deterministic) → chunk. The last stage is the same function
in both, deliberately: chunking within a section and never across one is the
property that makes a chunk mean one thing, and a second chunker would be a
second set of boundaries to keep in agreement with the first.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..models.types import Chunk, Section
from ..sectioning_and_chunking.chunking import chunk_document
from .rendering import render_records
from .types import Record, RecordDocument, RecordProjection

logger = logging.getLogger("dsrag.dsparse")


def parse_and_chunk_records(
    record: Record,
    projection: RecordProjection,
    children: Optional[List[Record]] = None,
    chunking_config: Optional[Dict[str, Any]] = None,
    kb_id: str = "",
    doc_id: str = "",
) -> Tuple[List[Section], List[Chunk], RecordDocument]:
    """
    Turn one record — and, for ``aggregate`` grain, its children — into sections
    and chunks.

    Returns the rendered document alongside them, which is the one thing this
    path has that a file path does not: attributes and identifiers that were
    deliberately kept OUT of the embedded text, for the caller to put in a
    metadata filter and a keyword index.

    No LLM is called. A caller that wants an AutoContext document summary still
    gets one from ``auto_context``; what it does not pay for is a model
    rediscovering boundaries the loader already knew.
    """
    chunking_config = chunking_config or {}
    chunk_size = chunking_config.get("chunk_size", 800)
    min_length_for_chunking = chunking_config.get("min_length_for_chunking", 1600)

    base_extra = {"kb_id": kb_id, "doc_id": doc_id, "grain": projection.get("grain", "record")}
    started = time.perf_counter()

    document = render_records(record=record, projection=projection, children=children)
    chunks = chunk_document(
        sections=document["sections"],
        document_lines=document["lines"],
        chunk_size=chunk_size,
        min_length_for_chunking=min_length_for_chunking,
    )

    logger.info(
        "Record projection complete",
        extra={
            **base_extra,
            "duration_s": round(time.perf_counter() - started, 4),
            "num_sections": len(document["sections"]),
            "num_chunks": len(chunks),
            "num_children": len(children or []),
        },
    )

    return document["sections"], chunks, document
