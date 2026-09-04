"""
Types for the record source — a third way into dsparse, beside a file and a
string.

A file arrives as bytes whose structure has to be discovered; that is what
semantic sectioning is for, and it costs an LLM call per window. A stream of
records arrives with its structure ALREADY KNOWN: the loader that produced it
knew where one ticket ended and the next began, which columns were prose and
which were measurements, and which value identified the row.

Throwing that away, serialising the stream to one blob and asking a model to
find the boundaries again, is the defect this module exists to remove. It is
also the expensive one — sectioning is the only generative stage on the ingest
path, and on an export-sized document it is spent finding the boundaries the
loader already handed us.

A projection is DATA. It is not a subclass, a plugin, or a callback: an install
that needed code here would be a per-install code path, and there is no version
of that which stays maintainable across a few hundred sources.
"""

from typing import Any, Dict, List, Optional, TypedDict

Record = Dict[str, Any]

#: What one document is, for one stream.
#:
#: ``fact`` is deliberately absent. A stream whose rows are measurements should
#: never reach this module at all — it produces no documents, and the caller
#: routes it to a tabular store instead. Accepting it here would mean embedding
#: three hundred invoice rows and answering an aggregate question from whichever
#: nine of them a reranker liked.
GRAINS = ("record", "aggregate")

#: Where a column's value goes. Exactly one role per column.
#:
#: ``narrative`` is the only role whose text is embedded. ``attribute`` and
#: ``identifier`` are returned to the caller as structured metadata instead:
#: embedding ``status=paid`` places a point next to every other row that says
#: paid, which crowds the index and discriminates nothing, and an opaque id
#: embeds to noise while the query that wants it wants it exactly.
FIELD_ROLES = ("narrative", "title", "attribute", "identifier", "ignore")


class ChildProjection(TypedDict, total=False):
    """A child stream folded into its root's document, for ``aggregate`` grain."""

    #: The child column joining to the root's key.
    on: str
    #: The child column the rendered sections are ordered by. Adjacency has to
    #: mean something: relevant segment extraction reads a RANGE of chunks, so
    #: an arbitrary order makes the chunks between two matches meaningless.
    order: str
    fields: Dict[str, str]
    #: Interpolates ``{column}`` from the child record. Becomes the section
    #: title, which is what a citation locator shows the reader.
    section_title_template: str


class RecordProjection(TypedDict, total=False):
    """How one stream becomes documents."""

    grain: str
    #: The natural key. Never a loader-generated surrogate: those change on
    #: every sync, so every citation breaks on the next full refresh and the
    #: corpus silently doubles.
    key: List[str]
    fields: Dict[str, str]
    #: Interpolates ``{column}`` from the root record.
    title_template: str
    child: Optional[ChildProjection]


class RecordDocument(TypedDict):
    """
    A rendered document, with the structure the projection already knew.

    ``attributes`` and ``identifiers`` never appear in ``lines``: they are the
    two roles that must not be embedded, and returning them separately is what
    lets the caller put them in a metadata filter and a keyword index instead.
    """

    doc_id: str
    title: str
    lines: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    attributes: Dict[str, Any]
    identifiers: List[str]
