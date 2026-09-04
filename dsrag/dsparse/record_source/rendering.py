"""
Records to a document whose structure was never guessed.

Everything here is deterministic. That is the point rather than an
implementation detail: the sections a projection produces are the ones the
source already had, so the generative sectioning stage is SKIPPED rather than
run, and a citation locator names a real thing ("Comment by dana, 2026-03-04")
instead of a model's paraphrase of one.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from ..sectioning_and_chunking.semantic_sectioning import str_to_lines
from .types import FIELD_ROLES, GRAINS, Record, RecordDocument, RecordProjection

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


class ProjectionError(ValueError):
    """
    A projection that cannot produce a sound document.

    Raised rather than worked around, per record. A skipped field or a
    part-rendered document indexes successfully, reports success, and then
    abstains forever — which is the failure mode that leaves no evidence.
    """


def _columns_with_role(fields: Dict[str, str], role: str) -> List[str]:
    return [column for column, assigned in fields.items() if assigned == role]


def _validate_fields(fields: Dict[str, str], where: str) -> None:
    for column, role in fields.items():
        if role not in FIELD_ROLES:
            raise ProjectionError(
                f"{where} field '{column}' has role '{role}'; expected one of {', '.join(FIELD_ROLES)}"
            )


def _require_title_columns_are_used(fields: Dict[str, str], title_template: str) -> None:
    """
    A column marked ``title`` must appear in the title template.

    Otherwise the role is silently inert: the column is not embedded, not
    returned as metadata, and not used to name anything, so a misassignment
    reads as a deliberate choice and the column simply disappears.
    """
    for column in _columns_with_role(fields, "title"):
        if "{" + column + "}" not in title_template:
            raise ProjectionError(
                f"field '{column}' has role 'title' but the title template does not name it; "
                "the column would reach nothing"
            )


def _interpolate(template: str, record: Record, where: str) -> str:
    def substitute(match: "re.Match[str]") -> str:
        column = match.group(1)
        if column not in record:
            # Schema drift, caught at the one moment it is still cheap. A
            # renamed column that silently renders as an empty title produces
            # documents nobody can find and nothing goes red.
            raise ProjectionError(f"{where} template names column '{column}', which the record does not carry")
        return _stringify(record[column])

    return _PLACEHOLDER.sub(substitute, template).strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _humanise(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()


def external_id(record: Record, key: Sequence[str]) -> str:
    """
    The stable identity of one record, from its natural key.

    The caller qualifies this with its own source id. It must be derived from
    columns the source will never renumber: a surrogate that changes on every
    sync breaks every citation pointing at the document and silently doubles
    the corpus on the next full refresh.
    """
    if not key:
        raise ProjectionError("projection declares no key; a document with no stable id cannot be re-synced or cited")

    values = []
    for column in key:
        if column not in record:
            raise ProjectionError(f"key column '{column}' is absent from the record")
        value = _stringify(record[column])
        if value == "":
            raise ProjectionError(f"key column '{column}' is empty; the document would have no stable id")
        values.append(value.replace("\\", "\\\\").replace(":", "\\:"))
    # Escaped before joining, because a plain join is not injective: ("a:b", "c")
    # and ("a", "b:c") would name the SAME document, and the collision surfaces
    # as one record overwriting another with no error anywhere.
    return ":".join(values)


class _Builder:
    """
    Accumulates lines and the section boundaries over them.

    The section TITLE is deliberately not written into the lines. It reaches the
    embedding anyway, through the AutoContext chunk header, so writing it into
    the body would embed it twice — and it would smuggle whatever the title
    names past the field roles, which is the one thing they exist to decide.

    The sections PARTITION the lines: every line belongs to exactly one, and
    they run start-to-end with no gap. A separator line between them would read
    better as a document and belong to no section, which is text a consumer
    mapping sections back onto source offsets cannot place — and text that is
    part of no section is text no chunk can carry.
    """

    def __init__(self) -> None:
        self.lines: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []

    def add_section(self, title: str, body: str) -> None:
        body = body.strip()
        if body == "":
            return

        start = len(self.lines)
        self.lines.extend(str_to_lines(body))
        end = len(self.lines) - 1

        content = "\n".join(line["content"] for line in self.lines[start : end + 1])
        self.sections.append({"title": title, "content": content, "start": start, "end": end})


def _narrative_body(record: Record, fields: Dict[str, str], columns: Sequence[str]) -> List[Dict[str, str]]:
    """One entry per narrative column that actually carried text."""
    written = []
    for column in columns:
        if column not in record:
            raise ProjectionError(f"narrative column '{column}' is absent from the record")
        text = _stringify(record[column]).strip()
        if text != "":
            written.append({"title": _humanise(column), "body": text})
    return written


def render_records(
    record: Record,
    projection: RecordProjection,
    children: Optional[List[Record]] = None,
) -> RecordDocument:
    """
    Render one root record — and, for ``aggregate`` grain, its children — into a
    document carrying its own sections.

    Attributes and identifiers are returned BESIDE the text rather than written
    into it. They are the two roles that must not be embedded, and keeping them
    out of the lines is what lets the caller put them in a metadata filter and a
    keyword index, where the question that wants them can actually use them.
    """
    grain = projection.get("grain", "record")
    if grain not in GRAINS:
        raise ProjectionError(f"grain '{grain}' is not one of {', '.join(GRAINS)}")

    fields = projection.get("fields", {})
    _validate_fields(fields, "root")

    child_projection = projection.get("child")
    if grain == "aggregate" and not child_projection:
        raise ProjectionError("aggregate grain declares no child stream; there is nothing to aggregate")
    if grain != "aggregate" and child_projection:
        # Refused rather than ignored: silently folding children into a record
        # document, or silently dropping them, both read as working.
        raise ProjectionError(f"grain '{grain}' declares a child stream; only aggregate grain folds children")
    if child_projection:
        _validate_fields(child_projection.get("fields", {}), "child")

    root_narrative = _columns_with_role(fields, "narrative")
    child_narrative = _columns_with_role(child_projection.get("fields", {}), "narrative") if child_projection else []
    if not root_narrative and not child_narrative:
        # The corollary the whole design rests on: if nothing on the row is
        # meant to be read, nothing on the row belongs in a retrieval index.
        raise ProjectionError("projection names no narrative field; this stream produces facts, not documents")

    title_template = projection.get("title_template", "")
    _require_title_columns_are_used(fields, title_template)
    title = _interpolate(title_template, record, "title")
    if title == "":
        raise ProjectionError("title template rendered empty; a document the reader cannot name is one nobody opens")

    # Resolved HERE rather than at the return, because the child join
    # dereferences the key column below and `external_id` is what validates it.
    # Reversed, a root record that lost its key raised a bare KeyError — the one
    # shape of drift this module promises to fail closed on.
    doc_id = external_id(record, projection.get("key", []))

    builder = _Builder()
    for entry in _narrative_body(record, fields, root_narrative):
        builder.add_section(entry["title"], entry["body"])

    if child_projection:
        _require_title_columns_are_used(
            child_projection.get("fields", {}), child_projection.get("section_title_template", "")
        )
        mine = _joined_children(children or [], child_projection, record, projection.get("key", []))
        for child in _ordered_children(mine, child_projection):
            child_fields = child_projection.get("fields", {})
            section_title = _interpolate(
                child_projection.get("section_title_template", ""), child, "child section title"
            )
            body = "\n\n".join(entry["body"] for entry in _narrative_body(child, child_fields, child_narrative))
            builder.add_section(section_title, body)

    if not builder.sections:
        raise ProjectionError(
            "every narrative field was empty; the document would be a title with no content, "
            "which indexes successfully and then abstains forever"
        )

    # A declared metadata column that stopped arriving is drift, and dropping it
    # quietly ships a document whose filters no longer match what the operator
    # configured — retrievable by nobody, and reported as a success.
    attributes: Dict[str, Any] = {}
    for column in _columns_with_role(fields, "attribute"):
        if column not in record:
            raise ProjectionError(f"attribute column '{column}' is absent from the record")
        attributes[column] = record[column]

    identifiers = []
    for column in _columns_with_role(fields, "identifier"):
        if column not in record:
            raise ProjectionError(f"identifier column '{column}' is absent from the record")
        value = _stringify(record[column])
        if value != "":
            identifiers.append(value)

    return RecordDocument(
        doc_id=doc_id,
        title=title,
        lines=builder.lines,
        sections=builder.sections,
        attributes=attributes,
        identifiers=identifiers,
    )


def _joined_children(
    children: List[Record], child_projection: Dict[str, Any], record: Record, key: Sequence[str]
) -> List[Record]:
    """
    The children of THIS root, by the declared join.

    Without the join every child in the batch renders into every document: one
    ticket's page carries another ticket's comments, which is a passage
    attributed to a record it did not come from. A caller may hand over the
    whole child stream; selecting from it is this function's job.
    """
    on = child_projection.get("on")
    if not on:
        raise ProjectionError("child stream declares no join column; every child would render into every document")

    if len(key) != 1:
        raise ProjectionError(
            f"child join '{on}' pairs with a single key column, but the projection declares {len(key)}; "
            "a composite join would have to guess the pairing"
        )

    root_value = _stringify(record[key[0]])
    for child in children:
        if on not in child:
            raise ProjectionError(f"child join column '{on}' is absent from a child record")

    return [child for child in children if _stringify(child[on]) == root_value]


def _ordered_children(children: List[Record], child_projection: Dict[str, Any]) -> List[Record]:
    """
    Children in the order the projection declared.

    Not cosmetic. Relevant segment extraction returns a RANGE of chunks, on the
    argument that the chunks between two matches carry the rest of the answer —
    the middle steps of a procedure rarely match the question on their own. That
    argument is only true if adjacency means something, so an unordered child
    stream would quietly make segment extraction assemble noise.
    """
    order = child_projection.get("order")
    if not order:
        raise ProjectionError("child stream declares no order column; chunk adjacency would be arbitrary")

    for child in children:
        if order not in child:
            raise ProjectionError(f"child order column '{order}' is absent from a child record")

    numeric = all(not isinstance(child[order], bool) and isinstance(child[order], (int, float)) for child in children)
    if numeric:
        return sorted(children, key=lambda child: child[order])
    return sorted(children, key=lambda child: _stringify(child[order]))
