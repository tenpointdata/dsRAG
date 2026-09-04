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
        values.append(value)
    return ":".join(values)


class _Builder:
    """Accumulates lines and the section boundaries over them."""

    def __init__(self) -> None:
        self.lines: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []

    def add_section(self, title: str, body: str) -> None:
        body = body.strip()
        if body == "":
            return

        if self.lines:
            self.lines.extend(str_to_lines(""))

        start = len(self.lines)
        self.lines.extend(str_to_lines(f"## {title}" if title else ""))
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
    if child_projection:
        _validate_fields(child_projection.get("fields", {}), "child")

    root_narrative = _columns_with_role(fields, "narrative")
    child_narrative = _columns_with_role(child_projection.get("fields", {}), "narrative") if child_projection else []
    if not root_narrative and not child_narrative:
        # The corollary the whole design rests on: if nothing on the row is
        # meant to be read, nothing on the row belongs in a retrieval index.
        raise ProjectionError("projection names no narrative field; this stream produces facts, not documents")

    title = _interpolate(projection.get("title_template", ""), record, "title")
    if title == "":
        raise ProjectionError("title template rendered empty; a document the reader cannot name is one nobody opens")

    builder = _Builder()
    for entry in _narrative_body(record, fields, root_narrative):
        builder.add_section(entry["title"], entry["body"])

    if child_projection:
        for child in _ordered_children(children or [], child_projection):
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

    attributes: Dict[str, Any] = {}
    for column in _columns_with_role(fields, "attribute"):
        if column in record:
            attributes[column] = record[column]

    identifiers = [
        value
        for value in (_stringify(record.get(column)) for column in _columns_with_role(fields, "identifier"))
        if value != ""
    ]

    return RecordDocument(
        doc_id=external_id(record, projection.get("key", [])),
        title=title,
        lines=builder.lines,
        sections=builder.sections,
        attributes=attributes,
        identifiers=identifiers,
    )


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

    return sorted(children, key=lambda child: _stringify(child[order]))
