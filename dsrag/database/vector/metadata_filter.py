"""
One metadata-filter vocabulary, and the dialect each vector store speaks.

Four stores each hand-wrote the same eight operators, and the copies drifted
where it mattered most — at the point an operator is not recognised. Chroma
raised ``KeyError``, Pinecone silently built ``{field: {None: value}}`` and
queried on it, Postgres raised ``ValueError``, and Milvus raised ``KeyError``.
The vocabulary lives here once, so adding a store means registering a dialect
rather than restating the list, and an unrecognised operator names itself the
same way everywhere.
"""

from typing import Any, Optional, get_args

from dsrag.database.vector.types import (
    MetadataFilter,
    MetadataFilterOperator,
    MetadataFilterValue,
)

SUPPORTED_OPERATORS: tuple[MetadataFilterOperator, ...] = get_args(MetadataFilterOperator)

# The Mongo-style comparison dialect, spoken by Chroma and Pinecone.
_MONGO_OPERATORS: dict[MetadataFilterOperator, str] = {
    "equals": "$eq",
    "not_equals": "$ne",
    "in": "$in",
    "not_in": "$nin",
    "greater_than": "$gt",
    "less_than": "$lt",
    "greater_than_equals": "$gte",
    "less_than_equals": "$lte",
}

# SQL comparison operators, for stores queried through SQL.
_SQL_OPERATORS: dict[MetadataFilterOperator, str] = {
    "equals": "=",
    "not_equals": "!=",
    "in": "IN",
    "not_in": "NOT IN",
    "greater_than": ">",
    "less_than": "<",
    "greater_than_equals": ">=",
    "less_than_equals": "<=",
}

# Milvus boolean-expression operators.
_EXPRESSION_OPERATORS: dict[MetadataFilterOperator, str] = {
    "equals": "==",
    "not_equals": "!=",
    "in": "in",
    "not_in": "not in",
    "greater_than": ">",
    "less_than": "<",
    "greater_than_equals": ">=",
    "less_than_equals": "<=",
}


def _translate(
    dialect: dict[MetadataFilterOperator, str], operator: str
) -> str:
    """Look an operator up in one dialect, or say which operators exist."""
    try:
        return dialect[operator]
    except KeyError:
        raise ValueError(
            f"Unsupported metadata filter operator: {operator!r}. "
            f"Supported operators are: {', '.join(SUPPORTED_OPERATORS)}"
        ) from None


def _parts(
    metadata_filter: MetadataFilter,
) -> tuple[str, MetadataFilterOperator, MetadataFilterValue]:
    return (
        metadata_filter["field"],
        metadata_filter["operator"],
        metadata_filter["value"],
    )


def to_mongo_style_filter(
    metadata_filter: MetadataFilter, *, bare_equality: bool = False
) -> dict[str, Any]:
    """
    Format a filter for a store that takes Mongo-style comparison operators.

    Args:
        metadata_filter: The filter to translate.
        bare_equality: Write equality as ``{field: value}`` rather than
            ``{field: {"$eq": value}}``. Pinecone accepts both; Chroma does not.

    Returns:
        The filter in the store's own shape.
    """
    field, operator, value = _parts(metadata_filter)
    formatted_operator = _translate(_MONGO_OPERATORS, operator)
    if bare_equality and operator == "equals":
        return {field: value}
    return {field: {formatted_operator: value}}


def to_sql_filter(metadata_filter: MetadataFilter) -> str:
    """
    Format a filter as a parameterised SQL predicate over a ``metadata`` column.

    The value itself is left as a placeholder — it is passed separately, so a
    filter value can never be read as SQL.

    Returns:
        A predicate such as ``metadata->>'year' >= %s``.
    """
    field, operator, value = _parts(metadata_filter)
    sql_operator = _translate(_SQL_OPERATORS, operator)
    placeholder = (
        f"({', '.join(['%s'] * len(value))})" if isinstance(value, list) else "%s"
    )
    return f"metadata->>'{field}' {sql_operator} {placeholder}"


def to_expression_filter(metadata_filter: Optional[MetadataFilter]) -> str:
    """
    Format a filter as a Milvus boolean expression.

    An absent filter is the empty expression, which Milvus reads as "no filter".
    """
    if not metadata_filter:
        return ""
    field, operator, value = _parts(metadata_filter)
    formatted_operator = _translate(_EXPRESSION_OPERATORS, operator)
    literal = f'"{value}"' if isinstance(value, str) else value
    return f"metadata['{field}'] {formatted_operator} {literal}"
