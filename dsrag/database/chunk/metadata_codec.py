"""Data-only decoding shared by chunk backends without changing stored formats."""

import ast
import json
from typing import Optional


def decode_chunk_metadata(value: object) -> Optional[dict]:
    """Read native maps, JSON, or legacy dictionary literals.

    Text must describe data, not an expression that needs a namespace. Keep the
    historical literal format readable while consumers migrate their stores.
    Native maps retain backend-specific scalar values, such as Decimal.
    This is not a resource-limit boundary; callers still own input-size policy.
    """
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Chunk metadata must be a dictionary or null")
    if not value.strip():
        return None

    try:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError, TypeError, RecursionError):
        raise ValueError("Chunk metadata must contain dictionary data or null") from None

    if decoded is not None and not isinstance(decoded, dict):
        raise ValueError("Chunk metadata must be a dictionary or null")
    return decoded
