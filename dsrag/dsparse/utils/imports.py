"""
Lazy imports for the optional dependencies dsparse itself reaches for.

The loader class is the shared one; only the table below is dsparse's own. A
missing dependency names ``dsparse`` in its install hint, because that is the
distribution a dsparse user installed.
"""
from functools import partial

from dsrag.utils.imports import LazyLoader

_loader = partial(LazyLoader, extra_of="dsparse")

# Create lazy loaders for dependencies used in dsparse
instructor = _loader("instructor")
openai = _loader("openai")
anthropic = _loader("anthropic")
genai = _loader("google.generativeai", "google-generativeai")
genai_new = _loader("google.genai", "google-genai")
vertexai = _loader("vertexai")
boto3 = _loader("boto3")
