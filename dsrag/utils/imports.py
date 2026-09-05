"""
Utilities for lazy imports of optional dependencies.

This module owns the one ``LazyLoader`` implementation. ``dsparse`` re-uses it
rather than keeping a fork — the two copies had already drifted apart in the
install hint they print, which is the one sentence a user actually reads when a
dependency is missing.
"""
import importlib
from typing import Optional


class LazyLoader:
    """
    Lazily import a module only when its attributes are accessed.

    This allows optional dependencies to be imported only when actually used,
    rather than at module import time.

    Usage:
        # Instead of: import chromadb
        chromadb = LazyLoader("chromadb")

        # Then use chromadb as normal - it will only be imported when accessed
        # If the module is not installed, a helpful error message is shown
    """

    def __init__(
        self,
        module_name: str,
        package_name: Optional[str] = None,
        extra_of: str = "dsrag",
    ):
        """
        Initialize a lazy loader for a module.

        Args:
            module_name: The name of the module to import
            package_name: Optional package name for pip install instructions
                         (defaults to module_name if not provided)
            extra_of: The distribution whose extras offer this dependency, named
                      in the install hint. ``dsparse`` loads under its own name.
        """
        self._module_name = module_name
        self._package_name = package_name or module_name
        self._extra_of = extra_of
        self._module = None

    def __getattr__(self, name):
        """Called when an attribute is accessed."""
        if self._module is None:
            try:
                # Use importlib.import_module instead of __import__ for better handling of nested modules
                self._module = importlib.import_module(self._module_name)
            except ImportError:
                raise ImportError(
                    f"The '{self._module_name}' module is required but not installed. "
                    f"Please install it with: pip install {self._package_name} "
                    f"or pip install {self._extra_of}[{self._package_name}]"
                )

        # Try to get the attribute from the module
        try:
            return getattr(self._module, name)
        except AttributeError:
            # If the attribute is not found, it might be a nested module
            try:
                # Try to import the nested module
                nested_module = importlib.import_module(f"{self._module_name}.{name}")
                # Cache it on the module for future access
                setattr(self._module, name, nested_module)
                return nested_module
            except ImportError:
                # If that fails, re-raise the original AttributeError
                raise AttributeError(
                    f"Module '{self._module_name}' has no attribute '{name}'"
                )


# Create lazy loaders for commonly used optional dependencies
instructor = LazyLoader("instructor")
openai = LazyLoader("openai")
cohere = LazyLoader("cohere")
voyageai = LazyLoader("voyageai")
ollama = LazyLoader("ollama")
anthropic = LazyLoader("anthropic")
genai = LazyLoader("google.generativeai", "google-generativeai")
genai_new = LazyLoader("google.genai", "google-genai")
boto3 = LazyLoader("boto3")
faiss = LazyLoader("faiss", "faiss-cpu")
psycopg2 = LazyLoader("psycopg2", "psycopg2-binary")
pgvector = LazyLoader("pgvector")

# On-device inference. Both backends are optional and mutually exclusive in
# practice: onnxruntime is the appliance path (CPU, quantized, no torch), and
# sentence-transformers is the convenience path for a machine that already has
# torch installed.
onnxruntime = LazyLoader("onnxruntime")
tokenizers = LazyLoader("tokenizers")
sentence_transformers = LazyLoader("sentence_transformers")
huggingface_hub = LazyLoader("huggingface_hub")
