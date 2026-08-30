"""
Shared configuration for the Hoonify provider.

Three classes talk to Hoonify — a chat model, an embedder and a reranker — and
they all resolve the same base URL and the same key. Keeping that resolution in
one place is not tidiness: a deployment that points two of the three at a
private endpoint and leaves the third on the public default is a data-egress
bug that no test would catch, because each class works perfectly on its own.
"""
import os

#: Hoonify's public OpenAI-compatible endpoint. Overridden per deployment by
#: HOONIFY_BASE_URL — a customer running Hoonify in their own VPC is a base URL
#: change and nothing more.
DEFAULT_BASE_URL = "https://api.hoonify.ai/v1"

API_KEY_ENV = "HOONIFY_API_KEY"
BASE_URL_ENV = "HOONIFY_BASE_URL"


def base_url() -> str:
    """The configured endpoint, without a trailing slash."""
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def api_key() -> str:
    """
    The configured key.

    Raises rather than returning None: a missing key surfaces here, naming the
    variable, instead of as a 401 from a vendor three frames down.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            f"{API_KEY_ENV} is not set. Hoonify needs a bearer token; set it, or "
            f"choose a different provider."
        )
    return key
