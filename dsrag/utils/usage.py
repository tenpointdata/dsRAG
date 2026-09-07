"""Request-local inference accounting, including work submitted to worker pools."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace
import math
from threading import Lock
from time import perf_counter
from typing import Optional


@dataclass
class UsageTotal:
    provider: str
    model: str
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    input_tokens_complete: bool = True
    output_tokens_complete: bool = True
    cached_input_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


class UsageCollector:
    def __init__(self):
        self._lock = Lock()
        self._totals = {}

    def record(self, provider, model, operation, input_tokens=0, output_tokens=0, calls=1,
               *, input_tokens_complete=True, output_tokens_complete=True,
               cached_input_tokens=None, latency_ms=None):
        cached_input_tokens = cached_token_count(cached_input_tokens, input_tokens)
        latency_ms = measured_latency(latency_ms)
        # Missing measurements cannot be represented by an explicit zero in a
        # combined receipt. Keep their buckets distinct through both collectors.
        key = (provider, model, operation, input_tokens_complete, output_tokens_complete,
               cached_input_tokens is not None, latency_ms is not None)
        with self._lock:
            total = self._totals.setdefault(key, UsageTotal(
                provider, model, operation, input_tokens_complete=input_tokens_complete,
                output_tokens_complete=output_tokens_complete,
                cached_input_tokens=0 if cached_input_tokens is not None else None,
                latency_ms=0.0 if latency_ms is not None else None))
            total.input_tokens += max(0, int(input_tokens or 0))
            total.output_tokens += max(0, int(output_tokens or 0))
            total.calls += calls
            if cached_input_tokens is not None:
                total.cached_input_tokens += cached_input_tokens
            if latency_ms is not None:
                total.latency_ms += latency_ms

    def totals(self):
        with self._lock:
            return [replace(total) for total in self._totals.values()]


_active = ContextVar("dsrag_usage", default=None)


@contextmanager
def collect_usage():
    collector = UsageCollector()
    token = _active.set(collector)
    try:
        yield collector
    finally:
        _active.reset(token)


def record_usage(provider, model, operation, input_tokens=0, output_tokens=0, calls=1,
                 *, input_tokens_complete=True, output_tokens_complete=True,
                 cached_input_tokens=None, latency_ms=None):
    collector = _active.get()
    if collector is not None:
        collector.record(provider, model, operation, input_tokens, output_tokens, calls,
                         input_tokens_complete=input_tokens_complete,
                         output_tokens_complete=output_tokens_complete,
                         cached_input_tokens=cached_input_tokens, latency_ms=latency_ms)


def field(value, name, default=None):
    """SDKs return both attribute objects and dictionaries; absent counts stay zero."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def token_count_complete(value):
    """Zero is a provider receipt; a missing, negative, or malformed count is not."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def cached_token_count(value, input_tokens):
    """Cached tokens are a known subset of inclusive input, never extra tokens."""
    if token_count_complete(input_tokens) and token_count_complete(value) and value <= input_tokens:
        return value
    return None


def measured_latency(value):
    """Missing/invalid timing stays unknown, including non-finite SDK values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value
    return None


def record_response(provider, model, operation, response, *,
                    usage_field="usage", input_field="prompt_tokens",
                    output_field="completion_tokens", latency_ms=None):
    counts = field(response, usage_field) if usage_field else response
    input_tokens = field(counts, input_field)
    output_tokens = field(counts, output_field) if operation == "generate" else 0
    input_complete = token_count_complete(input_tokens)
    output_complete = token_count_complete(output_tokens)
    missing_cache = object()
    cached = field(field(counts, "prompt_tokens_details"), "cached_tokens", missing_cache)
    cached_input_tokens = cached_token_count(cached, input_tokens)
    cache_complete = cached is missing_cache or cached_input_tokens is not None
    record_usage(provider, model, operation,
                 input_tokens if input_complete else 0,
                 output_tokens if output_complete else 0,
                 input_tokens_complete=input_complete and cache_complete,
                 output_tokens_complete=output_complete,
                 cached_input_tokens=cached_input_tokens, latency_ms=latency_ms)


@contextmanager
def metered_call(provider, model, operation):
    """Meter dispatched calls even when the provider fails before a response.

    Only explicit usage attached to SDK errors is retained. Missing receipts
    report unknown token coverage, with measured time and no estimated tokens.
    """
    started = perf_counter()
    reported = False

    def report(response):
        nonlocal reported
        record_response(provider, model, operation, response,
                        latency_ms=(perf_counter() - started) * 1000)
        reported = True

    try:
        yield report
    except Exception as error:
        if not reported:
            candidates = (field(error, "body"), field(error, "response"))
            response = next((candidate for candidate in candidates
                             if field(candidate, "usage") is not None), None)
            report(response)
        raise


class UsageThreadPoolExecutor(ThreadPoolExecutor):
    """Copy each submitter's context, including sidecar meters, into its worker.

    A fresh context per submission avoids entering one Context concurrently.
    Context.run restores the worker afterwards, preventing tenant leakage when
    a pool is reused. Inherited map() also dispatches through submit().
    """
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)
