"""Request-local inference accounting, including work submitted to worker pools."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace
from threading import Lock


@dataclass
class UsageTotal:
    provider: str
    model: str
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class UsageCollector:
    def __init__(self):
        self._lock = Lock()
        self._totals = {}

    def record(self, provider, model, operation, input_tokens=0, output_tokens=0, calls=1):
        key = (provider, model, operation)
        with self._lock:
            total = self._totals.setdefault(key, UsageTotal(*key))
            total.input_tokens += max(0, int(input_tokens or 0))
            total.output_tokens += max(0, int(output_tokens or 0))
            total.calls += calls

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


def record_usage(provider, model, operation, input_tokens=0, output_tokens=0, calls=1):
    collector = _active.get()
    if collector is not None:
        collector.record(provider, model, operation, input_tokens, output_tokens, calls)


def field(value, name):
    """SDKs return both attribute objects and dictionaries; absent counts stay zero."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def record_response(provider, model, operation, response, *,
                    usage_field="usage", input_field="prompt_tokens",
                    output_field="completion_tokens"):
    counts = field(response, usage_field) if usage_field else response
    record_usage(provider, model, operation, field(counts, input_field),
                 field(counts, output_field) if operation == "generate" else 0)


class UsageThreadPoolExecutor(ThreadPoolExecutor):
    """Copy each submitter's context, including sidecar meters, into its worker.

    A fresh context per submission avoids entering one Context concurrently.
    Context.run restores the worker afterwards, preventing tenant leakage when
    a pool is reused. Inherited map() also dispatches through submit().
    """
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)
