"""
Token accounting for every model call dsRAG makes.

A RAG pipeline's running cost is almost entirely tokens, and they are spent in
two places that are easy to forget: the embedder, which is called once per
chunk of every document ever ingested, and the small generative calls behind
semantic sectioning and AutoContext, which run per window and per section. A
caller that meters only its own frontier calls is measuring the visible half.

So the counting happens at the boundary rather than at the call sites: every
`LLM.make_llm_call` and every `Embedding.get_embeddings` reports what the
provider said it spent, and a caller collects the total for a unit of work it
chooses — one document, one query, one whole ingest run.

Two properties this keeps:

  1. **Collecting is opt-in and free when nobody is.** `record` with no active
     meter is a dictionary lookup and a return, so a library user who never
     asks for usage pays nothing and sees no behaviour change.
  2. **A provider that reports no counts is visible as such.** Its calls are
     still counted; its tokens stay zero. `calls > 0` with `input_tokens == 0`
     means "this ran and nobody counted it" — which is the truth, and is worth
     more than a plausible estimate nobody can reconcile against an invoice.
"""
import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

#: What a metered call did. `rerank` is here because a cross-encoder is
#: charged per token by every hosted provider that serves one.
OPERATIONS = ("generate", "embed", "rerank")


@dataclass
class UsageTotal:
    """One provider, one model, one kind of call, summed."""

    provider: str
    model: str
    operation: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "calls": self.calls,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
        }


class UsageMeter:
    """
    Totals by (provider, model, operation), safe to share between threads.

    Grouped rather than kept per call because a single document ingest makes
    hundreds of embedding calls and the useful answer is one number per model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: Dict[Tuple[str, str, str], UsageTotal] = {}

    def record(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        calls: int = 1,
    ) -> None:
        if operation not in OPERATIONS:
            raise ValueError(f"unknown usage operation {operation!r}")
        key = (provider, model, operation)
        with self._lock:
            total = self._totals.get(key)
            if total is None:
                total = UsageTotal(provider=provider, model=model, operation=operation)
                self._totals[key] = total
            total.calls += calls
            total.input_tokens += max(0, int(input_tokens))
            total.output_tokens += max(0, int(output_tokens))

    def totals(self) -> List[UsageTotal]:
        with self._lock:
            return [
                UsageTotal(
                    provider=total.provider,
                    model=total.model,
                    operation=total.operation,
                    calls=total.calls,
                    input_tokens=total.input_tokens,
                    output_tokens=total.output_tokens,
                )
                for total in self._totals.values()
            ]

    def to_list(self) -> List[dict]:
        return [total.to_dict() for total in self.totals()]


_active: contextvars.ContextVar[Optional[UsageMeter]] = contextvars.ContextVar(
    "dsrag_usage_meter", default=None
)


@contextmanager
def collect_usage() -> Iterator[UsageMeter]:
    """
    Collect what every model call inside this block spent.

    Nested blocks are legal and the innermost one wins, which is what a caller
    metering one document inside a run that meters the whole batch wants.
    """
    meter = UsageMeter()
    token = _active.set(meter)
    try:
        yield meter
    finally:
        _active.reset(token)


def active_meter() -> Optional[UsageMeter]:
    return _active.get()


def record_usage(
    *,
    provider: str,
    model: str,
    operation: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    calls: int = 1,
) -> None:
    """Report one metered call. A no-op when nobody is collecting."""
    meter = _active.get()
    if meter is None:
        return
    meter.record(
        provider=provider,
        model=model,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=calls,
    )


def in_context(function: Callable) -> Callable:
    """
    Wrap `function` so a worker thread runs under the CALLER's context.

    `ThreadPoolExecutor` does not propagate context variables, so a section
    summary generated on a pool thread would report into no meter at all — and
    those are the calls a document makes most of. Call this once per
    submission: a `Context` cannot be entered twice at the same time, so one
    wrapper shared between workers would raise instead of running.
    """
    context = contextvars.copy_context()

    def run(*args: Any, **kwargs: Any) -> Any:
        return context.run(function, *args, **kwargs)

    return run


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _attribute(carrier: Any, *names: str) -> Any:
    for name in names:
        if isinstance(carrier, dict):
            if name in carrier:
                return carrier[name]
            continue
        found = getattr(carrier, name, None)
        if found is not None:
            return found
    return None


def usage_tokens(response: Any) -> Tuple[int, int]:
    """
    Input and output tokens off a provider response, whatever it calls them.

    Four vocabularies for one pair of numbers: OpenAI's `prompt_tokens` /
    `completion_tokens`, Anthropic's `input_tokens` / `output_tokens`, Ollama's
    `prompt_eval_count` / `eval_count`, and Gemini's `prompt_token_count` /
    `candidates_token_count`. Reading all four here is what lets the recording
    line in each client stay one line, and an unrecognised shape reports zero
    rather than guessing.
    """
    usage = _attribute(response, "usage", "usage_metadata")
    if usage is None:
        usage = response
    return (
        _int(_attribute(usage, "prompt_tokens", "input_tokens", "prompt_eval_count", "prompt_token_count")),
        _int(_attribute(usage, "completion_tokens", "output_tokens", "eval_count", "candidates_token_count")),
    )


def record_response_usage(
    response: Any, *, provider: str, model: str, operation: str
) -> None:
    """Record whatever token counts a provider response carries. Never raises."""
    try:
        input_tokens, output_tokens = usage_tokens(response)
    except Exception:  # noqa: BLE001 - accounting must not break a working call
        input_tokens, output_tokens = 0, 0
    record_usage(
        provider=provider,
        model=model,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
