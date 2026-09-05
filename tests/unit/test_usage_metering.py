"""
Token accounting at the model boundary.

Every one of these asserts a property a bill is reconciled against, not a
wiring detail: what a caller collects, what a provider that reports nothing
looks like, and that the calls made on a worker thread are counted at all —
which is where a document spends most of its generative tokens.
"""
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dsrag.utils.usage import (  # noqa: E402
    UsageMeter,
    collect_usage,
    in_context,
    record_response_usage,
    record_usage,
    usage_tokens,
)


def openai_response(prompt: int, completion: int):
    return types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )


def test_collects_only_inside_the_block():
    record_usage(provider="hoonify", model="m", operation="generate", input_tokens=9)

    with collect_usage() as meter:
        record_usage(provider="hoonify", model="m", operation="generate", input_tokens=3)

    record_usage(provider="hoonify", model="m", operation="generate", input_tokens=9)

    assert [total.to_dict() for total in meter.totals()] == [
        {
            "provider": "hoonify",
            "model": "m",
            "operation": "generate",
            "calls": 1,
            "inputTokens": 3,
            "outputTokens": 0,
        }
    ]


def test_groups_by_provider_model_and_operation():
    with collect_usage() as meter:
        record_usage(provider="p", model="a", operation="generate", input_tokens=10, output_tokens=2)
        record_usage(provider="p", model="a", operation="generate", input_tokens=5, output_tokens=1)
        record_usage(provider="p", model="a", operation="embed", input_tokens=7)
        record_usage(provider="p", model="b", operation="generate", input_tokens=1)

    by_key = {(t.model, t.operation): t for t in meter.totals()}
    assert by_key[("a", "generate")].calls == 2
    assert by_key[("a", "generate")].input_tokens == 15
    assert by_key[("a", "generate")].output_tokens == 3
    assert by_key[("a", "embed")].input_tokens == 7
    assert by_key[("b", "generate")].calls == 1


def test_a_provider_that_reports_nothing_is_still_counted():
    """`calls > 0` with zero tokens is the honest reading of an unmetered call."""
    with collect_usage() as meter:
        record_response_usage(
            types.SimpleNamespace(), provider="ondevice", model="bge", operation="embed"
        )

    total = meter.totals()[0]
    assert total.calls == 1
    assert (total.input_tokens, total.output_tokens) == (0, 0)


@pytest.mark.parametrize(
    "usage, expected",
    [
        (types.SimpleNamespace(prompt_tokens=4, completion_tokens=5), (4, 5)),
        (types.SimpleNamespace(input_tokens=6, output_tokens=7), (6, 7)),
        ({"prompt_eval_count": 8, "eval_count": 9}, (8, 9)),
        (types.SimpleNamespace(prompt_token_count=1, candidates_token_count=2), (1, 2)),
    ],
)
def test_reads_every_provider_vocabulary(usage, expected):
    assert usage_tokens(types.SimpleNamespace(usage=usage)) == expected


def test_gemini_usage_metadata_is_read_from_its_own_field():
    response = types.SimpleNamespace(
        usage_metadata=types.SimpleNamespace(prompt_token_count=11, candidates_token_count=3)
    )
    assert usage_tokens(response) == (11, 3)


def test_a_worker_thread_reports_into_the_collecting_meter():
    """
    The property section summaries and semantic windows depend on.

    Without `in_context` a pool thread starts with no context variables, and
    every generative call a document makes in parallel would be invisible.
    """
    with collect_usage() as meter:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    in_context(record_usage),
                    provider="hoonify",
                    model="m",
                    operation="generate",
                    input_tokens=2,
                )
                for _ in range(8)
            ]
            for future in futures:
                future.result()

    total = meter.totals()[0]
    assert total.calls == 8
    assert total.input_tokens == 16


def test_meter_totals_are_a_snapshot_not_a_live_view():
    meter = UsageMeter()
    meter.record(provider="p", model="m", operation="embed", input_tokens=1)
    snapshot = meter.totals()
    meter.record(provider="p", model="m", operation="embed", input_tokens=1)
    assert snapshot[0].input_tokens == 1


def test_an_unknown_operation_is_refused():
    with pytest.raises(ValueError):
        UsageMeter().record(provider="p", model="m", operation="summarise")


def test_nested_collection_reports_to_the_innermost_meter():
    with collect_usage() as outer:
        record_usage(provider="p", model="m", operation="embed", input_tokens=1)
        with collect_usage() as inner:
            record_usage(provider="p", model="m", operation="embed", input_tokens=4)
        record_usage(provider="p", model="m", operation="embed", input_tokens=1)

    assert inner.totals()[0].input_tokens == 4
    assert outer.totals()[0].input_tokens == 2


def test_the_hoonify_embedder_records_what_the_endpoint_charged():
    """The embedder is the call a corpus makes once per chunk, forever."""
    from unittest import mock

    from dsrag.embedding import HoonifyEmbedding

    embedder = HoonifyEmbedding(api_key="test-key", model="bge-m3", dimension=1024)
    embedder.client = mock.Mock()
    embedder.client.embeddings.create.return_value = types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=[0.0] * 1024)],
        usage=types.SimpleNamespace(prompt_tokens=42),
    )

    with collect_usage() as meter:
        embedder.get_embeddings(["one chunk"])

    total = meter.totals()[0]
    assert (total.provider, total.model, total.operation) == ("hoonify", "bge-m3", "embed")
    assert total.input_tokens == 42


def test_the_hoonify_chat_client_records_both_directions():
    from unittest import mock

    from dsrag.llm import HoonifyChatAPI

    model = HoonifyChatAPI(model="llama-3.3-70b-instruct", api_key="k", base_url="http://x")
    completion = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=" hi "))],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )

    with mock.patch("dsrag.llm.openai") as openai_module:
        openai_module.OpenAI.return_value.chat.completions.create.return_value = completion
        with collect_usage() as meter:
            assert model.make_llm_call([{"role": "user", "content": "hey"}]) == "hi"

    total = meter.totals()[0]
    assert (total.input_tokens, total.output_tokens) == (100, 20)
