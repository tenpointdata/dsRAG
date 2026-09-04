"""
Narrowing a Matryoshka embedding on the client, when the server will not.

The failure these guard is silent in both directions. A `dimensions` parameter
sent to an endpoint that rejects it is a 400 on every embed call — loud, at
least. Not truncating when the server ignored the request is worse: a 4,096-wide
vector handed to a 768-wide collection, rejected on every upsert, with a corpus
that simply never fills.

No network. The OpenAI client is stubbed, because what is under test is what
goes on the wire and what comes back off it, not whether Hoonify is up.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.embedding import HoonifyEmbedding, truncate_to_width


class StubEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [mock.Mock(embedding=list(vector)) for vector in self.vectors]
        return mock.Mock(data=data)


def stubbed(vectors, **kwargs):
    """A HoonifyEmbedding whose endpoint is a recorded stub."""
    embedder = HoonifyEmbedding(api_key="test-key", **kwargs)
    embedder.client = mock.Mock(embeddings=StubEmbeddings(vectors))
    return embedder


class TestTruncateToWidth(unittest.TestCase):
    def test_keeps_the_significant_prefix(self):
        # Matryoshka training packs the most significant dimensions first, so
        # the prefix is the embedding — not a sample of it.
        self.assertEqual(len(truncate_to_width([1.0] * 4096, 768)), 768)

    def test_renormalises_what_it_keeps(self):
        # Truncation drops magnitude with the dimensions it removes. A store
        # comparing by dot product would then rank by how much of each vector
        # survived rather than by similarity — and cosine distance, being
        # scale-invariant, would never report the difference.
        narrowed = truncate_to_width([3.0, 4.0, 100.0, 100.0], 2)
        self.assertAlmostEqual(sum(value * value for value in narrowed) ** 0.5, 1.0)
        self.assertAlmostEqual(narrowed[0], 0.6)
        self.assertAlmostEqual(narrowed[1], 0.8)

    def test_a_width_at_or_above_the_vector_is_the_identity(self):
        # Asking a 4,096-wide model to index at 4,096 has narrowed nothing, and
        # a caller that has not narrowed anything must not be told it did —
        # renormalising here would silently rescale every vector the server
        # served.
        vector = [3.0, 4.0]
        self.assertEqual(truncate_to_width(vector, 2), vector)
        self.assertEqual(truncate_to_width(vector, 8), vector)

    def test_an_all_zero_prefix_is_returned_rather_than_divided_by(self):
        # Not a vector any real embedder returns; dividing by its magnitude
        # would be a crash rather than a wrong answer.
        self.assertEqual(truncate_to_width([0.0, 0.0, 5.0], 2), [0.0, 0.0])


class TestHoonifyEmbeddingWidth(unittest.TestCase):
    def test_asks_the_endpoint_when_it_takes_the_parameter(self):
        embedder = stubbed(
            [[1.0] * 768],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=True,
        )

        embedder.get_embeddings("hello")

        self.assertEqual(embedder.client.embeddings.calls[0]["dimensions"], 768)

    def test_never_sends_the_parameter_to_an_endpoint_that_rejects_it(self):
        # Hoonify's Qwen3 endpoint does today. A parameter sent there is a 400
        # on every call, so the width has to be taken on this side instead.
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=False,
        )

        embedder.get_embeddings("hello")

        self.assertNotIn("dimensions", embedder.client.embeddings.calls[0])

    def test_narrows_what_the_endpoint_served_wide(self):
        # The half that matters: without it a 4,096-wide vector reaches a
        # 768-wide collection and every upsert is rejected.
        embedder = stubbed(
            [[1.0] * 4096, [2.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=False,
        )

        vectors = embedder.get_embeddings(["one", "two"])

        self.assertEqual([len(vector) for vector in vectors], [768, 768])

    def test_narrows_a_server_that_ignored_the_parameter_it_accepted(self):
        # Truncation is unconditional rather than a second branch on the flag,
        # so an endpoint that takes `dimensions` and serves something else
        # anyway still lands at the width the collection was built for.
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=True,
        )

        self.assertEqual(len(embedder.get_embeddings("hello")), 768)

    def test_a_single_string_still_answers_with_one_vector(self):
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=False,
        )

        vector = embedder.get_embeddings("hello")

        self.assertIsInstance(vector[0], float)

    def test_the_width_survives_to_dict(self):
        # `to_dict` is written to a KnowledgeBase's config on disk, and a config
        # that lost this flag would rebuild an embedder sending the parameter to
        # an endpoint that rejects it.
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=768,
            dimension_parameter=False,
        )

        self.assertIs(embedder.to_dict()["dimension_parameter"], False)


if __name__ == "__main__":
    unittest.main()
