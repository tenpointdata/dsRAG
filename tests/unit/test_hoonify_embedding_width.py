"""
Which width goes on the wire, and which width comes back.

Hoonify's Qwen3 endpoint rejects the `dimensions` request parameter, so the
only honest thing to do with it is send nothing and index at the width the
endpoint serves. Nothing narrows the vector on this side: a prefix would rest
on the model being Matryoshka-trained, and one taken from a model that is not
is a vector of the right length carrying an arbitrary slice of the wrong space
— which retrieves badly and reports nothing at all.

So `dimension` has to be the width the server actually returns. These pin the
two ways that can go wrong: a parameter sent where it is refused, and a vector
quietly reshaped on the way back.

No network. The OpenAI client is stubbed, because what is under test is what
goes on the wire and what comes back off it, not whether Hoonify is up.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.embedding import HoonifyEmbedding, dimensionality


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


class TestHoonifyEmbeddingWidth(unittest.TestCase):
    def test_never_sends_the_parameter_to_an_endpoint_that_rejects_it(self):
        # Hoonify's Qwen3 endpoint does. A parameter sent there is a 400 on
        # every call, so the width is simply not asked for.
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=4096,
            dimension_parameter=False,
        )

        embedder.get_embeddings("hello")

        self.assertNotIn("dimensions", embedder.client.embeddings.calls[0])

    def test_asks_the_endpoint_when_it_takes_the_parameter(self):
        embedder = stubbed(
            [[1.0] * 768],
            model="some/matryoshka-model",
            dimension=768,
            dimension_parameter=True,
        )

        embedder.get_embeddings("hello")

        self.assertEqual(embedder.client.embeddings.calls[0]["dimensions"], 768)

    def test_returns_the_served_vector_untouched(self):
        # The half that matters after dropping truncation: what the endpoint
        # served is what gets indexed. Reshaping it here would mean claiming a
        # width the server never produced.
        served = [float(index) for index in range(4096)]
        embedder = stubbed(
            [served],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=4096,
            dimension_parameter=False,
        )

        vector = embedder.get_embeddings("hello")

        self.assertEqual(len(vector), 4096)
        self.assertEqual(vector, served)

    def test_a_batch_comes_back_in_order_and_at_full_width(self):
        embedder = stubbed(
            [[1.0] * 4096, [2.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=4096,
            dimension_parameter=False,
        )

        vectors = embedder.get_embeddings(["one", "two"])

        self.assertEqual([len(vector) for vector in vectors], [4096, 4096])
        self.assertEqual(vectors[0][0], 1.0)
        self.assertEqual(vectors[1][0], 2.0)

    def test_a_single_string_answers_with_one_vector(self):
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=4096,
            dimension_parameter=False,
        )

        vector = embedder.get_embeddings("hello")

        self.assertIsInstance(vector[0], float)

    def test_the_qwen_default_width_is_the_one_it_serves(self):
        # Looked up when the caller passes none. Getting it wrong is not a
        # runtime error — it is a vector store built at the wrong width,
        # discovered at the first query.
        self.assertEqual(dimensionality["Qwen/Qwen3-Embedding-8B"], 4096)
        self.assertEqual(
            HoonifyEmbedding(model="Qwen/Qwen3-Embedding-8B", api_key="k").dimension,
            4096,
        )

    def test_the_flag_survives_to_dict(self):
        # `to_dict` is written to a KnowledgeBase's config on disk, and a config
        # that lost this flag would rebuild an embedder sending the parameter to
        # an endpoint that rejects it.
        embedder = stubbed(
            [[1.0] * 4096],
            model="Qwen/Qwen3-Embedding-8B",
            dimension=4096,
            dimension_parameter=False,
        )

        self.assertIs(embedder.to_dict()["dimension_parameter"], False)


if __name__ == "__main__":
    unittest.main()
