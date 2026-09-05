"""
Which width goes on the wire, and which width comes back.

Hoonify's Qwen3 endpoint rejects the `dimensions` request parameter, so the
only honest thing to do with it is send nothing and take the width the endpoint
serves. Nothing narrows the vector on this side EXCEPT `truncate_to`, and that
is refused for any model not in `MATRYOSHKA_MODELS`: a prefix taken from a model
that was not trained for one is a vector of the right length carrying an
arbitrary slice of the wrong space — which retrieves badly and reports nothing
at all.

These pin the ways that can go wrong: a parameter sent where it is refused, a
vector quietly reshaped on the way back, a prefix taken from a model that cannot
give one, and a prefix left un-normalised — which ranks by vector length rather
than by meaning, and is the failure that looks like a working search.

No network. The OpenAI client is stubbed, because what is under test is what
goes on the wire and what comes back off it, not whether Hoonify is up.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.embedding import MATRYOSHKA_MODELS, HoonifyEmbedding, dimensionality


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


class TestMatryoshkaTruncation(unittest.TestCase):
    """
    Narrowing on this side, which is allowed only where the model was trained
    for it. The width is a real cost — a 4,096-dimension float32 vector is
    16 KB, and 1,024 is a quarter of that in the index and in the search — so
    the operation is worth having and worth refusing everywhere else.
    """

    def test_refuses_a_model_that_is_not_declared_matryoshka_trained(self):
        # The whole guard. A prefix here is not a smaller embedding, it is a
        # different space, and nothing downstream can see the difference.
        with self.assertRaises(ValueError) as refused:
            HoonifyEmbedding(model="bge-m3", api_key="k", truncate_to=256)

        self.assertIn("Matryoshka", str(refused.exception))

    def test_refuses_a_width_outside_what_the_card_declares(self):
        with self.assertRaises(ValueError):
            HoonifyEmbedding(model="Qwen/Qwen3-Embedding-8B", api_key="k", truncate_to=16)
        with self.assertRaises(ValueError):
            HoonifyEmbedding(model="Qwen/Qwen3-Embedding-8B", api_key="k", truncate_to=8192)

    def test_refuses_a_dimension_that_contradicts_the_truncation(self):
        # `dimension` is the width this embedder produces. Two answers to that
        # is a vector store built at one width and written at the other.
        with self.assertRaises(ValueError):
            HoonifyEmbedding(
                model="Qwen/Qwen3-Embedding-8B",
                api_key="k",
                dimension=4096,
                truncate_to=1024,
            )

    def test_the_produced_width_is_the_width_to_build_the_store_at(self):
        embedder = HoonifyEmbedding(
            model="Qwen/Qwen3-Embedding-8B", api_key="k", truncate_to=1024
        )

        self.assertEqual(embedder.dimension, 1024)

    def test_takes_the_prefix_and_renormalises_it(self):
        # Renormalisation is not tidying. The served vectors are unit length; a
        # prefix is shorter, and cosine similarity over vectors of mixed norm
        # ranks by length rather than by meaning.
        # 3 and 4 inside the kept prefix, everything beyond it large enough that
        # keeping any of it would show.
        served = [3.0, 4.0] + [0.0] * 30 + [100.0] * 4064
        embedder = stubbed(
            [served],
            model="Qwen/Qwen3-Embedding-8B",
            dimension_parameter=False,
            truncate_to=32,
        )

        vector = embedder.get_embeddings("hello")

        self.assertEqual(len(vector), 32)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)

    def test_a_zero_prefix_is_left_alone_rather_than_divided_by_its_length(self):
        embedder = stubbed(
            [[0.0] * 32 + [1.0] * 4064],
            model="Qwen/Qwen3-Embedding-8B",
            dimension_parameter=False,
            truncate_to=32,
        )

        self.assertEqual(embedder.get_embeddings("hello"), [0.0] * 32)

    def test_every_vector_in_a_batch_is_narrowed(self):
        embedder = stubbed(
            [[3.0, 4.0] + [0.0] * 4094, [0.0, 5.0] + [0.0] * 4094],
            model="Qwen/Qwen3-Embedding-8B",
            dimension_parameter=False,
            truncate_to=32,
        )

        vectors = embedder.get_embeddings(["one", "two"])

        self.assertEqual([len(vector) for vector in vectors], [32, 32])
        self.assertAlmostEqual(vectors[1][1], 1.0)

    def test_the_truncation_survives_to_dict(self):
        # `to_dict` is written to a KnowledgeBase's config on disk. A config that
        # lost this rebuilds an embedder producing 4,096-wide vectors against a
        # store built at 1,024 — which is not an error anything reports.
        embedder = HoonifyEmbedding(
            model="Qwen/Qwen3-Embedding-8B", api_key="k", truncate_to=1024
        )

        self.assertEqual(embedder.to_dict()["truncate_to"], 1024)
        self.assertEqual(embedder.to_dict()["dimension"], 1024)

    def test_an_untruncated_embedder_records_that_too(self):
        embedder = HoonifyEmbedding(model="Qwen/Qwen3-Embedding-8B", api_key="k")

        self.assertIsNone(embedder.to_dict()["truncate_to"])
        self.assertEqual(embedder.dimension, 4096)

    def test_the_declared_range_matches_the_model_card(self):
        # This table is the licence to narrow a vector. It is short on purpose.
        qwen = MATRYOSHKA_MODELS["Qwen/Qwen3-Embedding-8B"]

        self.assertEqual(qwen.native, dimensionality["Qwen/Qwen3-Embedding-8B"])
        self.assertEqual(qwen.minimum, 32)
        self.assertNotIn("bge-m3", MATRYOSHKA_MODELS)


if __name__ == "__main__":
    unittest.main()
