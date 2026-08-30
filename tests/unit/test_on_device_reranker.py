import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.reranker import Reranker, OnDeviceReranker


def result(text, header=""):
    return {"metadata": {"chunk_header": header, "chunk_text": text}}


class StubbedOnDeviceReranker(OnDeviceReranker):
    """
    The real class loads a cross-encoder on first use. These tests are about the
    ordering and the calibration ABOVE that model, so the model is replaced with
    a lookup — which is what `score_pairs` is a separate method for.
    """

    def __init__(self, scores, **kwargs):
        super().__init__(**kwargs)
        self.scores = scores
        self.pairs_seen = None

    def score_pairs(self, pairs):
        self.pairs_seen = pairs
        return [self.scores[passage] for _, passage in pairs]


class TestOnDeviceReranker(unittest.TestCase):
    def test_orders_by_score_descending(self):
        reranker = StubbedOnDeviceReranker(
            {"\n\nlow": -4.0, "\n\nhigh": 5.0, "\n\nmiddle": 0.5}
        )
        reranked = reranker.rerank_search_results(
            "q", [result("low"), result("high"), result("middle")]
        )
        self.assertEqual(
            [r["metadata"]["chunk_text"] for r in reranked], ["high", "middle", "low"]
        )

    def test_similarity_follows_the_passage_it_scored(self):
        """
        The bug this guards: writing similarities back in score order rather
        than in result order, which silently pairs each passage with its
        neighbour's score.
        """
        reranker = StubbedOnDeviceReranker({"\n\nlow": -4.0, "\n\nhigh": 5.0})
        reranked = reranker.rerank_search_results("q", [result("low"), result("high")])

        by_text = {r["metadata"]["chunk_text"]: r["similarity"] for r in reranked}
        self.assertAlmostEqual(by_text["high"], float(reranker.transform(5.0)))
        self.assertAlmostEqual(by_text["low"], float(reranker.transform(-4.0)))

    def test_pairs_carry_the_header_and_the_text(self):
        reranker = StubbedOnDeviceReranker({"Policy 4.1\n\nthe body": 1.0})
        reranker.rerank_search_results("q", [result("the body", header="Policy 4.1")])
        self.assertEqual(reranker.pairs_seen, [["q", "Policy 4.1\n\nthe body"]])

    def test_empty_results_short_circuit(self):
        reranker = StubbedOnDeviceReranker({})
        self.assertEqual(reranker.rerank_search_results("q", []), [])

    def test_transform_is_monotonic_and_neutral_at_zero(self):
        reranker = OnDeviceReranker()
        values = [float(reranker.transform(x)) for x in (-8, -2, 0, 2, 8)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[2], 0.5, places=6)
        self.assertGreater(values[0], 0.0)
        self.assertLess(values[-1], 1.0)

    def test_unknown_backend_is_refused_at_construction(self):
        """Fails naming the backend, not later with an obscure load error."""
        with self.assertRaises(ValueError) as raised:
            OnDeviceReranker(backend="tensorflow")
        self.assertIn("tensorflow", str(raised.exception))

    def test_constructing_loads_no_model(self):
        """
        `from_dict` constructs a reranker just to read a KnowledgeBase's config.
        If that downloaded a model, reading a config would need a network.
        """
        reranker = OnDeviceReranker(model="does-not-exist", backend="onnx")
        self.assertIsNone(reranker._session)
        self.assertIsNone(reranker._cross_encoder)
        self.assertIsNone(reranker._resolved_backend)

    def test_save_and_load_from_dict(self):
        reranker = OnDeviceReranker(
            model="BAAI/bge-reranker-base",
            backend="onnx",
            model_dir="/opt/models/reranker",
            max_length=256,
            batch_size=4,
        )
        loaded = Reranker.from_dict(reranker.to_dict())

        self.assertIsInstance(loaded, OnDeviceReranker)
        self.assertEqual(loaded.model, "BAAI/bge-reranker-base")
        self.assertEqual(loaded.backend, "onnx")
        self.assertEqual(loaded.model_dir, "/opt/models/reranker")
        self.assertEqual(loaded.max_length, 256)
        self.assertEqual(loaded.batch_size, 4)


if __name__ == "__main__":
    unittest.main()
