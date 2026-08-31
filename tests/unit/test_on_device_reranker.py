import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag import reranker as reranker_module
from dsrag.reranker import Reranker, OnDeviceReranker


class Encoding:
    def __init__(self, index):
        self.ids = [index]
        self.attention_mask = [1]
        self.type_ids = [0]


class RecordingTokenizer:
    """Records the batch it was handed, which is the shape the session sees."""

    def __init__(self):
        self.batches = []

    def encode_batch(self, batch):
        self.batches.append(list(batch))
        return [Encoding(index) for index in range(len(batch))]


class StubRuntime:
    """Stands in for the lazily-imported onnxruntime, which CI does not have."""

    def __init__(self, providers):
        self.providers = providers

    def get_available_providers(self):
        return self.providers


class StubSession:
    """One logit per encoded row, in row order."""

    class Input:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self.Input("input_ids"), self.Input("attention_mask")]

    def run(self, _outputs, feed):
        return [[[float(index)] for index in range(len(feed["input_ids"]))]]


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

    def test_threads_must_be_positive(self):
        with self.assertRaises(ValueError):
            OnDeviceReranker(threads=0)

    def test_warm_loads_the_model_and_runs_a_pair_through_it(self):
        """
        The first question must not be the one that pays the model load.

        A deployment calls this at startup precisely because a download and a
        graph compile are far outside any rerank timeout — so warming has to
        touch the model, not merely construct it.
        """
        reranker = StubbedOnDeviceReranker({"warm": 0.0})
        reranker._resolved_backend = "onnx"

        self.assertEqual(reranker.warm(), "onnx")
        self.assertEqual(reranker.pairs_seen, [["warm", "warm"]])

    def test_save_and_load_from_dict(self):
        reranker = OnDeviceReranker(
            model="BAAI/bge-reranker-base",
            backend="onnx",
            model_dir="/opt/models/reranker",
            max_length=256,
            batch_size=4,
            threads=2,
            providers=["CPUExecutionProvider"],
        )
        loaded = Reranker.from_dict(reranker.to_dict())

        self.assertIsInstance(loaded, OnDeviceReranker)
        self.assertEqual(loaded.model, "BAAI/bge-reranker-base")
        self.assertEqual(loaded.backend, "onnx")
        self.assertEqual(loaded.model_dir, "/opt/models/reranker")
        self.assertEqual(loaded.max_length, 256)
        self.assertEqual(loaded.batch_size, 4)
        self.assertEqual(loaded.threads, 2)
        self.assertEqual(loaded.providers, ["CPUExecutionProvider"])


class TestOnDeviceRerankerHardwareSelection(unittest.TestCase):
    """
    Which provider, which graph, and which shape — decided from the hardware.

    None of it can be exercised on the machine CI runs on, and all of it is
    exactly what a machine-dependent path gets wrong silently: an accelerator
    handed a quantized graph runs it on the CPU anyway, and an accelerator
    handed a fresh input shape per batch recompiles instead of inferring.
    """

    def test_an_explicit_provider_list_wins(self):
        reranker = OnDeviceReranker(providers=["CPUExecutionProvider"])
        self.assertEqual(reranker.resolve_providers(), ["CPUExecutionProvider"])

    def test_cpu_elsewhere(self):
        reranker = OnDeviceReranker()
        with mock.patch.object(reranker_module, "is_apple_silicon", return_value=False):
            self.assertEqual(reranker.resolve_providers(), ["CPUExecutionProvider"])

    def test_coreml_first_on_apple_silicon(self):
        reranker = OnDeviceReranker()
        with mock.patch.object(
            reranker_module, "is_apple_silicon", return_value=True
        ), mock.patch.object(
            reranker_module,
            "onnxruntime",
            StubRuntime(["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        ):
            self.assertEqual(
                reranker.resolve_providers(),
                ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )

    def test_cpu_on_an_apple_build_without_coreml(self):
        """A conda or a source build may have no CoreML provider at all."""
        reranker = OnDeviceReranker()
        with mock.patch.object(
            reranker_module, "is_apple_silicon", return_value=True
        ), mock.patch.object(
            reranker_module, "onnxruntime", StubRuntime(["CPUExecutionProvider"])
        ):
            self.assertEqual(reranker.resolve_providers(), ["CPUExecutionProvider"])

    def test_the_quantized_graph_is_preferred_on_a_cpu(self):
        candidates = OnDeviceReranker().graph_candidates(["CPUExecutionProvider"])
        self.assertIn("quantized", candidates[0])

    def test_the_float_graph_is_preferred_on_an_accelerator(self):
        """CoreML has no int8 kernels: a quantized graph goes back to the CPU."""
        candidates = OnDeviceReranker().graph_candidates(
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertNotIn("quantized", candidates[0])

    def test_a_short_batch_is_padded_to_one_shape_on_an_accelerator(self):
        reranker = OnDeviceReranker(batch_size=4)
        reranker._resolved_backend = "onnx"
        reranker._fixed_shape = True
        reranker._tokenizer = RecordingTokenizer()
        reranker._session = StubSession()

        scores = reranker._score_onnx([["q", "a"], ["q", "b"]])

        self.assertEqual(scores, [0.0, 1.0])
        self.assertEqual(len(reranker._tokenizer.batches[0]), 4)

    def test_a_short_batch_is_not_padded_on_a_cpu(self):
        """Padding is pure waste where every shape is compiled on the fly."""
        reranker = OnDeviceReranker(batch_size=4)
        reranker._resolved_backend = "onnx"
        reranker._tokenizer = RecordingTokenizer()
        reranker._session = StubSession()

        scores = reranker._score_onnx([["q", "a"], ["q", "b"]])

        self.assertEqual(scores, [0.0, 1.0])
        self.assertEqual(len(reranker._tokenizer.batches[0]), 2)


if __name__ == "__main__":
    unittest.main()
