import os
import sys
import tempfile
import unittest
from unittest import mock

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

    def test_onnx_download_excludes_unneeded_training_checkpoints(self):
        reranker = OnDeviceReranker(model="BAAI/bge-reranker-base", backend="onnx")
        with mock.patch(
            "dsrag.reranker.huggingface_hub.snapshot_download",
            return_value="/models/reranker",
        ) as download, mock.patch(
            "dsrag.reranker.os.path.isfile",
            side_effect=lambda path: path.endswith("onnx/model.onnx")
            or path.endswith("tokenizer.json"),
        ), mock.patch(
            "dsrag.reranker.onnxruntime.InferenceSession"
        ), mock.patch(
            "dsrag.reranker.tokenizers.Tokenizer.from_file", return_value=mock.Mock()
        ):
            reranker._load_onnx()

        patterns = download.call_args.kwargs["allow_patterns"]
        self.assertIn("onnx/model.onnx", patterns)
        self.assertIn("tokenizer.json", patterns)
        self.assertNotIn("model.safetensors", patterns)

    def test_existing_huggingface_snapshot_needs_no_network_probe(self):
        with tempfile.TemporaryDirectory() as home:
            repository = os.path.join(home, "models--BAAI--bge-reranker-base")
            snapshot = os.path.join(repository, "snapshots", "revision-1")
            os.makedirs(snapshot)
            os.makedirs(os.path.join(repository, "refs"))
            with open(os.path.join(repository, "refs", "main"), "w", encoding="utf-8") as ref:
                ref.write("revision-1")

            reranker = OnDeviceReranker(model="BAAI/bge-reranker-base", backend="onnx")
            with mock.patch.dict(os.environ, {"HF_HOME": home}), mock.patch(
                "dsrag.reranker.huggingface_hub.snapshot_download"
            ) as download:
                resolved = reranker._resolve_model_dir()

        self.assertEqual(resolved, snapshot)
        download.assert_not_called()

    def _snapshot(self, home, *files):
        """A Hugging Face cache holding exactly `files`, as a snapshot dir."""
        repository = os.path.join(home, "models--BAAI--bge-reranker-base")
        snapshot = os.path.join(repository, "snapshots", "revision-1")
        os.makedirs(os.path.join(snapshot, "onnx"), exist_ok=True)
        os.makedirs(os.path.join(repository, "refs"), exist_ok=True)
        with open(os.path.join(repository, "refs", "main"), "w", encoding="utf-8") as ref:
            ref.write("revision-1")
        for name in files:
            with open(os.path.join(snapshot, name), "w", encoding="utf-8") as handle:
                handle.write("x")
        return snapshot

    def test_interrupted_download_is_not_mistaken_for_a_cached_model(self):
        # A dropped connection leaves the snapshot directory and refs/main
        # behind with the graph still `.incomplete`. Reading that as a cache hit
        # skipped a download that would have RESUMED from the partial blob, so
        # every later run failed identically — one transient fault made
        # permanent, recoverable only by deleting the cache by hand.
        with tempfile.TemporaryDirectory() as home:
            self._snapshot(home, "tokenizer.json")
            finished = self._snapshot(os.path.join(home, "finished"),
                                      "tokenizer.json", "onnx/model.onnx")

            reranker = OnDeviceReranker(model="BAAI/bge-reranker-base", backend="onnx")
            with mock.patch.dict(os.environ, {"HF_HOME": home}), mock.patch(
                "dsrag.reranker.huggingface_hub.snapshot_download",
                return_value=finished,
            ) as download, mock.patch(
                "dsrag.reranker.onnxruntime.InferenceSession"
            ), mock.patch(
                "dsrag.reranker.tokenizers.Tokenizer.from_file", return_value=mock.Mock()
            ):
                reranker._load_onnx()

        download.assert_called_once()

    def test_a_complete_cache_still_needs_no_network_probe(self):
        # The other half of the same rule: the shortcut has to keep working
        # when the files really are there, or every start pays for a probe.
        with tempfile.TemporaryDirectory() as home:
            self._snapshot(home, "tokenizer.json", "onnx/model.onnx")

            reranker = OnDeviceReranker(model="BAAI/bge-reranker-base", backend="onnx")
            with mock.patch.dict(os.environ, {"HF_HOME": home}), mock.patch(
                "dsrag.reranker.huggingface_hub.snapshot_download"
            ) as download, mock.patch(
                "dsrag.reranker.onnxruntime.InferenceSession"
            ), mock.patch(
                "dsrag.reranker.tokenizers.Tokenizer.from_file", return_value=mock.Mock()
            ):
                reranker._load_onnx()

        download.assert_not_called()

    def test_warm_loads_the_selected_runtime_once(self):
        reranker = OnDeviceReranker(model="local", backend="onnx")
        reranker._load_onnx = mock.Mock()

        reranker.warm()
        reranker.warm()

        reranker._load_onnx.assert_called_once_with()
        self.assertEqual(reranker._resolved_backend, "onnx")

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
