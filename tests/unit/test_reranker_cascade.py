"""Relative ordering and absolute admission have independent, bounded contracts."""
import copy
import unittest
from unittest import mock

from dsrag.reranker import OnDeviceReranker, Reranker
from dsrag.reranker_cascade import CascadeReranker, RankedSubset

MODEL = "cross-encoder/ettin-reranker-32m-v1"


def candidate(index):
    return {
        "metadata": {"cortex_candidate_key": f"key-{index}", "chunk_header": f"Header {index}", "chunk_text": f"Document {index}"},
        "similarity": 0.9,
    }


class CascadeTests(unittest.TestCase):
    def test_relative_scores_cannot_become_absolute_relevance(self):
        for options in ({}, {"score_offset": 0.0, "score_scale": 1.0}):
            ranker = OnDeviceReranker(model=MODEL, **options)
            with mock.patch.object(ranker, "_ensure_loaded"), mock.patch.object(ranker, "_score_sentence_transformers", return_value=[13.43]):
                self.assertEqual(ranker.score_pairs([["q", "d"]]), [13.43])
            with self.assertRaisesRegex(ValueError, "relative ranking"):
                ranker.transform(13.43)
            with self.assertRaisesRegex(ValueError, "Cascade"):
                ranker.rerank_search_results("q", [candidate(0)])
            status = ranker.describe()
            self.assertEqual(status["calibration"], "ranking_only")
            self.assertIsNone(status["calibrationVersion"])
            self.assertIsNone(status["scoreOffset"])
            self.assertIsNone(status["scoreScale"])

    def test_scores_only_shortlist_and_preserves_ranking_order(self):
        reranker = CascadeReranker(max_candidates=2)
        inputs = [candidate(index) for index in range(4)]
        with mock.patch.object(reranker.ranker, "score_pairs", return_value=[4.0, 9.0, 8.999, 1.0]) as rank, mock.patch.object(reranker.admission, "score_pairs", return_value=[-2.0, 3.0]) as admission:
            results = reranker.rerank_search_results("query", inputs)
        self.assertEqual(len(rank.call_args.args[0]), 4)
        self.assertEqual(len(admission.call_args.args[0]), 2)
        self.assertEqual([result["metadata"]["cortex_candidate_key"] for result in results], ["key-1", "key-2"])
        self.assertEqual(results.ranking_scores, (9.0, 8.999))
        self.assertEqual(results.unscored_candidate_keys, ("key-0", "key-3"))
        self.assertLess(results[0]["similarity"], results[1]["similarity"])
        baseline = OnDeviceReranker(model="BAAI/bge-reranker-base")
        self.assertEqual(results[0]["similarity"], baseline.transform(-2.0))
        self.assertEqual(results[1]["similarity"], baseline.transform(3.0))
        self.assertEqual(inputs[0]["similarity"], 0.9)
        self.assertEqual(inputs[3]["similarity"], 0.9)
        self.assertIs(results[0], inputs[1])

    def test_default_budget_is_eight_with_complete_partition_and_stable_ties(self):
        reranker = CascadeReranker()
        inputs = [candidate(index) for index in range(32)]
        with mock.patch.object(reranker.ranker, "score_pairs", return_value=[1.0] * 32), mock.patch.object(reranker.admission, "score_pairs", return_value=[0.0] * 8) as admission:
            results = reranker.rerank_search_results("q", inputs)
        self.assertEqual(len(admission.call_args.args[0]), 8)
        self.assertEqual([item["metadata"]["cortex_candidate_key"] for item in results], [f"key-{index}" for index in range(8)])
        self.assertEqual(results.unscored_candidate_keys, tuple(f"key-{index}" for index in range(8, 32)))
        self.assertEqual(len(results) + len(results.unscored_candidate_keys), len(inputs))
        with self.assertRaises(AttributeError):
            results.ranking_scores = ()

    def test_stage_failures_never_mutate_or_fabricate_scores(self):
        for raw, admission in (([float("nan"), 1.0], [0.0]), ([3.0, 2.0], []), ([3.0, 2.0], [float("inf")])):
            reranker = CascadeReranker(max_candidates=1)
            inputs = [candidate(0), candidate(1)]
            before = copy.deepcopy(inputs)
            with mock.patch.object(reranker.ranker, "score_pairs", return_value=raw), mock.patch.object(reranker.admission, "score_pairs", return_value=admission), self.assertRaises(ValueError):
                reranker.rerank_search_results("q", inputs)
            self.assertEqual(inputs, before)
        with mock.patch.object(reranker.ranker, "score_pairs", side_effect=TimeoutError("model timeout")), self.assertRaises(TimeoutError):
            reranker.rerank_search_results("q", inputs)
        self.assertEqual(inputs, before)

    def test_missing_or_duplicate_keys_cannot_hide_unscored_candidates(self):
        reranker = CascadeReranker()
        for inputs in ([candidate(0), candidate(0)], [{"metadata": {"chunk_header": "", "chunk_text": "missing key"}}]):
            with mock.patch.object(reranker.ranker, "score_pairs") as rank, self.assertRaises(ValueError):
                reranker.rerank_search_results("q", inputs)
            rank.assert_not_called()

    def test_empty_and_smaller_input_never_invents_extra_candidates(self):
        reranker = CascadeReranker()
        with mock.patch.object(reranker.ranker, "score_pairs") as rank:
            empty = reranker.rerank_search_results("q", [])
            self.assertIsInstance(empty, RankedSubset)
            self.assertEqual(empty, [])
            self.assertEqual(empty.unscored_candidate_keys, ())
            rank.assert_not_called()
        with mock.patch.object(reranker.ranker, "score_pairs", return_value=[1.0]), mock.patch.object(reranker.admission, "score_pairs", return_value=[2.0]):
            result = reranker.rerank_search_results("q", [candidate(0)])
            self.assertEqual(len(result), 1)
            self.assertEqual(result.unscored_candidate_keys, ())

    def test_configuration_roundtrip_warm_and_semantic_status(self):
        reranker = CascadeReranker(max_candidates=4, max_length=384, onnx_threads=2)
        restored = Reranker.from_dict(reranker.to_dict())
        self.assertEqual(restored.to_dict(), reranker.to_dict())
        self.assertEqual(restored.admission.model_revision, "2cfc18c9415c912f9d8155881c133215df768a70")
        with mock.patch.object(reranker.ranker, "warm") as rank, mock.patch.object(reranker.admission, "warm") as admission:
            reranker.warm()
            rank.assert_called_once()
            admission.assert_called_once()
        status = reranker.describe()
        self.assertEqual(status["mode"], "cascade")
        self.assertEqual(status["ranking"]["model"], MODEL)
        self.assertEqual(status["admission"]["model"], "BAAI/bge-reranker-base")
        self.assertEqual(status["ranking"]["calibration"], "ranking_only")
        self.assertEqual(status["admission"]["calibration"], "logistic_beta")
        self.assertFalse(status["loaded"])
        for options in ({"max_candidates": 0}, {"max_candidates": 129}, {"admission_model": MODEL}):
            with self.assertRaises(ValueError):
                CascadeReranker(**options)


if __name__ == "__main__":
    unittest.main()
