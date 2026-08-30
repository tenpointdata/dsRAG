import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.reranker import Reranker, HoonifyReranker
from dsrag.utils import hoonify


def result(text, header=""):
    return {"metadata": {"chunk_header": header, "chunk_text": text}}


class TestHoonifyConfig(unittest.TestCase):
    def test_base_url_defaults_to_the_public_endpoint(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(hoonify.base_url(), hoonify.DEFAULT_BASE_URL)

    def test_base_url_override_loses_its_trailing_slash(self):
        """Otherwise every path built from it contains a double slash."""
        with mock.patch.dict(os.environ, {"HOONIFY_BASE_URL": "https://vpc.example/v1/"}):
            self.assertEqual(hoonify.base_url(), "https://vpc.example/v1")

    def test_missing_key_names_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as raised:
                hoonify.api_key()
            self.assertIn("HOONIFY_API_KEY", str(raised.exception))

    def test_empty_key_is_treated_as_missing(self):
        with mock.patch.dict(os.environ, {"HOONIFY_API_KEY": ""}):
            with self.assertRaises(ValueError):
                hoonify.api_key()


class TestHoonifyReranker(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ,
            {"HOONIFY_API_KEY": "test-key", "HOONIFY_BASE_URL": "https://vpc.example/v1"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _response(self, results):
        response = mock.Mock()
        response.json.return_value = {"results": results}
        response.raise_for_status.return_value = None
        return response

    def test_reorders_by_the_index_the_service_returned(self):
        reranker = HoonifyReranker()
        posted = self._response(
            [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        )
        with mock.patch("dsrag.reranker.requests.post", return_value=posted) as post:
            reranked = reranker.rerank_search_results(
                "q", [result("first"), result("second")]
            )

        self.assertEqual(
            [r["metadata"]["chunk_text"] for r in reranked], ["second", "first"]
        )
        self.assertGreater(reranked[0]["similarity"], reranked[1]["similarity"])

        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["query"], "q")
        self.assertEqual(kwargs["json"]["documents"], ["\n\nfirst", "\n\nsecond"])

    def test_posts_to_the_configured_endpoint(self):
        reranker = HoonifyReranker()
        with mock.patch(
            "dsrag.reranker.requests.post", return_value=self._response([])
        ) as post:
            reranker.rerank_search_results("q", [result("only")])
        self.assertEqual(post.call_args[0][0], "https://vpc.example/v1/rerank")

    def test_empty_results_make_no_request(self):
        reranker = HoonifyReranker()
        with mock.patch("dsrag.reranker.requests.post") as post:
            self.assertEqual(reranker.rerank_search_results("q", []), [])
        post.assert_not_called()

    def test_save_and_load_from_dict(self):
        reranker = HoonifyReranker(model="bge-reranker-base", timeout=5.0)
        loaded = Reranker.from_dict(reranker.to_dict())
        self.assertIsInstance(loaded, HoonifyReranker)
        self.assertEqual(loaded.model, "bge-reranker-base")
        self.assertEqual(loaded.timeout, 5.0)


if __name__ == "__main__":
    unittest.main()
