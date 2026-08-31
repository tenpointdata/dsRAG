import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tests.fetch_corpora import (
    CORPORA,
    as_document_name,
    download_file,
    explain_failure,
    is_safe_name,
    split_multihop_rag,
    split_squad,
    split_tatqa,
    write_documents,
)

DOCUMENT = "The quick brown fox. " * 60 + "\n"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("gone.txt"):
            body, status = b"Not Found", 404
        elif self.path.endswith("errorpage.txt"):
            # A raw host answers some missing paths with a 200 and a short body.
            body, status = b"404: Not Found", 200
        else:
            body, status = DOCUMENT.encode("utf-8"), 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestDocumentNames(unittest.TestCase):
    """A name becomes a file name, and dsRAG takes a document's title from it."""

    def test_rejects_names_that_would_escape_the_destination(self):
        self.assertTrue(is_safe_name("rfc9110 HTTP Semantics"))
        self.assertFalse(is_safe_name("../escape"))
        self.assertFalse(is_safe_name("nested/name"))
        self.assertFalse(is_safe_name(".hidden"))
        self.assertFalse(is_safe_name(""))
        self.assertFalse(is_safe_name(" padded "))

    def test_turns_a_fetched_headline_into_something_writable(self):
        name = as_document_name('AC/DC: "Back in Black" -- 1980', "fallback")
        self.assertTrue(is_safe_name(name))
        self.assertNotIn("/", name)

    def test_caps_the_name_in_bytes_not_characters(self):
        # 120 CJK characters is 360 bytes and ext4 stops at 255. A character cap
        # passes the check and throws at the write, on exactly the corpora most
        # likely to be non-Latin.
        name = as_document_name("東京" * 200, "fallback")
        self.assertTrue(is_safe_name(name))
        self.assertLessEqual(len(name.encode("utf-8")), 200)

    def test_falls_back_rather_than_returning_an_unwritable_name(self):
        self.assertEqual(as_document_name("///", "fallback"), "fallback")


class TestSplits(unittest.TestCase):
    def test_multihop_rag_carries_the_outlet_and_date_into_the_text(self):
        # Its queries condition on both, and a .txt has nowhere else to put them.
        documents = split_multihop_rag(
            json.dumps(
                [
                    {
                        "title": 'Chips/Deals: the "big" one',
                        "source": "TechCrunch",
                        "published_at": "2023-11-27T08:45:59+00:00",
                        "body": DOCUMENT,
                    }
                ]
            )
        )
        self.assertEqual(len(documents), 1)
        name, text = documents[0]
        self.assertTrue(name.startswith("0001 "))
        self.assertNotIn("/", name)
        self.assertIn("TechCrunch", text)
        self.assertIn("2023-11-27", text)

    def test_squad_rejoins_an_article_rather_than_emitting_loose_paragraphs(self):
        # Its unanswerable questions were written by people looking at the
        # paragraph. Loose paragraphs hand the retriever the answer's
        # neighbourhood for free, and abstention stops measuring anything.
        documents = split_squad(
            json.dumps(
                {
                    "data": [
                        {
                            "title": "Normans_in_France",
                            "paragraphs": [
                                {"context": "First paragraph."},
                                {"context": "Second paragraph."},
                            ],
                        }
                    ]
                }
            )
        )
        self.assertEqual(len(documents), 1)
        name, text = documents[0]
        self.assertEqual(name, "Normans in France")
        self.assertIn("First paragraph.", text)
        self.assertIn("Second paragraph.", text)

    def test_tatqa_renders_the_table_beneath_the_caption_that_names_its_figures(self):
        documents = split_tatqa(
            json.dumps(
                [
                    {
                        "table": {"table": [["", "2019 %"], ["Rate of inflation", "2.9"]]},
                        "paragraphs": [
                            {"text": "Actuarial assumptions"},
                            {"text": "The scheme liabilities are measured as follows:"},
                        ],
                    }
                ]
            )
        )
        self.assertEqual(len(documents), 1)
        _, text = documents[0]
        self.assertTrue(text.startswith("Actuarial assumptions"))
        self.assertIn("Rate of inflation | 2.9", text)

    def test_a_split_raises_on_a_shape_it_does_not_recognise(self):
        # A published schema that moved must not read as an empty corpus: an
        # empty corpus retrieves nothing and reports success.
        with self.assertRaises(ValueError):
            split_tatqa('{"not": "an array"}')


class TestWriteDocuments(unittest.TestCase):
    def setUp(self):
        self.destination = tempfile.mkdtemp(prefix="dsrag-corpora-")

    def tearDown(self):
        shutil.rmtree(self.destination, ignore_errors=True)

    def test_writes_each_document_under_its_own_name(self):
        outcome = write_documents([("first", DOCUMENT), ("second", DOCUMENT)], self.destination)
        self.assertEqual(outcome["written"], 2)
        self.assertEqual(sorted(os.listdir(self.destination)), ["first.txt", "second.txt"])

    def test_refuses_a_derived_name_that_would_escape_the_directory(self):
        # The name came out of a file on the internet, which is a boundary.
        outcome = write_documents([("../escaped", DOCUMENT)], self.destination)
        self.assertEqual(outcome["written"], 0)
        self.assertEqual(os.listdir(self.destination), [])

    def test_reports_a_collision_rather_than_overwriting_the_first(self):
        # Two outlets file the same headline. Overwriting silently removes the
        # near-duplicate pressure the corpus is kept for.
        outcome = write_documents(
            [("same", DOCUMENT), ("same", DOCUMENT + "different")], self.destination
        )
        self.assertEqual(outcome["written"], 1)
        self.assertIn("same name", outcome["failed"][0][1])
        with open(os.path.join(self.destination, "same.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), DOCUMENT)


class TestDownloadFile(unittest.TestCase):
    """Served over a real socket: every defect here is about what comes back."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.destination = tempfile.mkdtemp(prefix="dsrag-download-")

    def tearDown(self):
        shutil.rmtree(self.destination, ignore_errors=True)

    def test_writes_the_document_and_then_resumes_by_skipping_it(self):
        path = os.path.join(self.destination, "rfc9110.txt")
        self.assertEqual(download_file(f"{self.origin}/rfc9110.txt", path), "written")
        self.assertEqual(download_file(f"{self.origin}/rfc9110.txt", path), "skipped")

    def test_reports_the_status_rather_than_writing_the_error_body(self):
        path = os.path.join(self.destination, "gone.txt")
        self.assertEqual(download_file(f"{self.origin}/gone.txt", path), "HTTP 404")
        self.assertFalse(os.path.exists(path))

    def test_refuses_a_200_too_small_to_be_a_document(self):
        # An error page indexed as a document is indistinguishable afterwards
        # from a document that genuinely had little text.
        path = os.path.join(self.destination, "errorpage.txt")
        self.assertIn("error page", download_file(f"{self.origin}/errorpage.txt", path))
        self.assertFalse(os.path.exists(path))

    def test_leaves_no_partial_behind(self):
        download_file(f"{self.origin}/rfc9110.txt", os.path.join(self.destination, "a.txt"))
        self.assertEqual([n for n in os.listdir(self.destination) if n.endswith(".partial")], [])

    def test_names_the_network_when_the_request_never_reached_a_host(self):
        # A blocked CONNECT and a moved file both surface as a failed GET and
        # want opposite responses. On a corporate proxy this is what an operator
        # actually hits first.
        self.assertIn(
            "never reached the host",
            explain_failure("<urlopen error Tunnel connection failed: 403 Forbidden>"),
        )
        self.assertEqual(explain_failure("socket closed after 3 bytes"), "socket closed after 3 bytes")


class TestCatalogue(unittest.TestCase):
    def test_every_corpus_declares_one_shape_and_only_one(self):
        # Declaring both would leave which one is fetched to reading order, and
        # the corpus would arrive whole on one machine and halved on the next.
        for corpus_id, corpus in CORPORA.items():
            with self.subTest(corpus=corpus_id):
                self.assertEqual(
                    ("files" in corpus, "bundle" in corpus) in [(True, False), (False, True)],
                    True,
                )
                if "bundle" in corpus:
                    self.assertTrue(callable(corpus["split"]))

    def test_every_address_is_https_and_every_name_is_writable(self):
        # Corpus text edited in flight moves every retrieval result without
        # failing anything.
        for corpus_id, corpus in CORPORA.items():
            with self.subTest(corpus=corpus_id):
                urls = [corpus["bundle"]] if "bundle" in corpus else [u for _, u in corpus["files"]]
                for url in urls:
                    self.assertTrue(url.startswith("https://"), url)
                for name, _ in corpus.get("files", []):
                    self.assertTrue(is_safe_name(name), name)

    def test_no_two_documents_in_a_corpus_share_a_file_name(self):
        # Not a duplicate download -- the second overwrites the first, and the
        # corpus is quietly one document short.
        for corpus_id, corpus in CORPORA.items():
            names = [name for name, _ in corpus.get("files", [])]
            with self.subTest(corpus=corpus_id):
                self.assertEqual(len(names), len(set(names)))

    def test_the_committed_fixtures_are_not_a_fetch_target(self):
        # les_miserables.txt is a 34 KB excerpt that test_auto_context.py
        # asserts on. Fetching the 3.3 MB novel over it would change the cost
        # and the behaviour of every call in that suite.
        from tests.fetch_corpora import CORPORA_DIR, DATA_DIR

        self.assertNotEqual(os.path.abspath(CORPORA_DIR), os.path.abspath(DATA_DIR))
        self.assertTrue(CORPORA_DIR.startswith(DATA_DIR))


if __name__ == "__main__":
    unittest.main()
