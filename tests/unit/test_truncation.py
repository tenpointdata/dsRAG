import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import dsrag.auto_context as auto_context
from dsrag.auto_context import CHARS_READ_PER_TOKEN, truncate_content


class CharacterEncoder:
    """One token per character, so a test can state token counts exactly.

    A stub rather than the real BPE because these tests are about how much of
    the document reaches the encoder at all, which no vocabulary changes.
    """

    def __init__(self):
        self.encoded_lengths = []

    def encode(self, text, **_kwargs):
        self.encoded_lengths.append(len(text))
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


class TestTruncateContent(unittest.TestCase):
    """The prefix AutoContext sends, and what it costs to produce.

    These are the invariants a multi-megabyte export depends on: the prompt is
    bounded, the work to produce it is bounded by the BUDGET rather than by the
    document, and a caller can still tell whether it saw the whole document.
    """

    def setUp(self):
        self.encoder = CharacterEncoder()
        self.previous = auto_context._TOKEN_ENCODER
        auto_context._TOKEN_ENCODER = self.encoder

    def tearDown(self):
        auto_context._TOKEN_ENCODER = self.previous

    def test__short_content_is_returned_whole_with_its_real_token_count(self):
        content = "a short document"
        text, num_tokens = truncate_content(content, 4000)
        self.assertEqual(text, content)
        self.assertEqual(num_tokens, len(content))

    def test__long_content_is_cut_to_the_budget_and_reports_the_cap(self):
        text, num_tokens = truncate_content("x" * 5_000, 100)
        self.assertEqual(num_tokens, 100)
        self.assertEqual(text, "x" * 100)

    def test__a_prefix_the_char_window_cut_still_reports_truncation(self):
        # Content that tokenises far more coarsely than the slack allows for:
        # the window is full and still holds fewer tokens than the budget, but
        # the rest of the document is missing all the same.
        class CoarseEncoder(CharacterEncoder):
            def encode(self, text, **kwargs):
                super().encode(text, **kwargs)
                return list(text[: len(text) // 100])

        auto_context._TOKEN_ENCODER = CoarseEncoder()
        content = "y" * (200 * CHARS_READ_PER_TOKEN * 4)
        text, num_tokens = truncate_content(content, 200)
        self.assertEqual(num_tokens, 200)
        self.assertLess(len(text), len(content))

    def test__work_is_bounded_by_the_budget_not_the_document(self):
        truncate_content("a very long export line\n" * 400_000, 500)
        self.assertEqual(self.encoder.encoded_lengths, [500 * CHARS_READ_PER_TOKEN])

    def test__the_encoder_is_built_once(self):
        builds = []
        auto_context._TOKEN_ENCODER = None
        original = auto_context.tiktoken.encoding_for_model

        def counted(name):
            builds.append(name)
            return CharacterEncoder()

        auto_context.tiktoken.encoding_for_model = counted
        try:
            truncate_content("first", 10)
            truncate_content("second", 10)
        finally:
            auto_context.tiktoken.encoding_for_model = original
        self.assertEqual(len(builds), 1)


if __name__ == "__main__":
    unittest.main()
