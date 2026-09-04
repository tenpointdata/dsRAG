import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.sectioning_and_chunking.chunking import chunk_document, chunk_sub_section, find_lines_in_range

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sections = [
            {
                'title': 'Section 1',
                'start': 0,
                'end': 3,
                'content': 'This is the first section of the document.'
            },
            {
                'title': 'Section 2',
                'start': 4,
                'end': 7,
                'content': 'This is the second section of the document.'
            }
        ]

        self.document_lines = [
            {
                'content': 'This is the first line of the document. And here is another sentence.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the second line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the third line of the document........',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the fourth line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the fifth line of the document. With another sentence.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the sixth line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the seventh line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            },
            {
                'content': 'This is the eighth line of the document. And here is another sentence that is a bit longer',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'is_visual': False
            }
        ]

    def test__chunk_document(self):
        chunk_size = 90
        min_length_for_chunking = 120
        chunks = chunk_document(self.sections, self.document_lines, chunk_size, min_length_for_chunking)
        assert len(chunks[-1]["content"]) > 45

    def test__chunk_sub_section(self):
        chunk_size = 90
        chunks_text, chunk_line_indices = chunk_sub_section(4, 7, self.document_lines, chunk_size)
        assert len(chunks_text) == 3
        assert chunk_line_indices == [(4, 4), (5, 6), (7, 7)]

    def test__find_lines_in_range(self):
        # Test find_lines_in_range
        # (line_idx, start_char, end_char)
        line_char_ranges = [
            (0, 0, 49),
            (1, 50, 99),
            (2, 100, 149),
            (3, 150, 199),
            (4, 200, 249),
            (5, 250, 299),
            (6, 300, 349),
            (7, 350, 399)
        ]

        chunk_start = 50
        chunk_end = 82
        line_start = 0
        line_end = 0
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start, chunk_end, line_char_ranges, line_start, line_end)
        assert chunk_line_start == 1
        assert chunk_line_end == 1

        chunk_start = 50
        chunk_end = 150
        line_start = 0
        line_end = 0
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start, chunk_end, line_char_ranges, line_start, line_end)
        assert chunk_line_start == 1
        assert chunk_line_end == 3


def find_lines_in_range_by_full_scan(chunk_start, chunk_end, line_char_ranges, line_start, line_end):
    """The mapping as it read before the search was allowed to skip lines.

    Kept here as the reference the fast version is measured against: the point
    of the change was cost, so a difference in the ANSWER is a defect.
    """
    chunk_line_start = None
    chunk_line_end = None
    for line_idx, start, end in line_char_ranges:
        if start <= chunk_start <= end + 1:
            chunk_line_start = line_idx
        if start <= chunk_end <= end + 1:
            chunk_line_end = line_idx
        if chunk_start < start and chunk_end > end:
            if chunk_line_start is None:
                chunk_line_start = line_idx
            chunk_line_end = line_idx
    if chunk_line_start is None:
        chunk_line_start = line_start
    if chunk_line_end is None:
        chunk_line_end = line_end
    return (chunk_line_start, chunk_line_end)


def export_lines(count, width=64):
    return [
        {
            "content": f"row: {index} | sku: SKU-{index:06d} | amount: {index * 7 % 9973}".ljust(width, "."),
            "element_type": "NarrativeText",
            "page_number": None,
            "is_visual": False,
        }
        for index in range(count)
    ]


class TestChunkLineMapping(unittest.TestCase):
    """Chunking a section the size of a landed export.

    A dump split into parts still hands one part's worth of rows to the
    chunker, which is thousands of lines rather than the dozens a page of
    prose produces. The mapping from chunk back to line has to stay both
    correct and affordable at that size.
    """

    def test__mapping_matches_the_full_scan_on_a_long_section(self):
        document_lines = export_lines(600)
        chunks_text, chunk_line_indices = chunk_sub_section(0, 599, document_lines, 300)

        line_char_ranges = []
        offset = 0
        for index, line in enumerate(document_lines):
            line_char_ranges.append((index, offset, offset + len(line["content"])))
            offset += len(line["content"]) + 1

        expected = []
        cursor = 0
        for chunk_text in chunks_text:
            expected.append(
                find_lines_in_range_by_full_scan(cursor, cursor + len(chunk_text), line_char_ranges, 0, 599)
            )
            cursor += len(chunk_text) + 1

        self.assertEqual(chunk_line_indices, expected)

    def test__mapping_matches_the_full_scan_across_ragged_line_lengths(self):
        document_lines = export_lines(200)
        for index in range(0, 200, 7):
            document_lines[index]["content"] = ""
        for index in range(0, 200, 11):
            document_lines[index]["content"] = "x" * 190

        chunks_text, chunk_line_indices = chunk_sub_section(0, 199, document_lines, 250)

        line_char_ranges = []
        offset = 0
        for index, line in enumerate(document_lines):
            line_char_ranges.append((index, offset, offset + len(line["content"])))
            offset += len(line["content"]) + 1

        expected = []
        cursor = 0
        for chunk_text in chunks_text:
            expected.append(
                find_lines_in_range_by_full_scan(cursor, cursor + len(chunk_text), line_char_ranges, 0, 199)
            )
            cursor += len(chunk_text) + 1

        self.assertEqual(chunk_line_indices, expected)

    def test__one_chunk_does_not_walk_the_whole_section(self):
        class RecordingRanges(list):
            """Counts how many line ranges a single search actually reads."""

            consumed = 0

            def __getitem__(self, item):
                entries = super().__getitem__(item)
                if not isinstance(item, slice):
                    return entries

                def counting():
                    for entry in entries:
                        RecordingRanges.consumed += 1
                        yield entry

                return counting()

        ranges = RecordingRanges((index, index * 10, index * 10 + 9) for index in range(10_000))
        RecordingRanges.consumed = 0
        self.assertEqual(find_lines_in_range(0, 25, ranges, 0, 9_999), (0, 2))
        # Line 3 begins after the chunk ends, which is where the search stops.
        self.assertEqual(RecordingRanges.consumed, 4)

    def test__consecutive_chunks_resume_where_the_last_one_started(self):
        ranges = [(index, index * 10, index * 10 + 9) for index in range(200)]
        self.assertEqual(
            find_lines_in_range(1_000, 1_025, ranges, 0, 199, 100),
            find_lines_in_range(1_000, 1_025, ranges, 0, 199),
        )


if __name__ == "__main__":
    unittest.main()
