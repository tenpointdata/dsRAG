import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.record_source import parse_and_chunk_records, render_records
from dsparse.record_source.rendering import ProjectionError


TICKET = {
    "id": "4821",
    "subject": "Fryer alarm",
    "description": "The fryer alarm sounds at 03:00 every night.",
    "status": "open",
    "site_id": "214",
}

PROJECTION = {
    "grain": "aggregate",
    "key": ["id"],
    "fields": {
        "id": "identifier",
        "subject": "title",
        "description": "narrative",
        "status": "attribute",
        "site_id": "attribute",
        "_airbyte_raw_id": "ignore",
    },
    "title_template": "Ticket {id} — {subject}",
    "child": {
        "on": "ticket_id",
        "order": "created_at",
        "fields": {"body": "narrative", "author": "attribute"},
        "section_title_template": "Comment by {author}, {created_at}",
    },
}


def comment(created_at, author, body):
    return {"ticket_id": "4821", "created_at": created_at, "author": author, "body": body}


class TestRecordStructure(unittest.TestCase):
    """The structure the loader already knew, kept rather than rediscovered."""

    def test__one_section_per_record_no_model_call(self):
        # The sections are exactly the records. A generative sectioning pass
        # would produce a count that depends on a model; this one cannot.
        children = [comment("2026-03-0%d" % i, "dana", "Body %d" % i) for i in range(1, 6)]
        sections, _, _ = parse_and_chunk_records(TICKET, PROJECTION, children)

        self.assertEqual(len(sections), 1 + len(children))
        self.assertEqual(sections[0]["title"], "Description")

    def test__chunks_never_cross_a_record_boundary(self):
        # The property the whole design exists for. A chunk holding the tail of
        # one comment and the head of the next embeds to a point that means
        # neither, and retrieves for nothing.
        children = [comment("2026-03-0%d" % i, "dana", "Body %d. " % i * 60) for i in range(1, 5)]
        sections, chunks, _ = parse_and_chunk_records(
            TICKET, PROJECTION, children, chunking_config={"chunk_size": 200, "min_length_for_chunking": 200}
        )

        self.assertGreater(len(chunks), len(sections))
        for chunk in chunks:
            section = sections[chunk["section_index"]]
            self.assertGreaterEqual(chunk["line_start"], section["start"])
            self.assertLessEqual(chunk["line_end"], section["end"])

    def test__children_are_ordered_by_the_declared_column(self):
        # Segment extraction returns a RANGE, on the argument that the chunks
        # between two matches carry the rest of the answer. Unordered children
        # make that argument false.
        children = [comment("2026-03-09", "c", "third"), comment("2026-03-01", "a", "first"),
                    comment("2026-03-04", "b", "second")]
        sections, _, _ = parse_and_chunk_records(TICKET, PROJECTION, children)

        self.assertEqual(
            [section["title"] for section in sections[1:]],
            ["Comment by a, 2026-03-01", "Comment by b, 2026-03-04", "Comment by c, 2026-03-09"],
        )

    def test__section_titles_are_citation_locators(self):
        sections, _, _ = parse_and_chunk_records(TICKET, PROJECTION, [comment("2026-03-04", "dana", "Checked it.")])

        self.assertEqual(sections[1]["title"], "Comment by dana, 2026-03-04")

    def test__doc_id_comes_from_the_natural_key(self):
        document = render_records(TICKET, PROJECTION, [comment("2026-03-04", "dana", "Checked it.")])

        self.assertEqual(document["doc_id"], "4821")


class TestRolesDecideDestination(unittest.TestCase):
    """Three kinds of column, three destinations, and only one of them embeds."""

    def test__attributes_and_identifiers_are_returned_but_never_embedded(self):
        _, chunks, document = parse_and_chunk_records(TICKET, PROJECTION, [comment("2026-03-04", "d", "Checked it.")])

        self.assertEqual(document["attributes"], {"status": "open", "site_id": "214"})
        self.assertEqual(document["identifiers"], ["4821"])

        embedded = "\n".join(chunk["content"] for chunk in chunks)
        self.assertNotIn("open", embedded)
        self.assertNotIn("214", embedded)

    def test__ignored_columns_reach_nothing(self):
        record = dict(TICKET, _airbyte_raw_id="0f2c-loader-generated")
        _, chunks, document = parse_and_chunk_records(record, PROJECTION, [comment("2026-03-04", "d", "Checked.")])

        self.assertNotIn("_airbyte_raw_id", document["attributes"])
        self.assertNotIn("loader-generated", "\n".join(chunk["content"] for chunk in chunks))


class TestGuards(unittest.TestCase):
    """Each one is an acceptance criterion. A projection that breaks it is refused."""

    def test__a_stream_with_no_narrative_field_produces_no_document(self):
        projection = dict(PROJECTION, fields={"id": "identifier", "amount": "attribute"}, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records({"id": "1", "amount": 42}, projection)

    def test__all_narrative_empty_is_refused_rather_than_indexed(self):
        # A title with no content indexes successfully, reports success, and
        # then abstains forever — the failure that leaves no evidence.
        projection = dict(PROJECTION, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records(dict(TICKET, description="   "), projection)

    def test__a_renamed_column_fails_closed(self):
        projection = dict(PROJECTION, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records({k: v for k, v in TICKET.items() if k != "subject"}, projection)

    def test__an_absent_or_empty_key_is_refused(self):
        projection = dict(PROJECTION, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records(dict(TICKET, id=""), projection)

    def test__an_unknown_field_role_is_refused(self):
        projection = dict(PROJECTION, grain="record", child=None,
                          fields={"description": "narrative", "status": "embedme"})

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection)

    def test__aggregate_grain_without_a_child_stream_is_refused(self):
        projection = dict(PROJECTION, child=None)

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection, [])

    def test__a_child_stream_without_an_order_column_is_refused(self):
        child = dict(PROJECTION["child"])
        del child["order"]
        projection = dict(PROJECTION, child=child)

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection, [comment("2026-03-04", "d", "Checked.")])

    def test__an_unknown_grain_is_refused(self):
        with self.assertRaises(ProjectionError):
            render_records(TICKET, dict(PROJECTION, grain="fact"))


if __name__ == "__main__":
    unittest.main()
