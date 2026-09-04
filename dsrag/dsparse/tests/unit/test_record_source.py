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

    def test__sections_partition_every_line_of_the_document(self):
        # A line belonging to no section is a line no chunk can carry, and a
        # consumer mapping sections onto source offsets cannot place it.
        children = [comment("2026-03-0%d" % i, "dana", "Body %d" % i) for i in range(1, 4)]
        document = render_records(TICKET, PROJECTION, children)

        self.assertEqual(document["sections"][0]["start"], 0)
        self.assertEqual(document["sections"][-1]["end"], len(document["lines"]) - 1)
        for earlier, later in zip(document["sections"], document["sections"][1:]):
            self.assertEqual(later["start"], earlier["end"] + 1)

    def test__section_content_matches_the_lines_it_spans(self):
        document = render_records(TICKET, PROJECTION, [comment("2026-03-04", "d", "Checked it.")])

        for section in document["sections"]:
            spanned = "\n".join(
                line["content"] for line in document["lines"][section["start"] : section["end"] + 1]
            )
            self.assertEqual(section["content"], spanned)

    def test__section_titles_are_citation_locators(self):
        sections, _, _ = parse_and_chunk_records(TICKET, PROJECTION, [comment("2026-03-04", "dana", "Checked it.")])

        self.assertEqual(sections[1]["title"], "Comment by dana, 2026-03-04")

    def test__doc_id_comes_from_the_natural_key(self):
        document = render_records(TICKET, PROJECTION, [comment("2026-03-04", "dana", "Checked it.")])

        self.assertEqual(document["doc_id"], "4821")


class TestRolesDecideDestination(unittest.TestCase):
    """Three kinds of column, three destinations, and only one of them embeds."""

    def test__attributes_and_identifiers_are_returned_but_never_embedded(self):
        children = [comment("2026-03-04", "marguerite", "Checked it.")]
        _, chunks, document = parse_and_chunk_records(TICKET, PROJECTION, children)

        self.assertEqual(document["attributes"], {"status": "open", "site_id": "214"})
        self.assertEqual(document["identifiers"], ["4821"])

        embedded = "\n".join(chunk["content"] for chunk in chunks)
        self.assertNotIn("open", embedded)
        self.assertNotIn("214", embedded)
        # A child attribute reaches the section TITLE, which is a locator. It
        # must not thereby reach the embedded body.
        self.assertNotIn("marguerite", embedded)

    def test__a_section_title_is_a_locator_not_a_line_of_the_body(self):
        # The title reaches the embedding through the AutoContext chunk header.
        # Writing it into the body embeds it twice and smuggles whatever it
        # names past the field roles.
        sections, chunks, _ = parse_and_chunk_records(TICKET, PROJECTION, [comment("2026-03-04", "dana", "Checked it.")])

        self.assertEqual(sections[1]["title"], "Comment by dana, 2026-03-04")
        for chunk in chunks:
            self.assertNotIn("##", chunk["content"])
            self.assertNotIn("Comment by dana", chunk["content"])

    def test__ignored_columns_reach_nothing(self):
        record = dict(TICKET, _airbyte_raw_id="0f2c-loader-generated")
        _, chunks, document = parse_and_chunk_records(record, PROJECTION, [comment("2026-03-04", "d", "Checked.")])

        self.assertNotIn("_airbyte_raw_id", document["attributes"])
        self.assertNotIn("loader-generated", "\n".join(chunk["content"] for chunk in chunks))


class TestJoinAndOrder(unittest.TestCase):
    """A document carries its own children, in their own order."""

    def test__children_of_another_root_are_not_folded_in(self):
        # The caller may hand over the whole child stream. Selecting from it is
        # the projection's job — a comment rendered into the wrong ticket is a
        # passage attributed to a record it did not come from.
        mine = comment("2026-03-04", "dana", "Belongs to 4821.")
        theirs = dict(comment("2026-03-05", "sam", "Belongs to 9999."), ticket_id="9999")

        sections, chunks, _ = parse_and_chunk_records(TICKET, PROJECTION, [mine, theirs])

        self.assertEqual(len(sections), 2)
        self.assertNotIn("9999", "\n".join(chunk["content"] for chunk in chunks))

    def test__a_numeric_order_column_sorts_numerically(self):
        child = dict(PROJECTION["child"], order="position")
        projection = dict(PROJECTION, child=child)
        children = [
            {"ticket_id": "4821", "position": position, "created_at": "x", "author": "a", "body": "Step %d" % position}
            for position in (10, 2, 1)
        ]

        sections, _, _ = parse_and_chunk_records(TICKET, projection, children)

        self.assertEqual([section["content"] for section in sections[1:]], ["Step 1", "Step 2", "Step 10"])

    def test__composite_keys_that_would_collide_stay_distinct(self):
        # A plain join is not injective: ("a:b", "c") and ("a", "b:c") name the
        # same document, and one record overwrites the other with no error.
        projection = dict(PROJECTION, grain="record", child=None, key=["left", "right"],
                          title_template="{subject}")
        first = render_records(dict(TICKET, left="a:b", right="c"), projection)
        second = render_records(dict(TICKET, left="a", right="b:c"), projection)

        self.assertNotEqual(first["doc_id"], second["doc_id"])

    def test__a_composite_key_cannot_carry_a_child_join(self):
        projection = dict(PROJECTION, key=["id", "site_id"])

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection, [comment("2026-03-04", "d", "Checked.")])


class TestGuards(unittest.TestCase):
    """Each one is an acceptance criterion. A projection that breaks it is refused."""

    def test__a_stream_with_no_narrative_field_produces_no_document(self):
        projection = dict(PROJECTION, fields={"id": "identifier", "amount": "attribute"}, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records({"id": "1", "subject": "A ticket", "amount": 42}, projection)

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

    def test__record_grain_carrying_a_child_stream_is_refused(self):
        projection = dict(PROJECTION, grain="record")

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection, [comment("2026-03-04", "d", "Checked.")])

    def test__a_title_role_the_template_never_names_is_refused(self):
        # Otherwise the role is inert: not embedded, not returned, not used to
        # name anything, and the misassignment reads as a deliberate choice.
        projection = dict(PROJECTION, grain="record", child=None, title_template="Ticket {id}")

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection)

    def test__a_declared_attribute_that_stopped_arriving_is_drift(self):
        projection = dict(PROJECTION, grain="record", child=None)

        with self.assertRaises(ProjectionError):
            render_records({k: v for k, v in TICKET.items() if k != "site_id"}, projection)

    def test__a_child_title_role_the_section_template_never_names_is_refused(self):
        # The same inert role as on the root, one level down.
        child = dict(PROJECTION["child"], fields={"body": "narrative", "author": "title"},
                     section_title_template="Comment of {created_at}")
        projection = dict(PROJECTION, child=child)

        with self.assertRaises(ProjectionError):
            render_records(TICKET, projection, [comment("2026-03-04", "d", "Checked.")])

    def test__a_root_missing_its_key_fails_closed_on_an_aggregate(self):
        # The child join dereferences the key column, so a bare KeyError here
        # would escape the ProjectionError contract the module promises.
        projection = dict(PROJECTION, title_template="{subject}")
        record = {k: v for k, v in TICKET.items() if k != "id"}

        with self.assertRaises(ProjectionError):
            render_records(record, projection, [comment("2026-03-04", "d", "Checked.")])

    def test__an_unknown_grain_is_refused(self):
        with self.assertRaises(ProjectionError):
            render_records(TICKET, dict(PROJECTION, grain="fact"))


if __name__ == "__main__":
    unittest.main()
