"""
The one metadata-filter vocabulary, and the dialect each vector store speaks.

Four stores each hand-wrote these operators and disagreed about what an
unrecognised one meant. The vocabulary and the refusal are now single, and both
are worth holding still.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from dsrag.database.vector.metadata_filter import (
    SUPPORTED_OPERATORS,
    to_expression_filter,
    to_mongo_style_filter,
    to_sql_filter,
)


def a_filter(operator: str, value=2020, field: str = "year") -> dict:
    return {"field": field, "operator": operator, "value": value}


class TestVocabulary(unittest.TestCase):
    def test__every_operator_is_translatable_in_every_dialect(self):
        for operator in SUPPORTED_OPERATORS:
            with self.subTest(operator=operator):
                self.assertTrue(to_mongo_style_filter(a_filter(operator)))
                self.assertTrue(to_sql_filter(a_filter(operator)))
                self.assertTrue(to_expression_filter(a_filter(operator)))

    def test__an_unknown_operator_is_refused_by_every_dialect(self):
        # Chroma raised KeyError, Pinecone built `{field: {None: value}}` and
        # queried on it, Milvus raised KeyError, Postgres raised ValueError.
        for dialect in (to_mongo_style_filter, to_sql_filter, to_expression_filter):
            with self.subTest(dialect=dialect.__name__):
                with self.assertRaises(ValueError) as raised:
                    dialect(a_filter("greater_then"))
                self.assertIn("greater_then", str(raised.exception))
                self.assertIn("greater_than", str(raised.exception))


class TestMongoStyle(unittest.TestCase):
    def test__comparison_nests_under_the_operator(self):
        self.assertEqual(
            to_mongo_style_filter(a_filter("greater_than")), {"year": {"$gt": 2020}}
        )

    def test__equality_nests_by_default(self):
        self.assertEqual(
            to_mongo_style_filter(a_filter("equals")), {"year": {"$eq": 2020}}
        )

    def test__bare_equality_is_opt_in(self):
        # Pinecone accepts the shorthand; Chroma does not.
        self.assertEqual(
            to_mongo_style_filter(a_filter("equals"), bare_equality=True),
            {"year": 2020},
        )

    def test__bare_equality_does_not_flatten_other_operators(self):
        self.assertEqual(
            to_mongo_style_filter(a_filter("less_than"), bare_equality=True),
            {"year": {"$lt": 2020}},
        )


class TestSql(unittest.TestCase):
    def test__the_value_is_a_placeholder_never_inlined(self):
        predicate = to_sql_filter(a_filter("equals", value="2020; DROP TABLE"))
        self.assertNotIn("DROP TABLE", predicate)
        self.assertEqual(predicate, "metadata->>'year' = %s")

    def test__a_list_gets_one_placeholder_per_element(self):
        self.assertEqual(
            to_sql_filter(a_filter("in", value=[1, 2, 3])),
            "metadata->>'year' IN (%s, %s, %s)",
        )


class TestExpression(unittest.TestCase):
    def test__a_string_value_is_quoted(self):
        self.assertEqual(
            to_expression_filter(a_filter("equals", value="acme", field="tenant")),
            "metadata['tenant'] == \"acme\"",
        )

    def test__a_number_is_not_quoted(self):
        self.assertEqual(
            to_expression_filter(a_filter("greater_than_equals")),
            "metadata['year'] >= 2020",
        )

    def test__no_filter_is_the_empty_expression(self):
        self.assertEqual(to_expression_filter(None), "")
        self.assertEqual(to_expression_filter({}), "")


if __name__ == "__main__":
    unittest.main()
