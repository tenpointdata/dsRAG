"""Offline storage contract checks; no provider or external database is used."""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

from dsrag.database.chunk import dynamo_db, postgres_db
from dsrag.database.chunk.dynamo_db import DynamoDB
from dsrag.database.chunk.metadata_codec import decode_chunk_metadata
from dsrag.database.chunk.postgres_db import PostgresChunkDB
from dsrag.database.chunk.sqlite_db import SQLiteDB


def read_sqlite(metadata):
    with tempfile.TemporaryDirectory() as directory:
        database = SQLiteDB("metadata-contract", directory)
        database.add_document("document", {0: {"chunk_text": "fixture"}})
        with contextlib.closing(sqlite3.connect(database.db_path + "/metadata-contract.db")) as conn:
            conn.execute("UPDATE documents SET metadata = ?", (metadata,))
            conn.commit()
        return database.get_document("document")


class FakeCursor:
    def __init__(self, metadata):
        self.metadata = metadata

    def execute(self, *args):
        pass

    def fetchall(self):
        return [("", "Fixture", "", "1", self.metadata)]


class FakeConnection:
    def __init__(self, metadata):
        self.metadata = metadata

    def cursor(self):
        return FakeCursor(self.metadata)

    def close(self):
        pass


def read_postgres(metadata):
    database = PostgresChunkDB.__new__(PostgresChunkDB)
    database.database = "fixture"
    database.username = "fixture"
    database.password = "unused"
    database.host = "unused.invalid"
    database.port = 0
    database.table_name = "fixture_documents"
    driver = SimpleNamespace(connect=lambda **kwargs: FakeConnection(metadata))
    with mock.patch.object(postgres_db, "psycopg2", driver):
        return database.get_document("document")


class FakeKey:
    def __init__(self, name):
        self.name = name

    def eq(self, value):
        return self.name, value


def read_dynamo(metadata):
    item = {
        "doc_id": "document",
        "document_title": "Fixture",
        "created_on": Decimal(1),
        "metadata": metadata,
    }
    table = SimpleNamespace(query=lambda **kwargs: {"Items": [item]})
    resource = SimpleNamespace(Table=lambda name: table)
    database = DynamoDB("metadata-contract", table_name="fixture")
    with mock.patch.object(database, "create_dynamo_client", return_value=resource):
        with mock.patch.object(dynamo_db, "get_key", return_value=FakeKey):
            return database.get_document("document")


READERS = (read_sqlite, read_postgres, read_dynamo)


class ChunkMetadataContractTests(unittest.TestCase):
    def test_legacy_dictionary_text_preserves_values(self):
        expected = {"title": "L'été", "active": True, "empty": None, "nested": [1, 2]}
        for read in READERS:
            with self.subTest(adapter=read.__name__):
                self.assertEqual(read(repr(expected))["metadata"], expected)

    def test_json_dictionary_text_has_the_same_contract(self):
        expected = {"title": "L'été", "active": True, "empty": None, "nested": [1, 2]}
        for read in READERS:
            with self.subTest(adapter=read.__name__):
                self.assertEqual(read(json.dumps(expected))["metadata"], expected)

    def test_dictionary_decoding_does_not_invoke_the_general_evaluator(self):
        # The stored value is an ordinary inert fixture; no execution payload.
        for read in READERS:
            with self.subTest(adapter=read.__name__):
                with mock.patch("builtins.eval", side_effect=AssertionError("evaluator used")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        document = read("{'marker': 'ordinary fixture'}")
                self.assertIsNotNone(document)
                self.assertEqual(document["metadata"], {"marker": "ordinary fixture"})

    def test_non_object_metadata_is_rejected(self):
        for value in ("[]", "7", "'ordinary text'"):
            for read in (read_sqlite, read_postgres):
                with self.subTest(adapter=read.__name__, value=value):
                    with self.assertRaises(ValueError):
                        read(value)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(read_dynamo(value))

    def test_missing_metadata_stays_missing(self):
        for read in READERS:
            with self.subTest(adapter=read.__name__):
                self.assertIsNone(read(None)["metadata"])

    def test_dynamo_native_maps_preserve_decimal_values(self):
        expected = {"score": Decimal("1.25"), "enabled": True}
        self.assertEqual(read_dynamo(expected)["metadata"], expected)

    def test_sqlite_writer_and_legacy_read_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            database = SQLiteDB("roundtrip", directory)
            expected = {"title": "L'été", "empty": None, "active": False}
            database.add_document("document", {0: {"chunk_text": "fixture"}}, metadata=expected)
            self.assertEqual(database.get_document("document")["metadata"], expected)

    def test_nullable_and_empty_legacy_forms(self):
        for stored in (None, "", "  ", "null", "None"):
            with self.subTest(stored=stored):
                self.assertIsNone(decode_chunk_metadata(stored))

    def test_literal_container_values_remain_compatible(self):
        expected = {"tuple": (1, 2), "bytes": b"fixture", "set": {"tag"}}
        self.assertEqual(decode_chunk_metadata(repr(expected)), expected)

    def test_decoder_errors_do_not_include_stored_content(self):
        marker = "invalid private record marker"
        with self.assertRaises(ValueError) as caught:
            decode_chunk_metadata(marker)
        self.assertNotIn(marker, str(caught.exception))

    def test_ordinary_document_ids_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            database = SQLiteDB("identifiers", directory)
            for doc_id in ("simple-id", "with spaces", "unicode Ω", 'double"quote'):
                expected = {"source": doc_id}
                database.add_document(doc_id, {0: {"chunk_text": "fixture"}}, metadata=expected)
                self.assertEqual(database.get_document(doc_id)["metadata"], expected)


if __name__ == "__main__":
    unittest.main()
