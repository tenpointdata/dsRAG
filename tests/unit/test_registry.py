"""
The shared component registry — the invariants seven families now depend on.

Each of these was a real difference between the seven hand-written copies this
base replaced, so each is worth holding still.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from dsrag.database.chunk.db import ChunkDB
from dsrag.database.vector.db import VectorDB
from dsrag.dsparse.file_parsing.file_system import FileSystem, LocalFileSystem
from dsrag.embedding import Embedding
from dsrag.llm import LLM
from dsrag.reranker import NoReranker, Reranker
from dsrag.utils.registry import SerializableComponent


class ExampleFamily(SerializableComponent):
    pass


class ExampleMember(ExampleFamily):
    def __init__(self, size: int = 1):
        self.size = size

    def to_dict(self):
        return {**super().to_dict(), "size": self.size}


class OtherFamily(SerializableComponent):
    pass


class ExampleMemberSubclass(ExampleMember):
    pass


class TestSerializableComponent(unittest.TestCase):
    def test__a_subclass_registers_under_its_own_family(self):
        self.assertIs(ExampleFamily.subclasses["ExampleMember"], ExampleMember)
        self.assertNotIn("ExampleMember", OtherFamily.subclasses)

    def test__a_family_head_is_not_a_member_of_itself(self):
        self.assertNotIn("ExampleFamily", ExampleFamily.subclasses)

    def test__registration_reaches_below_the_first_level(self):
        self.assertIs(
            ExampleFamily.subclasses["ExampleMemberSubclass"], ExampleMemberSubclass
        )

    def test__from_dict_builds_the_named_subclass(self):
        built = ExampleFamily.from_dict({"subclass_name": "ExampleMember", "size": 7})
        self.assertIsInstance(built, ExampleMember)
        self.assertEqual(built.size, 7)

    def test__from_dict_works_from_a_member_as_well_as_the_head(self):
        built = ExampleMember.from_dict({"subclass_name": "ExampleMember", "size": 2})
        self.assertIsInstance(built, ExampleMember)

    def test__from_dict_leaves_the_callers_config_intact(self):
        # Six of the seven copies popped `subclass_name` out of the dict they
        # were handed. KnowledgeBase._load passes a live slice of the loaded
        # metadata, so loading a KB stripped the component names out of it.
        config = {"subclass_name": "ExampleMember", "size": 3}
        ExampleFamily.from_dict(config)
        self.assertEqual(config, {"subclass_name": "ExampleMember", "size": 3})
        ExampleFamily.from_dict(config)

    def test__a_missing_subclass_name_says_so(self):
        with self.assertRaises(ValueError) as raised:
            ExampleFamily.from_dict({})
        self.assertIn("subclass_name", str(raised.exception))

    def test__an_unknown_subclass_name_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as raised:
            ExampleFamily.from_dict({"subclass_name": "Nope"})
        self.assertIn("Nope", str(raised.exception))
        self.assertIn("ExampleMember", str(raised.exception))

    def test__to_dict_round_trips_through_from_dict(self):
        original = ExampleMember(size=9)
        rebuilt = ExampleFamily.from_dict(original.to_dict())
        self.assertEqual(rebuilt.size, 9)


class TestFamiliesAreSeparate(unittest.TestCase):
    """One registry per family: `Reranker.from_dict` must not build an `Embedding`."""

    FAMILIES = (Embedding, Reranker, LLM, VectorDB, ChunkDB, FileSystem)

    def test__every_family_has_its_own_registry(self):
        registries = [id(family.subclasses) for family in self.FAMILIES]
        self.assertEqual(len(set(registries)), len(self.FAMILIES))

    def test__a_family_holds_only_its_own_members(self):
        for family in self.FAMILIES:
            for name, member in family.subclasses.items():
                self.assertTrue(
                    issubclass(member, family),
                    f"{name} is registered under {family.__name__} but is not one",
                )

    def test__a_name_from_another_family_is_refused(self):
        with self.assertRaises(ValueError):
            Reranker.from_dict({"subclass_name": "OpenAIEmbedding"})


class TestFamilyRoundTrips(unittest.TestCase):
    def test__reranker_round_trips(self):
        rebuilt = Reranker.from_dict(NoReranker().to_dict())
        self.assertIsInstance(rebuilt, NoReranker)

    def test__file_system_keeps_its_base_path(self):
        original = LocalFileSystem(base_path="/tmp/dsrag-registry-test")
        rebuilt = FileSystem.from_dict(original.to_dict())
        self.assertIsInstance(rebuilt, LocalFileSystem)
        self.assertEqual(rebuilt.base_path, original.base_path)


if __name__ == "__main__":
    unittest.main()
