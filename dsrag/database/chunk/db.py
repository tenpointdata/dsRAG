from abc import ABC, abstractmethod
from typing import Any, Optional

from dsrag.database.chunk.types import FormattedDocument
from dsrag.utils.registry import SerializableComponent


class ChunkDB(SerializableComponent, ABC):

    @abstractmethod
    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        """
        Store all chunks for a given document.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        """
        Remove all chunks and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        """
        Retrieve the is_visual flag of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        """
        Retrieve the page numbers of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[FormattedDocument]:
        """
        Retrieve all chunks from a given document ID.
        """
        pass

    @abstractmethod
    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the document title of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the document summary of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the section title of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """
        Retrieve the section summary of a specific chunk from a given document ID.
        """
        pass

    @abstractmethod
    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        """
        Retrieve all document IDs.
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the chunk database.
        """
        pass
