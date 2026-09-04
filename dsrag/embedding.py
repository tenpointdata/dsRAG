import os
from abc import ABC, abstractmethod
from typing import Optional
from dsrag.database.vector.types import Vector
from dsrag.utils import hoonify
from dsrag.utils.imports import openai, cohere, voyageai, ollama


dimensionality = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "voyage-large-2": 1536,
    "voyage-law-2": 1024,
    "voyage-code-2": 1536,
    "llama2": 4096,
    "llama3": 4096,
    "all-minilm": 384,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "bge-large-en-v1.5": 1024,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


def truncate_to_width(vector: Vector, width: int) -> Vector:
    """
    A Matryoshka embedding, narrowed by the client rather than by the server.

    A model trained with Matryoshka representation learning packs its most
    significant dimensions first, so a truncated prefix of its vector is itself
    a usable embedding. Qwen3-Embedding is one, and that property is what makes
    a 768-wide index of a 4,096-wide model reasonable rather than lossy
    guesswork — a fifth of the RAM for very nearly the same retrieval.

    The prefix is RENORMALISED. Truncation drops magnitude along with the
    dimensions it removes, so the remainder is no longer a unit vector, and a
    store comparing by dot product would then rank by how much of each vector
    survived rather than by similarity. Cosine distance is scale-invariant and
    would not notice; a dot-product index would, silently.

    A width at or above the vector's own is returned unchanged rather than
    refused: asking a 4,096-wide model to index at 4,096 is the identity case,
    and a caller that has not narrowed anything must not be told it did.
    """
    if width >= len(vector):
        return vector
    prefix = vector[:width]
    magnitude = sum(value * value for value in prefix) ** 0.5
    # An all-zero prefix has no direction to preserve. It is not a vector any
    # real embedder returns, and dividing by its magnitude would be a crash
    # rather than a wrong answer, so it is returned as it is.
    if magnitude == 0:
        return prefix
    return [value / magnitude for value in prefix]


class Embedding(ABC):
    subclasses = {}

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {"subclass_name": self.__class__.__name__, "dimension": self.dimension}

    @classmethod
    def from_dict(cls, config) -> "Embedding":
        subclass_name = config.pop(
            "subclass_name", None
        )  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def get_embeddings(self, text: list[str], input_type: Optional[str]) -> list[Vector]:
        pass


class OpenAIEmbedding(Embedding):
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768):
        """
        Only v3 models are supported.
        """
        super().__init__(dimension)
        self.model = model
        base_url = os.environ.get("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_embeddings(self, text: list[str], input_type: Optional[str] = None) -> list[Vector]:
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.dimension
        )
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class CohereEmbedding(Embedding):
    def __init__(self, model: str = "embed-english-v3.0", dimension: Optional[int] = None):
        super().__init__()

        self.model = model
        base_url = os.environ.get("DSRAG_COHERE_BASE_URL", None)
        if base_url is not None:
            self.client = cohere.Client(api_key=os.environ["CO_API_KEY"], base_url=base_url)
        else:
            self.client = cohere.Client(api_key=os.environ["CO_API_KEY"])

        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        if input_type == "query":
            input_type = "search_query"
        elif input_type == "document":
            input_type = "search_document"
        response = self.client.embed(
            texts=[text] if isinstance(text, str) else text,
            input_type=input_type,
            model=self.model,
        )
        return response.embeddings[0] if isinstance(text, str) else response.embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class VoyageAIEmbedding(Embedding):
    def __init__(self, model: str = "voyage-large-2", dimension: Optional[int] = None):
        super().__init__()
        self.model = model
        self.client = voyageai.Client()

        # Set dimension if not provided
        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        response = self.client.embed(
            texts=[text] if isinstance(text, str) else text,
            model=self.model,
            input_type=input_type,
        )
        return response.embeddings[0] if isinstance(text, str) else response.embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict


class OllamaEmbedding(Embedding):

    def __init__(
        self,
        model: str = "llama3",
        dimension: Optional[int] = None,
        client: "ollama.Client" = None,
    ):
        super().__init__(dimension)
        self.model = model
        self.client = client or ollama.Client()
        ollama.pull(model)

        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str]):
        if isinstance(text, list):
            responses = []
            for text in text:
                response = self.client.embeddings(model=self.model, prompt=text)
                responses.append(response["embedding"])
            return responses
        else:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response["embedding"]

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"model": self.model})
        return base_dict

class HoonifyEmbedding(Embedding):
    """
    Hoonify's embedding endpoint, OpenAI-compatible.

    `dimension` is looked up from the table above when the caller does not pass
    one, because getting it wrong is not a runtime error — it is a vector store
    built at the wrong width, discovered at the first query.

    `dimension_parameter` says WHO narrows the vector, and the two answers are
    not interchangeable. True means the endpoint honours the `dimensions`
    request parameter and is asked for the width directly. False means it does
    not — Hoonify's Qwen3 endpoint rejects the parameter outright today — so the
    full vector is fetched and truncated here. Both produce a vector of
    `dimension` floats; sending the parameter to a server that has one answer is
    a 400 on every call, and NOT truncating when the server ignored the request
    is a 4,096-wide vector rejected by a 768-wide collection on every upsert.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        dimension: Optional[int] = None,
        dimension_parameter: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(dimension)
        self.model = model
        self.dimension_parameter = dimension_parameter
        self.base_url = base_url
        # `base_url` and `api_key` default to the environment. Passing them
        # explicitly is what lets one caller drive two endpoints in the same
        # process.
        self.client = openai.OpenAI(
            api_key=api_key or hoonify.api_key(),
            base_url=base_url or hoonify.base_url(),
        )

        if dimension is None:
            try:
                self.dimension = dimensionality[model]
            except KeyError:
                raise ValueError(
                    f"Dimension for model {model} is unknown. Please provide the dimension manually."
                )
        else:
            self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: Optional[str] = None) -> list[Vector]:
        # The parameter goes on the wire only where the endpoint takes it.
        # Several of the open embedders Hoonify serves emit a fixed width and
        # reject it, and one sent to those is a 400 on every call.
        width = {"dimensions": self.dimension} if self.dimension_parameter else {}
        response = self.client.embeddings.create(
            input=[text] if isinstance(text, str) else text, model=self.model, **width
        )
        embeddings = [item.embedding for item in response.data]
        # Truncation is a no-op when the server already served the width asked
        # for, so this is unconditional rather than a second branch on the same
        # flag — one place decides, and the other cannot disagree with it.
        if self.dimension:
            embeddings = [truncate_to_width(vector, self.dimension) for vector in embeddings]
        return embeddings[0] if isinstance(text, str) else embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        # The key is deliberately absent: to_dict output is written to a
        # KnowledgeBase's config on disk.
        base_dict.update({
            "model": self.model,
            "dimension_parameter": self.dimension_parameter,
            "base_url": self.base_url,
        })
        return base_dict
