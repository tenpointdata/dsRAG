import os
from abc import ABC, abstractmethod
from typing import Optional
from dsrag.database.vector.types import Vector
from dsrag.utils import hoonify
from dsrag.utils.imports import openai, cohere, voyageai, ollama
from dsrag.utils.usage import record_response_usage, record_usage


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
        record_response_usage(response, provider="openai", model=self.model, operation="embed")
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
        # Cohere reports the charge under `meta.billed_units` rather than
        # `usage`, and an SDK version that stops carrying it must not break a
        # working embed call.
        billed = getattr(getattr(response, "meta", None), "billed_units", None)
        record_usage(
            provider="cohere",
            model=self.model,
            operation="embed",
            input_tokens=getattr(billed, "input_tokens", 0) or 0,
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
        record_usage(
            provider="voyageai",
            model=self.model,
            operation="embed",
            input_tokens=getattr(response, "total_tokens", 0) or 0,
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
                record_response_usage(
                    response, provider="ollama", model=self.model, operation="embed"
                )
                responses.append(response["embedding"])
            return responses
        else:
            response = self.client.embeddings(model=self.model, prompt=text)
            record_response_usage(
                response, provider="ollama", model=self.model, operation="embed"
            )
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

    `dimension_parameter` says whether the endpoint takes a `dimensions`
    request parameter at all. True asks it for `dimension` directly. False sends
    nothing and takes the width the endpoint serves — Hoonify's Qwen3 endpoint
    rejects the parameter outright, so that is the only thing to do with it, and
    sending one to a server that has a single answer is a 400 on every call.

    Nothing narrows the vector on this side. A width the server did not serve is
    a width this client cannot honestly produce: truncating to one would rest on
    the model being Matryoshka-trained, and a prefix taken from a model that is
    not is a vector of the right length carrying an arbitrary slice of the wrong
    space — which retrieves badly and reports nothing. `dimension` must
    therefore be the width the endpoint actually returns.
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
        record_response_usage(response, provider="hoonify", model=self.model, operation="embed")
        embeddings = [item.embedding for item in response.data]
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
