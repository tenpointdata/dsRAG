import math
import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
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


class MatryoshkaWidths(NamedTuple):
    """The range a Matryoshka-trained model's own card says it supports."""

    native: int
    minimum: int


#: Models trained so that a PREFIX of the output vector is itself a usable
#: embedding, with the widths each one's card declares.
#:
#: This table is the licence to narrow a vector on the client side, and it is
#: deliberately short. A prefix of a model that was not trained for one is a
#: vector of the right length carrying an arbitrary slice of the wrong space:
#: it retrieves badly and reports nothing at all, which is why truncation is
#: refused for anything absent here rather than merely discouraged. Add a model
#: on its card's own word, never on the shape of its output.
MATRYOSHKA_MODELS = {
    # Qwen3-Embedding is MRL-trained with user-defined output dimensions from
    # 32 up to the native width — see the model card and technical report.
    "Qwen/Qwen3-Embedding-8B": MatryoshkaWidths(native=4096, minimum=32),
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

    `truncate_to` narrows the served vector on this side, and is the only thing
    that does. It is refused for any model not in `MATRYOSHKA_MODELS`, because a
    prefix taken from a model that was not trained for one is a vector of the
    right length carrying an arbitrary slice of the wrong space — which
    retrieves badly and reports nothing at all. That failure is why nothing
    narrowed here before; naming the models that support it is what makes the
    operation safe rather than merely available. Truncation is followed by
    renormalisation: the served vectors are unit length, a prefix of one is not,
    and cosine similarity over vectors of mixed norm ranks by length.

    `dimension` is always the width this embedder PRODUCES, so it stays the
    width to build the vector store at whether or not truncation is on.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        dimension: Optional[int] = None,
        dimension_parameter: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        truncate_to: Optional[int] = None,
    ):
        super().__init__(dimension)
        self.model = model
        self.dimension_parameter = dimension_parameter
        self.truncate_to = truncate_to
        self.base_url = base_url
        # `base_url` and `api_key` default to the environment. Passing them
        # explicitly is what lets one caller drive two endpoints in the same
        # process.
        self.client = openai.OpenAI(
            api_key=api_key or hoonify.api_key(),
            base_url=base_url or hoonify.base_url(),
        )

        served = dimensionality.get(model)
        if truncate_to is None:
            if dimension is None:
                if served is None:
                    raise ValueError(
                        f"Dimension for model {model} is unknown. Please provide the dimension manually."
                    )
                self.dimension = served
            else:
                self.dimension = dimension
        else:
            # Refused at construction, not at the first query: a store built at
            # a width whose vectors mean nothing is discovered a corpus later.
            supported = MATRYOSHKA_MODELS.get(model)
            if supported is None:
                raise ValueError(
                    f"{model} is not declared Matryoshka-trained, so a {truncate_to}-dimension "
                    f"prefix of its output is an arbitrary slice of the wrong space — a vector of "
                    f"the right length that retrieves badly and reports nothing. Add it to "
                    f"MATRYOSHKA_MODELS only on the model card's own word."
                )
            if truncate_to < supported.minimum or truncate_to > supported.native:
                raise ValueError(
                    f"{model} supports {supported.minimum}–{supported.native} dimensions; "
                    f"truncate_to={truncate_to} is outside that."
                )
            if dimension is not None and dimension != truncate_to:
                raise ValueError(
                    f"dimension={dimension} contradicts truncate_to={truncate_to}. `dimension` is "
                    f"the width this embedder produces, which truncation decides."
                )
            self.dimension = truncate_to

    def _narrowed(self, vector: list) -> list:
        """
        A Matryoshka prefix, renormalised.

        Both halves are required. The prefix is only meaningful because the
        model was trained so that one is; the renormalisation is required
        because the served vector is unit length and a prefix of it is shorter,
        and cosine similarity over vectors of mixed norm ranks by length rather
        than by meaning.
        """
        if self.truncate_to is None:
            return vector
        prefix = vector[: self.truncate_to]
        norm = math.sqrt(math.fsum(value * value for value in prefix))
        # A zero vector has no direction to preserve; scaling it would divide by
        # zero to produce one that is still zero.
        if norm == 0:
            return prefix
        return [value / norm for value in prefix]

    def get_embeddings(self, text: list[str], input_type: Optional[str] = None) -> list[Vector]:
        # The parameter goes on the wire only where the endpoint takes it.
        # Several of the open embedders Hoonify serves emit a fixed width and
        # reject it, and one sent to those is a 400 on every call.
        width = {"dimensions": self.dimension} if self.dimension_parameter else {}
        response = self.client.embeddings.create(
            input=[text] if isinstance(text, str) else text, model=self.model, **width
        )
        record_response_usage(response, provider="hoonify", model=self.model, operation="embed")
        embeddings = [self._narrowed(item.embedding) for item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        # The key is deliberately absent: to_dict output is written to a
        # KnowledgeBase's config on disk.
        base_dict.update({
            "model": self.model,
            "dimension_parameter": self.dimension_parameter,
            "base_url": self.base_url,
            # A config that lost this would rebuild an embedder producing full-
            # width vectors against a store built at the narrowed width.
            "truncate_to": self.truncate_to,
        })
        return base_dict
