from dsrag.utils.usage import field, metered_call, record_response, record_usage
import os
from abc import ABC, abstractmethod
from typing import Optional
from dsrag.database.vector.types import Vector
from dsrag.utils import hoonify
from dsrag.utils.imports import (
    openai,
    cohere,
    voyageai,
    ollama,
    onnxruntime,
    tokenizers,
    sentence_transformers,
    huggingface_hub,
)


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
        record_response("openai", self.model, "embed", response)
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
        record_response("cohere", self.model, "embed", field(field(response, "meta"), "billed_units"),
                        usage_field=None, input_field="input_tokens")
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
        record_response("voyage", self.model, "embed", response,
                        usage_field=None, input_field="total_tokens")
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
                record_response("ollama", self.model, "embed", response,
                                usage_field=None, input_field="prompt_eval_count")
                responses.append(response["embedding"])
            return responses
        else:
            response = self.client.embeddings(model=self.model, prompt=text)
            record_response("ollama", self.model, "embed", response,
                            usage_field=None, input_field="prompt_eval_count")
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
    built at the wrong width, discovered at the first query. It is also SENT on
    the request: a Matryoshka embedder serves several widths, and the one the
    store was built at is the one to ask for.

    `dimension_parameter` says whether the endpoint TAKES that width as a
    request parameter. It defaults to False because the safe mistake is the
    quiet one: an endpoint that emits a fixed width rejects the parameter
    outright — HuggingFace TEI's OpenAI-compatible route, the one that ships in
    the box, does — and a 400 on every embed call is worse than a width the
    caller can still check against the collection.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        dimension: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension_parameter: bool = False,
    ):
        super().__init__(dimension)
        self.model = model
        self.base_url = base_url
        self.dimension_parameter = dimension_parameter
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
        # `dimensions` is sent EXPLICITLY when the caller named a width. A
        # Matryoshka embedder serves several, and leaving the parameter off
        # means taking whatever the server defaults to today — which is how a
        # collection sized at one width ends up fed by another, silently.
        #
        # A server that emits one fixed width rejects the parameter, so a
        # caller pointed at one passes `dimension_parameter=False`.
        request = {"input": [text] if isinstance(text, str) else text, "model": self.model}
        if self.dimension_parameter and self.dimension is not None:
            request["dimensions"] = self.dimension
        with metered_call("hoonify", self.model, "embed") as report:
            response = self.client.embeddings.create(**request)
            report(response)
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    def to_dict(self):
        base_dict = super().to_dict()
        # The key is deliberately absent: to_dict output is written to a
        # KnowledgeBase's config on disk.
        base_dict.update({"model": self.model, "base_url": self.base_url})
        return base_dict


class OnDeviceEmbedding(Embedding):
    """
    An embedder that runs in THIS process, with no network call.

    The sibling of `OnDeviceReranker`, and here for the same reasons: an
    air-gapped install has no embedding endpoint to call, a confidentiality
    boundary is one that passages must not cross, and — the reason this class
    was written — the in-the-box embedding CONTAINER is amd64-only, so on an
    arm64 host it runs under emulation. onnxruntime has native wheels for both,
    so loading the graph here rather than calling a container is the difference
    between emulated and native on the machine most of this is developed on.

    Backends mirror the reranker's: `onnx` needs onnxruntime and tokenizers and
    no torch; `sentence_transformers` is for a machine that already has torch;
    `auto` prefers ONNX and falls back.

    The model loads LAZILY, on first use. A KnowledgeBase is routinely
    constructed only to read its config, and paying a model download for a
    constructor that may never embed anything is how a config read turns into a
    hundred-megabyte fetch.
    """

    #: Ordered candidates for the ONNX graph, as on the reranker. Quantized
    #: first: on-device is precisely where the size matters.
    ONNX_CANDIDATES = (
        "onnx/model_quantized.onnx",
        "onnx/model.onnx",
        "model_quantized.onnx",
        "model.onnx",
    )

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: Optional[int] = None,
        backend: str = "auto",
        model_dir: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        """
        - model: repo id, or a local directory when `model_dir` is not given.
        - dimension: the width to EMIT. A Matryoshka model is trained so that a
          truncated prefix is itself a usable embedding, so a width below the
          graph's output is served by truncating and renormalising. A width
          above it is a configuration error and raises on first use.
        - query_prefix / document_prefix: some embedders are trained with an
          instruction prefix and lose real accuracy without it — nomic's
          `search_query: ` / `search_document: ` are the case in point. They are
          arguments rather than a lookup table keyed by model name, because a
          table is a branch on the model that goes stale silently.
        """
        if backend not in ("auto", "onnx", "sentence_transformers"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'auto', 'onnx' or 'sentence_transformers'."
            )
        super().__init__(dimension)
        self.model = model
        self.backend = backend
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        self._session = None
        self._tokenizer = None
        self._sentence_transformer = None
        self._resolved_backend = None

    # ─── Model loading ────────────────────────────────────────────────────

    def _cached_snapshot_dir(self):
        """Reuse either Hugging Face cache layout without a network probe."""
        home = os.environ.get("HF_HOME")
        if not home or "/" not in self.model:
            return None

        repository = f"models--{self.model.replace('/', '--')}"
        for cache_root in (home, os.path.join(home, "hub")):
            repository_dir = os.path.join(cache_root, repository)
            ref = os.path.join(repository_dir, "refs", "main")
            try:
                with open(ref, encoding="utf-8") as handle:
                    revision = handle.read().strip()
            except OSError:
                continue
            snapshot = os.path.join(repository_dir, "snapshots", revision)
            if os.path.isdir(snapshot):
                return snapshot
        return None

    def _resolve_model_dir(self, allow_patterns=None) -> str:
        if self.model_dir is not None:
            return self.model_dir
        if os.path.isdir(self.model):
            return self.model
        cached = self._cached_snapshot_dir()
        if cached is not None:
            return cached
        return huggingface_hub.snapshot_download(
            repo_id=self.model, allow_patterns=allow_patterns
        )

    def _load_onnx(self):
        directory = self._resolve_model_dir(
            allow_patterns=[*self.ONNX_CANDIDATES, "tokenizer.json"]
        )

        graph = next(
            (
                os.path.join(directory, name)
                for name in self.ONNX_CANDIDATES
                if os.path.isfile(os.path.join(directory, name))
            ),
            None,
        )
        if graph is None:
            raise FileNotFoundError(
                f"No ONNX graph in {directory}. Looked for: {', '.join(self.ONNX_CANDIDATES)}. "
                f"Export one with optimum, or use backend='sentence_transformers'."
            )

        tokenizer_file = os.path.join(directory, "tokenizer.json")
        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(f"No tokenizer.json in {directory}.")

        self._session = onnxruntime.InferenceSession(
            graph, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        self._tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer.enable_padding()

    def _load_sentence_transformers(self):
        self._sentence_transformer = sentence_transformers.SentenceTransformer(
            self.model_dir or self.model, trust_remote_code=True
        )

    def _ensure_loaded(self):
        """Load once, on first use. Idempotent."""
        if self._resolved_backend is not None:
            return

        if self.backend == "onnx":
            self._load_onnx()
            self._resolved_backend = "onnx"
            return
        if self.backend == "sentence_transformers":
            self._load_sentence_transformers()
            self._resolved_backend = "sentence_transformers"
            return

        try:
            self._load_onnx()
            self._resolved_backend = "onnx"
        except Exception:
            self._load_sentence_transformers()
            self._resolved_backend = "sentence_transformers"

    def warm(self):
        """Load the graph without embedding anything meaningful."""
        self._ensure_loaded()

    # ─── Embedding ────────────────────────────────────────────────────────

    def _prefixed(self, texts: list, input_type: Optional[str]) -> list:
        prefix = self.query_prefix if input_type == "query" else self.document_prefix
        return [f"{prefix}{text}" for text in texts] if prefix else list(texts)

    @staticmethod
    def _mean_pool(token_vectors: list, mask: list) -> list:
        """
        Mean over the UNPADDED tokens.

        Averaging across padding drags every vector toward the pad embedding by
        an amount that depends on batch composition — so the same text embeds
        differently depending on what it was batched with, which is silent and
        which no downstream check can see.
        """
        totals = [0.0] * len(token_vectors[0])
        counted = 0
        for vector, keep in zip(token_vectors, mask):
            if not keep:
                continue
            counted += 1
            for index, value in enumerate(vector):
                totals[index] += value
        if counted == 0:
            return totals
        return [value / counted for value in totals]

    @staticmethod
    def _normalize(vector: list) -> list:
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

    def _truncated(self, vector: list) -> list:
        """
        Matryoshka truncation, then RE-normalise.

        A prefix of a unit vector is not a unit vector, and cosine similarity
        over vectors of differing magnitude is not cosine similarity. Skipping
        the second normalise degrades ranking rather than breaking it, which is
        the kind of bug that survives a demo.
        """
        if self.dimension is None or self.dimension >= len(vector):
            if self.dimension is not None and self.dimension > len(vector):
                raise ValueError(
                    f"{self.model} emits {len(vector)} dimensions; {self.dimension} were asked for. "
                    f"A Matryoshka width may only be narrower than the model's own."
                )
            return vector
        return self._normalize(vector[: self.dimension])

    def _embed_onnx(self, texts: list) -> list:
        encodings = self._tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": [encoding.ids for encoding in encodings],
            "attention_mask": [encoding.attention_mask for encoding in encodings],
        }
        # A graph that never trained on segment ids still declares the input,
        # and onnxruntime treats a missing declared input as a hard error
        # rather than an ignored key — the same asymmetry the reranker hit.
        expected = {declared.name for declared in self._session.get_inputs()}
        if "token_type_ids" in expected:
            inputs["token_type_ids"] = [
                [0] * len(encoding.ids) for encoding in encodings
            ]

        outputs = self._session.run(
            None, {name: value for name, value in inputs.items() if name in expected}
        )
        hidden = outputs[0]

        vectors = []
        for row, encoding in zip(hidden, encodings):
            pooled = self._mean_pool(
                [list(token) for token in row], encoding.attention_mask
            )
            vectors.append(self._truncated(self._normalize(pooled)))
        return vectors

    def _embed_sentence_transformers(self, texts: list) -> list:
        encoded = self._sentence_transformer.encode(
            texts, normalize_embeddings=True, batch_size=self.batch_size
        )
        return [self._truncated(list(map(float, row))) for row in encoded]

    def get_embeddings(self, text, input_type: Optional[str] = None):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        if not texts:
            return [] if not single else []

        self._ensure_loaded()
        prepared = self._prefixed(texts, input_type)

        vectors = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            if self._resolved_backend == "onnx":
                vectors.extend(self._embed_onnx(batch))
            else:
                vectors.extend(self._embed_sentence_transformers(batch))

            record_usage("ondevice", self.model, "embed")

        return vectors[0] if single else vectors

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model": self.model,
                "backend": self.backend,
                "max_length": self.max_length,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
            }
        )
        return base_dict
