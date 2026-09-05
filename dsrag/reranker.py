from abc import ABC, abstractmethod
import math
import os
from typing import Optional

import requests
from scipy.stats import beta

from dsrag.utils import hoonify
from dsrag.utils.hardware import is_apple_silicon, performance_cores
from dsrag.utils.imports import (
    cohere,
    voyageai,
    onnxruntime,
    tokenizers,
    sentence_transformers,
    huggingface_hub,
)
from dsrag.utils.registry import SerializableComponent


class Reranker(SerializableComponent, ABC):

    @abstractmethod
    def rerank_search_results(self, query: str, search_results: list) -> list:
        pass

class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model
        cohere_api_key = os.environ['CO_API_KEY']
        base_url = os.environ.get("DSRAG_COHERE_BASE_URL", None)
        if base_url is not None:
            self.client = cohere.Client(api_key=cohere_api_key)
        else:
            self.client = cohere.Client(api_key=cohere_api_key)

    def transform(self, x):
        """
        transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1
        - this is critical for the new version of RSE to work properly, because it utilizes the absolute relevance values to calculate the similarity scores
        """
        a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Cohere Rerank API to rerank the search results
        """
        documents = []
        for result in search_results:
            documents.append(f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}")

        reranked_results = self.client.rerank(model=self.model, query=query, documents=documents)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_similarity_scores = [result.relevance_score for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        for i, result in enumerate(reranked_search_results):
            result['similarity'] = self.transform(reranked_similarity_scores[i])
        return reranked_search_results
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model
        })
        return base_dict
    
class VoyageReranker(Reranker):
    def __init__(self, model: str = "rerank-2"):
        self.model = model
        voyage_api_key = os.environ['VOYAGE_API_KEY']
        self.client = voyageai.Client(api_key=voyage_api_key)

    def transform(self, x):
        """
        transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1
        - this is critical for the new version of RSE to work properly, because it utilizes the absolute relevance values to calculate the similarity scores
        """
        a, b = 0.5, 1.8  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Voyage Rerank API to rerank the search results
        """
        documents = []
        for result in search_results:
            documents.append(f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}")
        
        reranked_results = self.client.rerank(model=self.model, query=query, documents=documents)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_similarity_scores = [result.relevance_score for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        for i, result in enumerate(reranked_search_results):
            result['similarity'] = self.transform(reranked_similarity_scores[i])
        return reranked_search_results
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model
        })
        return base_dict
    
class NoReranker(Reranker):
    def __init__(self, ignore_absolute_relevance: bool = False):
        """
        - ignore_absolute_relevance: if True, the reranker will override the absolute relevance values and assign a default similarity score to each chunk. This is useful when using an embedding model where the absolute relevance values are not reliable or meaningful.
        """
        self.ignore_absolute_relevance = ignore_absolute_relevance

    def rerank_search_results(self, query: str, search_results: list) -> list:
        if self.ignore_absolute_relevance:
            for result in search_results:
                result['similarity'] = 0.8 # default similarity score (represents a moderately relevant chunk)
        return search_results

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'ignore_absolute_relevance': self.ignore_absolute_relevance,
        })
        return base_dict

class OnDeviceReranker(Reranker):
    """
    A cross-encoder that runs in THIS process, with no network call.

    Every other reranker here is an HTTP client, which makes reranking a hop
    whose latency and availability belong to somebody else. That is a poor fit
    for three situations this class exists to serve:

      - an air-gapped install, where there is no reranking endpoint to call;
      - a latency budget, where a cross-encoder over a few dozen candidates is
        single-digit milliseconds locally and a round trip is not;
      - a confidentiality boundary that passages must not cross.

    Two backends, because two genuinely different machines run this. `onnx`
    needs onnxruntime and tokenizers — no torch, and the quantized export is
    small enough for an appliance. `sentence_transformers` is for a machine
    that already has torch and would rather not manage an export. `auto`
    prefers onnx and falls back.

    NEITHER BACKEND IS CPU-ONLY BY ASSUMPTION. An arm64 Mac has a GPU and a
    Neural Engine, and each backend reaches them differently: sentence-
    transformers selects MPS on its own, while onnxruntime does not select
    CoreML on its own and is given it here, along with the float graph it needs
    and one input shape to compile. Thread counts are derived from the hardware
    for the same reason — onnxruntime's defaults count efficiency cores, and a
    container's core count is the host's. See `dsrag.utils.hardware`.

    The model loads LAZILY, on the first rerank rather than in `__init__`.
    A KnowledgeBase is routinely constructed to read its config — `from_dict`
    does exactly that — and paying a model load for a constructor that may
    never rerank anything is how a config read turns into a download.
    """

    #: Ordered candidates for the ONNX graph inside a model directory. Exports
    #: disagree on where they put it, and a quantized export is preferred
    #: because on-device is precisely where the size matters.
    ONNX_CANDIDATES = (
        "onnx/model_quantized.onnx",
        "onnx/model.onnx",
        "model_quantized.onnx",
        "model.onnx",
    )

    #: The same graphs in the order an ACCELERATOR wants them.
    #:
    #: CoreML runs float ops on the GPU and the Neural Engine and has no int8
    #: kernels, so a quantized graph is partitioned back onto the CPU node by
    #: node — which costs the partition boundaries on top of the CPU work it
    #: was trying to avoid. Where an accelerator is present the float export is
    #: the fast graph, and the size argument for quantization does not apply to
    #: a machine that has the memory.
    ACCELERATED_ONNX_CANDIDATES = (
        "onnx/model.onnx",
        "model.onnx",
        "onnx/model_quantized.onnx",
        "model_quantized.onnx",
    )

    #: Providers tried, in order, on an arm64 Mac. CPU stays on the list: a
    #: CoreML partition falls back per operator rather than failing, and a
    #: machine whose onnxruntime build has no CoreML support still runs.
    APPLE_SILICON_PROVIDERS = ("CoreMLExecutionProvider", "CPUExecutionProvider")

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        backend: str = "auto",
        model_dir: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        threads: Optional[int] = None,
        providers: Optional[list] = None,
        a: float = 0.4,
        b: float = 0.4,
    ):
        """
        - model: repo id, or a local directory when `model_dir` is not given.
        - backend: "auto" | "onnx" | "sentence_transformers".
        - model_dir: a local directory to load from. Set this for an air-gapped
          install; it is what makes the class work with no network at all.
        - max_length: token ceiling per query/passage pair. Pairs longer than
          this are truncated from the passage end.
        - batch_size: pairs scored per forward pass.
        - threads: onnxruntime intra-op threads. None derives it from the
          hardware, which is what onnxruntime's own default gets wrong on the
          two machines this runs on most — see `dsrag.utils.hardware`.
        - providers: explicit onnxruntime execution providers. None selects
          them from the hardware. Pass `["CPUExecutionProvider"]` to hold an
          accelerator out of a comparison.
        - a, b: Beta CDF shape parameters for `transform`, as on the hosted
          rerankers above. These can be adjusted to change the distribution
          shape. The defaults match CohereReranker's, and for the same reason:
          a cross-encoder's squashed output is bimodal near 0 and 1, and a
          U-shaped Beta is what spreads it back out. They are symmetric, so a
          neutral logit maps to a neutral 0.5.
        """
        if backend not in ("auto", "onnx", "sentence_transformers"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'auto', 'onnx' or 'sentence_transformers'."
            )
        if threads is not None and threads < 1:
            raise ValueError(f"threads must be positive or None, got {threads!r}")
        self.model = model
        self.backend = backend
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.threads = threads
        self.providers = list(providers) if providers is not None else None
        self.a = a
        self.b = b

        self._session = None
        self._tokenizer = None
        self._cross_encoder = None
        self._resolved_backend = None
        #: Set when the loaded session runs on an accelerator. See `_score_onnx`.
        self._fixed_shape = False

    # ─── Model loading ────────────────────────────────────────────────────

    def _resolve_model_dir(self) -> str:
        if self.model_dir is not None:
            return self.model_dir
        if os.path.isdir(self.model):
            return self.model
        return huggingface_hub.snapshot_download(repo_id=self.model)

    def resolve_providers(self) -> list:
        """
        Execution providers for this machine, most capable first.

        An arm64 Mac has a GPU and a Neural Engine that CoreML can reach and
        that the CPU provider cannot, and onnxruntime does not select it on its
        own. Everywhere else — a Linux server, and a Linux CONTAINER on an
        arm64 Mac, which has no access to the host's accelerators whatever the
        host is — the CPU provider is the only one that is actually there.
        """
        if self.providers is not None:
            return list(self.providers)
        if is_apple_silicon():
            available = set(onnxruntime.get_available_providers())
            chosen = [name for name in self.APPLE_SILICON_PROVIDERS if name in available]
            if chosen:
                return chosen
        return ["CPUExecutionProvider"]

    def graph_candidates(self, providers: list) -> tuple:
        """Graph filenames in the order the chosen providers want them."""
        accelerated = any(name != "CPUExecutionProvider" for name in providers)
        return self.ACCELERATED_ONNX_CANDIDATES if accelerated else self.ONNX_CANDIDATES

    def session_options(self):
        """
        Thread counts, which onnxruntime's own defaults get wrong here.

        It sizes its pool from the machine's cores. That over-counts twice: an
        Apple Silicon core count includes efficiency cores, which set the pace
        of a parallel-for rather than adding to it, and a container's core
        count is the host's while the cgroup admits a fraction of it. Both
        produce a pool whose threads mostly wait for each other.

        `inter_op_num_threads` is 1 deliberately. Op-level parallelism only
        pays on a graph with independent branches; a transformer is a chain,
        and a second pool would take threads from the one doing the work.
        """
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.threads or performance_cores()
        options.inter_op_num_threads = 1
        return options

    def _load_onnx(self):
        directory = self._resolve_model_dir()
        providers = self.resolve_providers()
        candidates = self.graph_candidates(providers)

        graph = next(
            (
                os.path.join(directory, name)
                for name in candidates
                if os.path.isfile(os.path.join(directory, name))
            ),
            None,
        )
        if graph is None:
            raise FileNotFoundError(
                f"No ONNX graph in {directory}. Looked for: {', '.join(candidates)}. "
                f"Export one with optimum, or use backend='sentence_transformers'."
            )

        tokenizer_file = os.path.join(directory, "tokenizer.json")
        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(f"No tokenizer.json in {directory}.")

        self._session = onnxruntime.InferenceSession(
            graph, sess_options=self.session_options(), providers=providers
        )
        self._fixed_shape = any(name != "CPUExecutionProvider" for name in providers)

        self._tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        self._tokenizer.enable_truncation(max_length=self.max_length)
        # An accelerator compiles the subgraph PER INPUT SHAPE, so padding each
        # batch to its own longest pair — the sensible thing on a CPU, where a
        # short batch is genuinely less work — makes almost every batch a fresh
        # compile. One length for the session is worth more than the tokens it
        # wastes.
        if self._fixed_shape:
            self._tokenizer.enable_padding(length=self.max_length)
        else:
            self._tokenizer.enable_padding()

    def _load_sentence_transformers(self):
        # No device argument on purpose: sentence-transformers selects CUDA,
        # then MPS, then CPU on its own, so this path already reaches an Apple
        # GPU. Naming a device here would take that away on every other
        # machine.
        self._cross_encoder = sentence_transformers.CrossEncoder(
            self.model_dir or self.model, max_length=self.max_length
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

        # auto: prefer onnx, and keep the reason the preferred path failed —
        # a fallback that swallows the first error leaves you debugging the
        # second backend for a problem that lives in the first.
        try:
            self._load_onnx()
            self._resolved_backend = "onnx"
        except Exception as onnx_error:
            try:
                self._load_sentence_transformers()
                self._resolved_backend = "sentence_transformers"
            except Exception as st_error:
                raise RuntimeError(
                    f"No on-device backend could load {self.model!r}. "
                    f"onnx: {onnx_error}. sentence_transformers: {st_error}."
                ) from st_error

    # ─── Scoring ──────────────────────────────────────────────────────────

    def _score_onnx(self, pairs: list) -> list:
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]

            # The batch dimension is part of the shape an accelerator compiles
            # for, so the last, short batch of every query would compile a
            # second model. Padding it out and dropping the padding rows keeps
            # one shape for the life of the session.
            wanted = len(batch)
            if self._fixed_shape and wanted < self.batch_size:
                batch = list(batch) + [["", ""]] * (self.batch_size - wanted)

            encodings = self._tokenizer.encode_batch(batch)

            # Feed only the inputs this graph actually declares. Exports differ
            # on token_type_ids, and passing an input the graph does not have
            # is a hard onnxruntime error rather than an ignored key.
            available = {
                "input_ids": lambda e: [x.ids for x in e],
                "attention_mask": lambda e: [x.attention_mask for x in e],
                "token_type_ids": lambda e: [x.type_ids for x in e],
            }
            feed = {
                declared.name: available[declared.name](encodings)
                for declared in self._session.get_inputs()
                if declared.name in available
            }

            logits = self._session.run(None, feed)[0]
            scores.extend(float(row[0]) for row in logits[:wanted])
        return scores

    def _score_sentence_transformers(self, pairs: list) -> list:
        predictions = self._cross_encoder.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in predictions]

    def score_pairs(self, pairs: list) -> list:
        """
        Raw cross-encoder logits for (query, passage) pairs, in input order.

        Separate from `rerank_search_results` so the ordering and calibration
        above it can be tested without a model on disk.
        """
        if not pairs:
            return []
        self._ensure_loaded()
        if self._resolved_backend == "onnx":
            return self._score_onnx(pairs)
        return self._score_sentence_transformers(pairs)

    def warm(self) -> str:
        """
        Load the model and push one pair through it, before anyone is waiting.

        Loading is lazy for a good reason — a KnowledgeBase constructed only to
        read its config must not download a model — but the cost does not
        disappear, it moves onto the FIRST question. That question pays a model
        download, a session build and, on an accelerator, a graph compile, all
        of which are far outside any rerank timeout: the one query that warms
        the process is the one query that silently loses its reranking, and the
        stack looks fine afterwards.

        Called at startup, this puts the cost where nobody is waiting. Returns
        the backend that actually loaded, so a caller can report which of the
        two paths this machine took.
        """
        self._ensure_loaded()
        self.score_pairs([["warm", "warm"]])
        return self._resolved_backend

    def transform(self, x):
        """
        Map a raw logit onto a value distributed between 0 and 1.

        Two steps, unlike the hosted rerankers: a cross-encoder emits an
        unbounded logit rather than a relevance probability, so it is squashed
        with a logistic first and only then reshaped by the Beta CDF that RSE's
        absolute-relevance arithmetic expects.
        """
        probability = 1.0 / (1.0 + math.exp(-x))
        return beta.cdf(probability, self.a, self.b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Rerank locally. Identical contract to the hosted rerankers above.
        """
        if not search_results:
            return []

        pairs = [
            [
                query,
                f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}",
            ]
            for result in search_results
        ]
        scores = self.score_pairs(pairs)

        order = sorted(range(len(search_results)), key=lambda i: scores[i], reverse=True)
        reranked_search_results = [search_results[i] for i in order]
        for position, i in enumerate(order):
            reranked_search_results[position]["similarity"] = self.transform(scores[i])
        return reranked_search_results

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model": self.model,
                "backend": self.backend,
                "model_dir": self.model_dir,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "threads": self.threads,
                "providers": self.providers,
                "a": self.a,
                "b": self.b,
            }
        )
        return base_dict


class HoonifyReranker(Reranker):
    """
    Hoonify's hosted reranker.

    Hoonify is OpenAI-compatible for chat and embeddings, but reranking is not
    part of that wire format — so this is the `/rerank` shape the hosted
    rerankers converged on (`model`, `query`, `documents` in, `results` with
    `index` and `relevance_score` out), spoken with `requests` rather than the
    OpenAI SDK, which has no method for it.
    """

    def __init__(
        self,
        model: str = "bge-reranker-v2-m3",
        timeout: float = 30.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        `base_url` and `api_key` default to the environment. Passing them
        explicitly is what lets one caller drive two endpoints in the same
        process — a sidecar serving a tenancy on a private endpoint alongside
        one on the public default.
        """
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.api_key = api_key

    def _endpoint(self) -> str:
        return (self.base_url.rstrip("/") if self.base_url else hoonify.base_url()) + "/rerank"

    def _key(self) -> str:
        return self.api_key or hoonify.api_key()

    def transform(self, x):
        """
        Map the absolute relevance value onto something more uniformly
        distributed between 0 and 1, as RSE's absolute-relevance arithmetic
        requires. These can be adjusted to change the distribution shape.
        """
        a, b = 0.4, 0.4
        return beta.cdf(x, a, b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        if not search_results:
            return []

        documents = [
            f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}"
            for result in search_results
        ]

        response = requests.post(
            self._endpoint(),
            headers={"Authorization": f"Bearer {self._key()}"},
            json={"model": self.model, "query": query, "documents": documents},
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()["results"]

        reranked_search_results = [search_results[r["index"]] for r in results]
        for i, result in enumerate(reranked_search_results):
            result["similarity"] = self.transform(results[i]["relevance_score"])
        return reranked_search_results

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model": self.model,
                "timeout": self.timeout,
                "base_url": self.base_url,
                # The key is deliberately absent: to_dict output is written to
                # a KnowledgeBase's config on disk.
            }
        )
        return base_dict
