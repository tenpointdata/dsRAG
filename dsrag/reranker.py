from dsrag.utils.usage import metered_call, record_response, record_usage
from abc import ABC, abstractmethod
import math
import os
import threading
from pathlib import Path
from typing import Optional

import requests
from scipy.stats import beta

from dsrag.reranker_profiles import LEGACY_ONNX_GRAPHS, TOKENIZER_ASSETS, profile_for
from dsrag.reranker_onnx import configure_tokenizer, scalar_scores, score_onnx, session_options
from dsrag.reranker_onnx_head import OnnxScoringHead, module_assets
from dsrag.utils import hoonify
from dsrag.utils.imports import (
    cohere,
    voyageai,
    onnxruntime,
    tokenizers,
    sentence_transformers,
    huggingface_hub,
)


class Reranker(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            'subclass_name': self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop('subclass_name', None)  # Remove subclass_name from config
        if subclass_name == "CascadeReranker" and subclass_name not in cls.subclasses:
            from dsrag.reranker_cascade import CascadeReranker
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

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
        record_usage("cohere", self.model, "rerank")
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
        record_response("voyage", self.model, "rerank", reranked_results,
                        usage_field=None, input_field="total_tokens")
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
    """Lazy local cross-encoder with reviewed ONNX profiles and raw-logit calibration.

    ONNX profiles load only pinned CPU artifacts and allowlisted scoring heads.
    Legacy models retain the optional Sentence Transformers fallback.
    """

    #: Ordered candidates for the ONNX graph inside a model directory. Exports
    #: disagree on where they put it, and a quantized export is preferred
    #: because on-device is precisely where the size matters.
    ONNX_CANDIDATES = LEGACY_ONNX_GRAPHS

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        backend: str = "auto",
        model_dir: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        a: float = 0.4,
        b: float = 0.4,
        model_revision: Optional[str] = None,
        score_offset: Optional[float] = None,
        score_scale: Optional[float] = None,
        onnx_threads: Optional[int] = None,
    ):
        """
        - model: repo id, or a local directory when `model_dir` is not given.
        - backend: "auto" | "onnx" | "sentence_transformers".
        - model_dir: a local directory to load from. Set this for an air-gapped
          install; it is what makes the class work with no network at all.
        - max_length: token ceiling per query/passage pair. Pairs longer than
          this reserve at most half the ceiling (capped at 128 tokens) for the
          query beginning, then truncate the passage end.
        - batch_size: pairs scored per forward pass.
        - a, b: Beta CDF shape parameters for legacy model profiles.
        - score_offset, score_scale: logistic calibration location and scale.
          Reviewed profiles provide defaults; explicit values override them.
        - model_revision: immutable Hub revision or an explicit caller override.
        - onnx_threads: bounded CPU workers, defaulting to at most four cores.
        """
        if backend not in ("auto", "onnx", "sentence_transformers"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'auto', 'onnx' or 'sentence_transformers'."
            )
        for name, value in (("max_length", max_length), ("batch_size", batch_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model
        self.backend = backend
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.a, self.b = a, b
        if not all(math.isfinite(value) and value > 0 for value in (a, b)):
            raise ValueError("Beta calibration shapes must be finite and positive")
        self._profile = profile_for(model)
        self.ranking_only = bool(self._profile and self._profile.ranking_only)
        self.model_revision = model_revision or (self._profile.revision if self._profile else None)
        if self._profile and not self.ranking_only and self.model_revision != self._profile.revision and (score_offset is None or score_scale is None):
            raise ValueError("A different model revision requires explicit score calibration")
        if self._profile and not self.ranking_only and self._profile.calibrated_max_length is not None and max_length != self._profile.calibrated_max_length and (score_offset is None or score_scale is None):
            raise ValueError("A different max_length requires explicit score calibration")
        self.score_offset = score_offset if score_offset is not None else (self._profile.score_offset if self._profile else 0.0)
        self.score_scale = score_scale if score_scale is not None else (self._profile.score_scale if self._profile else 1.0)
        if not self.ranking_only:
            if self.score_offset is None or self.score_scale is None:
                raise ValueError("This reranker requires reviewed score_offset and score_scale calibration")
            if not math.isfinite(self.score_offset) or not math.isfinite(self.score_scale) or self.score_scale <= 0:
                raise ValueError("Score calibration must be finite with positive scale")
        self.onnx_threads = onnx_threads if onnx_threads is not None else min(4, os.cpu_count() or 1)
        if not isinstance(self.onnx_threads, int) or isinstance(self.onnx_threads, bool) or not 1 <= self.onnx_threads <= 32:
            raise ValueError("onnx_threads must be an integer between 1 and 32")
        self._session = self._tokenizer = self._cross_encoder = self._head = None
        self._query_tokenizer = None
        self._query_limit = min(128, max(1, max_length // 2))
        self._resolved_backend = self._resolved_graph = None
        self._warmed = False
        self._lock = threading.RLock()

    # ─── Model loading ────────────────────────────────────────────────────

    def _onnx_graph_in(self, directory):
        """The first ONNX candidate actually present in `directory`, or None."""
        return next(
            (
                os.path.join(directory, name)
                for name in self._graph_candidates()
                if os.path.isfile(os.path.join(directory, name))
            ),
            None,
        )

    def _graph_candidates(self):
        return (self._profile.graph(),) if self._profile else self.ONNX_CANDIDATES

    def _onnx_assets(self):
        return [*self._graph_candidates(), *TOKENIZER_ASSETS, *(self._profile.assets if self._profile else ())]

    def _cached_snapshot_dir(self, usable=None):
        """
        Reuse either Hugging Face cache layout without a network probe.

        A snapshot directory only counts as cached when the files the caller
        needs are IN it. An interrupted download leaves the directory and
        `refs/main` behind with the blob still `.incomplete`, and calling that
        a hit turns one dropped connection into a permanent failure:
        `snapshot_download` resumes from the partial blob, but only if it is
        reached at all.
        """
        home = os.environ.get("HF_HOME")
        if not home or "/" not in self.model:
            return None

        repository = f"models--{self.model.replace('/', '--')}"
        for cache_root in (home, os.path.join(home, "hub")):
            repository_dir = os.path.join(cache_root, repository)
            revision = self.model_revision
            if revision is None or len(revision) != 40:
                ref = os.path.join(repository_dir, "refs", revision or "main")
                try:
                    with open(ref, encoding="utf-8") as handle:
                        revision = handle.read().strip()
                except OSError:
                    continue
            snapshot = os.path.join(repository_dir, "snapshots", revision)
            if not os.path.isdir(snapshot):
                continue
            if usable is None or usable(snapshot):
                return snapshot
        return None

    def _resolve_model_dir(self, allow_patterns=None, usable=None) -> str:
        if self.model_dir is not None:
            return self.model_dir
        if os.path.isdir(self.model):
            return self.model
        cached = self._cached_snapshot_dir(usable)
        if cached is not None:
            return cached
        return huggingface_hub.snapshot_download(
            repo_id=self.model, revision=self.model_revision, allow_patterns=allow_patterns
        )

    def _onnx_snapshot_usable(self, directory) -> bool:
        if self._onnx_graph_in(directory) is None or not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            return False
        if self._profile and not all(os.path.isfile(os.path.join(directory, name)) for name in (*TOKENIZER_ASSETS, *self._profile.assets)):
            return False
        manifest = Path(directory) / "modules.json"
        return not manifest.is_file() or all((Path(directory) / name).is_file() for name in module_assets(Path(directory)))

    def _load_onnx(self):
        directory = self._resolve_model_dir(allow_patterns=self._onnx_assets(), usable=self._onnx_snapshot_usable)
        if not self._onnx_snapshot_usable(directory):
            raise FileNotFoundError(f"Incomplete ONNX reranker artifacts in {directory}")
        graph = self._onnx_graph_in(directory)
        head = OnnxScoringHead(directory) if (Path(directory) / "modules.json").is_file() else None
        session = onnxruntime.InferenceSession(
            graph, sess_options=session_options(self.onnx_threads), providers=["CPUExecutionProvider"]
        )
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        effective_max_length = configure_tokenizer(tokenizer, directory, self.max_length)
        query_tokenizer = tokenizers.Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        query_tokenizer.no_truncation()
        query_tokenizer.no_padding()
        self._query_tokenizer = query_tokenizer
        self._query_limit = min(128, max(1, effective_max_length // 2))
        self._session, self._tokenizer, self._head = session, tokenizer, head
        self._resolved_graph = os.path.basename(graph)

    def _load_sentence_transformers(self):
        self._cross_encoder = sentence_transformers.CrossEncoder(
            self.model_dir or self.model, max_length=self.max_length, revision=self.model_revision, trust_remote_code=False
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
            if self._profile is not None:
                raise
            try:
                self._load_sentence_transformers()
                self._resolved_backend = "sentence_transformers"
            except Exception as st_error:
                raise RuntimeError(
                    f"No on-device backend could load {self.model!r}. "
                    f"onnx: {onnx_error}. sentence_transformers: {st_error}."
                ) from st_error

    def warm(self):
        """Load and validate a real forward pass once in the serving process."""
        with self._lock:
            if not self._warmed:
                self.score_pairs([["warmup", "A short warmup passage."]])
                self._warmed = True

    def _score_onnx(self, pairs: list) -> list:
        return score_onnx(
            self._session, self._tokenizer, self._head, pairs, self.batch_size,
            query_tokenizer=self._query_tokenizer, query_limit=self._query_limit,
        )

    def _score_sentence_transformers(self, pairs: list) -> list:
        # CrossEncoder defaults can apply sigmoid; calibration requires raw logits.
        from torch.nn import Identity
        predictions = self._cross_encoder.predict(pairs, batch_size=self.batch_size, activation_fn=Identity())
        return scalar_scores(predictions, len(pairs))

    def score_pairs(self, pairs: list) -> list:
        """Finite raw logits in input order; serialize shared tokenizer/runtime use."""
        if not pairs:
            return []
        with self._lock:
            self._ensure_loaded()
            scores = self._score_onnx(pairs) if self._resolved_backend == "onnx" else self._score_sentence_transformers(pairs)
            return scalar_scores(scores, len(pairs))

    def describe(self) -> dict:
        version = self._profile.calibration_version if self._profile else "legacy-beta-v1"
        if self._profile and not self.ranking_only and (
            self.model_revision != self._profile.revision
            or self.score_offset != self._profile.score_offset
            or self.score_scale != self._profile.score_scale
            or self.max_length != self._profile.calibrated_max_length
        ):
            version = "explicit"
        return {
            "mode": "direct",
            "model": self.model,
            "modelRevision": self.model_revision,
            "rankingOnly": self.ranking_only,
            "calibration": "ranking_only" if self.ranking_only else (self._profile.calibration if self._profile else "logistic_beta"),
            "calibrationVersion": None if self.ranking_only else version,
            "scoreOffset": None if self.ranking_only else self.score_offset,
            "scoreScale": None if self.ranking_only else self.score_scale,
            "maxLength": self.max_length, "queryTokenLimit": self._query_limit,
            "onnxThreads": self.onnx_threads, "graph": self._resolved_graph,
            "backend": self._resolved_backend, "loaded": self._resolved_backend is not None,
            "warmed": self._warmed,
        }

    def transform(self, x):
        """
        Map a raw logit onto a value distributed between 0 and 1.

        Reviewed distilled-score profiles use their fitted logistic directly.
        Legacy cross-encoders retain the existing logistic then Beta CDF.
        """
        if self.ranking_only:
            raise ValueError("This model provides relative ranking only; absolute relevance requires an admission scorer")
        if not math.isfinite(x):
            raise ValueError("Cannot calibrate a nonfinite reranker score")
        value = (x - self.score_offset) / self.score_scale
        probability = 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))
        if self._profile and self._profile.calibration == "logistic":
            return probability
        return beta.cdf(probability, self.a, self.b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Rerank locally. Identical contract to the hosted rerankers above.
        """
        if not search_results:
            return []

        if self.ranking_only:
            raise ValueError("Use CascadeReranker to admit candidates from a ranking-only model")

        pairs = [
            [
                query,
                f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}",
            ]
            for result in search_results
        ]
        scores = scalar_scores(self.score_pairs(pairs), len(pairs))
        record_usage("ondevice", self.model, "rerank")

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
                "a": self.a,
                "b": self.b,
                "model_revision": self.model_revision,
                "score_offset": self.score_offset,
                "score_scale": self.score_scale,
                "onnx_threads": self.onnx_threads,
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

        endpoint = self._endpoint()
        headers = {"Authorization": f"Bearer {self._key()}"}
        with metered_call("hoonify", self.model, "rerank") as report:
            response = requests.post(
                endpoint,
                headers=headers,
                json={"model": self.model, "query": query, "documents": documents},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            report(payload)
        results = payload["results"]

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
