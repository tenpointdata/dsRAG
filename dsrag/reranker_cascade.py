"""Fast relative ordering with bounded, independently scored absolute admission."""
import math
import threading
from typing import Optional, Tuple

from dsrag.reranker import OnDeviceReranker, Reranker
from dsrag.reranker_onnx import scalar_scores
from dsrag.reranker_profiles import profile_for
from dsrag.utils.usage import record_usage


class RankedSubset(list):
    """Per-call proof of the scored/unscored partition and relative ordering."""

    __slots__ = ("_unscored_candidate_keys", "_ranking_scores")

    def __init__(self, results: list, unscored_candidate_keys: Tuple[str, ...], ranking_scores: Tuple[float, ...]):
        super().__init__(results)
        self._unscored_candidate_keys = unscored_candidate_keys
        self._ranking_scores = ranking_scores

    @property
    def unscored_candidate_keys(self) -> Tuple[str, ...]:
        return self._unscored_candidate_keys

    @property
    def ranking_scores(self) -> Tuple[float, ...]:
        return self._ranking_scores


class CascadeReranker(Reranker):
    """Return only a scored shortlist; never invent relevance for omitted hits.

    The ranking model orders all input candidates. The admission model scores
    at most max_candidates and supplies the existing absolute relevance values
    RSE consumes. Returned order remains the ranking model's stable order.
    Failure propagates before candidate mutation so the caller can fall back.
    """

    concurrency = 1

    def __init__(
        self,
        model: str = "cross-encoder/ettin-reranker-32m-v1",
        model_revision: Optional[str] = None,
        model_dir: Optional[str] = None,
        backend: str = "onnx",
        max_length: int = 512,
        batch_size: int = 16,
        onnx_threads: Optional[int] = None,
        max_candidates: Optional[int] = None,
        admission_model: Optional[str] = None,
        admission_model_revision: Optional[str] = None,
        admission_model_dir: Optional[str] = None,
        admission_max_length: int = 512,
    ):
        profile = profile_for(model)
        admission_model = admission_model or (profile.admission_model if profile else None)
        if admission_model is None:
            raise ValueError("A cascade requires an absolute admission model")
        if admission_model_revision is None and profile and admission_model == profile.admission_model:
            admission_model_revision = profile.admission_revision
        self.max_candidates = max_candidates if max_candidates is not None else (profile.admission_max_candidates if profile else 8)
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or not 1 <= self.max_candidates <= 128:
            raise ValueError("max_candidates must be an integer between 1 and 128")
        self.ranker = OnDeviceReranker(
            model=model, model_revision=model_revision, model_dir=model_dir,
            backend=backend, max_length=max_length, batch_size=batch_size,
            onnx_threads=onnx_threads,
        )
        self.admission = OnDeviceReranker(
            model=admission_model, model_revision=admission_model_revision,
            model_dir=admission_model_dir, backend=backend,
            max_length=admission_max_length,
            batch_size=min(batch_size, self.max_candidates), onnx_threads=onnx_threads,
        )
        if self.admission.ranking_only:
            raise ValueError("The admission model must supply absolute relevance")
        self._lock = threading.RLock()

    def warm(self):
        with self._lock:
            self.ranker.warm()
            self.admission.warm()

    def rerank_search_results(self, query: str, search_results: list) -> list:
        if not search_results:
            return RankedSubset([], (), ())
        keys = [result["metadata"].get("cortex_candidate_key") for result in search_results]
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Cascade candidates require unique stable cortex_candidate_key identities")
        pairs = [
            [query, f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}"]
            for result in search_results
        ]
        with self._lock:
            raw = scalar_scores(self.ranker.score_pairs(pairs), len(pairs))
            record_usage("ondevice", self.ranker.model, "rerank")
            order = sorted(range(len(pairs)), key=lambda index: raw[index], reverse=True)[:self.max_candidates]
            admitted = scalar_scores(self.admission.score_pairs([pairs[index] for index in order]), len(order))
            record_usage("ondevice", self.admission.model, "rerank")
            relevance = [float(self.admission.transform(value)) for value in admitted]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in relevance):
                raise ValueError("Admission scorer returned invalid absolute relevance")
            # Both stages and every score are validated before mutating inputs.
            selected = [search_results[index] for index in order]
            for result, value in zip(selected, relevance):
                result["similarity"] = value
            selected_indices = set(order)
            return RankedSubset(
                selected,
                tuple(key for index, key in enumerate(keys) if index not in selected_indices),
                tuple(raw[index] for index in order),
            )

    def describe(self) -> dict:
        ranking = self.ranker.describe()
        admission = self.admission.describe()
        return {
            "mode": "cascade", "maxCandidates": self.max_candidates,
            "loaded": ranking["loaded"] and admission["loaded"],
            "warmed": ranking["warmed"] and admission["warmed"],
            "ranking": ranking, "admission": admission,
        }

    def to_dict(self):
        return {
            **super().to_dict(),
            "model": self.ranker.model, "model_revision": self.ranker.model_revision,
            "model_dir": self.ranker.model_dir, "backend": self.ranker.backend,
            "max_length": self.ranker.max_length, "batch_size": self.ranker.batch_size,
            "onnx_threads": self.ranker.onnx_threads, "max_candidates": self.max_candidates,
            "admission_model": self.admission.model,
            "admission_model_revision": self.admission.model_revision,
            "admission_model_dir": self.admission.model_dir,
            "admission_max_length": self.admission.max_length,
        }
