"""Reviewed model artifacts and calibration defaults for local reranking."""
from dataclasses import dataclass, field
from functools import lru_cache
import platform
from typing import FrozenSet, Mapping, Optional, Tuple


LEGACY_ONNX_GRAPHS = (
    "onnx/model_quantized.onnx", "onnx/model.onnx",
    "model_quantized.onnx", "model.onnx",
)
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json", "config.json")


@lru_cache(maxsize=1)
def cpu_features() -> FrozenSet[str]:
    """The ISA flags this CPU reports, which is what the runtime dispatches on.

    Architecture alone does not say which int8 kernel a host can run: every
    x86_64 CPU since 2013 answers to that name, and the ones with VNNI execute
    the same quantized graph several times faster. Only Linux publishes the
    flags; every other host falls back to the architecture baseline, which is
    what already ran there.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


@dataclass(frozen=True)
class RerankerProfile:
    revision: str
    graphs: Mapping[str, str]
    assets: Tuple[str, ...]
    #: CPU feature -> graph, most specific first, consulted before the
    #: architecture baseline. Read from the host only when the caller names
    #: neither argument; asking about an architecture is a hypothetical, and
    #: answering it with THIS machine's flags would hand an aarch64 question
    #: the graph an x86 CPU happens to support.
    accelerated_graphs: Mapping[str, str] = field(default_factory=dict)
    ranking_only: bool = False
    admission_model: Optional[str] = None
    admission_revision: Optional[str] = None
    admission_max_candidates: int = 8
    calibration: str = "logistic"
    calibrated_max_length: Optional[int] = None
    calibration_version: Optional[str] = None
    score_offset: Optional[float] = None
    score_scale: Optional[float] = None

    def graph(self, architecture: Optional[str] = None, features: Optional[FrozenSet[str]] = None) -> str:
        probing = architecture is None and features is None
        architecture = (architecture or platform.machine()).lower()
        aliases = {"arm64": "aarch64", "amd64": "x86_64"}
        architecture = aliases.get(architecture, architecture)
        if architecture not in self.graphs:
            raise ValueError(f"No reviewed reranker graph for architecture {architecture!r}")
        available = cpu_features() if probing else (features or frozenset())
        for flag, name in self.accelerated_graphs.items():
            if flag in available:
                return name
        return self.graphs[architecture]


# Relative ranking scores are never treated as calibrated absolute relevance.
PROFILES = {
    "cross-encoder/ettin-reranker-32m-v1": RerankerProfile(
        revision="b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd",
        graphs={
            "aarch64": "onnx/model_qint8_arm64.onnx",
            "x86_64": "onnx/model_quint8_avx2.onnx",
        },
        # Measured on us-east1 Compute Engine, 24 pairs at the corpus's own
        # passage length, four threads: 0.631 s against 0.984 s for the AVX2
        # graph on Cascade Lake, and 0.306 s against 0.533 s on Emerald
        # Rapids. Broadwell has neither flag and keeps the AVX2 baseline.
        accelerated_graphs={"avx512_vnni": "onnx/model_qint8_avx512_vnni.onnx"},
        ranking_only=True,
        admission_model="BAAI/bge-reranker-base",
        admission_revision="2cfc18c9415c912f9d8155881c133215df768a70",
        assets=(
            "modules.json", "1_Pooling/config.json", "2_Dense/config.json",
            "2_Dense/model.safetensors", "3_LayerNorm/config.json",
            "3_LayerNorm/model.safetensors", "4_Dense/config.json",
            "4_Dense/model.safetensors",
        ),
    ),
}


def profile_for(model: str) -> Optional[RerankerProfile]:
    return PROFILES.get(model)
