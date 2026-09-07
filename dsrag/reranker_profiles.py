"""Reviewed model artifacts and calibration defaults for local reranking."""
from dataclasses import dataclass
import platform
from typing import Mapping, Optional, Tuple


LEGACY_ONNX_GRAPHS = (
    "onnx/model_quantized.onnx", "onnx/model.onnx",
    "model_quantized.onnx", "model.onnx",
)
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json", "config.json")


@dataclass(frozen=True)
class RerankerProfile:
    revision: str
    graphs: Mapping[str, str]
    assets: Tuple[str, ...]
    ranking_only: bool = False
    admission_model: Optional[str] = None
    admission_revision: Optional[str] = None
    admission_max_candidates: int = 8
    calibration: str = "logistic"
    calibrated_max_length: Optional[int] = None
    calibration_version: Optional[str] = None
    score_offset: Optional[float] = None
    score_scale: Optional[float] = None

    def graph(self, architecture: Optional[str] = None) -> str:
        architecture = (architecture or platform.machine()).lower()
        aliases = {"arm64": "aarch64", "amd64": "x86_64"}
        architecture = aliases.get(architecture, architecture)
        if architecture not in self.graphs:
            raise ValueError(f"No reviewed reranker graph for architecture {architecture!r}")
        return self.graphs[architecture]


# Relative ranking scores are never treated as calibrated absolute relevance.
PROFILES = {
    "cross-encoder/ettin-reranker-32m-v1": RerankerProfile(
        revision="b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd",
        graphs={
            "aarch64": "onnx/model_qint8_arm64.onnx",
            "x86_64": "onnx/model_quint8_avx2.onnx",
        },
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
