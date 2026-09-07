"""Allowlisted Sentence Transformers scoring modules without Torch or remote code."""
import json
import math
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf


MODULE_TYPES = {
    "sentence_transformers.base.modules.transformer.Transformer": "transformer",
    "sentence_transformers.models.Transformer": "transformer",
    "sentence_transformers.sentence_transformer.modules.pooling.Pooling": "pooling",
    "sentence_transformers.models.Pooling": "pooling",
    "sentence_transformers.base.modules.dense.Dense": "dense",
    "sentence_transformers.models.Dense": "dense",
    "sentence_transformers.sentence_transformer.modules.layer_norm.LayerNorm": "norm",
    "sentence_transformers.models.LayerNorm": "norm",
}
ACTIVATIONS = {
    "torch.nn.modules.activation.GELU": "gelu",
    "torch.nn.modules.linear.Identity": "identity",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def module_path(directory: Path, path: str) -> Path:
    if not isinstance(path, str):
        raise ValueError("Module path must be a relative string")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path:
        raise ValueError(f"Unsafe reranker module path: {path!r}")
    return directory.joinpath(*relative.parts)


def module_assets(directory: Path) -> List[str]:
    """Validate manifest paths before requesting a missing head from the Hub."""
    descriptors = json.loads((directory / "modules.json").read_text(encoding="utf-8"))
    if not isinstance(descriptors, list) or not 2 <= len(descriptors) <= 16:
        raise ValueError("Reranker modules must be a bounded nonempty pipeline")
    assets = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or descriptor.get("idx") != index:
            raise ValueError("Reranker modules must have contiguous indices")
        kind = MODULE_TYPES.get(descriptor.get("type"))
        if kind is None:
            raise ValueError(f"Unsupported reranker module: {descriptor.get('type')!r}")
        path = descriptor.get("path")
        module_path(directory, path)
        if (index == 0) != (kind == "transformer") or (index == 0 and path != ""):
            raise ValueError("Reranker pipeline must start with one root Transformer")
        if index and not path:
            raise ValueError("Scoring modules require an explicit relative directory")
        if index:
            assets.append(f"{path}/config.json")
            if kind in ("dense", "norm"):
                assets.append(f"{path}/model.safetensors")
    return assets


def weights(path: Path, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    loaded = load_file(str(path / "model.safetensors"))
    if set(loaded) != set(shapes):
        raise ValueError(f"Unexpected scoring weights in {path.name}")
    for key, shape in shapes.items():
        value = loaded[key]
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Invalid reranker weight {path.name}/{key}")
        loaded[key] = value.astype(np.float32, copy=False)
    return loaded


class OnnxScoringHead:
    def __init__(self, directory: str):
        root = Path(directory)
        module_assets(root)
        descriptors = json.loads((root / "modules.json").read_text(encoding="utf-8"))
        hidden = read_json(root / "config.json").get("hidden_size")
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError("Reranker hidden_size must be a positive integer")
        self.hidden_size = hidden
        self.steps = []
        pooled = False
        scored = False
        dimension = hidden
        for descriptor in descriptors[1:]:
            kind = MODULE_TYPES[descriptor["type"]]
            path = module_path(root, descriptor["path"])
            config = read_json(path / "config.json")
            if scored:
                raise ValueError("Scores must be the final reranker module output")
            if kind == "pooling":
                if pooled or config.get("pooling_mode") not in ("cls", "mean"):
                    raise ValueError("Exactly one supported pooling module is required")
                if config.get("embedding_dimension") != dimension:
                    raise ValueError("Pooling dimension does not match the backbone")
                self.steps.append((kind, config, {}))
                pooled = True
                continue
            if not pooled:
                raise ValueError("Scoring modules require pooling first")
            if kind == "dense":
                output = config.get("out_features")
                if config.get("in_features") != dimension or not isinstance(output, int) or output <= 0:
                    raise ValueError("Invalid Dense scoring dimensions")
                activation = ACTIVATIONS.get(config.get("activation_function"))
                if activation is None or config.get("module_input_name", "sentence_embedding") != "sentence_embedding":
                    raise ValueError("Unsupported Dense activation or input")
                name = config.get("module_output_name", "sentence_embedding")
                if name not in ("sentence_embedding", "scores"):
                    raise ValueError("Unsupported Dense output")
                scored = name == "scores"
                if scored and (output != 1 or activation != "identity"):
                    raise ValueError("Final score must be one untransformed logit")
                bias = config.get("bias", True)
                if not isinstance(bias, bool):
                    raise ValueError("Dense bias must be a boolean")
                shapes = {"linear.weight": (output, dimension)}
                if bias:
                    shapes["linear.bias"] = (output,)
                config = {**config, "activation": activation}
                dimension = output
            elif kind == "norm":
                if config.get("dimension") != dimension:
                    raise ValueError("LayerNorm dimension does not match its input")
                epsilon = config.get("eps", 1e-5)
                if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("LayerNorm epsilon must be positive and finite")
                config = {**config, "eps": epsilon}
                shapes = {"norm.weight": (dimension,), "norm.bias": (dimension,)}
            else:
                raise ValueError("Unsupported scoring pipeline order")
            self.steps.append((kind, config, weights(path, shapes)))
        if not pooled or not scored:
            raise ValueError("Reranker pipeline does not produce scalar scores")

    def __call__(self, hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape or hidden.shape[2] != self.hidden_size:
            raise ValueError("Reranker backbone output shape does not match the scoring head")
        if not np.isfinite(hidden).all():
            raise ValueError("Reranker backbone returned nonfinite values")
        values = hidden
        for kind, config, tensors in self.steps:
            if kind == "pooling":
                if config["pooling_mode"] == "cls":
                    values = values[:, 0, :]
                else:
                    mask = attention_mask[:, :, None]
                    values = (values * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            elif kind == "dense":
                values = values @ tensors["linear.weight"].T
                if "linear.bias" in tensors:
                    values = values + tensors["linear.bias"]
                if config["activation"] == "gelu":
                    values = values * np.float32(0.5) * (np.float32(1) + erf(values / np.float32(math.sqrt(2))))
            elif kind == "norm":
                centered = values - values.mean(axis=-1, keepdims=True)
                values = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + np.float32(config["eps"]))
                values = values * tensors["norm.weight"] + tensors["norm.bias"]
        return values
