"""Bounded CPU execution and strict cross-encoder I/O validation."""
from pathlib import Path
from typing import Optional

import numpy as np

from dsrag.reranker_onnx_head import OnnxScoringHead, read_json
from dsrag.utils.imports import onnxruntime


def session_options(threads: int):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def configure_tokenizer(tokenizer, directory: str, max_length: int) -> int:
    root = Path(directory)
    config = read_json(root / "config.json") if (root / "config.json").is_file() else {}
    settings = read_json(root / "tokenizer_config.json") if (root / "tokenizer_config.json").is_file() else {}
    ceiling = config.get("max_position_embeddings")
    if isinstance(ceiling, int) and ceiling > 0:
        max_length = min(max_length, ceiling)
    pad_token = settings.get("pad_token")
    if isinstance(pad_token, dict):
        pad_token = pad_token.get("content")
    candidates = [pad_token] if isinstance(pad_token, str) else ["[PAD]", "<pad>"]
    for candidate in candidates:
        pad_id = tokenizer.token_to_id(candidate)
        if isinstance(pad_id, int) and pad_id >= 0:
            declared = config.get("pad_token_id")
            if declared is not None and declared != pad_id:
                raise ValueError("Reranker tokenizer and model disagree on padding ID")
            tokenizer.enable_padding(pad_id=pad_id, pad_token=candidate)
            tokenizer.enable_truncation(max_length=max_length, strategy="only_second", direction="right")
            return max_length
    raise ValueError("Reranker tokenizer has no recognized padding token")


def clip_queries(pairs: list, tokenizer, limit: int) -> list:
    """Reserve passage space; short queries keep their exact original text.

    The separate query tokenizer has no padding or truncation. Encoding without
    special tokens avoids only-second truncation errors on overlong queries.
    Decode/re-encode validation also handles token boundaries that normalize.
    """
    queries = {}
    clipped = []
    for query, passage in pairs:
        if query not in queries:
            ids = tokenizer.encode(query, add_special_tokens=False).ids
            value = query
            if len(ids) > limit:
                prefix = ids[:limit]
                while prefix:
                    value = tokenizer.decode(prefix, skip_special_tokens=False)
                    if len(tokenizer.encode(value, add_special_tokens=False).ids) <= limit:
                        break
                    prefix.pop()
                else:
                    value = ""
            queries[query] = value
        clipped.append((queries[query], passage))
    return clipped


def scalar_scores(output, count: int) -> list:
    values = np.asarray(output)
    if values.shape == (count, 1):
        values = values[:, 0]
    if values.shape != (count,) or values.dtype.kind not in "fi" or not np.isfinite(values).all():
        raise ValueError(f"Reranker must return {count} finite scalar scores; got {values.shape}")
    return [float(value) for value in values]


def score_onnx(
    session, tokenizer, head: Optional[OnnxScoringHead], pairs: list, batch_size: int,
    query_tokenizer=None, query_limit: int = 128,
) -> list:
    declared = {item.name for item in session.get_inputs()}
    supported = {"input_ids", "attention_mask", "token_type_ids"}
    if not {"input_ids", "attention_mask"}.issubset(declared) or declared - supported:
        raise ValueError(f"Unsupported reranker ONNX inputs: {sorted(declared)}")
    pairs = clip_queries(pairs, query_tokenizer, query_limit) if query_tokenizer is not None else pairs
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        encodings = tokenizer.encode_batch(batch)
        if len(encodings) != len(batch):
            raise ValueError("Reranker tokenizer returned the wrong pair count")
        arrays = {
            "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.asarray([item.attention_mask for item in encodings], dtype=np.int64),
            "token_type_ids": np.asarray([item.type_ids for item in encodings], dtype=np.int64),
        }
        shape = arrays["input_ids"].shape
        if len(shape) != 2 or shape[1] == 0 or any(array.shape != shape for array in arrays.values()):
            raise ValueError("Reranker tokenizer returned inconsistent tensor shapes")
        outputs = session.run(None, {key: value for key, value in arrays.items() if key in declared})
        if len(outputs) != 1:
            raise ValueError("Reranker ONNX graph must return exactly one output")
        logits = np.asarray(outputs[0])
        if head is not None:
            logits = head(logits, arrays["attention_mask"])
        scores.extend(scalar_scores(logits, len(batch)))
    return scores
