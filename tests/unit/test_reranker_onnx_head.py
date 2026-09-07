"""Numerical and rejection tests for the complete modular scoring head."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from dsrag.reranker_onnx_head import OnnxScoringHead, module_assets


class ScoringHeadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.descriptors = [
            {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
            {"idx": 3, "path": "3_LayerNorm", "type": "sentence_transformers.models.LayerNorm"},
            {"idx": 4, "path": "4_Dense", "type": "sentence_transformers.models.Dense"},
        ]
        self.write("modules.json", self.descriptors)
        self.write("config.json", {"hidden_size": 2})
        self.write("1_Pooling/config.json", {"embedding_dimension": 2, "pooling_mode": "cls"})
        self.write("2_Dense/config.json", {
            "in_features": 2, "out_features": 2, "bias": False,
            "activation_function": "torch.nn.modules.activation.GELU",
        })
        self.write("3_LayerNorm/config.json", {"dimension": 2})
        self.write("4_Dense/config.json", {
            "in_features": 2, "out_features": 1, "bias": True,
            "activation_function": "torch.nn.modules.linear.Identity", "module_output_name": "scores",
        })
        self.save("2_Dense", {"linear.weight": np.eye(2, dtype=np.float32)})
        self.save("3_LayerNorm", {"norm.weight": np.ones(2, dtype=np.float32), "norm.bias": np.zeros(2, dtype=np.float32)})
        self.save("4_Dense", {"linear.weight": np.array([[2, -3]], dtype=np.float32), "linear.bias": np.array([0.2], dtype=np.float32)})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def save(self, module, tensors):
        save_file(tensors, str(self.root / module / "model.safetensors"))

    def test_full_cls_dense_gelu_norm_dense_scores(self):
        head = OnnxScoringHead(str(self.root))
        hidden = np.array([[[1, -1], [200, -200]], [[-1, 1], [-200, 200]]], dtype=np.float32)
        actual = head(hidden, np.array([[1, 1], [1, 0]]))
        positive = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
        negative = -0.5 * (1 + math.erf(-1 / math.sqrt(2)))
        mean = (positive + negative) / 2
        variance = ((positive - mean) ** 2 + (negative - mean) ** 2) / 2
        normalized = (positive - mean) / math.sqrt(variance + 1e-5)
        np.testing.assert_allclose(actual, [[5 * normalized + 0.2], [-5 * normalized + 0.2]], atol=1e-5)

    def test_missing_head_weight_is_rejected(self):
        (self.root / "4_Dense/model.safetensors").unlink()
        with self.assertRaises(Exception):
            OnnxScoringHead(str(self.root))

    def test_untrusted_module_and_path_are_rejected(self):
        for field, value in (("path", "../outside"), ("path", "/tmp/outside"), ("type", "untrusted.RemoteCode")):
            with self.subTest(field=field, value=value):
                altered = [dict(item) for item in self.descriptors]
                altered[2][field] = value
                self.write("modules.json", altered)
                with self.assertRaises(ValueError):
                    module_assets(self.root)

    def test_malformed_weights_are_rejected_before_inference(self):
        for weight in (np.ones((2, 3), dtype=np.float32), np.full((2, 2), np.nan, dtype=np.float32)):
            self.save("2_Dense", {"linear.weight": weight})
            with self.assertRaises(ValueError):
                OnnxScoringHead(str(self.root))

    def test_backbone_shape_and_nonfinite_values_are_rejected(self):
        head = OnnxScoringHead(str(self.root))
        for hidden in (np.zeros((2, 2)), np.zeros((2, 2, 3)), np.full((2, 2, 2), np.inf)):
            with self.assertRaises(ValueError):
                head(hidden, np.ones((2, 2)))

    def test_unscored_pipeline_and_output_sigmoid_are_rejected(self):
        self.write("modules.json", self.descriptors[:-1])
        with self.assertRaisesRegex(ValueError, "scalar scores"):
            OnnxScoringHead(str(self.root))
        self.write("modules.json", self.descriptors)
        config = json.loads((self.root / "4_Dense/config.json").read_text())
        config["activation_function"] = "torch.nn.modules.activation.Sigmoid"
        self.write("4_Dense/config.json", config)
        with self.assertRaisesRegex(ValueError, "activation"):
            OnnxScoringHead(str(self.root))


if __name__ == "__main__":
    unittest.main()
