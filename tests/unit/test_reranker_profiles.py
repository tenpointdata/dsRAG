"""Artifact selection, calibration, readiness and backend safety contracts."""
import dataclasses
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from dsrag.reranker import OnDeviceReranker, Reranker
from dsrag.reranker_onnx import clip_queries, configure_tokenizer, scalar_scores, score_onnx, session_options
from dsrag.reranker_profiles import PROFILES, cpu_features, profile_for

MODEL = "cross-encoder/ettin-reranker-32m-v1"


def ettin(**kwargs):
    return OnDeviceReranker(model=MODEL, score_offset=8.0, score_scale=2.0, **kwargs)


class ProfileTests(unittest.TestCase):
    def test_architecture_specific_graph_and_revision(self):
        profile = profile_for(MODEL)
        self.assertEqual(profile.graph("arm64"), "onnx/model_qint8_arm64.onnx")
        self.assertEqual(profile.graph("amd64"), "onnx/model_quint8_avx2.onnx")
        self.assertEqual(profile.revision, "b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd")
        with self.assertRaises(ValueError):
            profile.graph("unsupported")
        with mock.patch("dsrag.reranker_profiles.platform.machine", return_value="arm64"):
            assets = ettin()._onnx_assets()
        self.assertIn("4_Dense/model.safetensors", assets)
        self.assertNotIn("model.safetensors", assets)
        self.assertNotIn("onnx/model.onnx", assets)

    def test_accelerated_graph_follows_the_cpu_not_the_architecture(self):
        """The int8 kernel the runtime dispatches on is a CPU flag, not a name."""
        profile = profile_for(MODEL)
        vnni = frozenset({"avx", "avx2", "avx512f", "avx512_vnni"})
        self.assertEqual(profile.graph("amd64", vnni), "onnx/model_qint8_avx512_vnni.onnx")
        self.assertEqual(profile.graph("amd64", frozenset({"avx", "avx2"})), "onnx/model_quint8_avx2.onnx")
        # A named architecture is a hypothetical and must never be answered
        # with the flags of whatever machine is running the question.
        with mock.patch("dsrag.reranker_profiles.cpu_features", return_value=vnni):
            self.assertEqual(profile.graph("arm64"), "onnx/model_qint8_arm64.onnx")
            self.assertEqual(profile.graph("amd64"), "onnx/model_quint8_avx2.onnx")

    def test_accelerated_graph_is_requested_from_the_hub(self):
        """The upgrade is only real if the asset list asks for that file."""
        vnni = frozenset({"avx2", "avx512_vnni"})
        with mock.patch("dsrag.reranker_profiles.platform.machine", return_value="x86_64"), \
             mock.patch("dsrag.reranker_profiles.cpu_features", return_value=vnni):
            assets = ettin()._onnx_assets()
        self.assertIn("onnx/model_qint8_avx512_vnni.onnx", assets)
        self.assertNotIn("onnx/model_quint8_avx2.onnx", assets)

    def test_cpu_features_never_raises_off_linux(self):
        cpu_features.cache_clear()
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            self.assertEqual(cpu_features(), frozenset())
        cpu_features.cache_clear()

    def test_pinned_cache_does_not_follow_main(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home) / "hub" / "models--cross-encoder--ettin-reranker-32m-v1"
            pinned = root / "snapshots" / profile_for(MODEL).revision
            pinned.mkdir(parents=True)
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text("some-other-revision")
            with mock.patch.dict(os.environ, {"HF_HOME": home}), mock.patch("dsrag.reranker.huggingface_hub.snapshot_download") as download:
                self.assertEqual(ettin()._resolve_model_dir(), str(pinned))
            download.assert_not_called()

    def test_download_is_pinned_and_partial_profile_is_not_usable(self):
        reranker = ettin()
        with mock.patch.object(reranker, "_cached_snapshot_dir", return_value=None), mock.patch("dsrag.reranker.huggingface_hub.snapshot_download", return_value="/unused") as download:
            reranker._resolve_model_dir(reranker._onnx_assets())
        self.assertEqual(download.call_args.kwargs["revision"], profile_for(MODEL).revision)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / reranker._graph_candidates()[0]
            graph.parent.mkdir()
            graph.touch()
            (root / "tokenizer.json").touch()
            self.assertFalse(reranker._onnx_snapshot_usable(directory))

    def test_calibration_requires_fit_and_is_not_beta_reshaped(self):
        profile = dataclasses.replace(profile_for(MODEL), ranking_only=False, score_offset=None, score_scale=None)
        with mock.patch.dict(PROFILES, {MODEL: profile}), self.assertRaisesRegex(ValueError, "reviewed"):
            OnDeviceReranker(model=MODEL)
        with mock.patch.dict(PROFILES, {MODEL: profile}):
            reranker = ettin()
        self.assertAlmostEqual(reranker.transform(8.0), 0.5)
        self.assertAlmostEqual(reranker.transform(10.0), 1 / (1 + math.exp(-1)))
        self.assertEqual(reranker.transform(-1e10), 0)
        self.assertEqual(reranker.transform(1e10), 1)
        restored = Reranker.from_dict(reranker.to_dict())
        self.assertEqual(restored.to_dict(), reranker.to_dict())
        for kwargs in ({"score_scale": 0}, {"score_scale": float("nan")}, {"score_offset": float("inf")}):
            with self.assertRaises(ValueError):
                OnDeviceReranker(**kwargs)

    def test_reviewed_calibration_is_bound_to_its_pair_ceiling(self):
        profile = dataclasses.replace(profile_for(MODEL), ranking_only=False, calibrated_max_length=512, score_offset=8.0, score_scale=2.0)
        with mock.patch.dict(PROFILES, {MODEL: profile}):
            OnDeviceReranker(model=MODEL, max_length=512)
            with self.assertRaisesRegex(ValueError, "max_length"):
                OnDeviceReranker(model=MODEL, max_length=384)
            with self.assertRaisesRegex(ValueError, "max_length"):
                OnDeviceReranker(model=MODEL, max_length=384, score_offset=8.0)
            reranker = ettin(max_length=384)
            self.assertEqual(reranker.describe()["calibrationVersion"], "explicit")
        OnDeviceReranker(model="BAAI/bge-reranker-base", max_length=384)

    def test_profile_failure_does_not_download_torch_fallback(self):
        reranker = ettin()
        with mock.patch.object(reranker, "_load_onnx", side_effect=ValueError("bad head")), mock.patch.object(reranker, "_load_sentence_transformers") as fallback:
            with self.assertRaisesRegex(ValueError, "bad head"):
                reranker.warm()
            fallback.assert_not_called()
        self.assertFalse(reranker.describe()["loaded"])

    def test_warm_requires_valid_forward_pass_and_retries_failure(self):
        reranker = ettin(backend="onnx")
        with mock.patch.object(reranker, "_load_onnx"), mock.patch.object(reranker, "_score_onnx", side_effect=[[float("nan")], [2.0]]) as score:
            with self.assertRaises(ValueError):
                reranker.warm()
            self.assertFalse(reranker.describe()["warmed"])
            reranker.warm()
            reranker.warm()
            self.assertEqual(score.call_count, 2)
        self.assertTrue(reranker.describe()["warmed"])

    def test_invalid_scores_do_not_mutate_candidates(self):
        reranker = ettin()
        candidates = [{"metadata": {"chunk_header": "", "chunk_text": "document"}}]
        reranker = OnDeviceReranker(model="BAAI/bge-reranker-base")
        with mock.patch.object(reranker, "score_pairs", return_value=[float("nan")]):
            with self.assertRaises(ValueError):
                reranker.rerank_search_results("query", candidates)
        self.assertNotIn("similarity", candidates[0])

    def test_sentence_transformers_requests_raw_logits(self):
        reranker = OnDeviceReranker(backend="sentence_transformers")
        reranker._cross_encoder = mock.Mock()
        reranker._cross_encoder.predict.return_value = [6.0]
        torch = types.ModuleType("torch")
        torch.nn = types.ModuleType("torch.nn")
        torch.nn.Identity = type("Identity", (), {})
        with mock.patch.dict("sys.modules", {"torch": torch, "torch.nn": torch.nn}):
            self.assertEqual(reranker._score_sentence_transformers([["q", "d"]]), [6.0])
        self.assertIsInstance(reranker._cross_encoder.predict.call_args.kwargs["activation_fn"], torch.nn.Identity)

    def test_shared_inference_is_serialized(self):
        reranker = ettin(backend="onnx")
        active = []
        overlap = []
        def predict(pairs):
            active.append(True)
            overlap.append(len(active))
            time.sleep(0.01)
            active.pop()
            return [1.0]
        with mock.patch.object(reranker, "_load_onnx") as loader, mock.patch.object(reranker, "_score_onnx", side_effect=predict):
            threads = [threading.Thread(target=reranker.score_pairs, args=([["q", "d"]],)) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            loader.assert_called_once()
        self.assertEqual(max(overlap), 1)


class OnnxContracts(unittest.TestCase):
    def test_threads_are_bounded_and_do_not_spin(self):
        with mock.patch("dsrag.reranker.os.cpu_count", return_value=64):
            self.assertEqual(OnDeviceReranker().onnx_threads, 4)
        options = session_options(3)
        self.assertEqual(options.intra_op_num_threads, 3)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")
        self.assertEqual(options.get_session_config_entry("session.inter_op.allow_spinning"), "0")

    def test_tokenizer_uses_real_padding_and_passage_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"pad_token_id":50283,"max_position_embeddings":7999}')
            (root / "tokenizer_config.json").write_text('{"pad_token":"[PAD]"}')
            tokenizer = mock.Mock(token_to_id=mock.Mock(return_value=50283))
            configure_tokenizer(tokenizer, directory, 512)
            tokenizer.enable_padding.assert_called_once_with(pad_id=50283, pad_token="[PAD]")
            tokenizer.enable_truncation.assert_called_once_with(max_length=512, strategy="only_second", direction="right")
            tokenizer.token_to_id.return_value = 0
            with self.assertRaisesRegex(ValueError, "disagree"):
                configure_tokenizer(tokenizer, directory, 512)

    def test_overlong_queries_reserve_passage_tokens_and_still_score(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "q": 4, "passage": 5}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
        query_tokenizer = Tokenizer.from_str(tokenizer.to_str())
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(configure_tokenizer(tokenizer, directory, 16), 16)
        session = mock.Mock()
        session.get_inputs.return_value = [mock.Mock(), mock.Mock()]
        for entry, name in zip(session.get_inputs.return_value, ["input_ids", "attention_mask"]):
            entry.name = name
        session.run.return_value = [np.array([[2.0]])]
        scores = score_onnx(
            session, tokenizer, None, [["q " * 1000, "passage " * 50]], 16,
            query_tokenizer=query_tokenizer, query_limit=8,
        )
        self.assertEqual(scores, [2.0])
        ids = session.run.call_args.args[1]["input_ids"][0].tolist()
        self.assertEqual(len(ids), 16)
        self.assertEqual(ids.count(vocab["q"]), 8)
        self.assertEqual(ids.count(vocab["passage"]), 5)
        query = " q   q "
        self.assertEqual(clip_queries([[query, "passage"]], query_tokenizer, 8), [(query, "passage")])
        # The new query path never changes the pair tokenizer's only-second rule.
        self.assertEqual(tokenizer.truncation["strategy"], "only_second")

    def test_scores_reject_hidden_states_count_mismatch_and_nonfinite(self):
        for output in (np.zeros((2, 4, 3)), np.zeros((3, 1)), [[0], [float("nan")]], [float("inf"), 0]):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                scalar_scores(output, 2)
        self.assertEqual(scalar_scores([[1], [2]], 2), [1.0, 2.0])

    def test_onnx_graph_inputs_and_outputs_fail_closed(self):
        session = mock.Mock()
        session.get_inputs.return_value = [mock.Mock(name="input")]
        session.get_inputs.return_value[0].name = "unknown"
        with self.assertRaisesRegex(ValueError, "inputs"):
            score_onnx(session, mock.Mock(), None, [["q", "d"]], 16)
        session.run.assert_not_called()
        session.get_inputs.return_value = [mock.Mock(), mock.Mock()]
        for entry, name in zip(session.get_inputs.return_value, ["input_ids", "attention_mask"]):
            entry.name = name
        tokenizer = mock.Mock()
        tokenizer.encode_batch.return_value = [mock.Mock(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0])]
        session.run.return_value = [np.zeros((1, 2, 384))]
        with self.assertRaises(ValueError):
            score_onnx(session, tokenizer, None, [["q", "d"]], 16)


if __name__ == "__main__":
    unittest.main()
