import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import hf_utils
from common.errors import ConfigError, UnsupportedError


def _write_safetensors(path: str, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Minimal safetensors writer for unit tests (dtype, shape, raw payload)."""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        blobs.append(raw)
        offset = end
    h = json.dumps(header).encode("utf-8")
    # 8-byte align header payload (common convention; readers tolerate padding spaces)
    pad = (8 - len(h) % 8) % 8
    h = h + (b" " * pad)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


class TestConfigFromHf(unittest.TestCase):
    def test_scalar_intermediate_size(self):
        got = hf_utils.config_from_hf(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "vocab_size": 100,
                "max_position_embeddings": 256,
                "rope_theta": 10000.0,
            }
        )
        self.assertEqual(got["intermediate_size"], 128)
        self.assertEqual(got["num_layers"], 2)

    def test_gemma3n_list_intermediate_size_from_text_config(self):
        # MatFormer: per-layer MLP widths (Gemma-3n); flatten to max for aria metadata.
        got = hf_utils.config_from_hf(
            {
                "model_type": "gemma3n",
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "intermediate_size": [8192, 16384, 8192, 4096],
                    "vocab_size": 262144,
                    "max_position_embeddings": 32768,
                    "rope_theta": 1000000.0,
                },
            }
        )
        self.assertEqual(got["hidden_size"], 2048)
        self.assertEqual(got["intermediate_size"], 16384)
        self.assertEqual(got["num_layers"], 4)

    def test_empty_intermediate_size_list_raises(self):
        with self.assertRaises(ConfigError):
            hf_utils.config_from_hf(
                {
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "intermediate_size": [],
                }
            )

    def test_gemma4_nested_fields_and_rope_parameters(self):
        got = hf_utils.config_from_hf(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "vision_config": {"hidden_size": 768, "intermediate_size": 3072},
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "intermediate_size": 6144,
                    "vocab_size": 262144,
                    "max_position_embeddings": 131072,
                    "head_dim": 256,
                    "num_kv_shared_layers": 20,
                    "use_double_wide_mlp": True,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "layer_types": ["sliding_attention", "full_attention"],
                    "rope_parameters": {
                        "full_attention": {"rope_theta": 1000000.0},
                        "sliding_attention": {"rope_theta": 10000.0},
                    },
                    "tie_word_embeddings": True,
                },
            }
        )
        self.assertEqual(got["hidden_size"], 1536)
        self.assertEqual(got["intermediate_size"], 6144)
        self.assertEqual(got["head_dim"], 256)
        self.assertEqual(got["num_kv_shared_layers"], 20)
        self.assertTrue(got["use_double_wide_mlp"])
        self.assertEqual(got["hidden_act"], "gelu_pytorch_tanh")
        self.assertEqual(got["rope_theta"], 1000000.0)
        self.assertEqual(got["layer_types"], ["sliding_attention", "full_attention"])
        self.assertTrue(got["tie_word_embeddings"])

    def test_lfm2_block_ff_dim(self):
        got = hf_utils.config_from_hf(
            {
                "hidden_size": 1024,
                "num_hidden_layers": 16,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "block_ff_dim": 6656,
                "vocab_size": 65536,
                "max_position_embeddings": 32768,
                "layer_types": ["conv", "full_attention", "conv"],
                "rope_parameters": {"rope_theta": 1000000.0},
            }
        )
        self.assertEqual(got["intermediate_size"], 6656)
        self.assertEqual(got["rope_theta"], 1000000.0)
        self.assertEqual(got["layer_types"][0], "conv")

    def test_qwen35_linear_attention_and_rope(self):
        got = hf_utils.config_from_hf(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "intermediate_size": 6144,
                    "vocab_size": 151936,
                    "max_position_embeddings": 262144,
                    "head_dim": 256,
                    "hidden_act": "silu",
                    "layer_types": ["linear_attention", "full_attention"],
                    "rope_parameters": {"mrope": {"rope_theta": 10000000.0}},
                },
            }
        )
        self.assertEqual(got["head_dim"], 256)
        self.assertEqual(got["rope_theta"], 10000000.0)
        self.assertIn("linear_attention", got["layer_types"])

    def test_inkling_model_max_length_and_moe(self):
        got = hf_utils.config_from_hf(
            {
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "intermediate_size": 2048,
                    "vocab_size": 128256,
                    "model_max_length": 1048576,
                    "num_experts": 256,
                    "num_experts_per_tok": 6,
                    "rope_theta": 500000.0,
                },
            }
        )
        self.assertEqual(got["context_length"], 1048576)
        self.assertEqual(got["num_experts"], 256)
        self.assertEqual(got["num_experts_per_tok"], 6)

    def test_openvla_llm_nested_config(self):
        got = hf_utils.config_from_hf(
            {
                "model_type": "openvla",
                "llm_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 32,
                    "intermediate_size": 11008,
                    "vocab_size": 32000,
                    "max_position_embeddings": 4096,
                    "rope_theta": 10000.0,
                    "hidden_act": "silu",
                },
            }
        )
        self.assertEqual(got["hidden_size"], 4096)
        self.assertEqual(got["num_layers"], 32)

    def test_vla_missing_geometry_raises(self):
        with self.assertRaises(ConfigError):
            hf_utils.config_from_hf({"vlm_family": "qwen3_vl"})


class TestHfUtilsSafetensors(unittest.TestCase):
    def test_iter_f32_and_f16(self):
        f32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        f16 = np.array([0.5, -1.5], dtype=np.float16)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "w.safetensors")
            _write_safetensors(
                path,
                {
                    "a": ("F32", f32.shape, f32.tobytes()),
                    "b": ("F16", f16.shape, f16.tobytes()),
                },
            )
            got = dict(hf_utils.iter_safetensors_f32(path))
        self.assertEqual(set(got), {"a", "b"})
        np.testing.assert_allclose(got["a"], f32)
        np.testing.assert_allclose(got["b"], f16.astype(np.float32), rtol=1e-3)
        self.assertEqual(got["a"].dtype, np.float32)

    def test_iter_bf16(self):
        # 1.0 and 2.0 in bfloat16 bit patterns
        u16 = np.array([0x3F80, 0x4000], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bf16.safetensors")
            _write_safetensors(path, {"w": ("BF16", (2,), u16.tobytes())})
            got = dict(hf_utils.iter_safetensors_f32(path))
        np.testing.assert_allclose(got["w"], np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(got["w"].dtype, np.float32)

    def test_bf16_size_mismatch(self):
        raw = np.array([0x3F80], dtype=np.uint16).tobytes()
        with self.assertRaises(UnsupportedError):
            hf_utils._bf16_raw_to_f32(raw, (2,))

    def test_unsupported_dtype(self):
        with self.assertRaises(UnsupportedError):
            hf_utils._safetensors_bytes_to_f32(b"\x00", "F8_E4M3", (1,))


if __name__ == "__main__":
    unittest.main()
