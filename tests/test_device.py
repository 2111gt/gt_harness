"""Tests for CUDA preference helpers (no GPU required)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.device import (
    force_cpu,
    preferred_n_gpu_layers,
    torch_device,
)
from src.models import LLMConfig


class TestDevicePreference(unittest.TestCase):
    def tearDown(self) -> None:
        # Clear cached probes between tests
        from src import device as d

        d.torch_cuda_available.cache_clear()
        d.nvidia_smi_available.cache_clear()
        for k in (
            "GT_FORCE_CPU",
            "GT_NO_GPU",
            "GT_N_GPU_LAYERS",
            "GT_TORCH_DEVICE",
        ):
            os.environ.pop(k, None)

    def test_force_cpu_zeros_layers(self):
        os.environ["GT_FORCE_CPU"] = "1"
        from src import device as d

        d.torch_cuda_available.cache_clear()
        d.nvidia_smi_available.cache_clear()
        self.assertTrue(force_cpu())
        self.assertEqual(preferred_n_gpu_layers(), 0)
        self.assertEqual(torch_device(), "cpu")

    def test_explicit_n_gpu_layers(self):
        os.environ["GT_N_GPU_LAYERS"] = "32"
        from src import device as d

        d.torch_cuda_available.cache_clear()
        d.nvidia_smi_available.cache_clear()
        self.assertEqual(preferred_n_gpu_layers(), 32)

    def test_llm_config_resolved_layers_nonneg(self):
        cfg = LLMConfig(n_gpu_layers=0)
        self.assertEqual(cfg.resolved_n_gpu_layers(), 0)
        cfg2 = LLMConfig(n_gpu_layers=16)
        self.assertEqual(cfg2.resolved_n_gpu_layers(), 16)

    def test_llm_config_auto_uses_helper(self):
        os.environ["GT_FORCE_CPU"] = "1"
        from src import device as d

        d.torch_cuda_available.cache_clear()
        d.nvidia_smi_available.cache_clear()
        cfg = LLMConfig(n_gpu_layers=-1)
        self.assertEqual(cfg.resolved_n_gpu_layers(), 0)

    def test_prefer_cuda_sets_high_layers(self):
        os.environ.pop("GT_FORCE_CPU", None)
        os.environ.pop("GT_N_GPU_LAYERS", None)
        from src import device as d

        d.torch_cuda_available.cache_clear()
        d.nvidia_smi_available.cache_clear()
        with mock.patch.object(d, "cuda_likely_available", return_value=True):
            self.assertEqual(d.preferred_n_gpu_layers(), 99)

    def test_llama_cli_runner_accepts_ngl(self):
        from src.llama_cli_backend import LlamaCliRunner

        r = LlamaCliRunner(
            model_path="x.gguf",
            cli_path="llama-cli.exe",
            n_gpu_layers=40,
        )
        self.assertEqual(r.n_gpu_layers, 40)


if __name__ == "__main__":
    unittest.main()
