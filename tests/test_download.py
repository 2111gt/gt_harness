"""Tests for auto-download helpers (no network required)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download import (
    downloads_enabled,
    ensure_all_models,
    ensure_gguf,
    tspulse_clf_local_dir,
    tspulse_local_dir,
    _local_model_ready,
    _safe_path_component,
)
from src.models import load_models
from src.utils import MODELS_DIR, TSPULSE_MODELS_DIR, find_gguf_model


class TestDownloadFlags(unittest.TestCase):
    def test_downloads_enabled_respects_env(self):
        old = os.environ.get("GT_NO_DOWNLOAD")
        try:
            os.environ["GT_NO_DOWNLOAD"] = "1"
            self.assertFalse(downloads_enabled())
            os.environ["GT_NO_DOWNLOAD"] = "0"
            self.assertTrue(downloads_enabled())
            del os.environ["GT_NO_DOWNLOAD"]
            self.assertTrue(downloads_enabled())
        finally:
            if old is None:
                os.environ.pop("GT_NO_DOWNLOAD", None)
            else:
                os.environ["GT_NO_DOWNLOAD"] = old

    def test_ensure_all_models_offline_no_crash(self):
        old = os.environ.get("GT_NO_DOWNLOAD")
        try:
            os.environ["GT_NO_DOWNLOAD"] = "1"
            status = ensure_all_models()
            self.assertEqual(status.get("downloads"), "disabled (GT_NO_DOWNLOAD)")
            self.assertIn("gguf", status)
        finally:
            if old is None:
                os.environ.pop("GT_NO_DOWNLOAD", None)
            else:
                os.environ["GT_NO_DOWNLOAD"] = old

    def test_load_models_auto_download_false_skips_network(self):
        """Unit-test path used by analysis tests — must not hang on HF."""
        bundle = load_models(load_llm=False, load_tspulse=False, auto_download=False)
        self.assertEqual(bundle.status.get("llm"), "skipped")
        self.assertIn("statistical", bundle.tspulse_mode or bundle.status.get("tspulse", "statistical"))

    def test_ensure_gguf_uses_existing_without_download(self):
        """If a gguf already exists, ensure_gguf must not call the hub."""
        # Create a tiny fake gguf so find_gguf_model succeeds
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fake = MODELS_DIR / "_test_fake.gguf"
        try:
            fake.write_bytes(b"FAKEGGUF" * 100)
            with mock.patch("src.download.downloads_enabled", return_value=True):
                # Should short-circuit before hub import / download
                path, msg = ensure_gguf()
            self.assertIsNotNone(path)
            self.assertIn("already present", msg.lower())
        finally:
            if fake.exists():
                fake.unlink()

    def test_tspulse_local_dir_under_models(self):
        p = tspulse_local_dir(
            "ibm-granite/granite-timeseries-tspulse-r1",
            "tspulse-block-ad",
        )
        self.assertTrue(str(p).replace("\\", "/").endswith(
            "models/tspulse/ibm-granite--granite-timeseries-tspulse-r1/tspulse-block-ad"
        ) or (
            p.is_relative_to(TSPULSE_MODELS_DIR)
            if hasattr(p, "is_relative_to")
            else str(TSPULSE_MODELS_DIR) in str(p)
        ))
        self.assertTrue(p.is_dir())
        self.assertIn("tspulse", p.parts)
        self.assertNotIn(".cache", p.parts)

    def test_tspulse_clf_local_dir_under_models(self):
        p = tspulse_clf_local_dir("ibm-granite/granite-timeseries-tspulse-r1", "dual")
        self.assertIn("tspulse_clf", p.parts)
        self.assertTrue(p.is_dir())

    def test_safe_path_and_ready_helper(self):
        self.assertEqual(
            _safe_path_component("ibm-granite/foo"),
            "ibm-granite--foo",
        )
        empty = MODELS_DIR / "_empty_tspulse_test"
        empty.mkdir(parents=True, exist_ok=True)
        try:
            self.assertFalse(_local_model_ready(empty))
            (empty / "config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(_local_model_ready(empty))
        finally:
            for child in empty.iterdir():
                child.unlink()
            empty.rmdir()


if __name__ == "__main__":
    unittest.main()
