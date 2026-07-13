"""Tests for UI backends (textual / gui) + JSON bridge."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import run_diagnosis
from src.bridge import result_to_bridge_dict, run_json_once
from src.models import ModelBundle
from src.ui_launch import VALID_UIS, resolve_ui


class TestResolveUi(unittest.TestCase):
    def test_defaults_and_aliases(self):
        old = os.environ.pop("GT_UI", None)
        try:
            self.assertEqual(resolve_ui(None), "textual")
            self.assertEqual(resolve_ui("textual"), "textual")
            self.assertEqual(resolve_ui("gui"), "gui")
            self.assertEqual(resolve_ui("desktop"), "gui")
            # Legacy ratatui alias maps to gui
            self.assertEqual(resolve_ui("ratatui"), "gui")
            os.environ["GT_UI"] = "gui"
            self.assertEqual(resolve_ui(None), "gui")
            with self.assertRaises(ValueError):
                resolve_ui("swing")
        finally:
            if old is None:
                os.environ.pop("GT_UI", None)
            else:
                os.environ["GT_UI"] = old

    def test_valid_uis(self):
        self.assertIn("textual", VALID_UIS)
        self.assertIn("gui", VALID_UIS)
        self.assertNotIn("ratatui", VALID_UIS)

    def test_ratatui_removed(self):
        self.assertFalse((ROOT / "tui_ratatui").exists())
        self.assertTrue((ROOT / "src" / "gui_app.py").is_file())
        gui_src = (ROOT / "src" / "gui_app.py").read_text(encoding="utf-8")
        self.assertIn("run_gui", gui_src)
        self.assertIn("GTHarnessGUI", gui_src)
        self.assertIn("customtkinter", gui_src.lower() + "customtkinter")


class TestBridge(unittest.TestCase):
    def test_result_to_bridge_dict_has_plots(self):
        bundle = ModelBundle(
            tspulse_mode="statistical",
            status={"tspulse": "test", "llm": "offline"},
            llm=None,
        )
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame(
            {
                "load_MW": 50 + rng.normal(0, 0.2, n),
                "EGT_spread_C": 12 + rng.normal(0, 0.3, n),
            }
        )
        df.loc[20:24, "EGT_spread_C"] = [40, 48, 55, 50, 42]
        result = run_diagnosis(
            df=df,
            mode="Alerts",
            context="bridge test",
            bundle=bundle,
            full_reflection=False,
        )
        payload = result_to_bridge_dict(result, bundle=bundle)
        self.assertTrue(payload["ok"])
        self.assertIn("evidence_ascii", payload)
        self.assertTrue(payload["evidence_ascii"] or payload["proof_channels"])
        self.assertIn("display_markdown", payload)
        self.assertIn("severity", payload)
        raw = json.dumps(payload)
        self.assertIn("schema_version", raw)

    def test_json_once_missing_file(self):
        code = run_json_once(
            str(ROOT / "samples" / "no_such_file_xyz.csv"),
            mode="Alerts",
            auto_download=False,
        )
        self.assertEqual(code, 2)


class TestAppEntryUiFlag(unittest.TestCase):
    def test_app_py_exposes_ui_and_json_once(self):
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("--ui", src)
        self.assertIn("gui", src)
        self.assertNotIn("ratatui", src)
        self.assertIn("--json-once", src)
        self.assertIn("launch_ui", src)


if __name__ == "__main__":
    unittest.main()
