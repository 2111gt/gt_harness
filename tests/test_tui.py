"""Structural + smoke tests for Textual TUI entry (no interactive loop)."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestTuiStructure(unittest.TestCase):
    def test_tui_module_exists_and_exports_run(self):
        tui_path = ROOT / "src" / "tui_app.py"
        self.assertTrue(tui_path.is_file())
        src = tui_path.read_text(encoding="utf-8")
        self.assertIn("GTDiagnosticTUI", src)
        self.assertIn("def run_tui", src)
        self.assertIn("Run diagnosis", src)
        self.assertIn("Save & Learn", src)
        self.assertIn("Alerts", src)
        self.assertIn("Trips/Event", src)
        self.assertIn("configure_tui_logging", src)
        # Progress / status bar wiring (message-based, non-blocking)
        self.assertIn("ProgressBar", src)
        self.assertIn("StepList", src)
        self.assertIn("progress=", src)
        self.assertIn("PipelineProgress", src)
        self.assertIn("post_message", src)
        self.assertIn("diagnosis_step_plan", src)
        self.assertIn("report-scroll", src)
        self.assertIn("to_display_markdown", src)
        self.assertIn("last_diagnosis_report.md", src)
        self.assertIn("RichLog", src)
        self.assertIn("LiveOutput", src)
        self.assertIn("live=", src)
        self.assertIn("live-panel", src)
        self.assertIn("live-log", src)
        self.assertIn("_append_live", src)
        self.assertIn("btn-browse", src)
        self.assertIn("normalize_csv_path", src)
        self.assertIn("action_browse_csv", src)
        self.assertIn("CsvDropZone", src)
        self.assertIn("csv-drop", src)
        self.assertIn("PathDropped", src)
        self.assertIn("_extract_dropped_paths", src)
        self.assertIn("btn-new", src)
        self.assertIn("action_new_session", src)
        self.assertIn("New session", src)
        # Path sanitizer unit check
        from src.tui_app import normalize_csv_path

        self.assertEqual(
            normalize_csv_path(r'"C:\data\unit.csv"'),
            r"C:\data\unit.csv" if not __import__("pathlib").Path(r"C:\data\unit.csv").exists()
            else str(__import__("pathlib").Path(r"C:\data\unit.csv").resolve()),
        )
        # Quoted non-existent path still strips quotes
        cleaned = normalize_csv_path('"Z:\\no_such_file_xyz.csv"')
        self.assertTrue(cleaned.endswith("no_such_file_xyz.csv"))
        self.assertFalse(cleaned.startswith('"'))
        # tqdm must not be patched with a lambda (breaks HF hub / TS Pulse)
        utils = (ROOT / "src" / "utils.py").read_text(encoding="utf-8")
        self.assertIn("_silence_tqdm_safely", utils)
        self.assertNotIn(
            "_tqdm.tqdm = lambda",
            utils,
            "lambda tqdm patch breaks huggingface_hub subclassing",
        )
        self.assertIn("run_hidden_subprocess", (ROOT / "src" / "utils.py").read_text(encoding="utf-8"))
        self.assertIn("CREATE_NO_WINDOW", (ROOT / "src" / "utils.py").read_text(encoding="utf-8"))
        # Parse AST — ensures file is valid Python
        tree = ast.parse(src)
        names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertIn("GTDiagnosticTUI", names)
        self.assertIn("run_tui", names)
        self.assertIn("StepList", names)
        self.assertIn("StatusBar", names)

    def test_app_entry_defaults_to_tui_not_gradio(self):
        app_src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("run_tui", app_src)
        self.assertIn("cli-once", app_src)
        # Gradio should not be the default launch path
        self.assertNotIn("demo.launch", app_src)
        self.assertNotIn("import gradio", app_src)

    def test_import_tui_app_module(self):
        from src import tui_app

        self.assertTrue(callable(tui_app.run_tui))
        self.assertTrue(hasattr(tui_app, "GTDiagnosticTUI"))


if __name__ == "__main__":
    unittest.main()
