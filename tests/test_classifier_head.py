"""Tests for optional TS Pulse classification head wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    DEFAULT_CLF_LABELS,
    ModelBundle,
    classify_signature,
    detect_anomalies,
    _clf_labels_from_env,
)


class TestClassifierScaffold(unittest.TestCase):
    def test_default_labels(self):
        labels = _clf_labels_from_env()
        self.assertEqual(labels, list(DEFAULT_CLF_LABELS))
        self.assertIn("cold_spot", labels)
        self.assertIn("hets", labels)

    def test_statistical_bundle_skips_clf_auto_load(self):
        """Intentional statistical tests must not pull the classifier."""
        bundle = ModelBundle(
            tspulse=None,
            tspulse_mode="statistical",
            status={"tspulse": "statistical"},
        )
        df = pd.DataFrame(
            {
                "EGT_spread_C": np.r_[np.ones(40) * 12, [40, 48, 55]],
                "vib_DE_mm_s": np.ones(43) * 3.0,
            }
        )
        out = detect_anomalies(df, bundle=bundle)
        self.assertEqual(out.get("mode"), "statistical")
        self.assertIsNone(bundle.tspulse_clf)
        # No classification block forced
        self.assertNotIn("classification", out)

    def test_classify_returns_none_without_model(self):
        bundle = ModelBundle(tspulse_clf=None)
        df = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.ones(20)})
        self.assertIsNone(classify_signature(df, ["a", "b"], bundle=bundle))


class TestClassifierStructure(unittest.TestCase):
    def test_models_source_has_classifier_api(self):
        src = (ROOT / "src" / "models.py").read_text(encoding="utf-8")
        self.assertIn("tspulse_clf", src)
        self.assertIn("TSPulseForClassification", src)
        self.assertIn("classify_signature", src)
        self.assertIn("GT_TSPULSE_CLF_PATH", src)
        self.assertIn("tspulse-block-dualhead", src)


if __name__ == "__main__":
    unittest.main()
