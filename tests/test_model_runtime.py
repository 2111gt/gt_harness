"""
Gating tests: real model load/generate/encode/anomaly paths.

These drive shipped ensure/load/generate functions. LLM smoke may take ~10–60s
when using the llama-cli CPU backend.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    detect_anomalies,
    generate_llm,
    is_offline_draft,
    llm_available,
    load_models,
)
from src.tools import KnowledgeRAG, load_sensor_csv  # noqa: E402
from src.utils import find_gguf_model  # noqa: E402


@unittest.skipIf(
    os.environ.get("GT_SKIP_HEAVY_MODELS") == "1",
    "GT_SKIP_HEAVY_MODELS=1",
)
class TestModelRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load once for the class — downloads CLI if needed
        cls.bundle = load_models(auto_download=True)

    def test_gguf_present_and_llm_loads(self):
        path = find_gguf_model()
        self.assertIsNotNone(path, "GGUF must be on disk")
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1_000_000)
        self.assertTrue(
            llm_available(self.bundle),
            f"LLM must load (status={self.bundle.status})",
        )
        self.assertIsNotNone(self.bundle.llm)

    def test_llm_generate_is_model_backed(self):
        self.assertTrue(llm_available(self.bundle), self.bundle.status)
        text = generate_llm(
            self.bundle,
            system_prompt="You are a concise assistant.",
            user_prompt="Reply with exactly one word: Paris",
        )
        self.assertTrue(text and text.strip(), "empty generation")
        self.assertFalse(
            is_offline_draft(text),
            f"expected model-backed text, got offline draft: {text[:200]}",
        )
        # Should not be pure silence
        self.assertGreater(len(text.strip()), 2)

    def test_embeddings_encode(self):
        from src.download import ensure_embeddings

        model, msg = ensure_embeddings()
        self.assertIsNotNone(model, msg)
        vec = model.encode(["gas turbine exhaust temperature spread"])
        # sentence-transformers returns ndarray
        import numpy as np

        arr = np.asarray(vec)
        self.assertGreaterEqual(arr.size, 8)
        self.assertIn("ready", msg.lower())

    def test_rag_uses_live_embeddings(self):
        import tempfile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            rag = KnowledgeRAG(persist_dir=Path(tmp) / "chroma", force_memory=False)
            # Force embedder load
            emb = rag._get_embedder()
            self.assertTrue(emb is not None and emb is not False, "embedder must load")
            status = rag.rebuild_index()
            self.assertTrue(status)
            hits = rag.query("exhaust temperature spread fuel nozzle", k=3)
            # With knowledge docs present we expect hits; embeddings path is live
            self.assertIsInstance(hits, list)

    def test_tspulse_mode_when_model_loads(self):
        # load_models defers TS Pulse; bind via shipped ensure_tspulse (real path)
        from src.download import ensure_tspulse

        model, mode, msg = ensure_tspulse(load_weights=True)
        if model is None or mode != "tspulse":
            # Only skip when package truly unavailable — not a silent false-negative
            try:
                from tsfm_public.models.tspulse import TSPulseForReconstruction  # noqa: F401
            except Exception as exc:
                self.skipTest(f"TS Pulse package not importable in this env: {exc}")
            self.fail(f"ensure_tspulse failed while package imports: mode={mode} msg={msg}")

        self.bundle.tspulse = model
        self.bundle.tspulse_mode = "tspulse"
        self.bundle.status["tspulse"] = msg
        self.assertEqual(self.bundle.tspulse_mode, "tspulse")
        sample = ROOT / "samples" / "gt_sensors_demo.csv"
        df, _ = load_sensor_csv(sample)
        out = detect_anomalies(df, bundle=self.bundle, z_threshold=3.0)
        self.assertEqual(out.get("mode"), "tspulse")
        self.assertIn("column_scores", out)
        self.assertTrue(out["column_scores"])
        self.assertGreater(out.get("reconstruction_channels", 0), 0)
        # No silent mean-fallback method tags
        for a in out.get("anomalies") or []:
            self.assertEqual(a.get("method"), "tspulse_residual")
        # Deterministic: same input twice → identical severity inputs
        out2 = detect_anomalies(df, bundle=self.bundle, z_threshold=3.0)
        self.assertEqual(out["column_scores"], out2["column_scores"])
        from src.analysis import score_severity

        self.assertEqual(score_severity(out)["severity"], score_severity(out2)["severity"])

    def test_tspulse_deferred_bundle_lazy_loads_on_detect(self):
        """
        TUI/startup path: tspulse deferred in status, model not pre-bound.
        First detect_anomalies must call ensure_tspulse and report mode=tspulse
        when granite-tsfm is importable (not permanent statistical lockout).
        """
        try:
            from tsfm_public.models.tspulse import TSPulseForReconstruction  # noqa: F401
        except Exception as exc:
            self.skipTest(f"TS Pulse package not importable: {exc}")

        from src.models import ModelBundle
        import numpy as np

        # Same deferred status string shape as load_models()
        bundle = ModelBundle(
            llm=None,
            tspulse=None,
            tspulse_mode="statistical",
            status={
                "tspulse": "deferred — loads on first diagnosis if package available",
            },
        )
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame(
            {
                "load_MW": 50 + rng.normal(0, 0.1, n),
                "EGT_spread_C": 12 + rng.normal(0, 0.2, n),
                "vib_DE_mm_s": 3.0 + rng.normal(0, 0.05, n),
            }
        )
        # Clear spike on one channel so residual path has signal
        df.loc[20:24, "EGT_spread_C"] = [40, 48, 55, 50, 42]

        out = detect_anomalies(df, bundle=bundle, z_threshold=3.0)
        self.assertEqual(
            out.get("mode"),
            "tspulse",
            f"expected tspulse after deferred lazy load; status={bundle.status.get('tspulse')} out={out.get('summary')}",
        )
        self.assertEqual(bundle.tspulse_mode, "tspulse")
        self.assertIsNotNone(bundle.tspulse)
        self.assertNotIn("not importable", str(bundle.status.get("tspulse", "")).lower())
        self.assertNotIn("failed permanently", str(bundle.status.get("tspulse", "")).lower())
        self.assertTrue(out.get("column_scores"))
        self.assertGreater(out.get("reconstruction_channels", 0), 0)
        methods = {a.get("method") for a in (out.get("anomalies") or [])}
        if methods:
            self.assertIn("tspulse_residual", methods)

    def test_stale_failed_status_recovers_when_package_importable(self):
        """False-negative cache must not permanently block a working package."""
        try:
            from tsfm_public.models.tspulse import TSPulseForReconstruction  # noqa: F401
        except Exception as exc:
            self.skipTest(f"TS Pulse package not importable: {exc}")

        from src.download import ensure_granite_tsfm
        import src.download as dl

        # Poison process cache the way a bad pip/import once did
        dl._TSFM_STATUS = "failed (statistical fallback): ok"
        recovered = ensure_granite_tsfm()
        self.assertTrue(
            recovered.startswith("ok") or recovered.startswith("installed"),
            f"stale failure must re-probe live import, got: {recovered}",
        )

        from src.models import ModelBundle
        import numpy as np

        bundle = ModelBundle(
            tspulse=None,
            tspulse_mode="statistical",
            status={
                "tspulse": (
                    "TS Pulse API not importable (pkg=failed (statistical fallback): ok); "
                    "statistical fallback [failed permanently this session]"
                ),
            },
        )
        df = pd.DataFrame(
            {
                "EGT_spread_C": list(range(30)),
                "vib_DE_mm_s": [3.0] * 30,
            }
        )
        out = detect_anomalies(df, bundle=bundle)
        self.assertEqual(out.get("mode"), "tspulse", bundle.status.get("tspulse"))
        self.assertIn("ready", str(bundle.status.get("tspulse", "")).lower())

    def test_intentional_statistical_bundle_not_forced_to_tspulse(self):
        """Unit-test / offline bundles without deferred must stay statistical."""
        from src.models import ModelBundle
        import numpy as np

        bundle = ModelBundle(
            tspulse=None,
            tspulse_mode="statistical",
            status={"tspulse": "statistical", "llm": "offline-test"},
        )
        df = pd.DataFrame(
            {
                "EGT_spread_C": np.r_[np.ones(20) * 12, [40, 48, 55]],
                "vib_DE_mm_s": np.ones(23) * 3.0,
            }
        )
        out = detect_anomalies(df, bundle=bundle, z_threshold=3.0)
        self.assertEqual(out.get("mode"), "statistical")
        self.assertIsNone(bundle.tspulse)
        self.assertEqual(bundle.tspulse_mode, "statistical")

    def test_anomaly_scores_differ_by_input_with_loaded_models(self):
        """Shipped detect_anomalies + score_severity on demo vs flat CSV."""
        sample = ROOT / "samples" / "gt_sensors_demo.csv"
        df, _ = load_sensor_csv(sample)
        df_flat = df.copy()
        for c in df_flat.select_dtypes(include="number").columns:
            df_flat[c] = float(df_flat[c].median())

        # Ensure TS Pulse bound for this comparison when package is present
        if self.bundle.tspulse is None:
            try:
                from src.download import ensure_tspulse

                model, mode, msg = ensure_tspulse(load_weights=True)
                if model is not None and mode == "tspulse":
                    self.bundle.tspulse = model
                    self.bundle.tspulse_mode = "tspulse"
                    self.bundle.status["tspulse"] = msg
            except Exception:
                pass

        a1 = detect_anomalies(df, bundle=self.bundle)
        a2 = detect_anomalies(df_flat, bundle=self.bundle)
        from src.analysis import score_severity

        s1 = score_severity(a1)
        s2 = score_severity(a2)
        self.assertNotEqual(s1["severity"], s2["severity"])
        # When TS Pulse loaded, mode should reflect it
        if self.bundle.tspulse is not None:
            self.assertEqual(a1.get("mode"), "tspulse")


if __name__ == "__main__":
    unittest.main()
