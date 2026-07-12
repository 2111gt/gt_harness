"""
Unit tests for shipped analysis / anomaly / severity logic.
Drives real functions — different CSV patterns → different scores/labels.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (
    _build_rag_queries,
    diagnosis_step_plan,
    run_diagnosis,
    score_severity,
)
from src.models import ModelBundle, detect_anomalies
from src.tools import KnowledgeRAG, load_sensor_csv, save_case
from src.utils import mode_label


def _clean_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "load_MW": 50 + rng.normal(0, 0.2, n),
            "EGT_spread_C": 12 + rng.normal(0, 0.3, n),
            "vib_DE_mm_s": 3.0 + rng.normal(0, 0.05, n),
        }
    )


def _spike_df(n: int = 40) -> pd.DataFrame:
    df = _clean_df(n)
    # Large spike mid-window on spread + vibration
    df.loc[20:24, "EGT_spread_C"] = [40, 48, 55, 50, 42]
    df.loc[20:24, "vib_DE_mm_s"] = [8.0, 9.5, 11.0, 9.0, 7.5]
    return df


class TestModeLabel(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(mode_label("Alerts"), "alerts")
        self.assertEqual(mode_label("Trips/Event"), "trips_event")
        # legacy aliases
        self.assertEqual(mode_label("Routine Check"), "alerts")
        self.assertEqual(mode_label("Event Investigation"), "trips_event")


class TestAnomalyAndSeverity(unittest.TestCase):
    def test_spike_scores_higher_than_clean(self):
        bundle = ModelBundle(tspulse_mode="statistical", status={"tspulse": "test"})
        clean = detect_anomalies(_clean_df(), bundle=bundle, z_threshold=3.0)
        spiked = detect_anomalies(_spike_df(), bundle=bundle, z_threshold=3.0)

        sev_clean = score_severity(clean)
        sev_spike = score_severity(spiked)

        self.assertGreater(sev_spike["severity"], sev_clean["severity"])
        self.assertNotEqual(sev_spike["label"], sev_clean["label"])
        self.assertIn(sev_spike["level"], {"elevated", "high", "critical", "mild"})
        # Spiked data should flag EGT_spread or vib among top
        self.assertIsNotNone(sev_spike["top_channel"])
        self.assertGreater(len(spiked["anomalies"]), len(clean["anomalies"]))

    def test_empty_frame(self):
        bundle = ModelBundle(tspulse_mode="statistical")
        out = detect_anomalies(pd.DataFrame(), bundle=bundle)
        self.assertEqual(out["mode"], "empty")
        sev = score_severity(out)
        self.assertEqual(sev["level"], "normal")


class TestCrossModeSignatureRag(unittest.TestCase):
    def test_trip_queries_include_prior_alerts_and_signature(self):
        anomaly = {
            "summary": "spread elevated",
            "classification": {
                "enabled": True,
                "top_label": "combustion_dynamics",
                "top_prob": 0.55,
                "trained": False,
            },
        }
        qs = _build_rag_queries(
            mode_key="trips_event",
            context="HETS first-out",
            anomaly=anomaly,
            top_channels=["EGT_spread_C", "comb_dyn_psi"],
        )
        self.assertGreaterEqual(len(qs), 2)
        joined = " ".join(qs).lower()
        self.assertIn("prior alert", joined)
        self.assertIn("dynamics", joined)
        self.assertIn("egt_spread", joined.lower() or joined)

    def test_alerts_query_uses_signature_process_terms(self):
        anomaly = {
            "summary": "cold sector",
            "classification": {
                "enabled": True,
                "top_label": "cold_spot",
                "top_prob": 0.6,
            },
        }
        qs = _build_rag_queries(
            mode_key="alerts",
            context="",
            anomaly=anomaly,
            top_channels=["TC1", "TC2"],
        )
        joined = " ".join(qs).lower()
        self.assertIn("cold", joined)


class TestProgressAndPlan(unittest.TestCase):
    def test_step_plan_default_vs_reflection(self):
        fast = diagnosis_step_plan(full_reflection=False)
        full = diagnosis_step_plan(full_reflection=True)
        self.assertEqual([s[0] for s in fast], ["csv", "anomaly", "rag", "llm", "finalize"])
        self.assertIn("reflect", [s[0] for s in full])
        self.assertGreater(len(full), len(fast))

    def test_progress_callback_fires_with_eta(self):
        bundle = ModelBundle(
            llm=None,
            tspulse=None,
            tspulse_mode="statistical",
            status={"llm": "offline-test"},
        )
        events = []

        def cb(sid, i, n, frac, msg):
            events.append((sid, i, n, frac, msg))

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            rag = KnowledgeRAG(persist_dir=Path(tmp) / "chroma", force_memory=True)
            rag.rebuild_index()
            run_diagnosis(
                mode="Alerts",
                context="progress test",
                bundle=bundle,
                rag=rag,
                df=_clean_df(),
                progress=cb,
            )
        self.assertGreaterEqual(len(events), 5)
        self.assertEqual(events[-1][0], "done")
        self.assertAlmostEqual(events[-1][3], 1.0, places=2)
        # Intermediate messages include step / ETA markers
        joined = " ".join(e[4] for e in events)
        self.assertTrue("step" in joined.lower() or "left" in joined.lower(), joined[:300])
        ids = {e[0] for e in events}
        self.assertIn("csv", ids)
        self.assertIn("llm", ids)


class TestRunDiagnosis(unittest.TestCase):
    def test_end_to_end_offline_llm_two_modes(self):
        bundle = ModelBundle(
            llm=None,
            tspulse=None,
            tspulse_mode="statistical",
            status={"llm": "offline-test", "tspulse": "statistical"},
        )
        # Isolated RAG against temp chroma + project knowledge still readable
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            rag = KnowledgeRAG(persist_dir=Path(tmp) / "chroma", force_memory=True)
            rag.rebuild_index()

            r1 = run_diagnosis(
                mode="Alerts",
                context="Baseload weekly review",
                bundle=bundle,
                rag=rag,
                df=_clean_df(),
            )
            r2 = run_diagnosis(
                mode="Trips/Event",
                context="High exhaust spread after fuel nozzle work",
                bundle=bundle,
                rag=rag,
                df=_spike_df(),
            )

            self.assertEqual(r1.mode, "alerts")
            self.assertEqual(r2.mode, "trips_event")
            self.assertTrue(r1.final_report)
            self.assertTrue(r2.final_report)
            # Different inputs → different anomaly summaries / severity path
            s1 = score_severity(r1.anomaly)
            s2 = score_severity(r2.anomaly)
            self.assertGreater(s2["severity"], s1["severity"])
            self.assertIn("GT Diagnostic", r2.to_markdown())

    def test_offline_final_report_is_clean_not_garbled(self):
        """Offline path must not dump reflection-prompt noise into final_report."""
        bundle = ModelBundle(
            llm=None,
            tspulse=None,
            tspulse_mode="statistical",
            status={"llm": "offline"},
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            rag = KnowledgeRAG(persist_dir=Path(tmp) / "chroma", force_memory=True)
            r = run_diagnosis(
                mode="Trips/Event",
                context="spread event",
                bundle=bundle,
                rag=rag,
                df=_spike_df(),
            )
            final = r.final_report
            self.assertIn("GT Diagnostic Final Report", final)
            self.assertIn("Severity:", final)
            self.assertTrue(
                "### Final diagnosis" in final
                or "### Diagnosis" in final
                or "Improved diagnosis" in final,
                final[:400],
            )
            # Must NOT look like a second raw offline draft of the reflection prompt
            self.assertNotIn("Draft diagnosis to improve", final)
            self.assertNotIn("Task\nProduce the improved", final)
            # Draft has structured reasoning trail
            self.assertTrue(r.draft)
            self.assertIn("Initial hypotheses", r.draft)
            self.assertTrue(r.reflection)
            # Display report puts Final first, then reasoning trail
            display = r.to_display_markdown()
            self.assertIn("1. Final report", display)
            self.assertIn("2. Reasoning", display)
            self.assertIn("3. Self-review", display)
            sev = score_severity(r.anomaly)
            self.assertIn(sev["level"], final)
            self.assertIn("score=", final)


class TestCsvAndFlywheel(unittest.TestCase):
    def test_load_sample_csv(self):
        sample = ROOT / "samples" / "gt_sensors_demo.csv"
        self.assertTrue(sample.is_file(), "sample CSV must exist")
        df, msg = load_sensor_csv(sample)
        self.assertFalse(df.empty)
        self.assertIn("rows", msg.lower())
        self.assertIn("EGT_spread_C", df.columns)

    def test_save_case_roundtrip_with_severity(self):
        from src import utils as u
        import src.tools as tools

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_path = Path(tmp)
            original = u.SAVED_CASES_DIR
            try:
                u.SAVED_CASES_DIR = tmp_path / "saved_cases"
                u.SAVED_CASES_DIR.mkdir()
                tools.SAVED_CASES_DIR = u.SAVED_CASES_DIR

                rag = KnowledgeRAG(persist_dir=tmp_path / "chroma", force_memory=True)
                sev = {
                    "severity": 12.5,
                    "level": "high",
                    "label": "High — prioritized review",
                    "top_channel": "EGT_spread_C",
                }
                case = save_case(
                    mode="event_investigation",
                    context="test context",
                    anomaly_summary="spread high",
                    analysis="draft",
                    reflection="reflection",
                    final_report="final report text",
                    user_corrections="Root cause was TC fault",
                    severity=sev,
                    rag=rag,
                    reindex=True,
                )
                self.assertTrue(Path(case["path"]).is_file())
                self.assertEqual(case["severity_score"], 12.5)
                loaded = tools.load_saved_cases(limit=5)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["user_corrections"], "Root cause was TC fault")
                self.assertIn("final report", loaded[0]["final_report"])
                self.assertEqual(loaded[0]["severity_score"], 12.5)
                self.assertEqual(loaded[0]["severity_level"], "high")
                self.assertEqual(tools.case_severity_score(loaded[0]), 12.5)
            finally:
                u.SAVED_CASES_DIR = original
                tools.SAVED_CASES_DIR = original

    def test_two_sessions_history_and_score_trend(self):
        """Two sessions with different scores → history + trend reflect the change."""
        from src import utils as u
        import src.tools as tools

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_path = Path(tmp)
            original = u.SAVED_CASES_DIR
            try:
                u.SAVED_CASES_DIR = tmp_path / "saved_cases"
                u.SAVED_CASES_DIR.mkdir()
                tools.SAVED_CASES_DIR = u.SAVED_CASES_DIR

                rag = KnowledgeRAG(persist_dir=tmp_path / "chroma", force_memory=True)
                high = {
                    "severity": 18.0,
                    "level": "critical",
                    "label": "Critical — immediate investigation",
                    "top_channel": "EGT_spread_C",
                }
                low = {
                    "severity": 2.0,
                    "level": "mild",
                    "label": "Mild deviation",
                    "top_channel": "load_MW",
                }
                c_high = save_case(
                    mode="event_investigation",
                    context="spike",
                    anomaly_summary="high spread",
                    analysis="a1",
                    reflection="r1",
                    final_report="final high",
                    severity=high,
                    rag=rag,
                    reindex=False,
                )
                c_low = save_case(
                    mode="routine_check",
                    context="stable",
                    anomaly_summary="quiet",
                    analysis="a2",
                    reflection="r2",
                    final_report="final low",
                    severity=low,
                    rag=rag,
                    reindex=False,
                )
                # Monotonic seq must increase regardless of clock ties
                self.assertEqual(c_high["seq"], 1)
                self.assertEqual(c_low["seq"], 2)
                self.assertLess(c_high["seq"], c_low["seq"])

                # Simulate app restart: new load from disk only
                reloaded = tools.load_saved_cases(limit=10)
                self.assertEqual(len(reloaded), 2)
                scores = sorted(tools.case_severity_score(c) for c in reloaded)
                self.assertEqual(scores, [2.0, 18.0])

                trend = tools.score_trend(limit=10)
                self.assertEqual(len(trend["scores"]), 2)
                # Chronological by seq: first saved high (seq=1), then low (seq=2)
                self.assertEqual(trend["seqs"], [1, 2])
                self.assertEqual(trend["scores"][0], 18.0)
                self.assertEqual(trend["scores"][1], 2.0)
                self.assertLess(trend["scores"][1], trend["scores"][0])

                md = tools.cases_history_markdown(limit=10)
                self.assertIn("Score trend", md)
                self.assertIn("18.000", md)
                self.assertIn("2.000", md)
                self.assertIn("Session history", md)
                self.assertIn("critical", md)
                self.assertIn("mild", md)
            finally:
                u.SAVED_CASES_DIR = original
                tools.SAVED_CASES_DIR = original

    def test_trend_order_stable_under_identical_timestamps(self):
        """Even if saved_at is forced equal, seq keeps chronological order."""
        from src import utils as u
        import src.tools as tools

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_path = Path(tmp)
            original = u.SAVED_CASES_DIR
            try:
                u.SAVED_CASES_DIR = tmp_path / "saved_cases"
                u.SAVED_CASES_DIR.mkdir()
                tools.SAVED_CASES_DIR = u.SAVED_CASES_DIR

                # Save two cases then rewrite saved_at to the exact same string
                c1 = save_case(
                    mode="event_investigation",
                    context="a",
                    anomaly_summary="a",
                    analysis="a",
                    reflection="a",
                    final_report="a",
                    severity={"severity": 9.0, "level": "high", "label": "H", "top_channel": "x"},
                    reindex=False,
                )
                c2 = save_case(
                    mode="routine_check",
                    context="b",
                    anomaly_summary="b",
                    analysis="b",
                    reflection="b",
                    final_report="b",
                    severity={"severity": 1.0, "level": "mild", "label": "M", "top_channel": "y"},
                    reindex=False,
                )
                same_ts = "2026-01-01T00:00:00+00:00"
                for path in tools.list_saved_case_files():
                    data = u.read_json(path)
                    data["saved_at"] = same_ts
                    u.write_json(path, data)

                trend = tools.score_trend(limit=10)
                self.assertEqual(trend["seqs"], [c1["seq"], c2["seq"]])
                self.assertEqual(trend["scores"], [9.0, 1.0])
                # Newest-first history still respects seq
                hist = tools.load_saved_cases(limit=10)
                self.assertEqual(hist[0]["seq"], c2["seq"])
                self.assertEqual(hist[1]["seq"], c1["seq"])
            finally:
                u.SAVED_CASES_DIR = original
                tools.SAVED_CASES_DIR = original


if __name__ == "__main__":
    unittest.main()
