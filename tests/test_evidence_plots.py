"""Tests for proof / evidence plots of anomaly-flagged channels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import run_diagnosis
from src.evidence_plots import (
    build_evidence_bundle,
    channel_common_name,
    channel_display_title,
    channel_xlabel,
    channel_ylabel,
    evidence_to_markdown,
    select_issue_channels,
)
from src.models import ModelBundle, detect_anomalies


def _spike_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "load_MW": 50 + rng.normal(0, 0.2, n),
            "EGT_spread_C": 12 + rng.normal(0, 0.3, n),
            "vib_DE_mm_s": 3.0 + rng.normal(0, 0.05, n),
            "fuel_flow_kg_s": 2.8 + rng.normal(0, 0.02, n),
        }
    )
    df.loc[25:30, "EGT_spread_C"] = [40, 48, 55, 50, 42, 38]
    df.loc[25:28, "vib_DE_mm_s"] = [8.0, 9.5, 11.0, 9.0]
    return df


class TestSelectChannels(unittest.TestCase):
    def test_top_channels_from_scores(self):
        anomaly = {
            "column_scores": {"a": 1.0, "b": 5.0, "c": 3.0},
            "anomalies": [
                {"row": 2, "column": "b", "score": 5.0, "method": "robust_zscore"},
                {"row": 3, "column": "c", "score": 3.0, "method": "robust_zscore"},
            ],
        }
        picks = select_issue_channels(anomaly, max_channels=2)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0][0], "b")
        self.assertEqual(picks[0][2], [2])


class TestChannelLabels(unittest.TestCase):
    def test_common_names_and_titles(self):
        self.assertIn("spread", channel_common_name("EGT_spread_C").lower())
        self.assertIn("thermocouple", channel_common_name("TC12").lower())
        self.assertIn("drive end", channel_common_name("vib_DE_mm_s").lower())
        title = channel_display_title("EGT_spread_C", score=12.5)
        self.assertIn("EGT_spread_C", title)
        self.assertIn("Exhaust", title)
        self.assertIn("12.5", title)
        self.assertTrue(channel_ylabel("EGT_spread_C").startswith("Y:"))
        self.assertIn("°C", channel_ylabel("EGT_spread_C"))
        self.assertTrue(channel_xlabel(window_start=10, window_end=50).startswith("X:"))
        self.assertIn("10", channel_xlabel(window_start=10, window_end=50))


class TestBuildPlots(unittest.TestCase):
    def test_spike_plots_include_egt_or_vib(self):
        bundle = ModelBundle(tspulse_mode="statistical", status={"tspulse": "test"})
        df = _spike_df()
        anomaly = detect_anomalies(df, bundle=bundle, z_threshold=3.0)
        ev = build_evidence_bundle(df, anomaly, max_channels=3, width=60, height=8)
        self.assertFalse(ev.is_empty())
        names = {c.name for c in ev.channels}
        self.assertTrue(names & {"EGT_spread_C", "vib_DE_mm_s"})
        art = ev.ascii_art
        self.assertIn("EGT_spread_C", art + "".join(names))
        self.assertGreater(len(art), 80)
        # Flag marker present when we have point anomalies
        if any(c.flag_rows for c in ev.channels):
            self.assertIn("▲", art)
        md = evidence_to_markdown(ev)
        self.assertIn("Proof plots", md)
        self.assertIn("```text", md)

    def test_empty_df(self):
        ev = build_evidence_bundle(pd.DataFrame(), {"column_scores": {}, "anomalies": []})
        self.assertTrue(ev.is_empty())

    def test_diagnosis_attaches_evidence(self):
        bundle = ModelBundle(
            tspulse_mode="statistical",
            status={"tspulse": "test", "llm": "offline"},
            llm=None,
        )
        df = _spike_df()
        result = run_diagnosis(
            df=df,
            mode="Alerts",
            context="test spike",
            bundle=bundle,
            full_reflection=False,
        )
        self.assertIsNotNone(result.evidence)
        self.assertTrue(result.evidence.channels)
        body = result.to_display_markdown()
        self.assertIn("Proof plots", body)
        self.assertTrue(result.evidence_ascii())

    def test_shipped_demo_csv_produces_proof_plots(self):
        """Drive real sample under samples/ through anomaly + plot builder."""
        sample = ROOT / "samples" / "gt_sensors_demo.csv"
        self.assertTrue(sample.is_file())
        df = pd.read_csv(sample)
        bundle = ModelBundle(tspulse_mode="statistical", status={"tspulse": "test"})
        anomaly = detect_anomalies(df, bundle=bundle, z_threshold=3.0)
        scores = anomaly.get("column_scores") or {}
        self.assertTrue(scores)
        self.assertIn("EGT_spread_C", scores)
        top = max(scores.items(), key=lambda kv: kv[1])[0]
        # Demo file spikes exhaust spread — top channel should be that family
        self.assertTrue(
            "EGT" in top or "spread" in top.lower() or "vib" in top.lower(),
            f"unexpected top channel {top} scores={scores}",
        )
        ev = build_evidence_bundle(df, anomaly, max_channels=4, width=64, height=8)
        self.assertFalse(ev.is_empty(), "demo CSV must yield plottable issue channels")
        self.assertTrue(ev.ascii_art.strip())
        names = {c.name for c in ev.channels}
        self.assertIn("EGT_spread_C", names)
        # Flags present on the spiked window when statistical engine fires
        if any(c.flag_rows for c in ev.channels):
            self.assertIn("▲", ev.ascii_art)
        # High-quality PNG path for GUI (matplotlib)
        if ev.image_paths:
            from pathlib import Path

            self.assertTrue(Path(ev.image_paths[0]).is_file())
            self.assertTrue(
                (ev.combined_image_path and Path(ev.combined_image_path).is_file())
                or Path(ev.image_paths[0]).suffix.lower() == ".png"
            )


if __name__ == "__main__":
    unittest.main()
