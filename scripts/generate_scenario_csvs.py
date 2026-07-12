#!/usr/bin/env python3
"""
Generate synthetic gas-turbine historian CSVs for GT Diagnostic Harness testing.

Scenarios
---------
- 10× combustion cold spot   — 2 months, 10-minute resolution
- 5×  HETS trips             — 2 months, 10-minute resolution
- 10× combustion dynamics    — 10 days, 1-minute resolution

Timestamps use real calendar dates (timezone-naive local plant time).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT_COLD = SAMPLES / "cold_spot"
OUT_HETS = SAMPLES / "hets_trip"
OUT_DYN = SAMPLES / "combustion_dynamics"

# Exhaust TC ring size for cold-spot cases (plant-style circumferential array)
N_EXHAUST_TC = 27
TC_COLS = [f"TC{i}" for i in range(1, N_EXHAUST_TC + 1)]

# Column set for HETS / dynamics (4 zone summary + process)
COLUMNS = [
    "timestamp",
    "load_MW",
    "CDP_bar",
    "CDT_C",
    "EGT_avg_C",
    "EGT_spread_C",
    "EGT_Z1_C",
    "EGT_Z2_C",
    "EGT_Z3_C",
    "EGT_Z4_C",
    "fuel_flow_kg_s",
    "IGV_deg",
    "vib_DE_mm_s",
    "vib_NDE_mm_s",
    "lube_oil_C",
    "comb_dyn_psi",
    "trip_active",
]

# Cold-spot columns: process + full TC1..TC27 ring (no EGT_Z*)
COLUMNS_COLD = (
    [
        "timestamp",
        "load_MW",
        "CDP_bar",
        "CDT_C",
        "EGT_avg_C",
        "EGT_spread_C",
    ]
    + TC_COLS
    + [
        "fuel_flow_kg_s",
        "IGV_deg",
        "vib_DE_mm_s",
        "vib_NDE_mm_s",
        "lube_oil_C",
        "comb_dyn_psi",
        "trip_active",
    ]
)


def _baseload_core(
    index: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Shared process signals; returns (partial_df_without_TCs, load, egt_mean)."""
    n = len(index)
    hour = np.asarray(index.hour, dtype=float) + np.asarray(index.minute, dtype=float) / 60.0
    load = 48.0 + 1.2 * np.sin(2 * np.pi * (hour - 8) / 24.0) + rng.normal(0, 0.15, n)
    load = np.asarray(np.clip(load, 35.0, 55.0), dtype=float)

    cdp = 11.1 + 0.04 * (load - 48) + rng.normal(0, 0.03, n)
    cdt = 345.0 + 0.8 * (load - 48) + rng.normal(0, 0.25, n)
    egt_mean = np.asarray(
        540.0 + 1.1 * (load - 48) + rng.normal(0, 0.4, n), dtype=float
    )

    fuel = 2.85 + 0.035 * (load - 48) + rng.normal(0, 0.01, n)
    igv = 62.0 + 0.25 * (load - 48) + rng.normal(0, 0.15, n)
    vib_de = 3.05 + rng.normal(0, 0.08, n)
    vib_nde = 2.95 + rng.normal(0, 0.08, n)
    lube = 58.0 + 0.05 * (load - 48) + rng.normal(0, 0.12, n)
    dyn = np.clip(0.35 + rng.normal(0, 0.05, n), 0.05, 1.2)

    base = pd.DataFrame(
        {
            "timestamp": index,
            "load_MW": np.round(load, 3),
            "CDP_bar": np.round(cdp, 3),
            "CDT_C": np.round(cdt, 3),
            "fuel_flow_kg_s": np.round(fuel, 4),
            "IGV_deg": np.round(igv, 3),
            "vib_DE_mm_s": np.round(vib_de, 3),
            "vib_NDE_mm_s": np.round(vib_nde, 3),
            "lube_oil_C": np.round(lube, 3),
            "comb_dyn_psi": np.round(dyn, 3),
            "trip_active": 0,
        }
    )
    return base, load, egt_mean


def _baseload_frame(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Healthy baseload with 4 zone EGTs (HETS / dynamics samples)."""
    base, load, egt_mean = _baseload_core(index, rng)
    n = len(index)
    z = np.column_stack([egt_mean + rng.normal(0, 1.2, n) for _ in range(4)])
    base["EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    base["EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    for i in range(4):
        base[f"EGT_Z{i + 1}_C"] = np.round(z[:, i], 3)
    return base[COLUMNS]


def _baseload_frame_cold(
    index: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Healthy baseload with full exhaust TC ring TC1..TC27."""
    base, _load, egt_mean = _baseload_core(index, rng)
    n = len(index)
    # Circumferential pattern: small spatial sine + independent noise
    angles = 2 * np.pi * np.arange(N_EXHAUST_TC) / N_EXHAUST_TC
    spatial = 1.5 * np.sin(angles)[None, :]  # mild fixed pattern
    tc = egt_mean[:, None] + spatial + rng.normal(0, 1.0, (n, N_EXHAUST_TC))
    for i, col in enumerate(TC_COLS):
        base[col] = np.round(tc[:, i], 3)
    base["EGT_avg_C"] = np.round(tc.mean(axis=1), 3)
    base["EGT_spread_C"] = np.round(tc.max(axis=1) - tc.min(axis=1), 3)
    return base[COLUMNS_COLD]


def _inject_cold_spot(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    cold_tc: int,
    onset_frac: float,
    depth_C: float,
    duration_frac: float,
    load_dip: bool,
    sector_width: int = 3,
) -> pd.DataFrame:
    """
    Inject a circumferential cold sector on the TC1..TC27 ring.

    cold_tc : 1..27 primary cold thermocouple (neighbors also cool slightly).
    sector_width : odd-ish count of adjacent TCs in the cold sector (wraps ring).
    """
    out = df.copy()
    n = len(out)
    start = int(n * onset_frac)
    end = min(n, start + max(12, int(n * duration_frac)))
    width = max(1, int(sector_width))
    # Adjacent indices on a 27-point ring (0-based)
    center = (int(cold_tc) - 1) % N_EXHAUST_TC
    half = width // 2
    sector = [((center + k) % N_EXHAUST_TC) for k in range(-half, half + 1)]
    # Center deepest; neighbors 40–70% of depth
    weights = []
    for idx in sector:
        dist = min((idx - center) % N_EXHAUST_TC, (center - idx) % N_EXHAUST_TC)
        weights.append(1.0 if dist == 0 else max(0.35, 1.0 - 0.28 * dist))

    ramp_n = min(36, end - start)
    ramp = np.linspace(0, 1.0, ramp_n)
    hold = np.ones(max(0, end - start - ramp_n))
    envelope = np.concatenate([ramp, hold])[: end - start]

    tc = out[TC_COLS].to_numpy(copy=True)
    for w, idx in zip(weights, sector):
        bias = depth_C * w * envelope + rng.normal(0, 0.35, len(envelope))
        tc[start:end, idx] = tc[start:end, idx] - bias

    for i, col in enumerate(TC_COLS):
        out[col] = np.round(tc[:, i], 3)
    out["EGT_avg_C"] = np.round(tc.mean(axis=1), 3)
    out["EGT_spread_C"] = np.round(tc.max(axis=1) - tc.min(axis=1), 3)

    # Mild fuel / performance coupling
    out.loc[out.index[start:end], "fuel_flow_kg_s"] = np.round(
        out.loc[out.index[start:end], "fuel_flow_kg_s"].to_numpy()
        + 0.02
        + rng.normal(0, 0.005, end - start),
        4,
    )
    if load_dip:
        out.loc[out.index[start:end], "load_MW"] = np.round(
            out.loc[out.index[start:end], "load_MW"].to_numpy() - 0.4 - rng.uniform(0, 0.6),
            3,
        )
    if depth_C >= 28:
        out.loc[out.index[start:end], "vib_DE_mm_s"] = np.round(
            out.loc[out.index[start:end], "vib_DE_mm_s"].to_numpy() + 0.4, 3
        )
    return out


def _inject_hets_trip(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    trip_frac: float,
    peak_spread_C: float,
    recovery: bool,
) -> pd.DataFrame:
    """
    High Exhaust Temperature Spread (HETS) trip signature:
    rapid spread rise → protection trip → load collapse → optional recovery.
    """
    out = df.copy()
    n = len(out)
    trip_i = int(n * trip_frac)
    pre = max(0, trip_i - 18)  # ~3 hours at 10-min (18*10m)
    # Build-up: sectors diverge
    build = np.linspace(0, 1.0, trip_i - pre)
    z = out[[f"EGT_Z{i}_C" for i in range(1, 5)]].to_numpy(copy=True)
    # Make Z1 hot, Z3 cold for true spread (not single TC)
    for j, idx in enumerate(range(pre, trip_i)):
        f = build[j]
        z[idx, 0] += f * (peak_spread_C * 0.55) + rng.normal(0, 0.5)
        z[idx, 2] -= f * (peak_spread_C * 0.45) + rng.normal(0, 0.5)
        z[idx, 1] += f * 2.0
        z[idx, 3] -= f * 1.5

    # Trip window: ~40 minutes
    trip_end = min(n, trip_i + 4)
    for idx in range(trip_i, trip_end):
        z[idx, 0] += peak_spread_C * 0.6 + rng.normal(0, 1.0)
        z[idx, 2] -= peak_spread_C * 0.5 + rng.normal(0, 1.0)
        out.loc[out.index[idx], "trip_active"] = 1
        out.loc[out.index[idx], "load_MW"] = max(0.0, float(out.loc[out.index[idx], "load_MW"]) * 0.15)
        out.loc[out.index[idx], "fuel_flow_kg_s"] = max(
            0.2, float(out.loc[out.index[idx], "fuel_flow_kg_s"]) * 0.2
        )
        out.loc[out.index[idx], "IGV_deg"] = 25.0 + rng.normal(0, 1.0)
        out.loc[out.index[idx], "vib_DE_mm_s"] = 6.5 + rng.normal(0, 0.3)
        out.loc[out.index[idx], "vib_NDE_mm_s"] = 5.8 + rng.normal(0, 0.3)
        out.loc[out.index[idx], "comb_dyn_psi"] = 1.8 + rng.normal(0, 0.2)

    if recovery and trip_end < n:
        # Coast / restart to baseload over ~1 day
        rec_end = min(n, trip_end + 144)  # 144*10min = 24h
        for k, idx in enumerate(range(trip_end, rec_end)):
            f = min(1.0, k / max(1, rec_end - trip_end - 1))
            # zones re-blend toward mean
            mean = z[idx].mean()
            z[idx] = mean + (z[idx] - mean) * (1.0 - 0.85 * f)
            out.loc[out.index[idx], "load_MW"] = 20.0 + f * 28.0 + rng.normal(0, 0.2)
            out.loc[out.index[idx], "trip_active"] = 0

    out[[f"EGT_Z{i}_C" for i in range(1, 5)]] = np.round(z, 3)
    out["EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    out["EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    # Clip physical-ish bounds
    out["load_MW"] = np.clip(out["load_MW"], 0, 60)
    out["vib_DE_mm_s"] = np.round(np.clip(out["vib_DE_mm_s"], 0.5, 15), 3)
    out["vib_NDE_mm_s"] = np.round(np.clip(out["vib_NDE_mm_s"], 0.5, 15), 3)
    out["comb_dyn_psi"] = np.round(np.clip(out["comb_dyn_psi"], 0.05, 8), 3)
    return out


def _inject_combustion_dynamics(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    onset_frac: float,
    severity: float,
    tone_period_min: float,
) -> pd.DataFrame:
    """
    Combustion dynamics (pulsation) burst at 1-minute resolution.
    severity: 0.5 mild → 3.0 severe (psi peak)
    """
    out = df.copy()
    n = len(out)
    start = int(n * onset_frac)
    # Event lasts 6–18 hours
    dur = int(rng.integers(6 * 60, 18 * 60))
    end = min(n, start + dur)
    t = np.arange(end - start, dtype=float)
    period = max(2.0, tone_period_min)
    osc = severity * (0.55 + 0.45 * np.sin(2 * np.pi * t / period))
    osc += 0.15 * severity * np.sin(2 * np.pi * t / (period * 0.37))
    osc += rng.normal(0, 0.08 * severity, len(t))
    osc = np.clip(osc, 0.1, 8.0)

    base = out.loc[out.index[start:end], "comb_dyn_psi"].to_numpy()
    out.loc[out.index[start:end], "comb_dyn_psi"] = np.round(base + osc, 3)

    # Couple to EGT spread flutter and mild vib
    zcols = [f"EGT_Z{i}_C" for i in range(1, 5)]
    z = out.loc[out.index[start:end], zcols].to_numpy(copy=True)
    flutter = (osc / max(severity, 1e-6)) * rng.uniform(3.0, 8.0)
    z[:, 0] += flutter * 0.6
    z[:, 2] -= flutter * 0.5
    out.loc[out.index[start:end], zcols] = np.round(z, 3)
    out.loc[out.index[start:end], "EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    out.loc[out.index[start:end], "EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    out.loc[out.index[start:end], "vib_DE_mm_s"] = np.round(
        out.loc[out.index[start:end], "vib_DE_mm_s"].to_numpy() + 0.25 * osc, 3
    )
    out.loc[out.index[start:end], "fuel_flow_kg_s"] = np.round(
        out.loc[out.index[start:end], "fuel_flow_kg_s"].to_numpy()
        + 0.01 * (osc / max(severity, 1e-6)),
        4,
    )
    return out


def _write(df: pd.DataFrame, path: Path, meta: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ISO-8601 timestamps without timezone (plant local)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  rows={len(out)}  {meta}")


def generate_all() -> None:
    OUT_COLD.mkdir(parents=True, exist_ok=True)
    OUT_HETS.mkdir(parents=True, exist_ok=True)
    OUT_DYN.mkdir(parents=True, exist_ok=True)

    # --- Cold spots: 2 months @ 10 min, staggered start dates, TC1..TC27 ---
    print("Combustion cold spots (10 files, ~2 months @ 10 min, TC1–TC27)…")
    cold_starts = [
        "2025-11-01",
        "2025-12-01",
        "2026-01-05",
        "2026-02-01",
        "2026-03-01",
        "2026-04-01",
        "2026-05-01",
        "2026-06-01",
        "2025-09-15",
        "2025-10-10",
    ]
    for i, start in enumerate(cold_starts, start=1):
        rng = np.random.default_rng(1000 + i)
        idx = pd.date_range(start=start, periods=60 * 24 * 6, freq="10min")  # 60 days
        df = _baseload_frame_cold(idx, rng)
        # Spread cold centers around the 27-TC ring
        cold_tc = 1 + ((i - 1) * 3) % N_EXHAUST_TC  # TC1,4,7,..., then wrap
        depth = 18 + (i % 5) * 4  # 18..34 C center depth
        onset = 0.35 + 0.03 * (i % 4)
        dur = 0.08 + 0.02 * (i % 3)  # multi-day event
        sector_w = 3 if i % 3 else 5  # 3- or 5-TC cold sector
        df = _inject_cold_spot(
            df,
            rng,
            cold_tc=cold_tc,
            onset_frac=onset,
            depth_C=float(depth),
            duration_frac=dur,
            load_dip=(i % 2 == 0),
            sector_width=sector_w,
        )
        _write(
            df,
            OUT_COLD / f"cold_spot_{i:02d}.csv",
            f"cold_center=TC{cold_tc} sector={sector_w} depth≈{depth}C start={start}",
        )

    # --- HETS trips: 2 months @ 10 min ---
    print("HETS trips (5 files, ~2 months @ 10 min)…")
    hets_starts = [
        "2025-08-01",
        "2025-11-15",
        "2026-01-10",
        "2026-03-20",
        "2026-05-12",
    ]
    for i, start in enumerate(hets_starts, start=1):
        rng = np.random.default_rng(2000 + i)
        idx = pd.date_range(start=start, periods=60 * 24 * 6, freq="10min")
        df = _baseload_frame(idx, rng)
        peak = 48 + i * 4  # trip-class spreads
        trip_frac = 0.55 + 0.02 * i
        df = _inject_hets_trip(
            df,
            rng,
            trip_frac=trip_frac,
            peak_spread_C=float(peak),
            recovery=(i != 3),  # one case leaves unit offline-ish longer
        )
        _write(
            df,
            OUT_HETS / f"hets_trip_{i:02d}.csv",
            f"peak_spread≈{peak}C trip≈{trip_frac:.2f} start={start}",
        )

    # --- Combustion dynamics: 10 days @ 1 min ---
    print("Combustion dynamics (10 files, 10 days @ 1 min)…")
    dyn_starts = [
        "2026-01-08",
        "2026-01-22",
        "2026-02-05",
        "2026-02-18",
        "2026-03-03",
        "2026-03-17",
        "2026-04-02",
        "2026-04-20",
        "2026-05-06",
        "2026-05-25",
    ]
    for i, start in enumerate(dyn_starts, start=1):
        rng = np.random.default_rng(3000 + i)
        idx = pd.date_range(start=start, periods=10 * 24 * 60, freq="1min")
        df = _baseload_frame(idx, rng)
        severity = 0.8 + 0.25 * i  # psi-class peaks
        period = 3.0 + (i % 5)  # minutes between pulsation peaks in envelope
        onset = 0.25 + 0.04 * (i % 5)
        df = _inject_combustion_dynamics(
            df,
            rng,
            onset_frac=onset,
            severity=float(severity),
            tone_period_min=float(period),
        )
        _write(
            df,
            OUT_DYN / f"comb_dyn_{i:02d}.csv",
            f"severity≈{severity:.2f}psi period≈{period}min start={start}",
        )

    # Manifest
    manifest = SAMPLES / "SCENARIO_README.md"
    manifest.write_text(
        """# Scenario sample CSVs

Generated by `scripts/generate_scenario_csvs.py`.

| Folder | Cases | Duration | Resolution | Signature |
|--------|------:|----------|------------|-----------|
| `cold_spot/` | 10 | ~60 days | 10 min | Cold sector on **TC1–TC27** ring → elevated `EGT_spread_C` |
| `hets_trip/` | 5 | ~60 days | 10 min | Rapid spread rise, `trip_active=1`, load/fuel collapse |
| `combustion_dynamics/` | 10 | 10 days | 1 min | Elevated `comb_dyn_psi` with EGT flutter / mild vib |

## Columns
**Cold spot:** `timestamp, load_MW, CDP_bar, CDT_C, EGT_avg_C, EGT_spread_C, TC1..TC27,
fuel_flow_kg_s, IGV_deg, vib_DE_mm_s, vib_NDE_mm_s, lube_oil_C, comb_dyn_psi, trip_active`

**HETS / dynamics:** same process channels with `EGT_Z1_C..EGT_Z4_C` instead of TC1–TC27

Timestamps are real calendar datetimes (`YYYY-MM-DD HH:MM:SS`).

## Process maps (RAG)
See `knowledge/process_map_cold_spot.md`, `process_map_hets_trip.md`,
`process_map_combustion_dynamics.md`.

## Suggested TUI tests
1. **Cold spot** — `samples/cold_spot/cold_spot_01.csv`, mode **Alerts**, paste cold-spot process map notes.
2. **HETS** — `samples/hets_trip/hets_trip_01.csv`, mode **Trips/Event**, note trip first-out HETS.
3. **Dynamics** — `samples/combustion_dynamics/comb_dyn_01.csv`, mode **Alerts** or **Trips/Event**.

Regenerate:
```bash
python scripts/generate_scenario_csvs.py
```
""",
        encoding="utf-8",
    )
    print(f"  wrote {manifest.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    generate_all()
