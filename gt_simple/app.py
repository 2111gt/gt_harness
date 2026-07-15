#!/usr/bin/env python3
"""
GT Simple — one-file local gas-turbine diagnostic mini-app.

Pipeline
--------
1) Load sensor CSV + free-text context
2) Anomaly model: Granite TS Pulse reconstruction (residual scores)
   + statistical median/MAD fallback
3) Signature model: second TS Pulse dual-head / classifier scaffold
4) RAG over knowledge/*.md + saved_cases/*.json
5) LLM pass 1 (draft diagnosis) → LLM pass 2 (critique / refine)
6) Save & Learn → JSON case re-indexed into RAG (flywheel)

UI
--
  streamlit run app.py          # browser: CSV + context + Run + Save
  python app.py --cli path.csv  # headless CLI
  python app.py --cli path.csv --context "..." --save

Engineering decision-support only — not OEM protection software.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (standalone folder next to this file; models may share parent harness)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
_LOCAL_MODELS = ROOT / "models"
_PARENT_MODELS = ROOT.parent / "models"


def _resolve_models_dir() -> Path:
    """Prefer local models/; if empty of GGUF/tspulse, use parent gt_harness/models/."""
    local = _LOCAL_MODELS
    parent = _PARENT_MODELS
    local.mkdir(parents=True, exist_ok=True)
    has_local = bool(list(local.glob("*.gguf"))) or (local / "tspulse").is_dir() and any(
        (local / "tspulse").rglob("config.json")
    )
    if has_local:
        return local
    if parent.is_dir() and (
        bool(list(parent.glob("*.gguf")))
        or ((parent / "tspulse").is_dir() and any((parent / "tspulse").rglob("config.json")))
        or (parent / "llama-cpp-bin").is_dir()
    ):
        return parent
    return local


MODELS_DIR = _resolve_models_dir()
KNOWLEDGE_DIR = ROOT / "knowledge"
SAVED_CASES_DIR = ROOT / "saved_cases"
CHROMA_DIR = ROOT / "chroma_db"
SAMPLES_DIR = ROOT / "samples"
LOG_DIR = ROOT / "logs"

TSPULSE_MODEL_ID = os.environ.get(
    "GT_TSPULSE_MODEL", "ibm-granite/granite-timeseries-tspulse-r1"
)
TSPULSE_AD_REVISION = os.environ.get("GT_TSPULSE_REVISION", "tspulse-block-ad")
TSPULSE_CLF_REVISION = os.environ.get(
    "GT_TSPULSE_CLF_REVISION", "tspulse-block-dualhead-512-p16-r1"
)
DEFAULT_GGUF_NAME = "granite-4.1-8b-Q4_K_M.gguf"
EMBEDDING_MODEL = os.environ.get("GT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLF_LABELS = ("normal", "cold_spot", "hets", "combustion_dynamics")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gt_simple")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, KNOWLEDGE_DIR, SAVED_CASES_DIR, CHROMA_DIR, SAMPLES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Seed knowledge (written once if empty)
# ---------------------------------------------------------------------------
SEED_KNOWLEDGE: Dict[str, str] = {
    "gt_overview.md": """# Gas turbine diagnostic overview

Engineering decision-support for industrial gas turbines. Typical sensors:
load/MW, EGT average and spread, fuel flow, vibration DE/NDE, lube oil temp,
speed, compressor discharge pressure/temperature.

Common families:
- **Cold spot / exhaust spread**: one or more TCs lag; combustion imbalance or
  thermocouple faults.
- **HETS**: high exhaust temperature spread trip logic / protection.
- **Combustion dynamics**: pressure pulsation / dynamics channels elevated.
- **Vibration**: bearing / rotor mechanical issues.

Always separate instrumentation fault from true process fault. Not OEM certified.
""",
    "process_map_exhaust_spread.md": """# Process map — exhaust temperature spread

Triggers: rising EGT_spread, uneven TC profile, often with fuel or dynamics cues.

Checks:
1. Compare individual exhaust TCs vs average.
2. Review fuel staging / nozzle group balance.
3. Rule out TC open/short or wiring.
4. Correlate with load and ambient.

Hypotheses: combustion imbalance, fuel manifold issue, sensor fault, transient load.
""",
    "process_map_hets_trip.md": """# Process map — HETS trip

High Exhaust Temperature Spread protection. Event mode: capture pre/post trip
windows, confirm which TC(s) drove the spread, check for true hot/cold sectors
vs sensor spike at trip instant.
""",
    "process_map_cold_spot.md": """# Process map — cold spot

One sector persistently colder. Can be real flame/fuel issue or failed TC.
Cross-check adjacent TCs, fuel valves, and whether anomaly is step-like (sensor)
or gradual (process).
""",
    "process_map_combustion_dynamics.md": """# Process map — combustion dynamics

Elevated dynamics / pulsation with possible EGT and load coupling. Review
operating point, fuel schedule, and whether dynamics precede other anomalies.
""",
    "process_map_vibration.md": """# Process map — vibration

DE/NDE vibration rise: bearing, misalignment, rotor, or resonance near certain
loads. Check lube oil temperature and whether vibration tracks load or is
independent.
""",
}


def seed_knowledge() -> None:
    ensure_dirs()
    for name, body in SEED_KNOWLEDGE.items():
        path = KNOWLEDGE_DIR / name
        if not path.exists():
            path.write_text(body.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_csv(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    if df.empty:
        raise ValueError("CSV is empty")
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    skip = {"index", "time", "timestamp", "datetime", "date", "t", "id"}
    cols: List[str] = []
    for c in df.columns:
        if str(c).strip().lower() in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


# ---------------------------------------------------------------------------
# Anomaly: statistical + TS Pulse recon
# ---------------------------------------------------------------------------
def statistical_anomalies(
    df: pd.DataFrame, cols: Sequence[str], z_threshold: float = 3.0
) -> Dict[str, Any]:
    anomalies: List[Dict[str, Any]] = []
    column_scores: Dict[str, float] = {}
    x = df[list(cols)].astype(float)
    for col in cols:
        series = x[col].to_numpy(dtype=float)
        med = float(np.nanmedian(series))
        mad = float(np.nanmedian(np.abs(series - med)))
        scale = 1.4826 * mad if mad > 1e-12 else (float(np.nanstd(series)) or 1.0)
        z = 0.6745 * (series - med) / scale
        score = float(np.nanmax(np.abs(z)))
        column_scores[col] = score
        for i, zi in enumerate(z):
            if abs(zi) >= z_threshold and np.isfinite(zi):
                anomalies.append(
                    {
                        "row": int(i),
                        "column": col,
                        "value": float(series[i]),
                        "score": float(abs(zi)),
                        "method": "robust_z",
                    }
                )
    anomalies.sort(key=lambda a: a["score"], reverse=True)
    top = sorted(column_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "mode": "statistical",
        "rows": len(df),
        "columns": list(cols),
        "anomalies": anomalies[:50],
        "column_scores": column_scores,
        "summary": (
            f"Statistical robust-z: top={top[0][0] if top else 'n/a'} "
            f"score={top[0][1]:.3f}" if top else "no columns"
        ),
    }


def _find_tspulse_local(revision: str) -> Optional[Path]:
    """Locate a local HF snapshot under models/tspulse or models/tspulse_clf."""
    for base in (MODELS_DIR / "tspulse", MODELS_DIR / "tspulse_clf"):
        if not base.exists():
            continue
        # common layout: models/tspulse/ibm-granite--.../revision/
        for p in base.rglob("config.json"):
            parent = p.parent
            if revision.replace("/", "-") in str(parent) or revision in str(parent):
                return parent
        # any config under base
        configs = list(base.rglob("config.json"))
        if configs:
            return configs[0].parent
    return None


def load_tspulse_recon() -> Tuple[Any, str]:
    """Load reconstruction TS Pulse (anomaly). Returns (model_or_None, status)."""
    try:
        from tsfm_public.models.tspulse import TSPulseForReconstruction  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"tsfm_public not importable: {exc}"

    local = _find_tspulse_local(TSPULSE_AD_REVISION)
    try:
        if local is not None:
            model = TSPulseForReconstruction.from_pretrained(str(local), local_files_only=True)
            model.eval()
            return model, f"tspulse recon from {local}"
        # hub / cache
        kwargs: Dict[str, Any] = {"revision": TSPULSE_AD_REVISION}
        model = TSPulseForReconstruction.from_pretrained(TSPULSE_MODEL_ID, **kwargs)
        model.eval()
        return model, f"tspulse recon hub {TSPULSE_MODEL_ID}@{TSPULSE_AD_REVISION}"
    except Exception as exc:  # noqa: BLE001
        return None, f"tspulse recon load failed: {exc}"


def load_tspulse_classifier() -> Tuple[Any, str]:
    """Load dual-head / classification scaffold (second TS Pulse model)."""
    # Prefer dual-head class if present; else reuse reconstruction weights as feature stub
    local = _find_tspulse_local(TSPULSE_CLF_REVISION)
    try:
        try:
            from tsfm_public.models.tspulse import TSPulseForClassification  # type: ignore

            cls = TSPulseForClassification
        except Exception:
            from tsfm_public.models.tspulse import TSPulseForReconstruction  # type: ignore

            cls = TSPulseForReconstruction

        if local is not None:
            model = cls.from_pretrained(str(local), local_files_only=True)
            model.eval()
            return model, f"tspulse clf from {local} ({cls.__name__})"
        model = cls.from_pretrained(
            TSPULSE_MODEL_ID, revision=TSPULSE_CLF_REVISION
        )
        model.eval()
        return model, f"tspulse clf hub @{TSPULSE_CLF_REVISION} ({cls.__name__})"
    except Exception as exc:  # noqa: BLE001
        return None, f"tspulse clf load failed: {exc}"


def _window_matrix(df: pd.DataFrame, cols: Sequence[str], max_len: int = 512) -> np.ndarray:
    x = df[list(cols)].astype(float).to_numpy()
    # simple impute
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])
    if len(x) > max_len:
        x = x[-max_len:]
    # pad channels to multiple of something small if needed — keep as-is
    return x.astype(np.float32)


def tspulse_recon_anomalies(
    df: pd.DataFrame, cols: Sequence[str], model: Any
) -> Dict[str, Any]:
    """Residual-based channel scores from reconstruction model (best-effort)."""
    import torch

    x = _window_matrix(df, cols)
    # TS Pulse often expects (batch, time, channels) — try a few shapes
    attempts = [
        torch.tensor(x[None, ...]),  # B,T,C
        torch.tensor(x.T[None, ...]),  # B,C,T
    ]
    last_err = None
    recon = None
    used = None
    model.eval()
    with torch.no_grad():
        for t in attempts:
            try:
                out = model(t)
                if hasattr(out, "reconstruction"):
                    recon = out.reconstruction
                elif isinstance(out, (tuple, list)):
                    recon = out[0]
                elif torch.is_tensor(out):
                    recon = out
                else:
                    recon = getattr(out, "logits", None) or getattr(out, "prediction", None)
                if recon is not None:
                    used = t
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
    if recon is None or used is None:
        raise RuntimeError(f"TS Pulse forward failed: {last_err}")

    r = recon.detach().cpu().numpy()
    # align to (T,C)
    r = np.squeeze(r)
    if r.ndim == 1:
        r = r[:, None]
    if r.shape[0] == len(cols) and r.shape[1] != len(cols):
        r = r.T
    # match length
    tlen = min(len(x), r.shape[0])
    xin = x[-tlen:]
    rin = r[-tlen:]
    n_ch = min(xin.shape[1], rin.shape[1], len(cols))
    resid = xin[:, :n_ch] - rin[:, :n_ch]
    scores = np.sqrt(np.mean(resid**2, axis=0))
    column_scores = {cols[i]: float(scores[i]) for i in range(n_ch)}
    # flag rows with high combined residual
    row_scores = np.sqrt(np.mean(resid**2, axis=1))
    thr = float(np.nanmedian(row_scores) + 3 * (np.nanmedian(np.abs(row_scores - np.nanmedian(row_scores))) * 1.4826 + 1e-6))
    anomalies: List[Dict[str, Any]] = []
    for i, rs in enumerate(row_scores):
        if rs >= thr:
            ch_i = int(np.argmax(np.abs(resid[i])))
            anomalies.append(
                {
                    "row": int(len(df) - tlen + i),
                    "column": cols[ch_i],
                    "value": float(xin[i, ch_i]),
                    "score": float(rs),
                    "method": "tspulse_residual",
                }
            )
    anomalies.sort(key=lambda a: a["score"], reverse=True)
    top = sorted(column_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "mode": "tspulse",
        "rows": len(df),
        "columns": list(cols)[:n_ch],
        "anomalies": anomalies[:50],
        "column_scores": column_scores,
        "reconstruction_channels": n_ch,
        "summary": (
            f"TS Pulse recon: top={top[0][0]} score={top[0][1]:.3f}" if top else "tspulse empty"
        ),
    }


def classify_signature(
    df: pd.DataFrame, cols: Sequence[str], model: Any = None
) -> Dict[str, Any]:
    """
    Second TS Pulse model → signature probabilities.
    Untrained dual-head scaffold → weak prior (flag trained=False).
    """
    if model is None:
        return _heuristic_clf(cols)

    import torch
    import torch.nn.functional as F

    x = _window_matrix(df, cols, max_len=512)
    labels = list(CLF_LABELS)
    try:
        model.eval()
    except Exception:
        return _heuristic_clf(cols)

    logits = None
    with torch.no_grad():
        for tensor in (torch.tensor(x[None, ...]), torch.tensor(x.T[None, ...])):
            try:
                out = model(tensor)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif torch.is_tensor(out):
                    logits = out
                elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
                    logits = out[0]
                if logits is not None:
                    break
            except Exception:
                continue

    if logits is None:
        h = _heuristic_clf(cols)
        h["note"] = "heuristic fallback (clf forward unavailable)"
        return h

    arr = logits.detach().cpu().float().reshape(-1)
    n = min(len(labels), int(arr.numel()))
    vec = torch.zeros(len(labels))
    vec[:n] = arr[:n]
    probs_t = F.softmax(vec, dim=0).numpy()
    probs = {labels[i]: float(probs_t[i]) for i in range(len(labels))}
    top = max(probs, key=probs.get)
    return {
        "enabled": True,
        "trained": False,
        "labels": labels,
        "probs": probs,
        "top_label": top,
        "top_prob": probs[top],
        "note": "dual-head scaffold (treat as weak prior until fine-tuned)",
    }


def score_severity(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    scores = anomaly.get("column_scores") or {}
    if not scores:
        return {
            "severity_score": 0.0,
            "severity_level": "none",
            "severity_label": "No score",
            "top_channel": "",
        }
    top_ch, top_s = max(scores.items(), key=lambda kv: kv[1])
    if top_s >= 10:
        level, label = "high", "Significant deviation"
    elif top_s >= 3:
        level, label = "moderate", "Moderate deviation"
    elif top_s >= 1.5:
        level, label = "mild", "Mild deviation"
    else:
        level, label = "low", "Near baseline"
    return {
        "severity_score": float(top_s),
        "severity_level": level,
        "severity_label": label,
        "top_channel": top_ch,
    }


# ---------------------------------------------------------------------------
# RAG (Chroma + embeddings, keyword fallback)
# ---------------------------------------------------------------------------
class SimpleRAG:
    """Vector RAG via Chroma; keyword memory if chromadb is missing."""

    def __init__(self) -> None:
        ensure_dirs()
        seed_knowledge()
        self.backend = "memory"
        self._docs: List[Dict[str, str]] = []
        self._collection = None
        self._embed_fn = None  # chromadb embedding function, if any

    def _load_docs(self) -> List[Dict[str, str]]:
        docs: List[Dict[str, str]] = []
        for p in sorted(KNOWLEDGE_DIR.glob("*.md")):
            docs.append(
                {
                    "id": f"knowledge:{p.name}",
                    "source": str(p.name),
                    "text": p.read_text(encoding="utf-8", errors="replace"),
                }
            )
        for p in sorted(SAVED_CASES_DIR.glob("*.json")):
            try:
                case = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = "\n".join(
                [
                    f"Case {case.get('case_id', p.stem)}",
                    f"Mode: {case.get('mode', '')}",
                    f"Context: {case.get('context', '')}",
                    f"Anomaly: {case.get('anomaly_summary', '')}",
                    f"Final: {case.get('final_report', '')[:2000]}",
                    f"Corrections: {case.get('user_corrections', '')}",
                ]
            )
            docs.append({"id": f"case:{p.name}", "source": p.name, "text": text})
        return docs

    def _hash_embed_fn(self, dim: int = 384):
        """Always-available local embedder (no onnx / sentence-transformers)."""

        class _HashEF:
            def __init__(self) -> None:
                self._dim = dim

            def name(self) -> str:  # chroma optional metadata
                return "gt-simple-hash"

            def __call__(self, input: List[str]) -> List[List[float]]:
                out: List[List[float]] = []
                for text in input:
                    v = np.zeros(dim, dtype=np.float32)
                    for tok in re.findall(r"[a-z0-9_]+", (text or "").lower()):
                        # stable across processes (Python hash is salted)
                        h = int.from_bytes(tok.encode("utf-8"), "little", signed=False)
                        v[h % dim] += 1.0
                        v[(h // dim) % dim] += 0.5
                    n = float(np.linalg.norm(v)) or 1.0
                    out.append((v / n).tolist())
                return out

        return _HashEF()

    def _make_embed_fn(self):
        """Pick an embedding function for Chroma.

        Default: pure-NumPy hash embeddings (reliable on Windows).

        Optional (opt-in — broken onnxruntime DLLs can hard-crash CPython):
          GT_SIMPLE_EMBED=onnx  or  GT_SIMPLE_EMBED=st
        """
        prefer = (os.environ.get("GT_SIMPLE_EMBED") or "hash").strip().lower()

        if prefer in {"onnx", "chroma", "default"}:
            try:
                import onnxruntime as ort

                _ = ort.get_available_providers()
                from chromadb.utils import embedding_functions

                ef = embedding_functions.DefaultEmbeddingFunction()
                _ = ef(["gt simple embed probe"])
                return ef, "chroma-default-onnx"
            except Exception as exc:  # noqa: BLE001
                log.warning("GT_SIMPLE_EMBED=onnx failed (%s); using hash", exc)

        if prefer in {"st", "sentence-transformers", "minilm"}:
            try:
                from chromadb.utils import embedding_functions

                name = (
                    EMBEDDING_MODEL.split("/")[-1]
                    if "/" in EMBEDDING_MODEL
                    else EMBEDDING_MODEL
                )
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=name
                )
                _ = ef(["gt simple embed probe"])
                return ef, "sentence-transformers"
            except Exception as exc:  # noqa: BLE001
                log.warning("GT_SIMPLE_EMBED=st failed (%s); using hash", exc)

        return self._hash_embed_fn(), "hash"

    def rebuild(self) -> str:
        self._docs = self._load_docs()
        self._collection = None
        self._embed_fn = None
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as exc:  # noqa: BLE001
            self.backend = "memory"
            return (
                f"memory keyword RAG ({len(self._docs)} docs); "
                f"install chromadb: pip install chromadb  ({exc})"
            )

        try:
            embed_fn, embed_name = self._make_embed_fn()
            self._embed_fn = embed_fn
            client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                client.delete_collection("gt_simple")
            except Exception:
                pass
            col = client.get_or_create_collection(
                name="gt_simple",
                metadata={"hnsw:space": "cosine"},
                embedding_function=embed_fn,
            )
            texts = [d["text"][:4000] for d in self._docs]
            ids = [d["id"] for d in self._docs]
            metas = [{"source": d["source"]} for d in self._docs]
            if texts:
                # Chroma embeds via embedding_function
                col.add(ids=ids, documents=texts, metadatas=metas)
            self._collection = col
            self.backend = "chromadb"
            return f"chromadb indexed {len(texts)} docs (embed={embed_name})"
        except Exception as exc:  # noqa: BLE001
            self.backend = "memory"
            self._collection = None
            self._embed_fn = None
            return f"memory keyword RAG ({len(self._docs)} docs); chroma/embed error: {exc}"

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._docs:
            self.rebuild()
        q = (text or "").strip()
        if not q:
            return []
        if self._collection is not None:
            try:
                n = min(k, max(1, len(self._docs)))
                res = self._collection.query(query_texts=[q], n_results=n)
                hits = []
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for i, doc in enumerate(docs):
                    hits.append(
                        {
                            "source": (metas[i] or {}).get("source", "") if i < len(metas) else "",
                            "text": doc,
                            "score": float(1.0 - dists[i]) if i < len(dists) else 0.0,
                        }
                    )
                return hits
            except Exception as exc:  # noqa: BLE001
                log.warning("vector query failed: %s", exc)

        # keyword fallback
        words = {w.lower() for w in re.findall(r"[a-zA-Z0-9_]{3,}", q)}
        scored = []
        for d in self._docs:
            body = d["text"].lower()
            score = sum(1 for w in words if w in body)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"source": d["source"], "text": d["text"][:2000], "score": float(s)}
            for s, d in scored[:k]
        ]


# ---------------------------------------------------------------------------
# LLM (llama-cpp-python → llama-cli → offline)
# ---------------------------------------------------------------------------
@dataclass
class LLM:
    backend: str = "offline"
    runner: Any = None
    status: str = "not loaded"


def find_gguf() -> Optional[Path]:
    env = os.environ.get("GT_GGUF_PATH")
    if env and Path(env).is_file():
        return Path(env)
    p = MODELS_DIR / DEFAULT_GGUF_NAME
    if p.is_file():
        return p
    found = list(MODELS_DIR.glob("*.gguf"))
    return found[0] if found else None


def find_llama_cli() -> Optional[Path]:
    """Prefer llama-completion (batch) over interactive llama-cli."""
    bin_dir = MODELS_DIR / "llama-cpp-bin"
    for name in (
        "llama-completion.exe",
        "llama-completion",
        "llama-cli.exe",
        "llama-cli",
    ):
        p = bin_dir / name
        if p.is_file():
            return p
    return None


def load_llm() -> LLM:
    if (os.environ.get("GT_SIMPLE_OFFLINE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return LLM(backend="offline", status="GT_SIMPLE_OFFLINE=1")

    gguf = find_gguf()
    if gguf is None:
        return LLM(backend="offline", status="no GGUF in models/")

    # try python binding (often illegal-instruction on older CPUs)
    try:
        from llama_cpp import Llama  # type: ignore

        n_threads = max(2, min(8, os.cpu_count() or 4))
        llm = Llama(
            model_path=str(gguf),
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        return LLM(backend="llama-cpp-python", runner=llm, status=f"python bind {gguf.name}")
    except Exception as exc:  # noqa: BLE001
        log.info("llama-cpp-python unavailable: %s", exc)

    cli = find_llama_cli()
    if cli is not None:
        return LLM(
            backend="llama-cli",
            runner={"cli": cli, "gguf": gguf},
            status=f"{cli.name} + {gguf.name}",
        )
    return LLM(backend="offline", status=f"GGUF present ({gguf.name}) but no runner")


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"<|{role}|>\n{m.get('content', '')}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _clean_llama_output(text: str) -> str:
    text = (text or "").strip()
    lines = [
        ln
        for ln in text.splitlines()
        if not re.match(
            r"^(ggml|llama_|main:|print_info|load_|system_info|build:|log_)",
            ln.strip(),
            re.I,
        )
    ]
    return "\n".join(lines).strip()


def llm_generate(llm: LLM, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    if llm.backend == "offline" or llm.runner is None:
        return ""

    if llm.backend == "llama-cpp-python":
        try:
            out = llm.runner.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("llama-cpp-python generate failed: %s", exc)
            return ""

    if llm.backend == "llama-cli":
        cli = Path(llm.runner["cli"]).resolve()
        gguf = Path(llm.runner["gguf"]).resolve()
        prompt = _messages_to_prompt(messages)
        n_threads = max(2, min(8, os.cpu_count() or 4))
        fd, prompt_path = tempfile.mkstemp(prefix="gt_simple_prompt_", suffix=".txt")
        os.close(fd)
        try:
            Path(prompt_path).write_text(prompt, encoding="utf-8", errors="replace")
            # Prefer completion binary next to chosen cli
            binaries = []
            for name in (
                "llama-completion.exe",
                "llama-completion",
                cli.name,
                "llama-cli.exe",
            ):
                p = cli.parent / name
                if p.is_file() and p not in binaries:
                    binaries.append(p)
            extras_list = [
                [
                    "-t",
                    str(n_threads),
                    "-tb",
                    str(n_threads),
                    "--temp",
                    "0.2",
                    "-no-cnv",
                    "--no-display-prompt",
                ],
                [
                    "-t",
                    str(n_threads),
                    "--temp",
                    "0.2",
                    "-no-cnv",
                ],
            ]
            env = os.environ.copy()
            env["LLAMA_LOG_COLORS"] = "0"
            env.setdefault("TERM", "dumb")
            timeout = int(os.environ.get("GT_SIMPLE_LLM_TIMEOUT", "900"))
            last_err = ""
            for binary in binaries[:2]:
                for extra in extras_list:
                    cmd = [
                        str(binary),
                        "-m",
                        str(gguf),
                        "-n",
                        str(max_tokens),
                        "-c",
                        "2048",
                        *extra,
                        "-f",
                        prompt_path,
                    ]
                    try:
                        kwargs: Dict[str, Any] = {
                            "cwd": str(binary.parent),
                            "capture_output": True,
                            "text": True,
                            "encoding": "utf-8",
                            "errors": "replace",
                            "timeout": timeout,
                            "env": env,
                        }
                        if sys.platform == "win32":
                            kwargs["creationflags"] = getattr(
                                subprocess, "CREATE_NO_WINDOW", 0
                            )
                        log.info("LLM via %s n=%s", binary.name, max_tokens)
                        proc = subprocess.run(cmd, **kwargs)
                        text = _clean_llama_output(proc.stdout or "")
                        if not text and proc.stderr:
                            text = _clean_llama_output(proc.stderr)
                        if text and len(text) > 40:
                            return text
                        last_err = (proc.stderr or "")[:300]
                    except subprocess.TimeoutExpired:
                        last_err = f"timeout after {timeout}s"
                        log.warning("%s", last_err)
                    except Exception as exc:  # noqa: BLE001
                        last_err = str(exc)
                        log.warning("llama generate error: %s", exc)
            log.warning("llama-cli produced no usable text (%s)", last_err)
            return ""
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

    return ""


def offline_draft(anomaly: Dict[str, Any], clf: Dict[str, Any], context: str) -> str:
    sev = score_severity(anomaly)
    top = sorted((anomaly.get("column_scores") or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
    sig = clf.get("top_label", "n/a") if clf else "n/a"
    lines = [
        "## Reasoning",
        f"Offline draft (no LLM runner). Anomaly engine={anomaly.get('mode')}; "
        f"severity={sev['severity_label']} ({sev['severity_score']:.3f}); "
        f"top channel={sev['top_channel']}; signature hint={sig}.",
        f"Operator context: {context or '(none)'}",
        "",
        "## Initial hypotheses",
        "1. Process deviation on top-scored channels (see table).",
        "2. Sensor / instrumentation artifact.",
        "3. Transient operating-point change.",
        "",
        "## Self-review",
        "Without LLM, confidence is limited to statistical/TS Pulse residuals + RAG.",
        "",
        "## Final diagnosis",
        f"Priority channels: {', '.join(f'{c}={s:.2f}' for c,s in top) or 'none'}. "
        "Validate on historian trends and process maps before action.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class DiagnosisResult:
    anomaly: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)
    severity: Dict[str, Any] = field(default_factory=dict)
    rag_hits: List[Dict[str, Any]] = field(default_factory=list)
    draft: str = ""
    reflection: str = ""
    final_report: str = ""
    status: Dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0


def run_diagnosis(
    df: pd.DataFrame,
    context: str = "",
    mode: str = "alerts",
    rag: Optional[SimpleRAG] = None,
    llm: Optional[LLM] = None,
    use_tspulse: bool = True,
) -> DiagnosisResult:
    t0 = time.monotonic()
    ensure_dirs()
    seed_knowledge()
    rag = rag or SimpleRAG()
    if not rag._docs:
        rag.rebuild()
    llm = llm or load_llm()

    cols = numeric_columns(df)
    status: Dict[str, str] = {}

    # 1) statistical always
    stats = statistical_anomalies(df, cols)
    anomaly = stats

    recon_model = None
    clf_model = None
    if use_tspulse:
        recon_model, st = load_tspulse_recon()
        status["tspulse_recon"] = st
        if recon_model is not None:
            try:
                anomaly = tspulse_recon_anomalies(df, cols, recon_model)
                anomaly["statistical_backup"] = {
                    "column_scores": stats["column_scores"],
                    "anomaly_count": len(stats["anomalies"]),
                }
            except Exception as exc:  # noqa: BLE001
                status["tspulse_recon_infer"] = f"fallback stats: {exc}"
                anomaly = stats
        clf_model, st2 = load_tspulse_classifier()
        status["tspulse_clf"] = st2
    else:
        status["tspulse_recon"] = "skipped"
        status["tspulse_clf"] = "skipped"

    try:
        clf = classify_signature(df, cols, clf_model)
    except Exception as exc:  # noqa: BLE001
        clf = _heuristic_clf(cols)
        clf["error"] = str(exc)

    anomaly["classification"] = clf
    sev = score_severity(anomaly)

    # RAG query from top channels + context + signature
    top_cols = sorted(
        (anomaly.get("column_scores") or {}).items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    q = " ".join(
        [
            context or "",
            " ".join(c for c, _ in top_cols),
            str(clf.get("top_label") or ""),
            "gas turbine exhaust vibration fuel",
        ]
    )
    hits = rag.query(q, k=5)
    status["rag"] = f"{rag.backend} hits={len(hits)}"
    status["llm"] = llm.status

    rag_block = "\n\n".join(
        f"### {h.get('source')}\n{h.get('text', '')[:1200]}" for h in hits
    ) or "(no RAG hits)"

    anomaly_block = (
        f"mode={anomaly.get('mode')} summary={anomaly.get('summary')}\n"
        f"severity={sev}\n"
        f"top_channels={top_cols}\n"
        f"flags={anomaly.get('anomalies', [])[:8]}\n"
        f"classifier={clf}\n"
    )

    system = (
        "You are a local gas-turbine diagnostic assistant. Decision-support only, "
        "not OEM protection software. Be concrete about channels and hypotheses. "
        "Use exactly these markdown sections:\n"
        "## Reasoning\n## Initial hypotheses\n## Self-review\n## Final diagnosis"
    )
    user1 = (
        f"Mode: {mode}\n"
        f"Operator context:\n{context or '(none)'}\n\n"
        f"Anomaly + classifier evidence:\n{anomaly_block}\n\n"
        f"RAG knowledge / past cases:\n{rag_block}\n\n"
        "Write the structured diagnosis."
    )

    draft = llm_generate(
        llm,
        [{"role": "system", "content": system}, {"role": "user", "content": user1}],
        max_tokens=700,
    )
    if not draft:
        draft = offline_draft(anomaly, clf, context)

    # Pass 2 — critique / refine (always when we have any draft)
    user2 = (
        "Critique and refine the draft diagnosis below. Correct overconfidence, "
        "separate sensor vs process faults, and produce an improved full report "
        "with the same four sections (Reasoning, Initial hypotheses, Self-review, "
        "Final diagnosis).\n\n"
        f"Evidence recap:\n{anomaly_block}\n\n"
        f"RAG:\n{rag_block[:2000]}\n\n"
        f"DRAFT:\n{draft}"
    )
    reflection = llm_generate(
        llm,
        [{"role": "system", "content": system}, {"role": "user", "content": user2}],
        max_tokens=700,
    )
    if not reflection:
        reflection = (
            "## Reasoning\nOffline second pass: kept draft with severity packaging.\n\n"
            "## Initial hypotheses\nSee draft.\n\n"
            "## Self-review\nNo LLM runner for critique; treat draft as provisional.\n\n"
            "## Final diagnosis\n"
            + draft.split("## Final diagnosis")[-1].strip()
            if "## Final diagnosis" in draft
            else draft
        )

    final = reflection if reflection.strip() else draft
    # package header
    header = (
        f"# GT Simple Diagnostic Report\n\n"
        f"**Mode:** `{mode}`  \n"
        f"**Severity:** {sev['severity_label']} · score={sev['severity_score']:.3f} · "
        f"top=`{sev['top_channel']}` · engine=`{anomaly.get('mode')}`  \n"
        f"**Signature:** `{clf.get('top_label')}` "
        f"({clf.get('top_prob', 0):.1%}, trained={clf.get('trained')})  \n"
        f"**LLM:** {llm.backend} — {llm.status}  \n\n---\n\n"
    )
    final_report = header + final

    return DiagnosisResult(
        anomaly=anomaly,
        classification=clf,
        severity=sev,
        rag_hits=hits,
        draft=draft,
        reflection=reflection,
        final_report=final_report,
        status=status,
        elapsed_s=time.monotonic() - t0,
    )


def _heuristic_clf(cols: Sequence[str]) -> Dict[str, Any]:
    labels = list(CLF_LABELS)
    scores = {lab: 0.05 for lab in labels}
    joined = " ".join(cols).lower()
    if "egt" in joined or "spread" in joined:
        scores["hets"] += 0.35
        scores["cold_spot"] += 0.25
    if "vib" in joined:
        scores["normal"] += 0.05
    if "dyn" in joined:
        scores["combustion_dynamics"] += 0.3
    s = sum(scores.values()) or 1.0
    probs = {k: v / s for k, v in scores.items()}
    top = max(probs, key=probs.get)
    return {
        "enabled": True,
        "trained": False,
        "labels": labels,
        "probs": probs,
        "top_label": top,
        "top_prob": probs[top],
        "note": "heuristic signature (no clf weights)",
    }


# ---------------------------------------------------------------------------
# Save & Learn (flywheel)
# ---------------------------------------------------------------------------
def save_case(
    result: DiagnosisResult,
    context: str,
    mode: str,
    user_corrections: str = "",
    rag: Optional[SimpleRAG] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    case_id = f"case_{utc_now()}_{uuid.uuid4().hex[:8]}"
    case = {
        "case_id": case_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "context": context or "",
        "anomaly_summary": result.anomaly.get("summary", ""),
        "analysis": result.draft,
        "reflection": result.reflection,
        "final_report": result.final_report,
        "user_corrections": user_corrections or "",
        "severity_score": result.severity.get("severity_score"),
        "severity_level": result.severity.get("severity_level"),
        "severity_label": result.severity.get("severity_label"),
        "top_channel": result.severity.get("top_channel"),
        "signature_label": (result.classification or {}).get("top_label"),
        "metadata": {
            "status": result.status,
            "classification": result.classification,
            "column_scores": result.anomaly.get("column_scores"),
        },
    }
    path = SAVED_CASES_DIR / f"{case_id}.json"
    path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    case["path"] = str(path)
    engine = rag or SimpleRAG()
    case["reindex_status"] = engine.rebuild()
    return case


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GT Simple — local GT diagnostics (1-file)")
    p.add_argument("--cli", metavar="CSV", help="Run one diagnosis on CSV path")
    p.add_argument("--context", default="", help="Operator / process context")
    p.add_argument("--mode", default="alerts", choices=["alerts", "trips"])
    p.add_argument("--save", action="store_true", help="Save & Learn after run")
    p.add_argument("--corrections", default="", help="User corrections when --save")
    p.add_argument("--no-tspulse", action="store_true", help="Statistical + heuristic only")
    p.add_argument("--rebuild-rag", action="store_true", help="Rebuild RAG index and exit")
    args = p.parse_args(argv)

    ensure_dirs()
    seed_knowledge()

    if args.rebuild_rag:
        print(SimpleRAG().rebuild())
        return 0

    if not args.cli:
        p.print_help()
        print("\nTip: streamlit run app.py")
        return 0

    df = load_csv(args.cli)
    print(f"Loaded {args.cli}: {df.shape[0]} rows × {df.shape[1]} cols")
    rag = SimpleRAG()
    print("RAG:", rag.rebuild())
    llm = load_llm()
    print("LLM:", llm.status)
    result = run_diagnosis(
        df,
        context=args.context,
        mode=args.mode,
        rag=rag,
        llm=llm,
        use_tspulse=not args.no_tspulse,
    )
    print(result.final_report)
    print(f"\n--- elapsed {result.elapsed_s:.1f}s status={result.status}")
    out = LOG_DIR / "last_report.md"
    out.write_text(result.final_report, encoding="utf-8")
    print(f"Wrote {out}")
    if args.save:
        case = save_case(result, args.context, args.mode, args.corrections, rag=rag)
        print(f"Saved {case['path']} reindex={case.get('reindex_status')}")
    return 0


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="GT Simple", layout="wide")
    st.title("GT Simple Diagnostic")
    st.caption(
        "CSV + context → dual TS Pulse (recon + signature) → RAG → LLM draft + 2nd refine → Save & Learn"
    )
    st.warning("Engineering decision-support only. Not an OEM-certified protection system.")

    ensure_dirs()
    seed_knowledge()

    if "result" not in st.session_state:
        st.session_state.result = None
    if "rag" not in st.session_state:
        st.session_state.rag = SimpleRAG()
        st.session_state.rag_status = st.session_state.rag.rebuild()

    with st.sidebar:
        st.header("Setup")
        st.write(f"**Root:** `{ROOT}`")
        st.write(f"**RAG:** {st.session_state.get('rag_status', '?')}")
        if st.button("Rebuild RAG index"):
            st.session_state.rag_status = st.session_state.rag.rebuild()
            st.success(st.session_state.rag_status)
        use_tspulse = st.checkbox("Use TS Pulse models", value=True)
        mode = st.selectbox("Mode", ["alerts", "trips"])
        st.markdown("---")
        st.markdown("**Models folder**")
        st.code(str(MODELS_DIR), language="text")
        gguf = find_gguf()
        st.write("GGUF:", gguf.name if gguf else "missing")
        st.write("llama-cli:", "yes" if find_llama_cli() else "no")

    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("Sensor CSV", type=["csv"])
        sample = SAMPLES_DIR / "gt_sensors_demo.csv"
        use_sample = st.checkbox("Use bundled demo CSV", value=up is None and sample.exists())
    with c2:
        context = st.text_area(
            "Operator / process context",
            height=160,
            placeholder="Alarms, process map notes, trip text, recent maintenance…",
        )

    run = st.button("Run diagnosis", type="primary")

    if run:
        try:
            if use_sample and sample.exists() and up is None:
                df = load_csv(sample)
                src_name = sample.name
            elif up is not None:
                df = load_csv(up)
                src_name = up.name
            else:
                st.error("Upload a CSV or enable the demo sample.")
                return
            with st.spinner(f"Running pipeline on {src_name}…"):
                llm = load_llm()
                result = run_diagnosis(
                    df,
                    context=context,
                    mode=mode,
                    rag=st.session_state.rag,
                    llm=llm,
                    use_tspulse=use_tspulse,
                )
                st.session_state.result = result
                st.session_state.context = context
                st.session_state.mode = mode
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return

    result: Optional[DiagnosisResult] = st.session_state.result
    if result is None:
        st.info("Upload CSV (or demo), add context, then Run diagnosis.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity", result.severity.get("severity_label", "—"))
    m2.metric("Score", f"{result.severity.get('severity_score', 0):.2f}")
    m3.metric("Top channel", result.severity.get("top_channel") or "—")
    m4.metric("Signature", str((result.classification or {}).get("top_label") or "—"))

    st.write("**Status:**", result.status, f"· {result.elapsed_s:.1f}s")

    tab_r, tab_a, tab_rag, tab_d, tab_s = st.tabs(
        ["Final report", "Anomaly", "RAG hits", "Draft / pass-2", "Save & Learn"]
    )
    with tab_r:
        st.markdown(result.final_report)
    with tab_a:
        st.json(
            {
                "mode": result.anomaly.get("mode"),
                "summary": result.anomaly.get("summary"),
                "column_scores": result.anomaly.get("column_scores"),
                "anomalies": result.anomaly.get("anomalies", [])[:20],
                "classification": result.classification,
            }
        )
    with tab_rag:
        for h in result.rag_hits:
            st.markdown(f"**{h.get('source')}** (score={h.get('score', 0):.3f})")
            st.text(h.get("text", "")[:1500])
    with tab_d:
        st.subheader("Pass 1 — draft")
        st.markdown(result.draft)
        st.subheader("Pass 2 — critique / refine")
        st.markdown(result.reflection)
    with tab_s:
        corrections = st.text_area(
            "Your corrections / ground truth (feeds flywheel RAG)",
            height=120,
            key="corrections",
        )
        if st.button("Save & Learn", type="primary"):
            case = save_case(
                result,
                context=st.session_state.get("context", ""),
                mode=st.session_state.get("mode", "alerts"),
                user_corrections=corrections,
                rag=st.session_state.rag,
            )
            st.session_state.rag_status = case.get("reindex_status")
            st.success(f"Saved {case['path']}\n{case.get('reindex_status')}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # CLI first — do not import streamlit (broken protobuf stacks can hard-crash).
    if argv:
        return run_cli(argv)

    # streamlit run app.py sets runtime env / script context
    if os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_RUNTIME_ENV"):
        run_streamlit()
        return 0
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            run_streamlit()
            return 0
    except Exception:
        pass

    # No args: launch Streamlit UI if available
    try:
        import streamlit  # noqa: F401

        print("Starting Streamlit UI…")
        print("  streamlit run app.py")
        os.execvp(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )
    except Exception:
        return run_cli(["-h"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
