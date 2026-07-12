"""
Machine-readable diagnosis bridge for alternate frontends (ratatui, etc.).

The Python pipeline stays the single source of truth. External TUIs call:

    python app.py --json-once path/to.csv --mode Alerts

and receive one JSON object on stdout (progress on stderr).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis import get_rag, run_diagnosis, score_severity
from .models import get_bundle, is_offline_draft, load_models, llm_available
from .tools import save_case
from .utils import PROJECT_ROOT, SAMPLES_DIR, ensure_directories, setup_logging

logger = setup_logging()


def _resolve_csv_path(csv_path: str) -> Optional[Path]:
    """
    Resolve CSV for bridge/TUI callers.

    Accepts absolute paths, project-relative paths, bare filenames under
    samples/, and quoted / file:// forms.
    """
    text = (csv_path or "").strip()
    if not text:
        return None
    # first line only
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if text.lower().startswith("file:"):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(text)
        path = unquote(parsed.path or "")
        if path.startswith("/") and len(path) >= 3 and path[2] == ":":
            path = path[1:]
        text = path
    candidates = [
        Path(text),
        PROJECT_ROOT / text,
        SAMPLES_DIR / Path(text).name,
    ]
    # scenario subfolders
    if SAMPLES_DIR.is_dir():
        name = Path(text).name
        for sub in SAMPLES_DIR.iterdir():
            if sub.is_dir():
                candidates.append(sub / name)
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def result_to_bridge_dict(result: Any, *, bundle: Any = None) -> Dict[str, Any]:
    """Serialize an AnalysisResult for JSON frontends."""
    sev = score_severity(result.anomaly or {})
    channels: List[Dict[str, Any]] = []
    evidence_ascii = ""
    if getattr(result, "evidence", None) is not None:
        evidence_ascii = (result.evidence.ascii_art or "").strip()
        for c in result.evidence.channels or []:
            channels.append(
                {
                    "name": c.name,
                    "score": float(c.score),
                    "flag_count": len(c.flag_rows or []),
                    "method": c.method or "",
                    "reason": c.reason or "",
                    "window_start": getattr(c, "window_start", 0),
                    "window_end": getattr(c, "window_end", 0),
                    "full_n": getattr(c, "full_n", 0),
                }
            )
    try:
        display = result.to_display_markdown()
    except Exception:
        display = result.to_markdown() if hasattr(result, "to_markdown") else ""

    return {
        "ok": True,
        "schema_version": 1,
        "mode": result.mode,
        "load_status": result.load_status,
        "anomaly_mode": (result.anomaly or {}).get("mode"),
        "anomaly_summary": (result.anomaly or {}).get("summary", ""),
        "severity": {
            "severity": sev.get("severity"),
            "level": sev.get("level"),
            "label": sev.get("label"),
            "top_channel": sev.get("top_channel"),
        },
        "final_report": result.final_report or "",
        "draft": result.draft or "",
        "reflection": result.reflection or "",
        "display_markdown": display,
        "evidence_ascii": evidence_ascii,
        "proof_channels": channels,
        "model_status": dict(result.model_status or {}),
        "llm_available": bool(llm_available(bundle)) if bundle is not None else None,
        "offline_draft": bool(is_offline_draft(result.draft or "")),
        "rag_hit_count": len(result.rag_hits or []),
    }


def run_json_once(
    csv_path: str,
    mode: str = "Alerts",
    context: str = "",
    *,
    auto_download: bool = True,
    save: bool = False,
    full_reflection: Optional[bool] = None,
) -> int:
    """
    Run one diagnosis and print a single JSON document to stdout.

    Exit codes: 0 success, 2 file missing, 1 pipeline error.
    """
    ensure_directories()
    path = _resolve_csv_path(csv_path)
    if path is None or not path.is_file():
        print(
            json.dumps(
                {
                    "ok": False,
                    "schema_version": 1,
                    "error": f"CSV not found: {csv_path}",
                    "hint": "Use an absolute path or a path relative to the project root "
                    "(e.g. samples/gt_sensors_demo.csv).",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2

    def _prog(sid: str, i: int, n: int, frac: float, msg: str) -> None:
        # Structured progress for frontends that scrape stderr
        print(
            f"[diag {int(frac * 100):3d}%] {sid} · {msg}",
            file=sys.stderr,
            flush=True,
        )

    t0 = time.monotonic()
    try:
        print("[load] loading models…", file=sys.stderr, flush=True)
        bundle = load_models(auto_download=auto_download, progress=None)
        try:
            get_rag().ensure_ready()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG warm: %s", exc)

        result = run_diagnosis(
            csv_file=str(path.resolve()),
            mode=mode,
            context=context or "",
            bundle=bundle,
            rag=get_rag(),
            progress=_prog,
            full_reflection=full_reflection,
        )
        payload = result_to_bridge_dict(result, bundle=bundle)
        payload["elapsed_s"] = round(time.monotonic() - t0, 2)
        payload["csv_path"] = str(path.resolve())

        if save:
            case = save_case(
                mode=result.mode,
                context=result.context_used,
                anomaly_summary=str(result.anomaly.get("summary", "")),
                analysis=result.draft,
                reflection=result.reflection,
                final_report=result.final_report,
                severity=score_severity(result.anomaly),
                metadata={"severity": score_severity(result.anomaly)},
                rag=get_rag(),
                reindex=True,
            )
            payload["saved_case_id"] = case.get("case_id")

        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("json-once failed")
        print(
            json.dumps(
                {
                    "ok": False,
                    "schema_version": 1,
                    "error": str(exc),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1
