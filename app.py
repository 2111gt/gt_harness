#!/usr/bin/env python3
"""
GT Diagnostic Harness — terminal UI entry point.

Launch
------
    cd gt_harness
    pip install -r requirements.txt
    python app.py                      # Textual TUI (default)
    python app.py --ui ratatui         # Rust ratatui TUI (swap)
    set GT_UI=ratatui && python app.py # same via env

Optional:
    python app.py --no-download
    python app.py --download-only
    python app.py --cli-once samples/gt_sensors_demo.csv
    python app.py --json-once samples/gt_sensors_demo.csv   # bridge for ratatui
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import get_rag, run_diagnosis, score_severity  # noqa: E402
from src.bridge import run_json_once  # noqa: E402
from src.models import get_bundle, is_offline_draft, load_models, llm_available  # noqa: E402
from src.tools import cases_history_markdown, save_case  # noqa: E402
from src.ui_launch import VALID_UIS, launch_ui, resolve_ui  # noqa: E402
from src.utils import ensure_directories, setup_logging  # noqa: E402

logger = setup_logging()


def run_cli_once(
    csv_path: str,
    mode: str = "Alerts",
    context: str = "",
    *,
    auto_download: bool = True,
    save: bool = False,
) -> int:
    """
    Non-interactive diagnosis (for tests / automation).
    Prints severity + report markdown to stdout.
    Progress / ETA lines go to stderr so stdout stays machine-readable.
    """
    import time

    ensure_directories()

    def _load_progress(sid: str, i: int, n: int, frac: float, msg: str) -> None:
        print(f"[load {int(frac * 100):3d}%] {msg}", file=sys.stderr, flush=True)

    t_load = time.monotonic()
    bundle = load_models(auto_download=auto_download, progress=_load_progress)
    try:
        print("[load] warming RAG index…", file=sys.stderr, flush=True)
        get_rag().ensure_ready()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG: %s", exc)
    print(
        f"[load] models ready in {time.monotonic() - t_load:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    def _diag_progress(sid: str, i: int, n: int, frac: float, msg: str) -> None:
        print(f"[diag {int(frac * 100):3d}%] {msg}", file=sys.stderr, flush=True)

    result = run_diagnosis(
        csv_file=csv_path,
        mode=mode,
        context=context,
        bundle=bundle,
        rag=get_rag(),
        progress=_diag_progress,
    )
    sev = score_severity(result.anomaly)
    print(f"SEVERITY\t{sev['severity']}\t{sev['level']}\t{sev['label']}\t{sev['top_channel']}")
    print(f"ANOMALY_MODE\t{result.anomaly.get('mode')}")
    print(f"LLM_AVAILABLE\t{llm_available(bundle)}")
    print(f"OFFLINE_DRAFT\t{is_offline_draft(result.draft)}")
    if getattr(result, "evidence", None) and result.evidence.channels:
        print(
            "PROOF_CHANNELS\t"
            + ",".join(f"{c.name}:{c.score:.3f}" for c in result.evidence.channels)
        )
        print("--- PROOF PLOTS ---")
        print(result.evidence_ascii() or result.evidence.ascii_art or "")
    print("--- REPORT ---")
    print(result.to_markdown())
    if save:
        case = save_case(
            mode=result.mode,
            context=result.context_used,
            anomaly_summary=str(result.anomaly.get("summary", "")),
            analysis=result.draft,
            reflection=result.reflection,
            final_report=result.final_report,
            severity=sev,
            metadata={"severity": sev},
            rag=get_rag(),
            reindex=True,
        )
        print(f"SAVED\t{case['case_id']}\t{sev['severity']}")
        print(cases_history_markdown(limit=5))
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GT Diagnostic Harness — Textual or ratatui TUI"
    )
    parser.add_argument(
        "--ui",
        choices=list(VALID_UIS),
        default=None,
        help="TUI backend: textual (Python, default) or ratatui (Rust). "
        "Also set via GT_UI=textual|ratatui",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-install packages or download model weights",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/install all models then exit",
    )
    parser.add_argument(
        "--cli-once",
        metavar="CSV",
        help="Run one non-interactive diagnosis on CSV and exit (no TUI)",
    )
    parser.add_argument(
        "--json-once",
        metavar="CSV",
        help="JSON diagnosis bridge for ratatui / external frontends (stdout = one JSON object)",
    )
    parser.add_argument(
        "--mode",
        default="Trips/Event",
        choices=["Alerts", "Trips/Event"],
        help="Mode for --cli-once / --json-once",
    )
    parser.add_argument("--context", default="", help="Context for --cli-once / --json-once")
    parser.add_argument(
        "--save",
        action="store_true",
        help="With --cli-once/--json-once, also Save & Learn the session",
    )
    parser.add_argument("--skip-model-load", action="store_true", help="(compat) same as defer")
    args = parser.parse_args(argv)

    ensure_directories()

    if args.no_download:
        os.environ["GT_NO_DOWNLOAD"] = "1"

    auto_download = not args.no_download

    if args.download_only:
        logger.info("Downloading / installing AI models…")
        from src.download import ensure_all_models, repair_environment

        repair_environment()
        status = ensure_all_models()
        for k, v in status.items():
            logger.info("  %s: %s", k, v)
        b = load_models(auto_download=True)
        logger.info("Load status: %s", b.status)
        print("Download complete.")
        for k, v in b.status.items():
            print(f"  {k}: {v}")
        return 0

    if args.json_once:
        return run_json_once(
            args.json_once,
            mode=args.mode,
            context=args.context,
            auto_download=auto_download,
            save=args.save,
        )

    if args.cli_once:
        return run_cli_once(
            args.cli_once,
            mode=args.mode,
            context=args.context,
            auto_download=auto_download,
            save=args.save,
        )

    try:
        ui = resolve_ui(args.ui)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    logger.info("Starting GT Diagnostic Harness · UI=%s", ui)
    if auto_download:
        logger.info("Auto-download enabled (GGUF / embeddings / TS Pulse if missing)")
    return launch_ui(ui, auto_download=auto_download)


if __name__ == "__main__":
    raise SystemExit(main())
