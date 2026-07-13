"""
GT Diagnostic Harness — modern desktop GUI (CustomTkinter).

Uses the same diagnosis pipeline as the Textual TUI:
CSV → anomaly → RAG → LLM → report + proof plots.

Launch
------
    python app.py --ui gui
    set GT_UI=gui && python app.py
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .utils import PROJECT_ROOT, SAMPLES_DIR, ensure_directories, setup_logging

logger = setup_logging()

DEFAULT_CSV = str((SAMPLES_DIR / "gt_sensors_demo.csv").resolve())


def list_sample_csvs() -> List[Tuple[str, str]]:
    """(label, absolute path) for sample CSVs."""
    opts: List[Tuple[str, str]] = []
    if not SAMPLES_DIR.is_dir():
        return opts
    demo = SAMPLES_DIR / "gt_sensors_demo.csv"
    if demo.is_file():
        opts.append(("Demo · gt_sensors_demo.csv", str(demo.resolve())))
    for sub in sorted(SAMPLES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for csv in sorted(sub.glob("*.csv")):
            opts.append((f"{sub.name} · {csv.name}", str(csv.resolve())))
    return opts


def run_gui(*, auto_download: bool = True) -> int:
    """Entry point for the desktop GUI."""
    try:
        import customtkinter as ctk  # noqa: F401
    except ImportError:
        print(
            "Desktop GUI requires customtkinter.\n"
            "  pip install customtkinter\n"
            "Or use the terminal UI:  python app.py --ui textual",
            flush=True,
        )
        return 2

    ensure_directories()
    app = GTHarnessGUI(auto_download=auto_download)
    app.mainloop()
    return 0


class GTHarnessGUI:
    """Modern dark industrial desktop shell for GT Diagnostic Harness."""

    def __init__(self, *, auto_download: bool = True) -> None:
        import customtkinter as ctk
        from tkinter import filedialog, messagebox

        self.ctk = ctk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.auto_download = auto_download
        self._bundle = None
        self._last_result = None
        self._busy = False
        self._models_ready = False
        self._q: queue.Queue = queue.Queue()
        self._sample_map = {lab: path for lab, path in list_sample_csvs()}

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("GT Diagnostic Harness")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 640)
        try:
            self.root.iconbitmap(default="")  # ignore if none
        except Exception:
            pass

        self._build()
        self.root.after(100, self._poll_queue)
        self.root.after(200, self._start_model_load)

    def mainloop(self) -> None:
        self.root.mainloop()

    # ── Layout ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        ctk = self.ctk
        root = self.root

        # Header
        header = ctk.CTkFrame(root, corner_radius=0, fg_color=("#1a1a2e", "#0f1419"), height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header,
            text="GT HARNESS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#22d3ee",
        ).pack(side="left", padx=16, pady=12)
        self.chips = ctk.CTkLabel(
            header,
            text="MODE · —   ·  SEV · —   ·  ENGINE · —   ·  MODELS · loading…",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            anchor="w",
        )
        self.chips.pack(side="left", fill="x", expand=True, padx=8)
        ctk.CTkLabel(
            header,
            text="local · offline-capable",
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
        ).pack(side="right", padx=16)

        # Body split
        body = ctk.CTkFrame(root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=8)

        left = ctk.CTkScrollableFrame(body, width=380, label_text="Setup", label_font=ctk.CTkFont(weight="bold"))
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # CSV
        ctk.CTkLabel(left, text="Sensor CSV", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(4, 2))
        path_row = ctk.CTkFrame(left, fg_color="transparent")
        path_row.pack(fill="x", pady=2)
        self.csv_var = ctk.StringVar(value=DEFAULT_CSV if Path(DEFAULT_CSV).is_file() else "")
        self.csv_entry = ctk.CTkEntry(path_row, textvariable=self.csv_var, height=36)
        self.csv_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(path_row, text="Browse…", width=90, command=self._browse).pack(side="left", padx=(6, 0))

        # Samples
        ctk.CTkLabel(left, text="Sample scenarios", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        labels = list(self._sample_map.keys()) or ["(no samples found)"]
        self.sample_var = ctk.StringVar(value=labels[0])
        self.sample_menu = ctk.CTkOptionMenu(
            left,
            variable=self.sample_var,
            values=labels,
            command=self._on_sample,
            height=34,
        )
        self.sample_menu.pack(fill="x", pady=2)
        if labels and labels[0] in self._sample_map:
            self.csv_var.set(self._sample_map[labels[0]])

        # Mode
        ctk.CTkLabel(left, text="Diagnostic mode", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        self.mode_var = ctk.StringVar(value="Alerts")
        mode_row = ctk.CTkFrame(left, fg_color="transparent")
        mode_row.pack(fill="x")
        ctk.CTkRadioButton(mode_row, text="Alerts", variable=self.mode_var, value="Alerts", command=self._refresh_chips).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(mode_row, text="Trips/Event", variable=self.mode_var, value="Trips/Event", command=self._refresh_chips).pack(side="left")

        # Context
        ctk.CTkLabel(left, text="Context / process maps / SOE", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        self.context_box = ctk.CTkTextbox(left, height=110, font=ctk.CTkFont(family="Consolas", size=12))
        self.context_box.pack(fill="x", pady=2)
        self.context_box.insert("1.0", "Alarm text, trip first-outs, SOE notes…")

        # Actions
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", pady=(14, 4))
        self.btn_run = ctk.CTkButton(
            btn_row,
            text="Run diagnosis",
            fg_color="#0891b2",
            hover_color="#0e7490",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            command=self._run,
        )
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_save = ctk.CTkButton(
            btn_row,
            text="Save & Learn",
            fg_color="#059669",
            hover_color="#047857",
            height=40,
            command=self._save,
        )
        self.btn_save.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(
            btn_row,
            text="New",
            fg_color="#b45309",
            hover_color="#92400e",
            width=64,
            height=40,
            command=self._new_session,
        ).pack(side="left", padx=(4, 0))

        # Corrections
        ctk.CTkLabel(left, text="User corrections (Save & Learn)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        self.corr_box = ctk.CTkTextbox(left, height=70, font=ctk.CTkFont(family="Consolas", size=12))
        self.corr_box.pack(fill="x", pady=2)

        ctk.CTkLabel(
            left,
            text="Engineering decision-support only. Not OEM protection software.",
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(16, 4))

        # Right: tabs
        self.tabs = ctk.CTkTabview(right, segmented_button_selected_color="#0891b2")
        self.tabs.pack(fill="both", expand=True)
        tab_report = self.tabs.add("Report")
        tab_plots = self.tabs.add("Proof plots")
        tab_live = self.tabs.add("Live log")

        self.report_box = ctk.CTkTextbox(tab_report, font=ctk.CTkFont(family="Consolas", size=13), wrap="word")
        self.report_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.report_box.insert(
            "1.0",
            "Modern desktop GUI ready.\n\n"
            "1. Pick a sample or Browse for a CSV\n"
            "2. Choose Alerts or Trips/Event\n"
            "3. Click Run diagnosis\n\n"
            "Proof plots and the full write-up appear in these tabs.",
        )

        # High-quality PNG proof plots (scrollable image stack)
        self.plots_scroll = ctk.CTkScrollableFrame(tab_plots, label_text="High-quality charts")
        self.plots_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self._plot_image_labels: List[Any] = []
        self._plot_photo_refs: List[Any] = []  # prevent GC of CTkImage/PhotoImage
        self.plots_placeholder = ctk.CTkLabel(
            self.plots_scroll,
            text="High-quality proof plots appear here after a run\n"
            "(matplotlib PNG · flagged samples as ▲ markers).",
            text_color="#94a3b8",
            justify="left",
        )
        self.plots_placeholder.pack(anchor="w", padx=8, pady=12)
        self.plots_ascii_fallback = ctk.CTkTextbox(
            tab_plots, height=100, font=ctk.CTkFont(family="Consolas", size=11), wrap="none"
        )
        # hidden until needed as fallback
        self._plots_ascii_packed = False

        self.live_box = ctk.CTkTextbox(tab_live, font=ctk.CTkFont(family="Consolas", size=12), wrap="word")
        self.live_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.live_box.insert("1.0", "Live pipeline output…\n")

        # Footer progress
        foot = ctk.CTkFrame(root, corner_radius=0, fg_color=("#1e293b", "#111827"), height=72)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        self.progress_label = ctk.CTkLabel(foot, text="Starting…", anchor="w", text_color="#94a3b8")
        self.progress_label.pack(fill="x", padx=14, pady=(8, 2))
        self.progress = ctk.CTkProgressBar(foot, height=12, progress_color="#22d3ee")
        self.progress.pack(fill="x", padx=14, pady=(0, 10))
        self.progress.set(0)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _set_status(self, text: str, *, frac: Optional[float] = None) -> None:
        self.progress_label.configure(text=text)
        if frac is not None:
            self.progress.set(max(0.0, min(1.0, frac)))

    def _refresh_chips(
        self,
        *,
        severity: str = "—",
        engine: str = "—",
        models: Optional[str] = None,
    ) -> None:
        mod = models if models is not None else ("ready" if self._models_ready else "…")
        self.chips.configure(
            text=f"MODE · {self.mode_var.get()}   ·  SEV · {severity}   ·  "
            f"ENGINE · {engine}   ·  MODELS · {mod}"
        )

    def _append_live(self, text: str) -> None:
        self.live_box.insert("end", text.rstrip() + "\n")
        self.live_box.see("end")

    def _browse(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select gas turbine sensor CSV",
            initialdir=str(SAMPLES_DIR if SAMPLES_DIR.is_dir() else PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _on_sample(self, choice: str) -> None:
        path = self._sample_map.get(choice)
        if path:
            self.csv_var.set(path)

    def _new_session(self) -> None:
        if self._busy:
            self.messagebox.showwarning("Busy", "Wait for the current diagnosis to finish.")
            return
        self._last_result = None
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", "New session — report cleared. Run a diagnosis.")
        self._clear_plot_images()
        self.plots_placeholder.configure(
            text="Proof plots cleared. Run a diagnosis to regenerate charts."
        )
        self.plots_placeholder.pack(anchor="w", padx=8, pady=12)
        self.live_box.delete("1.0", "end")
        self.live_box.insert("1.0", "Live log cleared.\n")
        self.progress.set(0)
        self._set_status("Ready · idle")
        self._refresh_chips(severity="—", engine="—")

    # ── Model load ──────────────────────────────────────────────────────

    def _start_model_load(self) -> None:
        self._set_status("Loading models…", frac=0.05)
        self._append_live("Startup: loading models…")

        def work() -> None:
            try:
                from .models import load_models

                def prog(sid: str, i: int, n: int, frac: float, msg: str) -> None:
                    self._q.put(("progress", frac * 0.4, f"[load] {msg}"))

                bundle = load_models(
                    auto_download=self.auto_download,
                    progress=prog,
                )
                self._q.put(("models_ok", bundle))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Model load failed")
                self._q.put(("models_err", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    # ── Diagnosis ───────────────────────────────────────────────────────

    def _run(self) -> None:
        if self._busy:
            self.messagebox.showinfo("Busy", "Diagnosis already running.")
            return
        if not self._models_ready and self._bundle is None:
            self.messagebox.showwarning("Models", "Models still loading — wait for Ready.")
            return
        csv_path = (self.csv_var.get() or "").strip().strip('"')
        if not csv_path:
            self.messagebox.showerror("CSV", "Enter or browse to a sensor CSV path.")
            return
        if not Path(csv_path).is_file():
            # try project-relative resolution via bridge helper if available
            try:
                from .bridge import _resolve_csv_path

                resolved = _resolve_csv_path(csv_path)
                if resolved is not None:
                    csv_path = str(resolved)
                    self.csv_var.set(csv_path)
            except Exception:
                pass
        if not Path(csv_path).is_file():
            self.messagebox.showerror("CSV", f"File not found:\n{csv_path}")
            return

        mode = self.mode_var.get()
        ctx = self.context_box.get("1.0", "end").strip()
        if ctx.startswith("Alarm text"):
            ctx = ""

        self._busy = True
        self.btn_run.configure(state="disabled")
        self._set_status(f"Running diagnosis ({mode})…", frac=0.45)
        self.live_box.delete("1.0", "end")
        self._append_live(f"Diagnosis started · mode={mode}")
        self._append_live(f"CSV: {csv_path}")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", "Diagnosing…\n\nLive output streams in the Live log tab.")
        self.tabs.set("Live log")

        def work() -> None:
            try:
                from .analysis import get_rag, run_diagnosis
                from .models import get_bundle

                bundle = self._bundle or get_bundle(auto_download=self.auto_download)

                def prog(sid: str, i: int, n: int, frac: float, msg: str) -> None:
                    # Map diagnosis 0–1 onto overall 0.45–0.98
                    overall = 0.45 + 0.53 * max(0.0, min(1.0, frac))
                    self._q.put(("progress", overall, msg))

                def live(section: str, text: str) -> None:
                    self._q.put(("live", section, text))

                result = run_diagnosis(
                    csv_file=csv_path,
                    mode=mode,
                    context=ctx,
                    bundle=bundle,
                    rag=get_rag(),
                    progress=prog,
                    live=live,
                )
                self._q.put(("diag_ok", result))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Diagnosis failed")
                self._q.put(("diag_err", f"{exc}\n{traceback.format_exc()[-800:]}"))

        threading.Thread(target=work, daemon=True).start()

    def _save(self) -> None:
        if self._last_result is None:
            self.messagebox.showwarning("Save", "Run a diagnosis first.")
            return
        try:
            from .analysis import get_rag, score_severity
            from .tools import save_case

            result = self._last_result
            corr = self.corr_box.get("1.0", "end").strip()
            sev = score_severity(result.anomaly)
            clf = (result.anomaly or {}).get("classification") or {}
            case = save_case(
                mode=result.mode,
                context=result.context_used,
                anomaly_summary=str(result.anomaly.get("summary", "")),
                analysis=result.draft,
                reflection=result.reflection,
                final_report=result.final_report,
                user_corrections=corr or "",
                severity=sev,
                metadata={
                    "severity": sev,
                    "anomaly_mode": result.anomaly.get("mode"),
                    "classification": clf,
                },
                rag=get_rag(),
                reindex=True,
            )
            self._append_live(f"Saved {case.get('case_id')}")
            self.messagebox.showinfo("Saved", f"Case saved: {case.get('case_id')}")
            self._set_status(f"Saved {case.get('case_id')} · score={sev.get('severity')}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save failed")
            self.messagebox.showerror("Save failed", str(exc))

    # ── Queue pump (main thread) ────────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, frac, msg = item
                    self._set_status(msg, frac=frac)
                    self._append_live(f"▸ {msg}")
                elif kind == "live":
                    _, section, text = item
                    self._append_live(f"── {section} ──")
                    # Cap very long chunks
                    lines = (text or "").splitlines()
                    for line in lines[:80]:
                        self._append_live(line)
                    if len(lines) > 80:
                        self._append_live(f"… ({len(lines) - 80} more lines)")
                elif kind == "models_ok":
                    self._bundle = item[1]
                    self._models_ready = True
                    st = getattr(self._bundle, "status", {}) or {}
                    llm = str(st.get("llm", ""))[:48]
                    self._set_status(f"Ready · {llm}", frac=1.0)
                    self._refresh_chips(models="ready")
                    self._append_live(f"Models ready: {st}")
                elif kind == "models_err":
                    self._models_ready = False
                    self._set_status(f"Model load error: {item[1]}", frac=0)
                    self._refresh_chips(models="error")
                    self.messagebox.showerror("Model load", item[1])
                elif kind == "diag_ok":
                    self._on_diag_done(item[1], None)
                elif kind == "diag_err":
                    self._on_diag_done(None, item[1])
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _on_diag_done(self, result: Any, err: Optional[str]) -> None:
        from .analysis import score_severity

        self._busy = False
        self.btn_run.configure(state="normal")
        if err:
            self._set_status(f"Error: {err}", frac=0)
            self.report_box.delete("1.0", "end")
            self.report_box.insert("1.0", f"Diagnosis failed\n\n{err}")
            self.tabs.set("Report")
            self.messagebox.showerror("Diagnosis failed", err[:500])
            return

        self._last_result = result
        sev = score_severity(result.anomaly)
        try:
            body = result.to_display_markdown()
        except Exception:
            body = result.to_markdown() if hasattr(result, "to_markdown") else str(result)
        header = (
            f"Severity: {sev.get('label')} · score={sev.get('severity')} · "
            f"top={sev.get('top_channel')} · engine={result.anomaly.get('mode')}\n"
            f"{'─' * 60}\n\n"
        )
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", header + body)

        plots = ""
        image_paths: List[str] = []
        if getattr(result, "evidence", None) is not None:
            plots = (result.evidence.ascii_art or "").strip()
            image_paths = list(result.evidence.image_paths or [])
            if result.evidence.combined_image_path:
                # ensure combined first
                cpath = result.evidence.combined_image_path
                image_paths = [cpath] + [p for p in image_paths if p != cpath]
        self._show_plot_images(image_paths, ascii_fallback=plots)

        # Persist last report
        try:
            from .utils import PROJECT_ROOT

            out = PROJECT_ROOT / "logs"
            out.mkdir(parents=True, exist_ok=True)
            (out / "last_diagnosis_report.md").write_text(header + body, encoding="utf-8")
            if plots:
                (out / "last_evidence_plots.txt").write_text(plots + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("report write: %s", exc)

        n_img = len(image_paths)
        self._set_status(
            f"Done · severity={sev.get('severity')} ({sev.get('level')}) · "
            f"{n_img} chart PNG(s) · Report + Proof plots",
            frac=1.0,
        )
        self._refresh_chips(
            severity=f"{sev.get('level')} ({sev.get('severity')})",
            engine=str(result.anomaly.get("mode") or "n/a"),
            models="ready",
        )
        self._append_live("✓ Diagnosis complete")
        if getattr(result, "evidence", None) and result.evidence.channels:
            names = ", ".join(
                f"{c.name}({c.score:.2f})" for c in result.evidence.channels
            )
            self._append_live(f"Proof channels: {names}")
        if image_paths:
            self._append_live(f"PNG charts: {image_paths[0]}")
            self.tabs.set("Proof plots")
        else:
            self.tabs.set("Report")

    def _clear_plot_images(self) -> None:
        for lab in self._plot_image_labels:
            try:
                lab.destroy()
            except Exception:
                pass
        self._plot_image_labels.clear()
        self._plot_photo_refs.clear()
        if self._plots_ascii_packed:
            try:
                self.plots_ascii_fallback.pack_forget()
            except Exception:
                pass
            self._plots_ascii_packed = False

    def _show_plot_images(
        self,
        image_paths: List[str],
        *,
        ascii_fallback: str = "",
    ) -> None:
        """Display matplotlib PNGs in the Proof plots tab (CTkImage)."""
        self._clear_plot_images()
        try:
            self.plots_placeholder.pack_forget()
        except Exception:
            pass

        valid = [p for p in image_paths if p and Path(p).is_file()]
        if not valid:
            self.plots_placeholder.configure(
                text="No PNG charts for this run.\n"
                "Install matplotlib for high-quality plots, or see ASCII below."
            )
            self.plots_placeholder.pack(anchor="w", padx=8, pady=12)
            if ascii_fallback:
                self.plots_ascii_fallback.delete("1.0", "end")
                self.plots_ascii_fallback.insert("1.0", ascii_fallback)
                self.plots_ascii_fallback.pack(fill="x", padx=4, pady=4)
                self._plots_ascii_packed = True
            return

        ctk = self.ctk
        try:
            from PIL import Image
        except ImportError:
            # customtkinter usually depends on Pillow; fall back to ascii
            self.plots_placeholder.configure(
                text="Pillow not available — showing ASCII fallback.\n"
                f"PNG files on disk: {valid[0]}"
            )
            self.plots_placeholder.pack(anchor="w", padx=8, pady=8)
            if ascii_fallback:
                self.plots_ascii_fallback.delete("1.0", "end")
                self.plots_ascii_fallback.insert("1.0", ascii_fallback)
                self.plots_ascii_fallback.pack(fill="x", padx=4, pady=4)
                self._plots_ascii_packed = True
            return

        # Prefer combined chart first; still show individuals if more than one file
        display_paths = valid[:6]
        for i, path in enumerate(display_paths):
            try:
                img = Image.open(path)
                # Fit width ~ 900 px while keeping aspect
                max_w = 960
                w, h = img.size
                if w > max_w:
                    ratio = max_w / float(w)
                    img = img.resize((max_w, max(1, int(h * ratio))), Image.Resampling.LANCZOS)
                    w, h = img.size
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self._plot_photo_refs.append(ctk_img)
                lab = ctk.CTkLabel(
                    self.plots_scroll,
                    image=ctk_img,
                    text="",
                )
                lab.pack(anchor="w", padx=6, pady=(6 if i else 4, 8))
                self._plot_image_labels.append(lab)
                cap = ctk.CTkLabel(
                    self.plots_scroll,
                    text=Path(path).name,
                    text_color="#64748b",
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                )
                cap.pack(anchor="w", padx=10, pady=(0, 4))
                self._plot_image_labels.append(cap)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not show plot image %s: %s", path, exc)
                err = ctk.CTkLabel(
                    self.plots_scroll,
                    text=f"Could not load {path}: {exc}",
                    text_color="#f87171",
                    anchor="w",
                )
                err.pack(anchor="w", padx=8, pady=4)
                self._plot_image_labels.append(err)

        hint = ctk.CTkLabel(
            self.plots_scroll,
            text=f"Saved under logs/evidence_plots/  ·  {len(valid)} file(s)",
            text_color="#64748b",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        hint.pack(anchor="w", padx=10, pady=(4, 12))
        self._plot_image_labels.append(hint)
