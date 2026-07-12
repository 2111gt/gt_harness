"""
Analysis orchestration for GT Diagnostic Harness.

Pipeline
--------
1. Load CSV sensor data
2. Anomaly detection (TS Pulse or statistical fallback)
3. RAG retrieval (knowledge + saved cases)
4. Main LLM draft (Granite 4.1 via llama.cpp)
5. Reflection pass (model critiques and improves its own answer)
6. Optional flywheel save (handled in app via tools.save_case)

All public functions are pure enough to unit-test with a ModelBundle
that has llm=None (offline draft) and statistical anomalies.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .models import (
    LLMConfig,
    ModelBundle,
    detect_anomalies,
    generate_llm,
    get_bundle,
    llm_available,
    offline_reflect,
)
from .tools import (
    KnowledgeRAG,
    dataframe_preview,
    format_rag_context,
    load_sensor_csv,
    numeric_profile,
)
from .utils import mode_label, setup_logging, truncate

logger = setup_logging()

ProgressCb = Callable[[str, int, int, float, str], None]
# (step_id, step_index, total_steps, frac_overall_0_1, message)

LiveCb = Callable[[str, str], None]
# (section_id, markdown_or_text) — stream partial outputs to the TUI as they appear

SYSTEM_ANALYST = """You are GT Diagnostic Harness, a senior gas turbine reliability engineer.
Use ONLY the provided sensor facts and knowledge snippets. Be precise and practical.
You MUST write these section headings exactly (in this order):

## Reasoning
Step-by-step reasoning about what the data is saying (why channels moved together, timing, load context).
If a signature classifier label is provided, say how the sensor pattern does or does not match that signature.

## Initial hypotheses
Ranked list of 3–5 plausible failure modes / causes with brief support and confidence High/Med/Low.
When a known alert signature is indicated, put related process-map failure modes near the top unless data contradicts them.

## Self-review
Critique your own hypotheses: what would disprove each, instrumentation traps, missing data.
If classifier is marked UNTRAINED, treat signature probs as weak hints only.

## Final diagnosis
Operator-facing diagnosis: executive summary, evidence bullets, immediate actions, longer-term checks, caveats.

Keep total length under ~500 words. Do not invent sensors that are not in the data."""

SYSTEM_ANALYST_TRIPS = """You are GT Diagnostic Harness investigating a TRIP / EVENT (not a routine baseload alert).
Use ONLY provided sensor facts, SOE/first-out style context, and knowledge/case snippets.

Cross-mode learning (required when data supports it):
- The same signature classifier and process maps used for day-to-day ALERTS also apply here.
- If a signature is indicated (e.g. combustion_dynamics, cold_spot, hets, vibration), reuse that
  alert process path and explicitly cite similar PRIOR ALERT cases from the retrieved snippets.
- Then ADD trip-specific reasoning: first-outs, protection/trip_active behavior, load/fuel collapse,
  restart holds — produce a trip-unique final diagnosis, not a copy of a routine alert note.
- If signature is normal/unknown/untrained, prioritize reconstruction anomalies + SOE over labels.

You MUST write these section headings exactly (in this order):

## Reasoning
Link pre-trip / event sensor pattern to any known alert signature and to trip timeline evidence.

## Initial hypotheses
Ranked causes: (1) signature-linked modes from alerts process maps, (2) trip-specific modes, (3) sensor faults.

## Self-review
What would disprove signature match vs true trip mechanism; instrumentation traps; missing SOE.

## Final diagnosis
Trip-focused executive summary; evidence; immediate post-trip actions; restart criteria; longer-term checks.
Mention similar prior alerts by case id/title when present in retrieved knowledge.

Keep total length under ~500 words. Do not invent sensors or alarms not in the data."""

SYSTEM_REFLECTOR = """You are a critical peer reviewer of gas turbine diagnostics.
Review the analyst draft (reasoning, hypotheses, self-review, diagnosis).
Produce an IMPROVED report using the SAME four headings:
## Reasoning
## Initial hypotheses
## Self-review
## Final diagnosis
Keep what is correct, fix weak logic, drop unsupported claims. Under ~500 words.
For trip/event drafts, preserve cross-links to prior alert signatures and cases when supported.
Do not mention that you are a reviewer."""

# Pipeline steps with nominal durations (seconds) for ETA — calibrated for CPU GGUF CLI
DIAGNOSIS_STEPS: Tuple[Tuple[str, str, float], ...] = (
    ("csv", "Loading CSV", 1.0),
    ("anomaly", "Anomaly detection (TS Pulse)", 12.0),
    ("rag", "Knowledge retrieval (RAG)", 4.0),
    ("llm", "LLM diagnosis (Granite GGUF)", 90.0),
    ("finalize", "Finalizing report", 2.0),
)


def diagnosis_step_plan(*, full_reflection: bool = False) -> List[Tuple[str, str, float]]:
    """Return ordered (id, label, nominal_seconds) steps."""
    steps = list(DIAGNOSIS_STEPS)
    if full_reflection:
        # Insert reflection before finalize
        steps = steps[:-1] + [
            ("reflect", "LLM self-review pass", 90.0),
            steps[-1],
        ]
    return steps


@dataclass
class AnalysisResult:
    """Structured output of one diagnostic run."""

    mode: str
    load_status: str
    anomaly: Dict[str, Any]
    rag_hits: List[Dict[str, Any]]
    draft: str
    reflection: str
    final_report: str
    context_used: str
    model_status: Dict[str, str] = field(default_factory=dict)
    preview: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Full report — operator-first order for TUI / CLI."""
        return self.to_display_markdown()

    def to_display_markdown(self) -> str:
        """
        Human-facing report: final answer first, then reasoning trail,
        then supporting evidence. Scrollable in the TUI.
        """
        top = self.anomaly.get("anomalies") or []
        top_lines = []
        for a in top[:12]:
            try:
                sc = float(a.get("score") or 0)
                sc_s = f"{sc:.3f}"
            except (TypeError, ValueError):
                sc_s = str(a.get("score"))
            top_lines.append(
                f"- row={a.get('row')} · **{a.get('column')}** · "
                f"value={a.get('value')} · score={sc_s} · {a.get('method')}"
            )
        anomaly_bullets = (
            "\n".join(top_lines)
            if top_lines
            else "_No point anomalies flagged above threshold._"
        )
        rag_bits = format_rag_context(self.rag_hits, max_chars=1600)
        status = "\n".join(
            f"- **{k}**: {v}" for k, v in (self.model_status or {}).items()
        )
        draft = (self.draft or "").strip() or "_No LLM draft produced._"
        reflection = (self.reflection or "").strip() or "_No self-review produced._"
        final = (self.final_report or "").strip() or "_No final report produced._"

        clf = self.anomaly.get("classification") or {}
        if clf.get("enabled") and clf.get("top_label"):
            conf = clf.get("top_prob")
            conf_s = f"{conf:.1%}" if isinstance(conf, (int, float)) else "n/a"
            rank = clf.get("ranking") or []
            rank_s = "\n".join(f"- {lab}: {p:.1%}" for lab, p in rank[:6]) if rank else "_n/a_"
            trained = "trained" if clf.get("trained") else "UNTRAINED scaffold"
            clf_block = (
                f"**Status:** {trained}  \n"
                f"**Top signature:** `{clf.get('top_label')}` ({conf_s})  \n"
                f"**Note:** {clf.get('note', '')}\n\n"
                f"{rank_s}"
            )
        else:
            clf_block = "_Classifier not run or unavailable._"

        return f"""# GT Diagnostic Report

**Mode:** `{self.mode}`  
**Data:** {self.load_status}  
**Anomaly engine:** `{self.anomaly.get('mode', 'n/a')}`

---

# 1. Final report

{final}

---

# 2. Reasoning, hypotheses & self-review (LLM draft)

The model’s first-pass reasoning trail (before packaging / reflection):

{draft}

---

# 3. Self-review / reflection notes

{reflection}

---

# 4. Evidence

## Anomaly summary
{self.anomaly.get('summary', '')}

### Flagged points
{anomaly_bullets}

## Signature classifier (TS Pulse head)
{clf_block}

## Retrieved knowledge & prior cases
{rag_bits}

## Data preview
{self.preview or '_n/a_'}

## Model status
{status or '_n/a_'}
"""


# Singleton RAG for the process lifetime
_RAG: Optional[KnowledgeRAG] = None


def get_rag(force_rebuild: bool = False) -> KnowledgeRAG:
    """Shared KnowledgeRAG instance."""
    global _RAG
    if _RAG is None:
        _RAG = KnowledgeRAG()
        _RAG.ensure_ready()
    elif force_rebuild:
        _RAG.rebuild_index()
    return _RAG


def run_diagnosis(
    csv_file: Any = None,
    mode: str = "Alerts",
    context: str = "",
    bundle: Optional[ModelBundle] = None,
    rag: Optional[KnowledgeRAG] = None,
    llm_config: Optional[LLMConfig] = None,
    z_threshold: float = 3.0,
    df: Optional[pd.DataFrame] = None,
    progress: Optional[ProgressCb] = None,
    full_reflection: Optional[bool] = None,
    live: Optional[LiveCb] = None,
) -> AnalysisResult:
    """
    End-to-end diagnostic pipeline.

    Parameters
    ----------
    csv_file :
        Path or file for sensor CSV (ignored if ``df`` is passed).
    mode :
        "Alerts" or "Trips/Event" (legacy: Routine Check / Event Investigation).
    context :
        Operator notes, process map excerpts, alarm text, etc.
    bundle / rag :
        Optional injected dependencies for tests.
    df :
        Pre-loaded DataFrame (preferred in unit tests).
    progress :
        Optional callback(step_id, step_index, total_steps, frac, message)
        for UI status bars / ETA.
    full_reflection :
        If True, run a second LLM pass (much slower on CPU). Default False
        unless env GT_FULL_REFLECTION=1 — uses fast offline packaging instead.
    live :
        Optional callback(section_id, text) to stream partial results (anomaly,
        draft, reflection, final) into a live UI feed as they are produced.
    """
    def _live(section: str, text: str) -> None:
        if live is None:
            return
        try:
            live(section, text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("live callback failed: %s", exc)

    bundle = bundle or get_bundle()
    rag = rag or get_rag()
    mode_key = mode_label(mode)
    cfg = llm_config or LLMConfig()
    # Ensure thread count is concrete for CLI; allow room for structured sections
    if not cfg.n_threads or cfg.max_tokens < 512:
        cfg = LLMConfig(
            n_ctx=max(cfg.n_ctx, 2048),
            n_threads=cfg.resolved_threads(),
            n_gpu_layers=cfg.n_gpu_layers,
            temperature=cfg.temperature,
            max_tokens=max(cfg.max_tokens, 512),
            top_p=cfg.top_p,
            chat_format=cfg.chat_format,
        )

    # Default: run a real LLM self-review when GGUF is loaded (user wants
    # hypotheses + review). Set GT_FULL_REFLECTION=0 to skip the 2nd pass.
    if full_reflection is None:
        env = (os.environ.get("GT_FULL_REFLECTION") or "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            full_reflection = False
        elif env in {"1", "true", "yes", "on"}:
            full_reflection = True
        else:
            full_reflection = llm_available(bundle)

    steps = diagnosis_step_plan(full_reflection=bool(full_reflection))
    # Shrink LLM nominal ETA when GGUF is offline (rule-based draft is instant)
    if not llm_available(bundle):
        steps = [
            (sid, label, (1.5 if sid in ("llm", "reflect") else nom))
            for sid, label, nom in steps
        ]
    # If TS Pulse will lazy-load (deferred), keep TS Pulse nominal duration.
    # Only shrink ETA when intentionally statistical with no deferred load.
    st_ts = str((bundle.status or {}).get("tspulse", "")).lower()
    will_lazy_tspulse = "deferred" in st_ts or (bundle.tspulse is not None and bundle.tspulse_mode == "tspulse")
    if (bundle.tspulse_mode or "") != "tspulse" and not will_lazy_tspulse:
        steps = [
            (sid, ("Anomaly detection (statistical)" if sid == "anomaly" else label), (2.0 if sid == "anomaly" else nom))
            for sid, label, nom in steps
        ]

    total = len(steps)
    t0 = time.monotonic()
    nominal_total = sum(s[2] for s in steps)
    # Actual durations per finished step (for adaptive ETA)
    actual_durs: List[float] = []
    step_t0 = t0

    def _report(step_i: int, detail: str = "", *, within: float = 0.05) -> None:
        """
        within : 0..1 how far through the current step (start≈0.05, mid≈0.5, end≈0.95).
        """
        if progress is None:
            return
        sid, label, nom = steps[step_i]
        done_nom = sum(s[2] for s in steps[:step_i]) + max(0.0, min(1.0, within)) * nom
        frac = min(0.99, done_nom / max(nominal_total, 1e-6))
        elapsed = time.monotonic() - t0
        remaining_nom = sum(s[2] for s in steps[step_i + 1 :]) + (1.0 - within) * nom
        # Scale remaining nominal by observed pace vs nominal so far
        if actual_durs:
            nom_done = sum(s[2] for s in steps[: len(actual_durs)])
            scale = (sum(actual_durs) / max(nom_done, 1e-6)) if nom_done else 1.0
            # Blend toward 1.0 early so one slow step doesn't explode ETA
            blend = min(1.0, len(actual_durs) / max(total, 1))
            scale = (1.0 - 0.6 * blend) + 0.6 * blend * scale
            eta = max(0.0, remaining_nom * max(0.25, min(scale, 4.0)))
        elif frac > 0.08 and elapsed > 1.5:
            pace = elapsed / max(frac, 1e-3)
            eta = max(0.0, pace * (1.0 - frac))
        else:
            eta = remaining_nom
        eta_s = int(round(eta))
        if eta_s >= 90:
            eta_txt = f"~{eta_s // 60}m{eta_s % 60:02d}s left"
        else:
            eta_txt = f"~{eta_s}s left"
        msg = f"{label}" + (f" — {detail}" if detail else "")
        msg += f" · {eta_txt} · step {step_i + 1}/{total} · elapsed {int(elapsed)}s"
        try:
            progress(sid, step_i, total, frac, msg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("progress callback failed: %s", exc)

    def _begin_step(step_i: int, detail: str = "") -> None:
        nonlocal step_t0
        step_t0 = time.monotonic()
        _report(step_i, detail, within=0.05)

    def _end_step(step_i: int, detail: str = "") -> None:
        dur = max(0.01, time.monotonic() - step_t0)
        if len(actual_durs) == step_i:
            actual_durs.append(dur)
        elif len(actual_durs) > step_i:
            actual_durs[step_i] = dur
        else:
            while len(actual_durs) < step_i:
                actual_durs.append(0.0)
            actual_durs.append(dur)
        _report(step_i, detail or f"done in {dur:.1f}s", within=1.0)

    # --- CSV ---
    _live("step", "### Step: loading CSV…")
    _begin_step(0, "reading file")
    if df is None:
        df, load_status = load_sensor_csv(csv_file)
    else:
        load_status = f"In-memory frame: {len(df)} rows × {len(df.columns)} columns."

    preview = dataframe_preview(df, n=5)
    profile = numeric_profile(df)
    _end_step(0, f"{len(df)} rows")
    _live("csv", f"**Data loaded:** {load_status}\n\n{preview}")

    # --- Anomaly ---
    _live("step", "### Step: anomaly detection (TS Pulse / statistical)…")
    _begin_step(1, "TS Pulse / statistical scan")
    anomaly = detect_anomalies(df, bundle=bundle, z_threshold=z_threshold)
    sev = score_severity(anomaly)
    _end_step(1, f"mode={anomaly.get('mode')}")
    top_flags = anomaly.get("anomalies") or []
    flag_md = "\n".join(
        f"- row={a.get('row')} **{a.get('column')}** score={a.get('score')} ({a.get('method')})"
        for a in top_flags[:8]
    ) or "_none_"
    clf = anomaly.get("classification") or {}
    clf_md = ""
    if clf.get("enabled"):
        top = clf.get("top_label")
        conf = clf.get("top_prob")
        conf_s = f"{conf:.1%}" if isinstance(conf, (int, float)) else "n/a"
        rank = clf.get("ranking") or []
        rank_s = ", ".join(f"{lab}={p:.1%}" for lab, p in rank[:4]) if rank else ""
        trained = "trained" if clf.get("trained") else "UNTRAINED scaffold"
        clf_md = (
            f"\n\n**Signature classifier ({trained}):** top=`{top}` ({conf_s})\n"
            f"{rank_s}\n"
            f"_{clf.get('note', '')}_"
        )
    _live(
        "anomaly",
        f"**Anomaly engine:** `{anomaly.get('mode')}`  \n"
        f"**Severity:** {sev.get('label')} (score={sev.get('severity')}, "
        f"top=`{sev.get('top_channel')}`)\n\n"
        f"{anomaly.get('summary', '')}\n\n"
        f"**Top flags:**\n{flag_md}"
        f"{clf_md}",
    )

    # --- RAG (signature-aware; trips also pull similar prior alerts) ---
    _live("step", "### Step: knowledge retrieval (RAG)…")
    _begin_step(2, "knowledge + prior cases")
    top_channels = _top_channels(anomaly, n=5)
    hits = _retrieve_knowledge(
        rag,
        mode_key=mode_key,
        context=context or "",
        anomaly=anomaly,
        top_channels=top_channels,
    )
    rag_block = format_rag_context(hits, max_chars=2200)

    user_prompt = _build_analyst_prompt(
        mode_key=mode_key,
        context=context or "",
        load_status=load_status,
        anomaly=anomaly,
        profile=profile,
        rag_block=rag_block,
        preview=preview,
    )
    _end_step(2, f"{len(hits)} hits")
    _live("rag", f"**RAG hits:** {len(hits)}\n\n{truncate(rag_block, 1200)}")

    # --- LLM (single pass by default — 2nd pass ~doubles wall time on CPU) ---
    llm_step = 3
    _live("step", "### Step: LLM diagnosis (Granite GGUF) — this can take 1–3 minutes…")
    _begin_step(llm_step, "Granite GGUF (largest step on CPU)")
    system = SYSTEM_ANALYST_TRIPS if mode_key == "trips_event" else SYSTEM_ANALYST
    draft = generate_llm(bundle, system, user_prompt, config=cfg, role="analyst")
    _end_step(llm_step, f"{len(draft or '')} chars")
    _live(
        "draft",
        "## Live: Reasoning / hypotheses / self-review (LLM draft)\n\n"
        + (draft or "_empty draft_"),
    )

    if full_reflection and llm_available(bundle):
        _live("step", "### Step: LLM self-review pass (2nd generate)…")
        _begin_step(4, "self-review pass (2nd GGUF generate)")
        reflect_prompt = _build_reflection_prompt(
            mode_key=mode_key,
            context=context or "",
            anomaly_summary=str(anomaly.get("summary", "")),
            draft=draft,
            rag_block=rag_block,
        )
        improved = generate_llm(
            bundle, SYSTEM_REFLECTOR, reflect_prompt, config=cfg, role="reflect"
        )
        reflection = _format_reflection_notes(draft, improved)
        final_report = (
            improved.strip()
            if improved and improved.strip()
            else offline_reflect(
                draft,
                mode_key=mode_key,
                anomaly_summary=str(anomaly.get("summary", "")),
                severity=sev,
            )
        )
        _end_step(4)
        _live(
            "reflection",
            "## Live: Self-review / reflection\n\n" + (reflection or "_empty_"),
        )
        fin_i = 5
        _begin_step(fin_i, "assembling display report")
        _end_step(fin_i)
    else:
        # Single LLM draft (already contains Reasoning / Hypotheses / Self-review
        # / Final diagnosis when model follows the system prompt) + packaging
        fin_i = 4
        _live("step", "### Step: packaging final report…")
        _begin_step(fin_i, "packaging final report")
        final_report = offline_reflect(
            draft,
            mode_key=mode_key,
            anomaly_summary=str(anomaly.get("summary", "")),
            severity=sev,
        )
        if llm_available(bundle):
            # Surface self-review section from the draft when no 2nd pass
            draft_self = _extract_markdown_section(draft, "Self-review")
            draft_hyp = _extract_markdown_section(draft, "Initial hypotheses")
            bits = [
                "Single LLM pass (set GT_FULL_REFLECTION=1 to force a second critique generate).",
            ]
            if draft_hyp:
                bits.append("### Hypotheses (from draft)\n" + draft_hyp)
            if draft_self:
                bits.append("### Self-review (from draft)\n" + draft_self)
            if not draft_hyp and not draft_self:
                bits.append(
                    "Draft did not use the expected section headings; full draft is shown above."
                )
            reflection = "\n\n".join(bits)
        else:
            reflection = (
                "Offline path (no GGUF): deterministic packaging of the rule-based draft. "
                f"Severity level={sev.get('level')}, score={sev.get('severity')}."
            )
        _end_step(fin_i)
        _live("reflection", "## Live: Self-review notes\n\n" + reflection)

    _live("final", "## Live: Final report (packaged)\n\n" + (final_report or "_empty_"))

    if progress is not None:
        try:
            elapsed = time.monotonic() - t0
            progress(
                "done",
                total,
                total,
                1.0,
                f"Complete in {elapsed:.0f}s · steps: "
                + ", ".join(f"{s[0]}={d:.1f}s" for s, d in zip(steps, actual_durs)),
            )
        except Exception:
            pass

    return AnalysisResult(
        mode=mode_key,
        load_status=load_status,
        anomaly=anomaly,
        rag_hits=hits,
        draft=draft,
        reflection=reflection,
        final_report=final_report,
        context_used=context or "",
        model_status=dict(bundle.status or {}),
        preview=preview,
        profile=profile,
    )


def score_severity(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic severity scoring from anomaly output.

    Used by tests and UI badges. Pure function of anomaly dict.
    """
    scores = anomaly.get("column_scores") or {}
    if not scores:
        return {
            "severity": 0.0,
            "level": "normal",
            "label": "No anomaly signal",
            "top_channel": None,
        }

    max_score = float(max(scores.values()))
    mean_top = float(sum(sorted(scores.values(), reverse=True)[:3]) / min(3, len(scores)))
    # Combine max and mean-top for stability
    severity = round(0.65 * max_score + 0.35 * mean_top, 4)

    # Thresholds depend slightly on method scale; robust z and residuals both
    # tend to flag ~3+ as interesting. Normalize gently.
    if severity >= 6.0:
        level, label = "critical", "Critical — immediate investigation"
    elif severity >= 4.0:
        level, label = "high", "High — prioritized review"
    elif severity >= 3.0:
        level, label = "elevated", "Elevated — monitor / plan checks"
    elif severity >= 1.5:
        level, label = "mild", "Mild deviation"
    else:
        level, label = "normal", "Within typical noise"

    top_channel = max(scores.items(), key=lambda kv: kv[1])[0]
    return {
        "severity": severity,
        "level": level,
        "label": label,
        "top_channel": top_channel,
        "max_score": max_score,
    }


def _top_channels(anomaly: Dict[str, Any], n: int = 6) -> List[str]:
    scores = anomaly.get("column_scores") or {}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in ordered[:n]]


# Map classifier labels → process-map / vocabulary for RAG
_SIGNATURE_RAG_TERMS: Dict[str, str] = {
    "normal": "baseload normal operation",
    "cold_spot": "combustion cold spot exhaust thermocouple sector process map cold spot",
    "hets": "HETS high exhaust temperature spread trip process map",
    "combustion_dynamics": "combustion dynamics pulsation comb_dyn process map",
    "vibration": "bearing vibration shaft process map vibration",
}


def _signature_label(anomaly: Dict[str, Any]) -> str:
    clf = anomaly.get("classification") or {}
    if not clf.get("enabled"):
        return ""
    lab = str(clf.get("top_label") or "").strip()
    return lab


def _build_rag_queries(
    *,
    mode_key: str,
    context: str,
    anomaly: Dict[str, Any],
    top_channels: Sequence[str],
) -> List[str]:
    """
    Primary query + optional trip-specific query for similar prior alerts.

    Shared signature vocabulary so trips retrieve the same process maps and
    alert cases the day-to-day path would use.
    """
    sig = _signature_label(anomaly)
    sig_terms = _SIGNATURE_RAG_TERMS.get(sig, sig.replace("_", " ") if sig else "")
    base_bits = [
        mode_key.replace("_", " "),
        context or "",
        "gas turbine",
        " ".join(top_channels),
        str(anomaly.get("summary", ""))[:280],
        sig_terms,
    ]
    if sig and sig != "normal":
        base_bits.append(f"signature {sig}")
    primary = " ".join(b for b in base_bits if b).strip()

    queries = [primary]
    if mode_key == "trips_event":
        # Second query: explicitly hunt prior ALERT learnings for this signature
        trip_bits = [
            "prior alert case",
            "saved alert",
            "routine alert findings",
            "similar historical alert",
            sig_terms or "process anomaly",
            " ".join(top_channels[:4]),
            "process map",
        ]
        if sig and sig != "normal":
            trip_bits.append(f"alert signature {sig}")
        queries.append(" ".join(trip_bits).strip())
    elif sig and sig != "normal":
        # Alerts: boost process-map language for the known signature
        queries.append(
            f"process map {sig_terms} alert monitoring standardized response"
        )
    return queries


def _retrieve_knowledge(
    rag: KnowledgeRAG,
    *,
    mode_key: str,
    context: str,
    anomaly: Dict[str, Any],
    top_channels: Sequence[str],
    k_primary: int = 4,
    k_secondary: int = 3,
) -> List[Dict[str, Any]]:
    """Run one or two RAG queries and merge unique hits (source+title key)."""
    rag.ensure_ready()
    queries = _build_rag_queries(
        mode_key=mode_key,
        context=context,
        anomaly=anomaly,
        top_channels=top_channels,
    )
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for i, q in enumerate(queries):
        k = k_primary if i == 0 else k_secondary
        try:
            hits = rag.query(q, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG query failed: %s", exc)
            hits = []
        for h in hits:
            key = (
                str(h.get("source") or ""),
                str(h.get("title") or ""),
                (str(h.get("text") or "")[:80]),
            )
            if key in seen:
                continue
            seen.add(key)
            # Tag why this hit was pulled (helps LLM + live feed)
            h = dict(h)
            h["query_role"] = "primary" if i == 0 else "prior_alerts"
            merged.append(h)
    return merged


def _build_analyst_prompt(
    *,
    mode_key: str,
    context: str,
    load_status: str,
    anomaly: Dict[str, Any],
    profile: Dict[str, Any],
    rag_block: str,
    preview: str,
) -> str:
    sev = score_severity(anomaly)
    profile_lines = []
    for col, stats in list(profile.items())[:8]:
        profile_lines.append(
            f"- {col}: mean={stats['mean']:.4g}, std={stats['std']:.4g}, "
            f"min={stats['min']:.4g}, max={stats['max']:.4g}"
        )
    profile_txt = "\n".join(profile_lines) or "_No numeric profile._"

    top = anomaly.get("anomalies") or []
    flag_lines = [
        f"- row={a.get('row')} col={a.get('column')} value={a.get('value')} score={a.get('score'):.3f}"
        for a in top[:10]
    ]
    flags = "\n".join(flag_lines) or "- none"

    clf = anomaly.get("classification") or {}
    if clf.get("enabled") and clf.get("top_label") is not None:
        conf = clf.get("top_prob")
        conf_s = f"{conf:.3f}" if isinstance(conf, (int, float)) else "n/a"
        rank = clf.get("ranking") or []
        rank_s = ", ".join(f"{lab}={p:.3f}" for lab, p in rank[:5]) if rank else ""
        trained = "trained" if clf.get("trained") else "UNTRAINED"
        clf_txt = (
            f"top={clf.get('top_label')} p={conf_s} ({trained}); ranking: {rank_s}. "
            f"{clf.get('note', '')}"
        )
    else:
        clf_txt = "not available"

    if mode_key == "trips_event":
        mode_guidance = """## Mode guidance (TRIPS / EVENT)
- Shared signature engine with day-to-day alerts: use the classifier label when data supports it.
- Cite similar PRIOR ALERT cases from retrieved knowledge (case ids / titles) when present.
- Combine alert process-map steps for that signature with trip first-out / SOE / load-collapse evidence.
- Final diagnosis must be trip-specific (restart holds, protection validation), not a routine alert copy.
"""
    else:
        mode_guidance = """## Mode guidance (ALERTS)
- Prefer standardized signature process maps when the classifier indicates a known pattern.
- Keep actions proportional to alert monitoring (not full trip restart criteria) unless data shows trip risk.
"""

    return f"""## Diagnostic request
Mode: {mode_key}
Severity: {sev['label']} (score={sev['severity']}, top_channel={sev['top_channel']})
Data: {load_status}

{mode_guidance}
## Operator / process context
{truncate(context or '(none provided)', 600)}

## Anomaly summary
{anomaly.get('summary', '')}
Method mode: {anomaly.get('mode')}

### Signature classifier (TS Pulse head — shared for alerts and trips)
{clf_txt}

### Flagged points
{flags}

## Channel profile (top)
{profile_txt}

## Retrieved knowledge & prior cases (includes process maps; trips query prior alerts)
{truncate(rag_block, 1600)}

## Required output format
Write exactly these headings:
## Reasoning
## Initial hypotheses
## Self-review
## Final diagnosis
"""


def _build_reflection_prompt(
    *,
    mode_key: str,
    context: str,
    anomaly_summary: str,
    draft: str,
    rag_block: str,
) -> str:
    return f"""## Original mode
{mode_key}

## Operator context
{truncate(context, 1500)}

## Anomaly summary
{anomaly_summary}

## Knowledge snippets
{truncate(rag_block, 2000)}

## Draft diagnosis to improve
{draft}

## Task
Produce the improved final diagnostic report for operators.
"""


def _reflection_delta(draft: str, improved: str) -> str:
    """Legacy alias for reflection notes."""
    return _format_reflection_notes(draft, improved)


def _format_reflection_notes(draft: str, improved: str) -> str:
    """
    Human-readable self-review notes comparing draft → improved.
    """
    if not improved:
        return "Self-review pass returned empty output; draft retained as final."
    if improved.strip() == draft.strip():
        return (
            "Self-review retained the draft with no substantive changes.\n\n"
            + improved.strip()
        )
    parts = [
        f"Self-review pass produced an improved report "
        f"(draft {len(draft or '')} chars → review {len(improved)} chars).",
        "",
        "### Revised report from self-review",
        improved.strip(),
    ]
    # Highlight section-level deltas when both use expected headings
    for title in ("Initial hypotheses", "Self-review", "Final diagnosis"):
        d_sec = _extract_markdown_section(draft, title)
        i_sec = _extract_markdown_section(improved, title)
        if d_sec and i_sec and d_sec.strip() != i_sec.strip():
            parts.append(f"\n### Delta — {title}")
            parts.append(f"_Was:_ {truncate(d_sec, 400)}")
            parts.append(f"_Now:_ {truncate(i_sec, 400)}")
    return "\n".join(parts)


def _extract_markdown_section(text: str, heading: str) -> str:
    """
    Extract body under ``## heading`` (or ``### heading``) until the next
    same-or-higher level heading.
    """
    if not text or not heading:
        return ""
    import re

    pat = re.compile(
        rf"^(#{{1,3}})\s*{re.escape(heading)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        # Allow "1. Final diagnosis" style leftovers
        pat2 = re.compile(
            rf"^#{{1,3}}\s*\d+\.\s*{re.escape(heading)}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pat2.search(text)
        if not m:
            return ""
    start = m.end()
    rest = text[start:]
    nxt = re.search(r"^#{1,3}\s+\S", rest, re.MULTILINE)
    if nxt:
        return rest[: nxt.start()].strip()
    return rest.strip()


def run_diagnosis_from_path(
    csv_path: Union[str, Any],
    mode: str = "Alerts",
    context: str = "",
    **kwargs: Any,
) -> AnalysisResult:
    """Convenience wrapper used by CLI / tests."""
    return run_diagnosis(csv_file=csv_path, mode=mode, context=context, **kwargs)
