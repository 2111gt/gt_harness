"""
Tools layer: CSV I/O, RAG knowledge base, and flywheel case storage.

ChromaDB + SentenceTransformer (all-MiniLM-L6-v2) power retrieval over:
- knowledge/  (process maps, GT theory, OEM notes you drop in)
- saved_cases/ (past analyses + user corrections for continuous improvement)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .utils import (
    CHROMA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    RAG_COLLECTION_NAME,
    SAVED_CASES_DIR,
    ensure_directories,
    list_knowledge_files,
    list_saved_case_files,
    new_case_id,
    read_json,
    read_text_file,
    setup_logging,
    truncate,
    utc_now_iso,
    write_json,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_sensor_csv(
    source: Union[str, Path, Any],
    max_rows: Optional[int] = 50000,
) -> Tuple[pd.DataFrame, str]:
    """
    Load a gas-turbine sensor CSV into a DataFrame.

    Parameters
    ----------
    source :
        File path, Gradio file object, or file-like with a ``.name`` attribute.
    max_rows :
        Safety cap to keep local inference responsive.

    Returns
    -------
    (dataframe, status_message)
    """
    path = _resolve_file_path(source)
    if path is None:
        return pd.DataFrame(), "No CSV file provided."

    try:
        df = pd.read_csv(path, nrows=max_rows)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"Failed to read CSV: {exc}"

    if df.empty:
        return df, f"CSV loaded but empty: {path.name}"

    # Strip column whitespace
    df.columns = [str(c).strip() for c in df.columns]
    msg = f"Loaded {path.name}: {len(df)} rows × {len(df.columns)} columns."
    return df, msg


def _resolve_file_path(source: Any) -> Optional[Path]:
    if source is None or source == "":
        return None
    if isinstance(source, Path):
        return source if source.is_file() else None
    if isinstance(source, str):
        p = Path(source)
        return p if p.is_file() else None
    # Gradio NamedString / tempfile
    name = getattr(source, "name", None) or getattr(source, "path", None)
    if name:
        p = Path(str(name))
        return p if p.is_file() else None
    return None


def dataframe_preview(df: pd.DataFrame, n: int = 8) -> str:
    """Return a short markdown preview of a sensor frame."""
    if df is None or df.empty:
        return "_No data._"
    head = df.head(n)
    try:
        return head.to_markdown(index=False)
    except Exception:
        return head.to_string(index=False)


def numeric_profile(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Basic per-column stats for prompt context."""
    profile: Dict[str, Dict[str, float]] = {}
    if df is None or df.empty:
        return profile
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        profile[str(col)] = {
            "min": float(s.min()) if s.notna().any() else 0.0,
            "max": float(s.max()) if s.notna().any() else 0.0,
            "mean": float(s.mean()) if s.notna().any() else 0.0,
            "std": float(s.std()) if s.notna().any() else 0.0,
        }
    return profile


# ---------------------------------------------------------------------------
# RAG — ChromaDB + SentenceTransformer
# ---------------------------------------------------------------------------

class KnowledgeRAG:
    """
    Vector store over knowledge docs and saved diagnostic cases.

    Usage
    -----
    >>> rag = KnowledgeRAG()
    >>> rag.rebuild_index()
    >>> hits = rag.query("high exhaust temperature spread", k=5)
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = RAG_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        force_memory: bool = False,
    ) -> None:
        ensure_directories()
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.force_memory = force_memory
        self._client = None
        self._collection = None
        self._embedder = None
        self._memory_docs: List[Dict[str, str]] = []  # fallback if chromadb missing
        self.backend: str = "uninitialized"

    # -- lazy loaders -------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            try:
                from .compat import ensure_ml_compat

                ensure_ml_compat()
            except Exception:
                pass

            # Prefer shared download helper (auto-install + HF cache)
            try:
                from .download import ensure_embeddings

                model, msg = ensure_embeddings(self.embedding_model_name)
                if model is not None:
                    logger.info("%s", msg)
                    self._embedder = model
                    return self._embedder
            except Exception as inner:  # noqa: BLE001
                logger.debug("ensure_embeddings fallback: %s", inner)

            from sentence_transformers import SentenceTransformer

            try:
                from .device import torch_device

                dev = torch_device()
                self._embedder = SentenceTransformer(
                    self.embedding_model_name, device=dev
                )
                logger.info("Embeddings on device=%s", dev)
            except TypeError:
                self._embedder = SentenceTransformer(self.embedding_model_name)
            return self._embedder
        except Exception as exc:  # noqa: BLE001
            logger.warning("SentenceTransformer unavailable: %s", exc)
            self._embedder = False  # sentinel
            return None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if self.force_memory:
            self.backend = "memory"
            self._collection = None
            return None
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chromadb"
            return self._collection
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB unavailable, using in-memory keyword RAG: %s", exc)
            self.backend = "memory"
            self._collection = None
            return None

    def embed_texts(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        """Embed a list of strings; None if embedder missing."""
        embedder = self._get_embedder()
        if embedder is None or embedder is False:
            return None
        vectors = embedder.encode(list(texts), show_progress_bar=False)
        return [v.tolist() for v in vectors]

    # -- indexing -----------------------------------------------------------

    def rebuild_index(self) -> str:
        """
        Re-scan knowledge/ and saved_cases/ and (re)build the vector index.

        Returns a short human-readable status string.
        """
        docs = collect_rag_documents()
        self._memory_docs = docs

        collection = self._get_collection()
        if collection is None:
            self.backend = "memory"
            return f"Indexed {len(docs)} document(s) in memory (ChromaDB offline)."

        # Drop & recreate for a clean rebuild (small corpora)
        try:
            if self._client is not None:
                try:
                    self._client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                collection = self._collection
        except Exception as exc:  # noqa: BLE001
            logger.warning("Collection reset failed: %s", exc)

        if not docs:
            self.backend = "chromadb"
            return "ChromaDB ready — no documents found yet in knowledge/ or saved_cases/."

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"], "kind": d["kind"], "title": d.get("title", "")} for d in docs]

        embeddings = self.embed_texts(texts)
        try:
            if embeddings is not None:
                collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            else:
                # Chroma default embedding function (if available)
                collection.add(ids=ids, documents=texts, metadatas=metadatas)
            self.backend = "chromadb"
            return f"Indexed {len(docs)} document(s) into ChromaDB at {self.persist_dir}."
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chroma add failed")
            self.backend = "memory"
            return f"Chroma add failed ({exc}); using memory index with {len(docs)} doc(s)."

    def ensure_ready(self) -> str:
        """Build index if empty / first use."""
        collection = self._get_collection()
        if self.backend == "memory" or collection is None:
            if not self._memory_docs:
                return self.rebuild_index()
            return f"Memory RAG ready ({len(self._memory_docs)} docs)."
        try:
            count = collection.count()
        except Exception:
            count = 0
        if count == 0:
            return self.rebuild_index()
        return f"ChromaDB ready with {count} chunk(s)."

    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a natural-language question.
        """
        question = (question or "").strip()
        if not question:
            return []

        collection = self._get_collection()
        if collection is not None and self.backend == "chromadb":
            try:
                emb = self.embed_texts([question])
                kwargs: Dict[str, Any] = {"n_results": max(1, k)}
                if emb is not None:
                    kwargs["query_embeddings"] = emb
                else:
                    kwargs["query_texts"] = [question]
                res = collection.query(**kwargs)
                hits: List[Dict[str, Any]] = []
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for i, doc in enumerate(docs):
                    hits.append(
                        {
                            "text": doc,
                            "source": (metas[i] or {}).get("source", ""),
                            "kind": (metas[i] or {}).get("kind", ""),
                            "title": (metas[i] or {}).get("title", ""),
                            "distance": dists[i] if i < len(dists) else None,
                        }
                    )
                return hits
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chroma query failed: %s", exc)

        # Memory / keyword fallback
        if not self._memory_docs:
            self._memory_docs = collect_rag_documents()
        return _keyword_retrieve(self._memory_docs, question, k=k)


def collect_rag_documents() -> List[Dict[str, str]]:
    """
    Walk knowledge/ and saved_cases/ and produce chunked documents for RAG.
    """
    docs: List[Dict[str, str]] = []

    for path in list_knowledge_files():
        text = read_text_file(path)
        if not text.strip():
            continue
        try:
            rel = str(path.relative_to(KNOWLEDGE_DIR.parent))
        except ValueError:
            rel = path.name
        for i, chunk in enumerate(_chunk_text(text)):
            docs.append(
                {
                    "id": _doc_id(path, i),
                    "text": chunk,
                    "source": rel,
                    "kind": "knowledge",
                    "title": path.stem,
                }
            )

    for path in list_saved_case_files():
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        text = format_case_for_rag(data)
        if not text.strip():
            continue
        for i, chunk in enumerate(_chunk_text(text, max_chars=1200)):
            docs.append(
                {
                    "id": _doc_id(path, i),
                    "text": chunk,
                    "source": path.name,
                    "kind": "saved_case",
                    "title": data.get("case_id", path.stem),
                }
            )
    return docs


def format_case_for_rag(case: Dict[str, Any]) -> str:
    """Flatten a saved case JSON into searchable text."""
    meta = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    sig = (
        case.get("signature_label")
        or meta.get("signature_label")
        or (meta.get("classification") or {}).get("top_label")
        or ""
    )
    mode = str(case.get("mode") or "")
    # Explicit "prior alert" language so trip queries can find alert cases
    mode_hint = ""
    if mode in {"alerts", "alert", "routine_check"} or "alert" in mode.lower():
        mode_hint = "prior alert case day-to-day alert signature library"
    elif "trip" in mode.lower() or mode in {"trips_event", "event_investigation"}:
        mode_hint = "trip event case investigation"

    parts = [
        f"Case ID: {case.get('case_id', '')}",
        f"Mode: {mode}",
        f"Case type: {mode_hint}",
        f"Signature label: {sig}",
        f"Alert signature: {sig}" if sig else "",
        f"Timestamp: {case.get('saved_at', '')}",
        f"Severity score: {case.get('severity_score', case_severity_score(case))}",
        f"Severity level: {case.get('severity_level', '')}",
        f"Severity label: {case.get('severity_label', '')}",
        f"Top channel: {case.get('top_channel', '')}",
        f"User corrections: {case.get('user_corrections', '')}",
        f"Anomaly summary: {case.get('anomaly_summary', '')}",
        f"Analysis: {case.get('analysis', '')}",
        f"Reflection: {case.get('reflection', '')}",
        f"Final report: {case.get('final_report', '')}",
        f"Context: {case.get('context', '')}",
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": "))


def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 120) -> List[str]:
    """Simple character chunking with overlap (no extra deps)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _doc_id(path: Path, chunk_index: int) -> str:
    raw = f"{path.resolve()}::{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _keyword_retrieve(docs: List[Dict[str, str]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Very small TF-ish scorer used when embeddings/Chroma are offline."""
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return []
    scored: List[Tuple[float, Dict[str, str]]] = []
    for d in docs:
        text_l = d["text"].lower()
        score = sum(text_l.count(t) for t in terms)
        if score > 0:
            scored.append((float(score), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits: List[Dict[str, Any]] = []
    for score, d in scored[:k]:
        hits.append(
            {
                "text": d["text"],
                "source": d.get("source", ""),
                "kind": d.get("kind", ""),
                "title": d.get("title", ""),
                "distance": 1.0 / (1.0 + score),
            }
        )
    return hits


def format_rag_context(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    """Join retrieved chunks into a single prompt block."""
    if not hits:
        return "No relevant knowledge retrieved."
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] ({h.get('kind')}: {h.get('source')})\n{h.get('text', '').strip()}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Flywheel — save & learn
# ---------------------------------------------------------------------------

def _next_case_seq() -> int:
    """
    Monotonic integer sequence for save order.

    Independent of wall-clock resolution so two saves in the same second
    still sort deterministically (oldest → newest by increasing seq).
    """
    max_seq = 0
    for path in list_saved_case_files():
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        try:
            max_seq = max(max_seq, int(data.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return max_seq + 1


def case_sort_key(case: Dict[str, Any]) -> tuple:
    """
    Sort key for chronological order (oldest first).

    Primary: monotonic ``seq`` (1, 2, 3…).
    Fallback for legacy cases without seq: saved_at, then path/case_id.
    """
    try:
        seq = int(case.get("seq") or 0)
    except (TypeError, ValueError):
        seq = 0
    return (
        seq,
        str(case.get("saved_at") or ""),
        str(case.get("path") or case.get("case_id") or ""),
    )


def save_case(
    *,
    mode: str,
    context: str,
    anomaly_summary: str,
    analysis: str,
    reflection: str,
    final_report: str,
    user_corrections: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    severity: Optional[Dict[str, Any]] = None,
    rag: Optional[KnowledgeRAG] = None,
    reindex: bool = True,
) -> Dict[str, Any]:
    """
    Persist a diagnostic case as JSON under saved_cases/ and refresh RAG.

    Severity is stored as **first-class fields** (severity_score, severity_level,
    severity_label, top_channel) so history/trend survive reload even if
    metadata is omitted by a caller.

    Each case also gets a monotonic ``seq`` so score_trend order is stable
    even when two saves share the same timestamp second.

    Returns the saved case dict (includes case_id and path).
    """
    ensure_directories()
    case_id = new_case_id()
    meta = dict(metadata or {})
    sev = _normalize_severity(severity, meta)
    seq = _next_case_seq()

    # Promote signature for RAG / trip cross-lookup
    sig_label = (
        meta.get("signature_label")
        or (meta.get("classification") or {}).get("top_label")
        or ""
    )
    if sig_label:
        meta.setdefault("signature_label", sig_label)

    case: Dict[str, Any] = {
        "case_id": case_id,
        "seq": seq,
        "saved_at": utc_now_iso(),
        "mode": mode,
        "context": context or "",
        "anomaly_summary": anomaly_summary or "",
        "analysis": analysis or "",
        "reflection": reflection or "",
        "final_report": final_report or "",
        "user_corrections": user_corrections or "",
        # First-class score fields (required for history + trend)
        "severity_score": sev["severity_score"],
        "severity_level": sev["severity_level"],
        "severity_label": sev["severity_label"],
        "top_channel": sev["top_channel"],
        "signature_label": sig_label or None,
        "metadata": meta,
    }
    # Keep nested severity in metadata for older readers
    meta.setdefault(
        "severity",
        {
            "severity": sev["severity_score"],
            "level": sev["severity_level"],
            "label": sev["severity_label"],
            "top_channel": sev["top_channel"],
        },
    )
    case["metadata"] = meta

    path = SAVED_CASES_DIR / f"{case_id}.json"
    write_json(path, case)
    case["path"] = str(path)

    if reindex:
        engine = rag or KnowledgeRAG()
        status = engine.rebuild_index()
        case["reindex_status"] = status
        logger.info("Saved case %s; %s", case_id, status)

    return case


def _normalize_severity(
    severity: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Coerce severity from explicit arg or metadata into flat fields."""
    src: Dict[str, Any] = {}
    if isinstance(severity, dict) and severity:
        src = severity
    elif isinstance(metadata.get("severity"), dict):
        src = metadata["severity"]
    score = src.get("severity", src.get("severity_score", 0.0))
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    return {
        "severity_score": score_f,
        "severity_level": str(src.get("level") or src.get("severity_level") or "unknown"),
        "severity_label": str(src.get("label") or src.get("severity_label") or "n/a"),
        "top_channel": src.get("top_channel"),
    }


def case_severity_score(case: Dict[str, Any]) -> float:
    """Read severity score from a saved case (first-class, nested, or recovered)."""
    if case.get("severity_score") is not None:
        try:
            val = float(case["severity_score"])
            # 0.0 may mean "missing" on legacy saves — try recovery if no label
            if val != 0.0 or case.get("severity_level") not in (None, "", "—", "unknown"):
                return val
        except (TypeError, ValueError):
            pass
    meta = case.get("metadata") or {}
    sev = meta.get("severity") if isinstance(meta, dict) else None
    if isinstance(sev, dict) and sev.get("severity") is not None:
        try:
            return float(sev["severity"])
        except (TypeError, ValueError):
            pass
    # Recover from text written by older offline reports / anomaly summaries
    recovered = _recover_severity_from_text(case)
    if recovered is not None:
        return recovered
    return 0.0


def _recover_severity_from_text(case: Dict[str, Any]) -> Optional[float]:
    """Parse score=… from final_report / anomaly_summary for legacy JSON cases."""
    import re

    blobs = [
        str(case.get("final_report") or ""),
        str(case.get("anomaly_summary") or ""),
        str(case.get("analysis") or ""),
        str(case.get("reflection") or ""),
    ]
    text = "\n".join(blobs)
    # Prefer explicit "score=20.9545" (severity engine)
    m = re.search(r"score\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fall back to largest channel=xx.xx in anomaly summary tops
    nums = [float(x) for x in re.findall(r"=\s*([0-9]+(?:\.[0-9]+)?)", text)]
    if nums:
        return round(max(nums), 4)
    return None


def load_saved_cases(limit: int = 50) -> List[Dict[str, Any]]:
    """Load saved cases sorted newest-first (highest seq first)."""
    cases: List[Dict[str, Any]] = []
    for path in list_saved_case_files():
        data = read_json(path)
        if isinstance(data, dict):
            data = dict(data)
            data["path"] = str(path)
            # Normalize score fields for older JSON without first-class keys
            if "severity_score" not in data:
                data["severity_score"] = case_severity_score(data)
            cases.append(data)
    # Newest first: reverse chronological key (seq primary — never random case_id)
    cases.sort(key=case_sort_key, reverse=True)
    return cases[:limit]


def score_trend(limit: int = 50) -> Dict[str, Any]:
    """
    Chronological severity trend for saved cases (oldest → newest by seq).

    Order is driven by monotonic ``seq``, not wall-clock ties, so two saves
    in the same second never invert.

    Returns
    -------
    dict with keys:
      dates, scores, levels, labels, case_ids, modes, seqs
    """
    # load_saved_cases is newest-first; reverse for chronological trend
    cases = list(reversed(load_saved_cases(limit=limit)))
    return {
        "dates": [c.get("saved_at", "") for c in cases],
        "scores": [case_severity_score(c) for c in cases],
        "levels": [
            c.get("severity_level")
            or ((c.get("metadata") or {}).get("severity") or {}).get("level", "")
            for c in cases
        ],
        "labels": [
            c.get("severity_label")
            or ((c.get("metadata") or {}).get("severity") or {}).get("label", "")
            for c in cases
        ],
        "case_ids": [c.get("case_id", "") for c in cases],
        "modes": [c.get("mode", "") for c in cases],
        "seqs": [c.get("seq") for c in cases],
    }


def format_score_trend_markdown(trend: Optional[Dict[str, Any]] = None, limit: int = 50) -> str:
    """Simple text/sparkline trend of severity scores over time."""
    t = trend if trend is not None else score_trend(limit=limit)
    scores = t.get("scores") or []
    if not scores:
        return "_No score trend yet — save at least one case._"

    # ASCII sparkline (deterministic, no plotting deps)
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(
        blocks[min(len(blocks) - 1, int((s - lo) / span * (len(blocks) - 1)))]
        for s in scores
    )
    first, last = scores[0], scores[-1]
    delta = last - first
    direction = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "→")

    lines = [
        "### Score trend (severity over time)",
        f"`{spark}`  {len(scores)} session(s) · first **{first:.3f}** → latest **{last:.3f}** ({direction} {delta:+.3f})",
        "",
        "| # | Saved (UTC) | Score | Level | Mode | Case |",
        "|---|---|---|---|---|---|",
    ]
    for i, score in enumerate(scores):
        lines.append(
            f"| {i + 1} | {t['dates'][i]} | **{score:.3f}** | "
            f"{t['levels'][i] or '—'} | {t['modes'][i] or '—'} | "
            f"`{t['case_ids'][i]}` |"
        )
    return "\n".join(lines)


def cases_history_markdown(limit: int = 20) -> str:
    """
    Render history of past cases **with severity scores** and a score trend.

    Newest sessions appear first in the table; trend is chronological.
    """
    cases = load_saved_cases(limit=limit)
    if not cases:
        return "_No saved cases yet. Run an analysis and click **Save & Learn**._"

    lines = [
        "### Session history",
        "",
        "| Case ID | Saved (UTC) | Mode | **Score** | Level | Label | Top channel | Corrections |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        corr = (c.get("user_corrections") or "").replace("\n", " ")
        corr = truncate(corr, 60)
        score = case_severity_score(c)
        level = c.get("severity_level") or "—"
        label = c.get("severity_label") or "—"
        top = c.get("top_channel") or "—"
        lines.append(
            f"| `{c.get('case_id', '')}` | {c.get('saved_at', '')} | "
            f"{c.get('mode', '')} | **{score:.3f}** | {level} | {label} | "
            f"{top} | {corr or '—'} |"
        )

    lines.append("")
    lines.append(format_score_trend_markdown(score_trend(limit=limit)))
    return "\n".join(lines)
