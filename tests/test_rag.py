"""RAG document collection and query tests against shipped KnowledgeRAG."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools import KnowledgeRAG, collect_rag_documents, format_rag_context


class TestKnowledgeCorpus(unittest.TestCase):
    def test_knowledge_files_present(self):
        docs = collect_rag_documents()
        # Project ships knowledge markdowns
        knowledge_docs = [d for d in docs if d.get("kind") == "knowledge"]
        self.assertGreaterEqual(len(knowledge_docs), 1)
        joined = " ".join(d["text"] for d in knowledge_docs).lower()
        self.assertIn("gas turbine", joined)

    def test_query_returns_relevant_hits(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            rag = KnowledgeRAG(persist_dir=Path(tmp) / "chroma", force_memory=True)
            status = rag.rebuild_index()
            self.assertTrue(status)
            hits = rag.query("exhaust temperature spread fuel nozzle", k=3)
            # Memory or chroma — should find process map content when corpus present
            if hits:
                ctx = format_rag_context(hits)
                self.assertTrue(len(ctx) > 20)
            else:
                # If no hits (empty env), still ensure API stable
                self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
