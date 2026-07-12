"""
GT Diagnostic Harness — local gas turbine diagnostic toolkit.

Modules
-------
models   : LLM (Granite 4.1 via llama.cpp) + TS Pulse anomaly models
tools    : CSV loaders, RAG (ChromaDB), case flywheel
analysis : Orchestration pipeline (anomaly → RAG → LLM → reflection)
utils    : Paths, logging helpers, text formatting
"""

__version__ = "1.0.0"
__app_name__ = "GT Diagnostic Harness"
