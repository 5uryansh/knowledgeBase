"""Shared configuration for the retrieval comparison benchmark.

Importing this module also puts the project root on sys.path (so `from src...`
resolves) and loads the project-root .env (for GEMINI_API_KEY).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# --- Index locations -------------------------------------------------------
# The hybrid index is what main.py builds (bge-small embeddings + graph).
VAULT_ROOT = Path("/mnt/c/Users/Suryansh/Documents/KnowledgeBase")
INDEX_DIR = VAULT_ROOT / "indexes"
BASELINE_INDEX_DIR = VAULT_ROOT / "indexes_baseline"

# --- Fixtures / outputs ----------------------------------------------------
QUESTIONS_PATH = PROJECT_ROOT / "benchmarks" / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# --- Models ----------------------------------------------------------------
HYBRID_EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # what main.py's Embedder uses
BASELINE_EMBED_MODEL = "BAAI/bge-large-en-v1.5"    # the "just use a bigger model" baseline

# Judge model pool, ordered by preference. The judge sticks to one model until
# it hits a limit, then rotates to the next — so free-tier daily quotas (RPD)
# don't block the run. Lite models lead because they have far higher RPD (~500)
# than the full Flash models (~20). Reorder/trim to taste; wrong ids self-heal
# (they just error and rotation moves on).
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",   # ~500 RPD
    "gemini-3.1-flash-lite",   # ~500 RPD
    "gemini-3.6-flash",        # ~20 RPD (fallback)
    "gemini-3-flash",          # ~20 RPD (fallback)
    "gemini-2.5-flash",        # ~20 RPD (fallback)
]

# --- Retrieval params ------------------------------------------------------
TOP_K = 10
CANDIDATE_K = 50

# --- Judge -----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JUDGE_PASSAGE_CHARS = 700          # truncate each passage shown to the judge
JUDGE_DELAY_SECONDS = 4.0          # throttle between calls (free-tier RPM ~15/min)
JUDGE_RETRIES_PER_MODEL = 2        # brief retries on a model before rotating to the next
