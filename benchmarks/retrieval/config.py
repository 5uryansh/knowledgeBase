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
GEMINI_MODEL = "gemini-3.5-flash"

# --- Retrieval params ------------------------------------------------------
TOP_K = 10
CANDIDATE_K = 50

# --- Judge -----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JUDGE_PASSAGE_CHARS = 700       # truncate each passage shown to the judge
JUDGE_DELAY_SECONDS = 4.0       # throttle between calls (free-tier RPM limits)
JUDGE_MAX_RETRIES = 5
