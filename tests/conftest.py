from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "sample_exports"
sys.path.insert(0, str(ROOT))  # for `benchmarks`


def sample_exports() -> dict[str, str]:
    return {
        "chatgpt": str(SAMPLES / "chatgpt" / "conversations.json"),
        "claude": str(SAMPLES / "claude" / "conversations.json"),
        "gemini": str(SAMPLES / "gemini" / "MyActivity.json"),
        "claude-code": str(SAMPLES / "claude-code" / "projects"),
    }


@pytest.fixture(scope="session")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")


@pytest.fixture(scope="session")
def demo_vault(tmp_path_factory, nlp) -> Path:
    """Build the bundled sample knowledge base once for the whole test session (no dense model)."""
    from knowledgebase.config import Settings
    from knowledgebase.enrichment.concepts import ConceptExtractor
    from knowledgebase.pipeline import build

    vault = tmp_path_factory.mktemp("vault")
    settings = Settings(vault_dir=vault, entity_extractor="spacy", embedder="none")
    settings.exports = sample_exports()
    build(settings, embedder=None, extractor=ConceptExtractor(settings, nlp=nlp), show_progress=False)
    return vault


@pytest.fixture(scope="session")
def index(demo_vault):
    from knowledgebase.retrieval.index import KnowledgeIndex

    return KnowledgeIndex.load(demo_vault / "indexes", embedder=None)
