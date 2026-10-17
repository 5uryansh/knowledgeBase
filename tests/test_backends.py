"""The transformer backends can't download models in CI; check our wrappers against fakes."""
import sys
import types

import numpy as np


def test_sentence_transformer_wrapper_normalises_and_prefixes_queries(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, name):
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            calls.append(texts)
            if isinstance(texts, str):
                return np.array([3.0, 4.0, 0.0])
            return np.array([[1.0, 1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    from knowledgebase.retrieval.embedders import build_embedder

    emb = build_embedder("sentence-transformers", "BAAI/bge-small-en-v1.5")
    assert emb.name == "st:BAAI/bge-small-en-v1.5" and emb.dim == 3
    q = emb.embed_query("where did I fix ghosting")
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert calls[-1].startswith("Represent this sentence for searching relevant passages:")
    m = emb.embed(["a", "b"])
    assert m.shape == (2, 3) and np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_auto_embedder_falls_back_when_transformers_are_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    from knowledgebase.retrieval.embedders import build_embedder

    emb = build_embedder("auto")
    assert emb is None or emb.name.startswith("wordllama")


def test_gliner_wrapper(monkeypatch):
    class FakeGLiNER:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            return cls()

        def predict_entities(self, text, labels):
            return [{"text": "FAISS", "label": "technology"}] if "FAISS" in text else []

    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=FakeGLiNER))
    from knowledgebase.enrichment.entities import GlinerEntityExtractor, build_entity_extractor

    ex = build_entity_extractor("auto", nlp=None, gliner_model="x", labels=["technology"])
    assert isinstance(ex, GlinerEntityExtractor)
    hits = ex.extract(["We indexed the chunks with FAISS. " * 3])
    assert ("FAISS", "technology", 2.0) in hits


def test_claude_code_parser_skips_sidechains(tmp_path):
    import json

    from knowledgebase.parsers import ClaudeCodeChatParser

    proj = tmp_path / "proj"
    proj.mkdir()
    lines = [
        {"type": "user", "message": {"role": "user", "content": "Fix the build"}, "timestamp": "2026-01-01T10:00:00Z"},
        {"type": "assistant", "isSidechain": True, "message": {"content": [{"type": "text", "text": "agent noise"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Fixed it."}]}, "timestamp": "2026-01-01T10:01:00Z"},
    ]
    (proj / "s1.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    (conv,) = ClaudeCodeChatParser(tmp_path).parse()
    assert [m.text for m in conv.messages] == ["Fix the build", "Fixed it."]
