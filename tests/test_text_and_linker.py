from knowledgebase.enrichment.concepts import Concept
from knowledgebase.enrichment.linker import Linker
from knowledgebase.text import normalize_node_name, strip_wikilinks, tokenize


def concept(node, display, surfaces=(), score=1.0):
    return Concept(node=node, display=display, kind="entity", label="technology", score=score, surfaces=set(surfaces))


def test_normalize_node_name():
    assert normalize_node_name("Personalized PageRank") == "personalized_pagerank"
    assert normalize_node_name("  Node.js ") == "nodejs"
    assert normalize_node_name("C++") == "cplusplus"


def test_tokenize_stems_plurals_and_drops_stopwords():
    assert tokenize("The kernels are fused") == ["kernel", "fus"]


def test_linker_links_longest_surface_first_and_never_nests():
    linker = Linker([concept("cuda", "CUDA"), concept("cuda_kernel", "CUDA kernel", {"CUDA kernels"})])
    out = linker.link("Fused CUDA kernels beat CUDA defaults.")
    assert "[[cuda_kernel|CUDA kernels]]" in out
    assert "[[cuda|CUDA]] defaults" in out
    assert "[[[[" not in out
    # Idempotent: linking already-linked text changes nothing.
    assert linker.link(out) == out


def test_linker_skips_code_and_urls():
    linker = Linker([concept("faiss", "FAISS")])
    text = "Use FAISS.\n```python\nimport faiss  # FAISS\n```\nSee `faiss.IndexFlatIP` and https://faiss.ai/FAISS"
    out = linker.link(text)
    assert out.count("[[faiss|") == 1


def test_linker_short_surfaces_are_case_sensitive():
    linker = Linker([concept("ai", "AI")])
    assert linker.link("AI is everywhere") == "[[ai|AI]] is everywhere"
    assert linker.link("an ai in lowercase") == "an ai in lowercase"


def test_linker_matches_simple_plural():
    linker = Linker([concept("support_group", "support group")])
    assert "[[support_group|support groups]]" in linker.link("I joined two support groups.")


def test_strip_wikilinks():
    assert strip_wikilinks("[[cuda|CUDA]] and [[faiss]]") == "CUDA and faiss"
