from knowledgebase.enrichment.vault import END, START


def test_vault_layout(demo_vault):
    convs = list((demo_vault / "conversations").rglob("*.md"))
    assert len(convs) == 36
    assert {p.parent.name for p in convs} == {"chatgpt", "claude", "gemini", "claude-code"}
    assert (demo_vault / "entities").exists() and (demo_vault / "topics").exists()
    text = convs[0].read_text()
    assert text.startswith("---\nplatform:")
    assert "## Prompt" in text and "[[" in text


def test_concept_notes_have_generated_section(demo_vault):
    note = next((demo_vault / "entities").glob("*.md"))
    text = note.read_text()
    assert START in text and END in text and "## Mentioned In" in text


def test_rebuild_is_idempotent_and_keeps_user_notes(demo_vault, nlp):
    from conftest import sample_exports

    from knowledgebase.config import Settings
    from knowledgebase.enrichment.concepts import ConceptExtractor
    from knowledgebase.pipeline import build

    note = demo_vault / "entities" / "esp32.md"
    assert note.exists()
    note.write_text(note.read_text() + "\nMy own note about the ESP32.\n")
    before = sorted(p.name for p in (demo_vault / "conversations").rglob("*.md"))

    settings = Settings(vault_dir=demo_vault, entity_extractor="spacy", embedder="none")
    settings.exports = sample_exports()
    build(settings, embedder=None, extractor=ConceptExtractor(settings, nlp=nlp), show_progress=False)

    after = sorted(p.name for p in (demo_vault / "conversations").rglob("*.md"))
    assert before == after  # no duplicates on re-import
    text = note.read_text()
    assert "My own note about the ESP32." in text
    assert text.count(START) == 1


def test_graph_has_cross_platform_concepts(index):
    g = index.graph
    esp32 = g.nodes["esp32"]
    assert esp32["conversation_count"] >= 3
    assert len(esp32["platforms"]) >= 2
    assert g.communities and g.discoveries
    assert any(r["relation"] for r in g.relations)
