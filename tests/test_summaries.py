from knowledgebase.summaries import Summarizer, load_summaries


class FakeLLM:
    name = "fake-llm"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        return "- You fixed e-paper ghosting with periodic full refreshes [ESP32 e-paper weather station, 2025-11-20]"


def test_summarizer_uses_evidence_caches_and_feeds_the_api(index, demo_vault):
    from fastapi.testclient import TestClient

    from knowledgebase.config import Settings
    from knowledgebase.pipeline import refresh_concept_notes
    from knowledgebase.server.app import create_app

    llm = FakeLLM()
    s = Summarizer(index, llm, demo_vault / "indexes")
    evidence = s.evidence("esp32")
    assert evidence and all(e["text"] and e["title"] for e in evidence)
    assert evidence == sorted(evidence, key=lambda e: e["date"], reverse=True)

    assert s.summarize(["esp32"]) == 1
    assert "ESP32" in llm.prompts[0] and "EXCERPTS" in llm.prompts[0]
    assert s.summarize(["esp32"]) == 0  # unchanged evidence -> cached, no new LLM call
    assert len(llm.prompts) == 1
    assert load_summaries(demo_vault / "indexes")["esp32"]["model"] == "fake-llm"

    refresh_concept_notes(Settings(vault_dir=demo_vault), index, only={"esp32"})
    assert "## Summary" in (demo_vault / "entities" / "esp32.md").read_text()

    detail = TestClient(create_app(demo_vault / "indexes", embedder="none")).get("/api/graph/node/esp32").json()
    assert detail["summary"]["summary"].startswith("- You fixed")
