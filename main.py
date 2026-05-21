from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher
from src.cooccurrence.graph_exporter import GraphExporter

VAULT_ROOT = Path("C:/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase")

CONVERSATION_PATH_GEMINI = VAULT_ROOT / "conversations" / "gemini"

def main():

    parser = GeminiChatParser(
        input_file=Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        output_dir=CONVERSATION_PATH_GEMINI
    )
    parser.parse()

    enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH_GEMINI)
    enricher.enrich()

    graph_exporter = GraphExporter(output_dir=VAULT_ROOT.parent)
    graph_exporter.export_edges(enricher.get_edges())

if __name__ == "__main__":
    main()