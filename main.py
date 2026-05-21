from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher
from src.cooccurrence.graph_exporter import GraphExporter
from src.parser.claude import ClaudeChatParser
from src.cooccurrence.edge_store import EdgeStore

VAULT_ROOT = Path("C:/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase")

CONVERSATION_PATH_GEMINI = VAULT_ROOT / "conversations" / "gemini"
CONVERSATION_PATH_CLAUDE = VAULT_ROOT / "conversations" / "claude"

def main():
    global_edge_store = EdgeStore()
    gemini_parser = GeminiChatParser(
        input_file=Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        output_dir=CONVERSATION_PATH_GEMINI
    )
    gemini_parser.parse()
    gemini_enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH_GEMINI, edge_store=global_edge_store)
    gemini_enricher.enrich()

    claude_parser = ClaudeChatParser(
        input_file=Path("data/claude/conversations.json"),
        output_dir=CONVERSATION_PATH_CLAUDE
    )
    claude_parser.parse()
    claude_enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH_CLAUDE, edge_store=global_edge_store)
    claude_enricher.enrich()

    graph_exporter = GraphExporter(output_dir=VAULT_ROOT)
    graph_exporter.export_edges(global_edge_store.get_edges())


if __name__ == "__main__":
    main()