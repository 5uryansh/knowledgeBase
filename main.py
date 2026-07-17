from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher
from src.cooccurrence.graph_exporter import GraphExporter
from src.parser.claude import ClaudeChatParser
from src.cooccurrence.edge_store import EdgeStore
from src.relations.relation_store import RelationStore
from src.relations.relation_exporter import RelationExporter
from src.relations.relation_markdown_writer import RelationMarkdownWriter

VAULT_ROOT = Path("C:/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase")

CONVERSATION_PATH_GEMINI = VAULT_ROOT / "conversations" / "gemini"
CONVERSATION_PATH_CLAUDE = VAULT_ROOT / "conversations" / "claude"

def main():
    global_edge_store = EdgeStore()
    global_relation_store = RelationStore()
    gemini_parser = GeminiChatParser(
        input_file=Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        output_dir=CONVERSATION_PATH_GEMINI
    )
    gemini_parser.parse()
    gemini_enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH_GEMINI, edge_store=global_edge_store, relation_store=global_relation_store)
    gemini_enricher.enrich()

    claude_parser = ClaudeChatParser(
        input_file=Path("data/claude/conversations.json"),
        output_dir=CONVERSATION_PATH_CLAUDE
    )
    claude_parser.parse()
    claude_enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH_CLAUDE, edge_store=global_edge_store, relation_store=global_relation_store)
    claude_enricher.enrich()

    graph_exporter = GraphExporter(output_dir=VAULT_ROOT)
    graph_exporter.export_edges(global_edge_store.get_edges())

    relation_exporter = RelationExporter(output_dir=VAULT_ROOT)
    relation_exporter.export_relations(global_relation_store.get_relations())

    relation_writer = RelationMarkdownWriter(vault_root=VAULT_ROOT)
    relation_writer.write_relations(global_relation_store.get_relations())


if __name__ == "__main__":
    main()