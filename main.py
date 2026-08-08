from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher
from src.utils.memory_guard import check_memory_or_exit
from src.utils.graph_style_writer import GraphStyleWriter
from src.cooccurrence.graph_exporter import GraphExporter
from src.parser.claude import ClaudeChatParser
from src.parser.claude_code import ClaudeCodeChatParser
from src.cooccurrence.edge_store import EdgeStore
from src.relations.relation_store import RelationStore
from src.relations.relation_exporter import RelationExporter
from src.relations.relation_markdown_writer import RelationMarkdownWriter
from src.parser.chatgpt import ChatGPTChatParser
from src.cooccurrence.edge_builder import EdgeBuilder
from src.relations.relation_extractor import RelationExtractor
from src.retrieval.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.indexer import Indexer
from src.utils.text_utils import find_markdown_files

VAULT_ROOT = Path("/mnt/c/Users/Suryansh/Documents/KnowledgeBase")
INDEX_DIR = VAULT_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CONVERSATION_PATH_GEMINI = VAULT_ROOT / "conversations" / "gemini"
CONVERSATION_PATH_CLAUDE = VAULT_ROOT / "conversations" / "claude"
CONVERSATION_PATH_CHATGPT = VAULT_ROOT / "conversations" / "chatgpt"
CONVERSATION_PATH_CLAUDECODE = VAULT_ROOT / "conversations" / "claude-code"


def process_source(parser_cls, input_file: Path, output_dir: Path, name: str, edge_store: EdgeStore, relation_store: RelationStore):
    if not input_file.exists():
        print(f"Skipping {name}: input file not found.")
        return
    print(f"Processing {name}...")

    parser = parser_cls(input_file=input_file, output_dir=output_dir)
    parser.parse()
    check_memory_or_exit(context=f"after parsing {name}")

    enricher = ObsidianEnricher(vault_dir=output_dir, edge_store=edge_store, relation_store=relation_store)
    enricher.enrich()
    check_memory_or_exit(context=f"after enriching {name}")


def main():
    global_edge_store = EdgeStore()
    global_relation_store = RelationStore()

    process_source(
        GeminiChatParser,
        Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        CONVERSATION_PATH_GEMINI,
        "Gemini",
        global_edge_store,
        global_relation_store,
    )

    process_source(
        ClaudeChatParser,
        Path("data/claude/conversations.json"),
        CONVERSATION_PATH_CLAUDE,
        "Claude",
        global_edge_store,
        global_relation_store,
    )

    process_source(
        ChatGPTChatParser,
        Path("data/chatgpt/conversations.json"),
        CONVERSATION_PATH_CHATGPT,
        "ChatGPT",
        global_edge_store,
        global_relation_store,
    )

    process_source(
        ClaudeCodeChatParser,
        Path("data/claude-code/projects"),
        CONVERSATION_PATH_CLAUDECODE,
        "Claude Code",
        global_edge_store,
        global_relation_store,
    )

    edge_builder = EdgeBuilder(global_edge_store)
    relation_extractor = RelationExtractor(global_relation_store)

    for conv_dir in [CONVERSATION_PATH_CHATGPT, CONVERSATION_PATH_GEMINI, CONVERSATION_PATH_CLAUDE, CONVERSATION_PATH_CLAUDECODE]:
        if conv_dir.exists():
            for md_file in find_markdown_files(conv_dir):
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                edge_builder.process_document(content)
                relation_extractor.process_document(content)

    graph_exporter = GraphExporter(output_dir=INDEX_DIR)
    graph_exporter.export_edges(global_edge_store.get_edges())

    relation_exporter = RelationExporter(output_dir=INDEX_DIR)
    relation_exporter.export_relations(global_relation_store.get_relations())

    relation_writer = RelationMarkdownWriter(vault_root=VAULT_ROOT)
    relation_writer.write_relations(global_relation_store.get_relations())

    graph_style_writer = GraphStyleWriter(vault_root=VAULT_ROOT)
    graph_style_writer.apply()

    chunker = Chunker()
    embedder = Embedder()
    indexer = Indexer(chunker=chunker, embedder=embedder)

    markdown_files = find_markdown_files(CONVERSATION_PATH_CHATGPT)
    markdown_files += find_markdown_files(CONVERSATION_PATH_GEMINI)
    markdown_files += find_markdown_files(CONVERSATION_PATH_CLAUDE)
    markdown_files += find_markdown_files(CONVERSATION_PATH_CLAUDECODE)

    indexer.build(markdown_files=markdown_files, output_dir=INDEX_DIR)


if __name__ == "__main__":
    main()