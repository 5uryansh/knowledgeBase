from collections import defaultdict
import json
from pathlib import Path
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.embedder import Embedder
from src.retrieval.entity_index import EntityIndex
from src.retrieval.graph_expander import GraphExpander
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

VAULT_ROOT = Path("/mnt/c/Users/Suryansh/Documents/KnowledgeBase")
INDEX_DIR = VAULT_ROOT / "indexes"

embedder = Embedder()

vector_store = VectorStore.load(INDEX_DIR / "embeddings.faiss", INDEX_DIR / "chunk_ids.txt",)
chunk_store = ChunkStore.load(INDEX_DIR / "chunks.json")

entity_index = EntityIndex.load(INDEX_DIR / "entity_index.json")

graph_expander = GraphExpander.__new__(GraphExpander)
with open(INDEX_DIR / "graph_edges.json", "r", encoding="utf-8") as f:
    graph_expander._graph = defaultdict(list, json.load(f))

retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    chunk_store=chunk_store,
    entity_index=entity_index,
    graph_expander=graph_expander,
)

query = "How did I implement Zero Sync?"
results = retriever.retrieve(query)

for i, chunk in enumerate(results, 1):
    print("=" * 100)
    print(f"Result {i}")
    print(f"Chunk ID : {chunk.id}")
    print(f"Source   : {chunk.source}")
    print(f"Links    : {chunk.links}")
    print("-" * 100)
    print(chunk.text)
    print()