from collections import defaultdict
import json
from pathlib import Path

class GraphExpander:
    def __init__(self, graph_path: Path) -> None:
        with graph_path.open("r", encoding="utf-8") as f:
            edges = json.load(f)
        self._graph = defaultdict(list)
        for edge in edges:
            self._graph[edge["source"]].append(edge["target"])
            self._graph[edge["target"]].append(edge["source"])  # optional if undirected

    def expand(self, seed_nodes: list[str], max_neighbors: int = 5) -> set[str]:
        expanded = set(seed_nodes)
        for node in seed_nodes:
            expanded.update(self._graph.get(node, [])[:max_neighbors])

        return expanded