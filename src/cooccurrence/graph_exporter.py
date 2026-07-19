import json
from pathlib import Path
from collections import defaultdict

class GraphExporter:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def export_edges(self, edges):
        graph = defaultdict(list)
        for edge in edges:
            graph[edge.source].append(edge.target)
            graph[edge.target].append(edge.source) 
        with (self.output_dir / "graph_edges.json").open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)