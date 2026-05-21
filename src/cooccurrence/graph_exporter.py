import json
from pathlib import Path


class GraphExporter:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_edges(self, edges, filename="graph_edges.json"):
        output_path = (self.output_dir / filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2, ensure_ascii=False)
        print(f"Exported graph: {output_path}")