from collections import defaultdict


class EdgeStore:

    def __init__(self):
        self.edges = defaultdict(
            lambda: {
                "weight": 0,
                "contexts": set()
            })

    def add_edge(self, source, target, weight=1, context="unknown"):
        if source == target:
            return
        edge = tuple(sorted([source, target]))
        self.edges[edge]["weight"] += weight
        self.edges[edge]["contexts"].add(context)

    def get_edges(self):
        formatted_edges = []
        for (source, target), data in self.edges.items():
            formatted_edges.append({
                "source": source,
                "target": target,
                "weight": data["weight"],
                "contexts": sorted(
                    list(data["contexts"])
                )
            })
        return formatted_edges