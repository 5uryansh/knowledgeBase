import re
from src.cooccurrence.edge_weighting import get_context_weight


class EdgeBuilder:

    def __init__(self, edge_store):
        self.edge_store = edge_store

    def _extract_links(self, text):
        matches = re.findall(r'\[\[([^\]|]+)', text)
        normalized = []

        for match in matches:
            normalized.append(match.strip().lower())

        return list(set(normalized))

    def _build_edges_from_nodes(self, nodes, context):

        weight = get_context_weight(context)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):

                source = nodes[i]
                target = nodes[j]

                self.edge_store.add_edge(source=source, target=target, weight=weight, context=context)

    def process_document(self, text):
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            nodes = self._extract_links(paragraph)

            if len(nodes) < 2:
                continue

            self._build_edges_from_nodes(nodes, context="paragraph")

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            nodes = self._extract_links(sentence)

            if len(nodes) < 2:
                continue

            self._build_edges_from_nodes(nodes, context="sentence")

        conversation_nodes = self._extract_links(text)

        if len(conversation_nodes) >= 2:

            self._build_edges_from_nodes(conversation_nodes,context="conversation")