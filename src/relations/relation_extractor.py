import re
from src.relations.relation_patterns import RELATION_PATTERNS


class RelationExtractor:

    def __init__(self, relation_store):
        self.relation_store = relation_store

    def _normalize(self, text):
        text = text.strip().lower()
        text = re.sub(r"\s+", "_", text)
        return text

    def _extract_nodes(self, sentence):
        pattern = r'\[\[([^\|\]]+)(?:\|.*?)?\]\]'
        matches = re.findall(pattern, sentence)
        nodes = []
        for match in matches:
            nodes.append(self._normalize(match))
        return list(set(nodes))

    def process_document(self, text, context="conversation"):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            nodes = self._extract_nodes(sentence)

            if len(nodes) < 2:
                continue

            for relation, triggers in RELATION_PATTERNS.items():
                matched = any(
                    trigger in sentence_lower
                    for trigger in triggers
                )

                if not matched:
                    continue

                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        source = nodes[i]
                        target = nodes[j]
                        if source == target:
                            continue

                        self.relation_store.add_relation(source=source, relation=relation, target=target, context=context)