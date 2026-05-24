from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class RelationMarkdownWriter:

    def __init__(self, vault_root):
        self.vault_root = Path(vault_root)
        self.entities_dir = self.vault_root / "entities"
        self.topics_dir = self.vault_root / "topics"

    def _find_node_path(self, node):
        entity_path = self.entities_dir / f"{node}.md"
        if entity_path.exists():
            return entity_path
        topic_path = self.topics_dir / f"{node}.md"
        if topic_path.exists():
            return topic_path
        return None

    def write_relations(self, relations):
        grouped = defaultdict(list)

        for relation in relations:
            grouped[relation["source"]].append(relation)

        for source, source_relations in tqdm(grouped.items(), total=len(grouped.items()), desc="Updating relations"):
            source_path = self._find_node_path(source)

            if not source_path:
                continue

            lines = []
            lines.append("\n## Relations\n")

            for relation in source_relations:
                target = relation["target"]
                relation_type = relation["relation"]
                lines.append(f"- {relation_type} → [[{target}]]")
            with open(source_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))