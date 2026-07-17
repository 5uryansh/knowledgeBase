from pathlib import Path

class MentionWriter:

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

    def write_mentions(self, markdown_file, topics, entities):
        conversation_name = markdown_file.stem
        all_nodes = topics + entities

        for item in all_nodes:
            node = item["node"]
            node_path = self._find_node_path(node)

            if not node_path:
                continue

            mention_line = (f"- [[conversations/{markdown_file.parent.name}/{conversation_name}]]")

            with open(node_path, "r", encoding="utf-8") as f:
                content = f.read()

            if mention_line in content:
                continue

            with open(node_path, "a", encoding="utf-8") as f:
                if "## Mentioned In" not in content:
                    f.write("\n\n## Mentioned In\n")
                f.write(f"{mention_line}\n")