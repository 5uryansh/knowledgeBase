import json
from pathlib import Path


class GraphStyleWriter:

    DEFAULT_GROUPS = [
        {"query": "path:entities", "hex": "4C8DFF"},
        {"query": "path:topics", "hex": "B57BEE"},
        {"query": 'path:"conversations/claude"', "hex": "E8833A"},
        {"query": 'path:"conversations/chatgpt"', "hex": "3FBF8F"},
        {"query": 'path:"conversations/gemini"', "hex": "E85D9E"},
    ]

    def __init__(self, vault_root, groups=None):
        self.vault_root = Path(vault_root)
        self.graph_config_path = self.vault_root / ".obsidian" / "graph.json"
        self.groups = groups or self.DEFAULT_GROUPS

    def _hex_to_rgb_int(self, hex_color):
        return int(hex_color.lstrip("#"), 16)

    def apply(self):
        if not self.graph_config_path.exists():
            print(f"Skipping graph styling: {self.graph_config_path} not found.")
            return

        with open(self.graph_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        existing = {group["query"]: group for group in config.get("colorGroups", [])}

        for group in self.groups:
            existing[group["query"]] = {
                "query": group["query"],
                "color": {"a": 1, "rgb": self._hex_to_rgb_int(group["hex"])},
            }

        config["colorGroups"] = list(existing.values())

        with open(self.graph_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Updated graph color groups: {self.graph_config_path}")
