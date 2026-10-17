"""Colour groups for Obsidian's graph view (only touches an existing .obsidian/graph.json)."""
from __future__ import annotations

import json
from pathlib import Path


class ObsidianGraphStyle:
    DEFAULT_GROUPS = [
        {"query": "path:entities", "hex": "4C8DFF"},
        {"query": "path:topics", "hex": "B57BEE"},
        {"query": 'path:"conversations/claude"', "hex": "E8833A"},
        {"query": 'path:"conversations/chatgpt"', "hex": "3FBF8F"},
        {"query": 'path:"conversations/gemini"', "hex": "E85D9E"},
        {"query": 'path:"conversations/claude-code"', "hex": "008300"},
    ]

    def __init__(self, vault_root: Path, groups: list[dict] | None = None):
        self.graph_config_path = Path(vault_root) / ".obsidian" / "graph.json"
        self.groups = groups or self.DEFAULT_GROUPS

    def apply(self) -> bool:
        if not self.graph_config_path.exists():
            return False
        config = json.loads(self.graph_config_path.read_text(encoding="utf-8"))
        existing = {g["query"]: g for g in config.get("colorGroups", [])}
        for group in self.groups:
            existing[group["query"]] = {"query": group["query"], "color": {"a": 1, "rgb": int(group["hex"], 16)}}
        config["colorGroups"] = list(existing.values())
        self.graph_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return True
