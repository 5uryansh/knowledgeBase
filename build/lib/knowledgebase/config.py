"""Settings for the whole pipeline.

Resolution order (later wins):
  1. built-in defaults
  2. ``kb.yaml`` in the current directory (or the file passed with ``--config``)
  3. ``KB_*`` environment variables (e.g. ``KB_VAULT_DIR``, ``KB_EMBEDDER``)
  4. explicit CLI flags
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPORTS = {
    "chatgpt": "data/chatgpt/conversations.json",
    "claude": "data/claude/conversations.json",
    "gemini": "data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json",
    "claude-code": "data/claude-code/projects",
}


@dataclass
class Settings:
    vault_dir: Path = Path("vault")
    exports: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EXPORTS))
    # "auto" -> sentence-transformers if installed, else wordllama, else tfidf
    embedder: str = "auto"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # "auto" -> gliner if installed and the model is available, else spacy
    entity_extractor: str = "auto"
    gliner_model: str = "urchade/gliner_small-v2.1"
    spacy_model: str = "en_core_web_sm"
    host: str = "127.0.0.1"
    port: int = 3001
    max_chunk_chars: int = 2400
    stop_words_file: Path | None = None
    aliases_file: Path | None = None
    entity_labels_file: Path | None = None

    @property
    def index_dir(self) -> Path:
        return self.vault_dir / "indexes"

    @property
    def conversations_dir(self) -> Path:
        return self.vault_dir / "conversations"


def _coerce(name: str, value: Any) -> Any:
    if name in {"vault_dir", "stop_words_file", "aliases_file", "entity_labels_file"} and value is not None:
        return Path(value).expanduser()
    if name in {"port", "max_chunk_chars"}:
        return int(value)
    return value


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    settings = Settings()
    path = Path(config_path) if config_path else Path("kb.yaml")
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for f in fields(Settings):
            if f.name in data:
                value = data[f.name]
                if f.name == "exports":
                    value = {**DEFAULT_EXPORTS, **(value or {})}
                setattr(settings, f.name, _coerce(f.name, value))
    for f in fields(Settings):
        env = os.environ.get(f"KB_{f.name.upper()}")
        if env is not None and f.name != "exports":
            setattr(settings, f.name, _coerce(f.name, env))
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, _coerce(key, value))
    return settings


def _resource_text(name: str) -> str:
    return resources.files("knowledgebase.resources").joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def _load_yaml(path: str | None, default_name: str) -> Any:
    text = Path(path).read_text(encoding="utf-8") if path else _resource_text(default_name)
    return yaml.safe_load(text) or {}


def stop_words(settings: Settings | None = None) -> frozenset[str]:
    path = str(settings.stop_words_file) if settings and settings.stop_words_file else None
    return frozenset(str(w).lower() for w in _load_yaml(path, "stop_words.yaml").get("words", []))


def entity_labels(settings: Settings | None = None) -> list[str]:
    path = str(settings.entity_labels_file) if settings and settings.entity_labels_file else None
    return list(_load_yaml(path, "entity_labels.yaml").get("labels", []))


def aliases(settings: Settings | None = None) -> dict[str, str]:
    """Map of surface form (lowercase) -> canonical display name."""
    path = str(settings.aliases_file) if settings and settings.aliases_file else None
    data = _load_yaml(path, "aliases.yaml").get("aliases", {})
    mapping: dict[str, str] = {}
    for canonical, variants in data.items():
        mapping[canonical.lower()] = canonical
        for variant in variants or []:
            mapping[str(variant).lower()] = canonical
    return mapping
