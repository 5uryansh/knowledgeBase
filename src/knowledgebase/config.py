"""Settings for the whole pipeline.

Resolution order (later wins):
  1. built-in defaults
  2. ``kb.yaml`` in the current directory (or the file passed with ``--config``)
  3. choices saved from the app's settings (``<app_dir>/settings.json``, e.g. the output folder)
  4. ``KB_*`` environment variables (e.g. ``KB_VAULT_DIR``, ``KB_EMBEDDER``)
  5. explicit CLI flags
"""
from __future__ import annotations

import json
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
    # App state: uploaded export zips (unpacked) and settings chosen in the UI.
    app_dir: Path = Path("data/app")
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
    if name in {"vault_dir", "app_dir", "stop_words_file", "aliases_file", "entity_labels_file"} and value is not None:
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
    app_dir = os.environ.get("KB_APP_DIR") or overrides.get("app_dir") or settings.app_dir
    for key, value in saved_app_settings(Path(app_dir).expanduser()).items():
        setattr(settings, key, _coerce(key, value))
    for f in fields(Settings):
        env = os.environ.get(f"KB_{f.name.upper()}")
        if env is not None and f.name != "exports":
            setattr(settings, f.name, _coerce(f.name, env))
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, _coerce(key, value))
    return settings


# Settings the app's UI may change and persist (anything else stays in kb.yaml).
APP_SETTABLE = ("vault_dir",)


def app_settings_path(app_dir: Path) -> Path:
    return Path(app_dir) / "settings.json"


def saved_app_settings(app_dir: Path) -> dict[str, Any]:
    path = app_settings_path(app_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k in APP_SETTABLE and v is not None} if isinstance(data, dict) else {}


def save_app_settings(app_dir: Path, **values: Any) -> dict[str, Any]:
    data = {**saved_app_settings(app_dir), **{k: str(v) for k, v in values.items() if k in APP_SETTABLE}}
    path = app_settings_path(app_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return data


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
