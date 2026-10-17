"""Exports uploaded through the app, and the output folder chosen there.

An upload is a zip (or a bare ``conversations.json`` / ``MyActivity.json``) plus the
:class:`~knowledgebase.parsers.platform.Platform` the user says it came from. We

1. look inside the zip for the files that platform's parser needs (nothing else is
   extracted, so photos and audio in a ChatGPT export never touch the disk),
2. extract those safely (no path traversal, size-capped against zip bombs),
3. run the real parser over them, and only accept the import if it yields
   conversations, otherwise explaining why (wrong platform picked, Takeout exported
   as HTML, empty export, ...).

Accepted imports live under ``<app_dir>/imports/<id>/`` and are listed in
``imports.json``. :func:`knowledgebase.pipeline.parse_exports` reads them alongside
the ``exports:`` configured in kb.yaml, so ``kb build`` and the app's Build button
index the same thing.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from knowledgebase.config import Settings, save_app_settings
from knowledgebase.parsers.base import Conversation
from knowledgebase.parsers.platform import Platform, sniff_json

MAX_EXTRACTED_BYTES = int(os.environ.get("KB_MAX_IMPORT_BYTES", 8 * 1024 ** 3))  # 8 GB of chat text is plenty
SAMPLE_TITLES = 5
VAULT_ENTRIES = {"conversations", "entities", "topics", "indexes", ".obsidian"}  # what a build writes


class ImportRejected(ValueError):
    """The upload can't be indexed; ``str(exc)`` says why in words a user can act on."""

    def __init__(self, message: str, detected: Platform | None = None):
        super().__init__(message)
        self.detected = detected


@dataclass
class ImportRecord:
    id: str
    platform: str
    filename: str
    files: list[str]                 # relative to the import's folder ("." = the folder itself)
    conversations: int
    messages: int
    date_range: list[str | None]
    imported_at: str
    sample_titles: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return Platform.parse(self.platform).label

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}


class Workspace:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.app_dir = Path(settings.app_dir).expanduser()
        self.imports_dir = self.app_dir / "imports"
        self.registry_path = self.app_dir / "imports.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ registry
    def imports(self) -> list[ImportRecord]:
        try:
            rows = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for row in rows if isinstance(rows, list) else []:
            try:
                out.append(ImportRecord(**{k: row[k] for k in ImportRecord.__dataclass_fields__ if k in row}))
            except TypeError:
                continue
        return out

    def _save(self, records: list[ImportRecord]) -> None:
        self.app_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(r) for r in records], indent=1), encoding="utf-8")
        tmp.replace(self.registry_path)

    def remove(self, import_id: str) -> bool:
        with self._lock:
            records = self.imports()
            keep = [r for r in records if r.id != import_id]
            if len(keep) == len(records):
                return False
            self._save(keep)
        shutil.rmtree(self.imports_dir / import_id, ignore_errors=True)
        return True

    def import_paths(self) -> Iterator[tuple[Platform, Path]]:
        """(platform, path-for-its-parser) for every accepted import, oldest first, so newer exports win."""
        for rec in sorted(self.imports(), key=lambda r: r.imported_at):
            base = self.imports_dir / rec.id
            for rel in rec.files:
                yield Platform.parse(rec.platform), (base / rel)

    # ------------------------------------------------------------------ importing
    def import_file(self, upload: Path, platform: Platform | str, filename: str = "",
                    replace: bool = False) -> ImportRecord:
        """Validate an uploaded export and keep it if it parses; raises :class:`ImportRejected` otherwise."""
        platform = Platform.parse(platform)
        filename = Path(filename or upload.name).name
        import_id = uuid.uuid4().hex[:12]
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{import_id}-", dir=self.imports_dir))
        try:
            files = self._unpack(upload, platform, filename, staging)
            convs = self._parse(platform, staging, files)
            dates = sorted(c.date for c in convs if c.created_at)
            record = ImportRecord(
                id=import_id, platform=platform.value, filename=filename, files=files,
                conversations=len(convs), messages=sum(len(c.messages) for c in convs),
                date_range=[dates[0] if dates else None, dates[-1] if dates else None],
                imported_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                sample_titles=[c.title for c in sorted(convs, key=lambda c: c.date, reverse=True)[:SAMPLE_TITLES]],
            )
            staging.rename(self.imports_dir / import_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        with self._lock:
            records = self.imports()
            replaced = [r for r in records if replace and r.platform == platform.value]
            self._save([r for r in records if r not in replaced] + [record])
        for r in replaced:
            shutil.rmtree(self.imports_dir / r.id, ignore_errors=True)
        return record

    def _unpack(self, upload: Path, platform: Platform, filename: str, dest: Path) -> list[str]:
        if not zipfile.is_zipfile(upload):
            return self._single_file(upload, platform, filename, dest)
        with zipfile.ZipFile(upload) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            wanted = [m for m in members if _safe_parts(m.filename) and platform.wants(m.filename)]
            if not wanted:
                raise ImportRejected(*self._explain_missing(zf, members, platform))
            budget = MAX_EXTRACTED_BYTES
            for m in wanted:
                target = dest.joinpath(*_safe_parts(m.filename))
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, target.open("wb") as out:
                    while chunk := src.read(1 << 20):
                        budget -= len(chunk)
                        if budget < 0:
                            raise ImportRejected("This export unpacks to more than "
                                                 f"{MAX_EXTRACTED_BYTES // 1024 ** 3} GB of chat files; "
                                                 "that is more than one import can take.")
                        out.write(chunk)
        if platform.reads_directory:
            return ["."]
        return sorted(str(PurePosixPath(*_safe_parts(m.filename))) for m in wanted)

    def _single_file(self, upload: Path, platform: Platform, filename: str, dest: Path) -> list[str]:
        if platform.reads_directory:
            if not filename.endswith(".jsonl"):
                raise ImportRejected("Claude Code history is a folder of .jsonl files: zip your "
                                     "~/.claude/projects folder and upload the zip.")
            name = filename
        else:
            name = "MyActivity.json" if platform is Platform.GEMINI else "conversations.json"
            if not filename.lower().endswith(".json"):
                raise ImportRejected(f"That isn't a zip or a .json file. {platform.export_hint}")
        shutil.copyfile(upload, dest / name)
        return ["."] if platform.reads_directory else [name]

    def _explain_missing(self, zf: zipfile.ZipFile, members: list[zipfile.ZipInfo],
                         platform: Platform) -> tuple[str, Platform | None]:
        names = [m.filename for m in members]
        if platform is Platform.GEMINI and any(n.endswith("MyActivity.html") for n in names):
            return ("Your Takeout exported My Activity as HTML. Export again with the format set to JSON "
                    "(Takeout → My Activity → Multiple formats → Activity records: JSON).", None)
        other = _guess_from_names(members) or self._sniff_zip(zf, members)
        if other and other is not platform:
            return (f"This looks like a {other.label} export, not {platform.label}. "
                    f"Pick {other.label} and upload it again.", other)
        return f"Couldn't find {platform.label} conversations in this zip. {platform.export_hint}", None

    @staticmethod
    def _sniff_zip(zf: zipfile.ZipFile, members: list[zipfile.ZipInfo]) -> Platform | None:
        for m in members:
            name = PurePosixPath(m.filename).name
            if name.endswith(".json") and m.file_size < 512 * 1024 ** 2 and _safe_parts(m.filename):
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / name
                    with zf.open(m) as src, p.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    found = sniff_json(p)
                if found:
                    return found
        return None

    def _parse(self, platform: Platform, base: Path, files: list[str]) -> list[Conversation]:
        convs: dict[str, Conversation] = {}
        for rel in files:
            path = base / rel
            detected = None if platform.reads_directory else sniff_json(path)
            if detected and detected is not platform:
                raise ImportRejected(f"This looks like a {detected.label} export, not {platform.label}. "
                                     f"Pick {detected.label} and upload it again.", detected)
            try:
                for conv in platform.parser(path).parse():
                    convs[conv.uid] = conv
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError) as exc:
                raise ImportRejected(f"{Path(rel).name} isn't a readable {platform.label} export "
                                     f"({exc.__class__.__name__}). {platform.export_hint}", detected) from exc
        if not convs:
            raise ImportRejected(f"The {platform.label} export opened fine but has no conversations with text in it.")
        return list(convs.values())

    # ------------------------------------------------------------------ output folder
    def set_output_dir(self, raw: str) -> dict:
        """Point the vault (markdown notes + graph/index files) somewhere else and remember the choice."""
        if not raw or not str(raw).strip():
            raise ValueError("Choose a folder.")
        path = Path(str(raw).strip()).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if path.exists() and not path.is_dir():
            raise ValueError(f"{path} is a file, not a folder.")
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".kb-write-test-{uuid.uuid4().hex[:6]}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise ValueError(f"Can't write to {path}: {exc.strerror or exc}") from exc
        save_app_settings(self.app_dir, vault_dir=path)
        self.settings.vault_dir = path
        return self.output_info()

    def output_info(self) -> dict:
        vault = Path(self.settings.vault_dir).expanduser()
        path = vault if vault.is_absolute() else Path.cwd() / vault
        has_index = (path / "indexes" / "manifest.json").exists()
        foreign = path.exists() and not has_index and any(
            p.name not in VAULT_ENTRIES for p in path.iterdir())
        return {
            "path": str(path.resolve() if path.exists() else path),
            "exists": path.exists(),
            "has_index": has_index,
            "note": ("This folder already has other files in it. knowledgeBase only adds its own "
                     "conversations/, entities/, topics/ and indexes/ folders next to them.") if foreign else None,
        }


def _safe_parts(name: str) -> tuple[str, ...]:
    """Path parts of a zip member, or () if it tries to escape the destination."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or any(part in ("..", "") for part in p.parts) or ":" in (p.parts[0] if p.parts else ""):
        return ()
    return tuple(p.parts)


def _guess_from_names(members: list[zipfile.ZipInfo]) -> Platform | None:
    names = [m.filename for m in members]
    if any(n.endswith("MyActivity.json") and "Gemini" in n for n in names):
        return Platform.GEMINI
    if sum(n.endswith(".jsonl") for n in names) and not any(n.endswith(".json") for n in names):
        return Platform.CLAUDE_CODE
    return None


def platforms_catalog() -> list[dict]:
    return [{"id": p.value, "label": p.label, "hint": p.export_hint} for p in Platform]
