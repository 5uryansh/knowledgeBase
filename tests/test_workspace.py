"""Uploading exports through the app: platform enum, zip validation, output folder, background builds."""
import json
import time
import zipfile

import pytest
from conftest import SAMPLES

from knowledgebase.config import Settings, load_settings
from knowledgebase.parsers import PARSERS
from knowledgebase.parsers.platform import Platform, sniff_json
from knowledgebase.workspace import ImportRejected, Workspace


def make_zip(path, files: dict):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, src in files.items():
            if isinstance(src, (bytes, str)):
                zf.writestr(name, src)
            else:
                zf.write(src, name)
    return path


def chatgpt_zip(tmp_path):
    return make_zip(tmp_path / "chatgpt.zip", {"conversations.json": SAMPLES / "chatgpt" / "conversations.json",
                                              "chat.html": "<html></html>", "file-abc.png": b"\x89PNG..."})


def claude_code_zip(tmp_path):
    files = {f"projects/{p.relative_to(SAMPLES / 'claude-code' / 'projects')}": p
             for p in (SAMPLES / "claude-code" / "projects").rglob("*.jsonl")}
    return make_zip(tmp_path / "cc.zip", files)


@pytest.fixture
def ws(tmp_path):
    return Workspace(Settings(vault_dir=tmp_path / "vault", app_dir=tmp_path / "app"))


# ---------------------------------------------------------------- Platform enum
def test_platform_enum_covers_every_parser():
    assert {p.value for p in Platform} == set(PARSERS)
    assert Platform.parse("Claude Code") is Platform.CLAUDE_CODE and Platform.parse("openai") is Platform.CHATGPT
    assert Platform.CHATGPT == "chatgpt"  # str enum: still works as a kb.yaml exports key
    with pytest.raises(ValueError):
        Platform.parse("myspace")


def test_platform_wants_only_chat_files():
    assert Platform.CHATGPT.wants("conversations.json") and Platform.CHATGPT.wants("x/conversations-003.json")
    assert not Platform.CHATGPT.wants("shared_conversations.json") and not Platform.CHATGPT.wants("file-1.png")
    assert Platform.GEMINI.wants("Takeout/My Activity/Gemini Apps/MyActivity.json")
    assert not Platform.GEMINI.wants("Takeout/My Activity/YouTube/MyActivity.json")
    assert not Platform.CLAUDE_CODE.wants("p/s/subagents/a.jsonl") and Platform.CLAUDE_CODE.wants("p/s.jsonl")


def test_sniff_json():
    assert sniff_json(SAMPLES / "chatgpt" / "conversations.json") is Platform.CHATGPT
    assert sniff_json(SAMPLES / "claude" / "conversations.json") is Platform.CLAUDE
    assert sniff_json(SAMPLES / "gemini" / "MyActivity.json") is Platform.GEMINI


# ---------------------------------------------------------------- importing
def test_chatgpt_zip_imports_only_conversations(ws, tmp_path):
    rec = ws.import_file(chatgpt_zip(tmp_path), Platform.CHATGPT, "chatgpt.zip")
    assert rec.conversations > 0 and rec.files == ["conversations.json"] and rec.sample_titles
    stored = {p.name for p in (ws.imports_dir / rec.id).rglob("*") if p.is_file()}
    assert stored == {"conversations.json"}  # the photo and chat.html were never extracted
    assert [r.id for r in ws.imports()] == [rec.id]


def test_claude_code_zip_and_gemini_takeout(ws, tmp_path):
    cc = ws.import_file(claude_code_zip(tmp_path), "claude-code", "cc.zip")
    assert cc.files == ["."] and cc.conversations >= 3
    takeout = make_zip(tmp_path / "takeout.zip", {
        "Takeout/My Activity/Gemini Apps/MyActivity.json": SAMPLES / "gemini" / "MyActivity.json",
        "Takeout/My Activity/Search/MyActivity.json": "[]"})
    gem = ws.import_file(takeout, "gemini", "takeout.zip")
    assert gem.conversations > 0 and len(gem.files) == 1


def test_bare_json_upload(ws, tmp_path):
    rec = ws.import_file(SAMPLES / "claude" / "conversations.json", "claude", "conversations.json")
    assert rec.conversations > 0


@pytest.mark.parametrize("picked,maker,detected", [
    ("claude", chatgpt_zip, "chatgpt"),          # same file name, different structure
    ("gemini", chatgpt_zip, "chatgpt"),          # no MyActivity.json: sniffed from the zip
    ("chatgpt", claude_code_zip, "claude-code"),
])
def test_wrong_platform_is_explained(ws, tmp_path, picked, maker, detected):
    with pytest.raises(ImportRejected) as err:
        ws.import_file(maker(tmp_path), picked, "x.zip")
    assert err.value.detected is Platform(detected)
    assert Platform(detected).label in str(err.value)
    assert ws.imports() == [] and not [p for p in ws.imports_dir.iterdir()]  # nothing left behind


def test_rejections(ws, tmp_path):
    html_takeout = make_zip(tmp_path / "t.zip", {"Takeout/My Activity/Gemini Apps/MyActivity.html": "<html/>"})
    with pytest.raises(ImportRejected, match="as HTML"):
        ws.import_file(html_takeout, "gemini")
    with pytest.raises(ImportRejected, match="no conversations"):
        ws.import_file(make_zip(tmp_path / "e.zip", {"conversations.json": "[]"}), "chatgpt")
    with pytest.raises(ImportRejected, match="isn't a readable"):
        ws.import_file(make_zip(tmp_path / "b.zip", {"conversations.json": "{not json"}), "chatgpt")
    with pytest.raises(ImportRejected):
        ws.import_file(make_zip(tmp_path / "z.zip", {"readme.txt": "hi"}), "claude")
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ImportRejected, match="isn't a zip"):
        ws.import_file(tmp_path / "notes.txt", "chatgpt", "notes.txt")


def test_zip_slip_members_are_ignored(ws, tmp_path):
    evil = make_zip(tmp_path / "evil.zip", {"../../conversations.json": SAMPLES / "chatgpt" / "conversations.json"})
    with pytest.raises(ImportRejected):
        ws.import_file(evil, "chatgpt")
    assert not (tmp_path / "conversations.json").exists()


def test_replace_and_remove(ws, tmp_path):
    a = ws.import_file(chatgpt_zip(tmp_path), "chatgpt")
    b = ws.import_file(chatgpt_zip(tmp_path), "chatgpt", replace=True)
    assert [r.id for r in ws.imports()] == [b.id] and not (ws.imports_dir / a.id).exists()
    assert ws.remove(b.id) and ws.imports() == [] and not ws.remove(b.id)


def test_parse_exports_reads_imports(tmp_path, monkeypatch):
    from knowledgebase.pipeline import parse_exports

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.claude/projects either
    settings = Settings(vault_dir=tmp_path / "v", app_dir=tmp_path / "app")
    Workspace(settings).import_file(chatgpt_zip(tmp_path), "chatgpt")
    convs = parse_exports(settings)
    assert convs and {c.platform for c in convs} == {"chatgpt"}


# ---------------------------------------------------------------- output folder
def test_output_folder_is_validated_and_remembered(ws, tmp_path, monkeypatch):
    out = ws.set_output_dir(str(tmp_path / "my notes" / "kb"))
    assert out["exists"] and not out["has_index"] and out["path"].endswith("kb")
    monkeypatch.delenv("KB_VAULT_DIR", raising=False)
    assert load_settings(tmp_path / "missing.yaml", app_dir=ws.app_dir).vault_dir == tmp_path / "my notes" / "kb"
    assert load_settings(tmp_path / "missing.yaml", app_dir=ws.app_dir, vault_dir="flag").vault_dir.name == "flag"
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(ValueError, match="is a file"):
        ws.set_output_dir(str(tmp_path / "file.txt"))
    with pytest.raises(ValueError):
        ws.set_output_dir("  ")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "thesis.docx").write_text("x")
    assert ws.set_output_dir(str(tmp_path / "busy"))["note"]


# ---------------------------------------------------------------- HTTP API
@pytest.fixture
def api(tmp_path, nlp, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from knowledgebase.server.app import create_app

    monkeypatch.setenv("HOME", str(tmp_path))  # don't pick up this machine's Claude Code sessions
    monkeypatch.chdir(tmp_path)
    settings = Settings(vault_dir=tmp_path / "vault", app_dir=tmp_path / "app",
                        entity_extractor="spacy", embedder="none")
    settings.exports = {}
    return TestClient(create_app(settings.index_dir, embedder="none", settings=settings))


def wait_for_build(api, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = api.get("/api/build").json()
        if st["status"] != "running":
            return st
        time.sleep(0.2)
    raise AssertionError("build did not finish")


def test_api_upload_build_and_switch_folder(api, tmp_path):
    assert [p["id"] for p in api.get("/api/platforms").json()] == ["chatgpt", "claude", "claude-code", "gemini"]
    assert api.get("/api/health").json()["indexed"] is False

    bad = api.post("/api/imports", params={"platform": "claude", "filename": "c.zip"},
                   content=chatgpt_zip(tmp_path).read_bytes())
    assert bad.status_code == 422 and bad.json()["detected"] == "chatgpt"
    assert api.post("/api/imports", params={"platform": "myspace"}, content=b"x").status_code == 422
    assert api.post("/api/imports", params={"platform": "claude"}, content=b"").status_code == 400

    ok = api.post("/api/imports", params={"platform": "chatgpt", "filename": "c.zip"},
                  content=chatgpt_zip(tmp_path).read_bytes())
    assert ok.status_code == 201 and ok.json()["conversations"] > 0 and ok.json()["label"] == "ChatGPT"
    ws = api.get("/api/workspace").json()
    assert len(ws["imports"]) == 1 and ws["output"]["has_index"] is False

    out = api.put("/api/settings/output", json={"path": str(tmp_path / "chosen")}).json()
    assert out["path"] == str((tmp_path / "chosen").resolve())

    assert api.post("/api/build").status_code == 202
    st = wait_for_build(api)
    assert st["status"] == "done", st
    assert (tmp_path / "chosen" / "indexes" / "manifest.json").exists()
    assert list((tmp_path / "chosen" / "conversations").rglob("*.md"))
    assert api.get("/api/health").json()["indexed"] is True
    assert api.get("/api/stats").json()["platforms"] == {"chatgpt": ok.json()["conversations"]}

    rid = ok.json()["id"]
    assert api.delete(f"/api/imports/{rid}").status_code == 200
    assert api.delete(f"/api/imports/{rid}").status_code == 404
    st = (api.post("/api/build"), wait_for_build(api))[1]
    assert st["status"] == "error" and "No conversations" in st["error"]


def test_api_writes_need_a_local_origin(api, tmp_path):
    evil = {"Origin": "https://evil.example"}
    assert api.put("/api/settings/output", json={"path": str(tmp_path / "x")}, headers=evil).status_code == 403
    assert api.post("/api/build", headers=evil).status_code == 403
    assert api.get("/api/workspace", headers=evil).status_code == 200  # reads are fine
    pre = api.options("/api/build", headers={"Origin": "tauri://localhost", "Access-Control-Request-Method": "POST"})
    assert pre.headers.get("access-control-allow-origin") == "tauri://localhost"
    (tmp_path / "a.txt").write_text("")
    local = {"Origin": "http://localhost:3001"}
    assert api.put("/api/settings/output", json={"path": str(tmp_path / "a.txt")}, headers=local).status_code == 400
    assert json.loads(api.get("/api/build").content)["status"] == "idle"
