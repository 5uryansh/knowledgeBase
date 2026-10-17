import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from knowledgebase.server.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(demo_vault):
    return TestClient(create_app(demo_vault / "indexes", embedder="none"))


def rpc(client, method, params=None, id_=1, **kw):
    body = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        body["id"] = id_
    if params is not None:
        body["params"] = params
    return client.post("/api/mcp", json=body, **kw)


# ---------------------------------------------------------------- REST
def test_health_and_stats(client):
    assert client.get("/api/health").json()["indexed"] is True
    stats = client.get("/api/stats").json()
    assert stats["conversations"] == 36 and stats["nodes"] > 100


def test_overview_is_small_and_diverse(client):
    data = client.get("/api/graph/overview?limit=40").json()
    assert 10 <= len(data["nodes"]) <= 40
    assert len({n["community"] for n in data["nodes"]}) >= 3
    ids = {n["id"] for n in data["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in data["edges"])


def test_node_detail(client):
    d = client.get("/api/graph/node/ESP32").json()  # labels resolve to ids
    assert d["node"]["id"] == "esp32"
    assert d["neighbors"] and d["mentions"] and d["timeline"]
    assert "worth_connecting" in d
    assert client.get("/api/graph/node/definitely-not-a-concept-xyz").status_code == 404


def test_search_path_discover_conversation(client):
    s = client.get("/api/search", params={"q": "vLLM out of memory", "k": 3}).json()
    assert s["results"][0]["title"].startswith("Serving a 7B model")
    p = client.get("/api/graph/path", params={"source": "esp32", "target": "quantization"}).json()
    assert p["path"][0]["id"] == "esp32" and p["path"][-1]["id"] == "quantization"
    assert set(client.get("/api/discover").json()) == {"bridges", "unexpected_links", "hidden_connections", "forgotten"}
    conv = client.get(f"/api/conversation/{s['results'][0]['conversation_id']}").json()
    assert conv["chunks"]
    assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "knowledgeBase" in r.text
    assert client.get("/assets/app.js").status_code == 200


# ---------------------------------------------------------------- MCP
def test_mcp_handshake_and_tools(client):
    init = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "pytest", "version": "0"}}).json()
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "knowledgebase"
    assert rpc(client, "notifications/initialized", id_=None).status_code == 202
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"search_knowledge", "explore_concept", "find_connection", "discover",
                                         "get_conversation", "top_concepts", "recent_activity"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def test_mcp_unknown_version_falls_back_to_latest(client):
    init = rpc(client, "initialize", {"protocolVersion": "1999-01-01"}).json()
    assert init["result"]["protocolVersion"] == "2025-06-18"


@pytest.mark.parametrize("name,args", [
    ("search_knowledge", {"query": "e-ink ghosting", "k": 3}),
    ("explore_concept", {"concept": "vLLM"}),
    ("find_connection", {"source": "ESP32", "target": "Quantization"}),
    ("discover", {}),
    ("top_concepts", {"limit": 10}),
    ("recent_activity", {"when": "2026"}),
])
def test_mcp_tool_calls(client, name, args):
    res = rpc(client, "tools/call", {"name": name, "arguments": args}).json()["result"]
    assert res["isError"] is False
    assert res["content"][0]["type"] == "text" and res["content"][0]["text"]
    assert "structuredContent" in res


def test_mcp_errors(client):
    bad_tool = rpc(client, "tools/call", {"name": "nope", "arguments": {}}).json()
    assert bad_tool["error"]["code"] == -32602
    bad_concept = rpc(client, "tools/call", {"name": "explore_concept", "arguments": {"concept": "zzzz-no"}}).json()
    assert bad_concept["result"]["isError"] is True
    assert rpc(client, "does/not/exist").json()["error"]["code"] == -32601
    assert client.post("/api/mcp", content=b"{not json").status_code == 400
    assert client.get("/api/mcp").status_code == 405


def test_mcp_batch_and_origin_check(client):
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    out = client.post("/api/mcp", json=batch).json()
    assert [r["id"] for r in out] == [1, 2]
    assert rpc(client, "ping", headers={"Origin": "https://evil.example"}).status_code == 403
    assert rpc(client, "ping", headers={"Origin": "http://localhost:5173"}).status_code == 200


def test_static_export(demo_vault, tmp_path):
    import json

    from knowledgebase.server.export import export_site

    info = export_site(demo_vault / "indexes", tmp_path / "site", max_nodes=200)
    site = tmp_path / "site"
    assert info["nodes"] == 200 and info["conversations"] == 0
    assert "window.KB_STATIC" in (site / "index.html").read_text()
    assert 'src="assets/app.js"' in (site / "index.html").read_text()
    overview = json.loads((site / "data" / "overview.json").read_text())
    node = overview["nodes"][0]["id"]
    detail = json.loads((site / "data" / "nodes" / f"{node}.json").read_text())
    exported = {p.stem for p in (site / "data" / "nodes").glob("*.json")}
    assert all(nb["id"] in exported for nb in detail["neighbors"])
    assert not (site / "data" / "conversations").exists()  # private text is opt-in


def test_activity(client):
    d = client.get("/api/activity", params={"when": "January 2026"}).json()
    assert d["period"] == "January 2026" and d["conversation_count"] >= 3
    assert all(c["date"].startswith("2026-01") for c in d["conversations"])
    assert d["concepts"]
    assert client.get("/api/activity", params={"when": "whenever"}).status_code == 400
