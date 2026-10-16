"""Model Context Protocol server (Streamable HTTP transport) at ``/api/mcp``.

Implements the stateless subset of the MCP spec that tool servers need:
``initialize``, ``notifications/initialized``, ``ping``, ``tools/list``,
``tools/call`` (+ empty ``resources/list`` / ``prompts/list``). Every POST gets a
single ``application/json`` response — no SSE stream is required, which keeps
it compatible with Claude Code, Claude Desktop (via ``mcp-remote``), Cursor,
OpenCode and any other MCP client.

No SDK dependency on purpose: the wire protocol is small and stable, while SDK
APIs have changed between major versions.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from knowledgebase import __version__
from knowledgebase.server.service import KnowledgeService, NotFound

log = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST = SUPPORTED_VERSIONS[0]

INSTRUCTIONS = (
    "This server is the user's personal knowledge graph, built from their past conversations with AI "
    "assistants (ChatGPT, Claude, Gemini, Claude Code). Use `search_knowledge` to find where something was "
    "discussed before answering questions about the user's own past work, decisions or ideas. Use "
    "`explore_concept` to see everything connected to a topic, `find_connection` to explain how two ideas "
    "relate, and `discover` to surface non-obvious links. Cite conversation titles and dates."
)

READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}

TOOLS: list[dict] = [
    {
        "name": "search_knowledge",
        "title": "Search past conversations",
        "description": (
            "Graph-aware search over the user's past AI conversations. Combines keyword (BM25), semantic and "
            "knowledge-graph retrieval, so it also finds passages that are only *connected* to the query. "
            "Returns ranked passages with conversation title, date, platform and the concepts that linked them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question or keywords."},
                "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
                "mode": {"type": "string", "enum": ["graph_hybrid", "hybrid", "bm25", "dense", "graph"],
                         "default": "graph_hybrid"},
            },
            "required": ["query"],
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "explore_concept",
        "title": "Explore a concept",
        "description": (
            "Everything the knowledge graph knows about one concept (technology, person, project, topic): "
            "strongest related concepts, typed relations with evidence, the conversations that mention it and "
            "how often it came up over time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "Concept name or id, e.g. 'PyTorch' or 'esp32'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 60, "default": 15},
            },
            "required": ["concept"],
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "find_connection",
        "title": "Find how two concepts connect",
        "description": "Strongest path between two concepts in the graph, with a supporting snippet for each hop.",
        "inputSchema": {
            "type": "object",
            "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
            "required": ["source", "target"],
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "discover",
        "title": "Discover non-obvious connections",
        "description": (
            "Insights computed from the graph structure: bridge concepts that connect separate areas, unexpected "
            "cross-topic links, hidden connections (concepts never discussed together but sharing many "
            "neighbours) and important topics not touched in a long time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["all", "bridges", "unexpected_links", "hidden_connections",
                                                    "forgotten"], "default": "all"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
            },
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "get_conversation",
        "title": "Read a past conversation",
        "description": "Read a conversation (or the turns around one passage) returned by search_knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
                "chunk_id": {"type": "string", "description": "Optional: only return the turns around this chunk."},
            },
            "required": ["conversation_id"],
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "recent_activity",
        "title": "What was I working on?",
        "description": (
            "Conversations and concepts from a period of time — 'last week', 'last month', 'March 2026', "
            "'last spring', '2025'. Shows what the user focused on, which topics recurred and which were new. "
            "Use it to catch up on the user's recent work before helping with it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "default": "last month",
                         "description": "Natural-language period, e.g. 'last week', 'March 2026', 'last spring'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
            },
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "top_concepts",
        "title": "Most important concepts",
        "description": "The most central concepts of the knowledge graph and its main communities (themes).",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25}},
        },
        "annotations": READ_ONLY,
    },
]


# ---------------------------------------------------------------------- tool implementations
def _fmt_search(data: dict) -> str:
    lines = [f"Results for: {data['query']}"]
    if data.get("time_window"):
        lines.append(f"Favouring conversations from: {data['time_window']}")
    if data["concepts"]:
        lines.append("Matched concepts: " + ", ".join(c["label"] for c in data["concepts"]))
    for r in data["results"]:
        via = f" · via {', '.join(r['via'])}" if r["via"] else ""
        lines.append(f"\n[{r['rank']}] {r['title']} — {r['date']} ({r['platform']}){via}\n"
                     f"conversation_id={r['conversation_id']} chunk_id={r['chunk_id']}\n{r['text']}")
    if not data["results"]:
        lines.append("No matching passages.")
    return "\n".join(lines)


def _fmt_concept(d: dict) -> str:
    n = d["node"]
    lines = [f"# {n['label']} ({n.get('type', n.get('kind'))})",
             f"Mentioned in {n.get('conversation_count', 0)} conversations, {n.get('first_seen')} → {n.get('last_seen')}."]
    if d.get("community"):
        lines.append(f"Theme: {d['community']['label']}")
    if d.get("summary"):
        lines.append("\n## Summary (from your conversations)\n" + d["summary"]["summary"])
    if d["neighbors"]:
        lines.append("\n## Related")
        for nb in d["neighbors"]:
            rel = f" [{', '.join(nb['relations'])}]" if nb["relations"] else ""
            lines.append(f"- {nb['label']} (strength {nb['edge']['strength']:.2f}, "
                         f"{nb['edge'].get('conversations', 0)} shared conversations){rel}")
    if d["relations"]:
        lines.append("\n## Relations")
        for r in d["relations"][:15]:
            ev = f' — "{r["evidence"][0]}"' if r.get("evidence") else ""
            lines.append(f"- {r['source_label']} {r['relation']} {r['target_label']}{ev}")
    if d.get("worth_connecting"):
        lines.append("\n## Worth connecting (never discussed together, many shared neighbours)")
        for w in d["worth_connecting"]:
            lines.append(f"- {w['label']} via {', '.join(v['label'] for v in w['via'])}")
    if d["mentions"]:
        lines.append("\n## Conversations")
        for m in d["mentions"][:12]:
            lines.append(f"- {m['title']} — {m['date']} ({m['platform']}) conversation_id={m['conversation_id']}: "
                         f"{m['snippet']}")
    return "\n".join(lines)


def _fmt_connection(d: dict) -> str:
    if not d["path"]:
        return f"No path found between {d['source']} and {d['target']}."
    lines = ["Path: " + " → ".join(p["label"] for p in d["path"])]
    for h in d["hops"]:
        ev = h.get("evidence")
        if ev:
            lines.append(f"- {h['source']} ↔ {h['target']}: \"{ev['snippet']}\" ({ev['title']}, {ev['date']})")
    return "\n".join(lines)


def _fmt_discoveries(d: dict, kind: str, limit: int) -> str:
    out = []
    if kind in ("all", "bridges") and d["bridges"]:
        out.append("## Bridge concepts (connect separate areas)")
        out += [f"- {b['label']}: links {' / '.join(x for x in b['community_labels'] if x)}" for b in d["bridges"][:limit]]
    if kind in ("all", "unexpected_links") and d["unexpected_links"]:
        out.append("## Unexpected links")
        out += [f"- {u['source_label']} ↔ {u['target_label']} (NPMI {u['npmi']:.2f}, {u['conversations']} conversations)"
                for u in d["unexpected_links"][:limit]]
    if kind in ("all", "hidden_connections") and d["hidden_connections"]:
        out.append("## Hidden connections (never discussed together)")
        out += [f"- {h['source_label']} ↔ {h['target_label']} via {', '.join(h['via_labels'])}"
                for h in d["hidden_connections"][:limit]]
    if kind in ("all", "forgotten") and d["forgotten"]:
        out.append("## Forgotten topics")
        out += [f"- {f['label']}: last discussed {f['last_seen']} ({f['idle_days']} days earlier than your latest chat)"
                for f in d["forgotten"][:limit]]
    return "\n".join(out) or "No discoveries yet — build a larger knowledge base."


def _fmt_conversation(d: dict) -> str:
    from knowledgebase.text import strip_wikilinks

    lines = [f"# {d['title']} — {d['date']} ({d['platform']})"]
    for c in d["chunks"]:
        lines.append(strip_wikilinks(c["text"]))
    return "\n\n".join(lines)[:20_000]


class MCPHandler:
    def __init__(self, service: KnowledgeService):
        self.service = service
        self.tools: dict[str, Callable[[dict], tuple[str, Any]]] = {
            "search_knowledge": self._search,
            "explore_concept": self._explore,
            "find_connection": self._connection,
            "discover": self._discover,
            "get_conversation": self._conversation,
            "top_concepts": self._top,
            "recent_activity": self._activity,
        }

    def _search(self, a: dict):
        d = self.service.search(a["query"], k=int(a.get("k", 8)), mode=a.get("mode", "graph_hybrid"))
        return _fmt_search(d), d

    def _explore(self, a: dict):
        d = self.service.node_detail(a["concept"], limit=int(a.get("limit", 15)))
        return _fmt_concept(d), d

    def _connection(self, a: dict):
        d = self.service.connection(a["source"], a["target"])
        return _fmt_connection(d), d

    def _discover(self, a: dict):
        d = self.service.discoveries()
        return _fmt_discoveries(d, a.get("kind", "all"), int(a.get("limit", 8))), d

    def _conversation(self, a: dict):
        d = self.service.conversation(a["conversation_id"], a.get("chunk_id"))
        return _fmt_conversation(d), d

    def _activity(self, a: dict):
        d = self.service.activity(a.get("when", "last month"), limit=int(a.get("limit", 15)))
        lines = [f"{d['period']}: {d['conversation_count']} conversations"]
        if d["concepts"]:
            lines.append("Recurring: " + ", ".join(f"{c['label']} ({c['conversations_in_period']})" for c in d["concepts"][:12]))
        if d["new_concepts"]:
            lines.append("New in this period: " + ", ".join(c["label"] for c in d["new_concepts"]))
        for c in d["conversations"]:
            lines.append(f"- {c['date']} · {c['title']} ({c['platform']}) — {', '.join(c['concepts'])} "
                         f"conversation_id={c['conversation_id']}")
        if not d["conversations"]:
            lines.append("No conversations in this period.")
        return "\n".join(lines), d

    def _top(self, a: dict):
        d = self.service.overview(limit=int(a.get("limit", 25)))
        text = "Top concepts: " + ", ".join(n["label"] for n in d["nodes"]) + "\n\nThemes:\n" + \
            "\n".join(f"- {c['label']} ({c['size']} concepts)" for c in d["communities"][:12])
        return text, {"nodes": d["nodes"], "communities": d["communities"][:12]}

    # ------------------------------------------------------------------ JSON-RPC
    @staticmethod
    def _result(msg_id, result) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def handle(self, message: Any) -> dict | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return self._error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
        method = message["method"]
        msg_id = message.get("id")
        params = message.get("params") or {}
        if msg_id is None:  # notification (e.g. notifications/initialized): no response
            return None
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                return self._result(msg_id, {
                    "protocolVersion": requested if requested in SUPPORTED_VERSIONS else LATEST,
                    "capabilities": {"tools": {"listChanged": False}, "resources": {}, "prompts": {}},
                    "serverInfo": {"name": "knowledgebase", "title": "knowledgeBase", "version": __version__},
                    "instructions": INSTRUCTIONS,
                })
            if method == "ping":
                return self._result(msg_id, {})
            if method == "tools/list":
                return self._result(msg_id, {"tools": TOOLS})
            if method == "resources/list":
                return self._result(msg_id, {"resources": []})
            if method == "resources/templates/list":
                return self._result(msg_id, {"resourceTemplates": []})
            if method == "prompts/list":
                return self._result(msg_id, {"prompts": []})
            if method == "tools/call":
                name = params.get("name")
                tool = self.tools.get(name)
                if tool is None:
                    return self._error(msg_id, -32602, f"Unknown tool: {name}")
                try:
                    text, structured = tool(params.get("arguments") or {})
                    return self._result(msg_id, {
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": json.loads(json.dumps(structured, default=str)),
                        "isError": False,
                    })
                except (NotFound, KeyError, ValueError, TypeError) as exc:
                    # Tool-level errors are reported in the result so the model can recover.
                    return self._result(msg_id, {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True})
            return self._error(msg_id, -32601, f"Method not found: {method}")
        except FileNotFoundError as exc:
            return self._error(msg_id, -32603, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("MCP request failed")
            return self._error(msg_id, -32603, f"Internal error: {exc.__class__.__name__}")
