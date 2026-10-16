"""Static export of the atlas: a folder you can put on GitHub Pages / any static host.

    kb export site/                       # concepts, relations, discoveries, short snippets
    kb export site/ --include-conversations   # + full conversation text (think before sharing!)

The page stays lazy: it loads ``data/overview.json`` first and one small
``data/nodes/<id>.json`` shard per concept you open — the same contract as the
live API, just pre-computed. Passage search needs the server and is hidden.
"""
from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

from knowledgebase.server.service import KnowledgeService


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")


def export_site(index_dir: Path, out_dir: Path, max_nodes: int = 3000, include_conversations: bool = False,
                title: str | None = None) -> dict:
    service = KnowledgeService(index_dir, embedder="none")
    graph = service.graph
    out_dir = Path(out_dir)
    data = out_dir / "data"
    if data.exists():
        shutil.rmtree(data)

    # Which concepts get a shard: the most important ones (the rest stay reachable via search).
    ranked = sorted(graph.nodes.values(), key=lambda n: -n.get("importance", 0))[:max_nodes]
    exported = {n["id"] for n in ranked}

    stats = service.stats()
    stats["static"] = True
    stats["exported_nodes"] = len(exported)
    _dump(data / "stats.json", stats)
    _dump(data / "overview.json", service.overview(limit=55))
    for community in graph.communities[:40]:
        _dump(data / f"overview_c{community['id']}.json", service.overview(limit=30, community=community["id"]))
    _dump(data / "node_index.json", [
        {k: n.get(k) for k in ("id", "label", "conversation_count", "importance")} | {"aliases": n.get("aliases", [])[:5]}
        for n in ranked
    ])

    for node_id in exported:
        detail = service.node_detail(node_id, limit=40)
        detail["neighbors"] = [nb for nb in detail["neighbors"] if nb["id"] in exported][:30]
        keep = {node_id} | {nb["id"] for nb in detail["neighbors"]}
        detail["edges"] = [e for e in detail["edges"] if e["source"] in keep and e["target"] in keep]
        detail["relations"] = [r for r in detail["relations"] if r["source"] in exported and r["target"] in exported]
        if not include_conversations:
            for m in detail["mentions"]:
                m.pop("chunk_id", None)
        _dump(data / "nodes" / f"{node_id}.json", detail)

    discoveries = service.discoveries()
    _dump(data / "discover.json", discoveries)
    pairs = [(d["source"], d["target"]) for key in ("hidden_connections", "unexpected_links")
             for d in discoveries.get(key, [])]
    for a, b in pairs:
        _dump(data / "paths" / f"{a}__{b}.json", service.connection(a, b))

    conversations = 0
    if include_conversations:
        for conv_id in service.index.by_conversation:
            _dump(data / "conversations" / f"{conv_id}.json", service.conversation(conv_id))
            conversations += 1

    # Copy the UI, switching it to static mode and relative URLs (works under a sub-path like /repo/).
    static = Path(str(resources.files("knowledgebase.server").joinpath("static")))
    if (out_dir / "assets").exists():
        shutil.rmtree(out_dir / "assets")
    shutil.copytree(static / "assets", out_dir / "assets")
    html = (static / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/assets/', 'href="assets/').replace('src="/assets/', 'src="assets/')
    flags = {"conversations": include_conversations}
    html = html.replace("<head>", f"<head>\n  <script>window.KB_STATIC = {json.dumps(flags)};</script>", 1)
    if title:
        html = html.replace("<title>knowledgeBase atlas</title>", f"<title>{title}</title>")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve files as-is
    return {"nodes": len(exported), "paths": len(pairs), "conversations": conversations, "out": str(out_dir)}
