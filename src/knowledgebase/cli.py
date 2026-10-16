"""``kb`` command line.

    kb build                      parse exports in data/, build vault + graph + index
    kb search "query"             graph-aware search from the terminal
    kb serve                      UI + REST API + MCP on http://127.0.0.1:3001
    kb stats                      what is in the index
    kb demo                       build the bundled sample knowledge base and serve it
    kb export site/               the atlas as a static site (GitHub Pages)
    kb summarize --top 50         optional LLM-written concept summaries (any OpenAI-compatible endpoint)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from knowledgebase.config import load_settings


def _settings(args):
    return load_settings(
        getattr(args, "config", None),
        vault_dir=getattr(args, "vault", None),
        embedder=getattr(args, "embedder", None),
        entity_extractor=getattr(args, "entities", None),
        port=getattr(args, "port", None),
        host=getattr(args, "host", None),
    )


def cmd_build(args) -> int:
    from knowledgebase.pipeline import build

    settings = _settings(args)
    for platform, path in (args.export or []):
        settings.exports[platform] = path
    report = build(settings, write_vault=not args.no_vault)
    print(f"✓ {report}")
    print_highlights(settings.index_dir)
    print(f"\n  vault: {settings.vault_dir.resolve()}")
    print("  next:  kb serve   (UI + MCP on http://%s:%d)" % (settings.host, settings.port))
    return 0


def print_highlights(index_dir: Path) -> None:
    """A first look at what the graph found, right in the terminal."""
    from knowledgebase.graph.model import KnowledgeGraph

    try:
        g = KnowledgeGraph.load(index_dir / "graph.json")
    except (OSError, ValueError):
        return
    label = lambda n: g.nodes[n]["label"] if n in g.nodes else n  # noqa: E731
    top = [n["label"] for n in g.top_nodes(12)]
    if top:
        print("\n  Most central: " + ", ".join(top))
    themes = [c["label"] for c in g.communities[:6] if c.get("size", 0) >= 3]
    if themes:
        print("  Main themes:  " + "; ".join(themes))
    d = g.discoveries or {}
    if d.get("hidden_connections"):
        h = d["hidden_connections"][0]
        print(f"  Never connected, but close: {label(h['source'])} and {label(h['target'])} "
              f"(both linked to {', '.join(label(v) for v in h['via'][:3])})")
    if d.get("forgotten"):
        f = d["forgotten"][0]
        print(f"  Not visited lately: {label(f['node'])} (last discussed {f['last_seen']})")


def cmd_search(args) -> int:
    from knowledgebase.server.service import KnowledgeService

    settings = _settings(args)
    service = KnowledgeService(settings.index_dir, embedder=settings.embedder)
    data = service.search(args.query, k=args.k, mode=args.mode, max_chars=args.chars)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    if data["concepts"]:
        print("concepts: " + ", ".join(c["label"] for c in data["concepts"]))
    if data.get("time_window"):
        print(f"favouring: {data['time_window']}")
    for r in data["results"]:
        via = f"  via {', '.join(r['via'])}" if r["via"] else ""
        print(f"\n[{r['rank']}] {r['title']} · {r['date']} · {r['platform']}{via}")
        print("    " + r["text"].replace("\n", "\n    "))
    return 0


def cmd_stats(args) -> int:
    from knowledgebase.server.service import KnowledgeService

    settings = _settings(args)
    print(json.dumps(KnowledgeService(settings.index_dir, embedder="none").stats(), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from knowledgebase.server.app import create_app

    settings = _settings(args)
    app = create_app(settings.index_dir, embedder=settings.embedder, embed_model=settings.embed_model,
                     allowed_origins=args.allow_origin)
    print(f"knowledgeBase on http://{settings.host}:{settings.port}  ·  MCP: http://{settings.host}:{settings.port}/api/mcp")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


def cmd_summarize(args) -> int:
    from knowledgebase.llm import build_llm
    from knowledgebase.pipeline import refresh_concept_notes
    from knowledgebase.retrieval.index import KnowledgeIndex
    from knowledgebase.summaries import Summarizer

    settings = _settings(args)
    llm = build_llm()
    if llm is None:
        print("No LLM configured. Set KB_LLM_BASE_URL + KB_LLM_MODEL (any OpenAI-compatible endpoint, e.g. Ollama, "
              "vLLM, a Hugging Face Space) or GEMINI_API_KEY.", file=sys.stderr)
        return 1
    index = KnowledgeIndex.load(settings.index_dir, embedder=None)
    if args.concept:
        from knowledgebase.server.service import KnowledgeService

        service = KnowledgeService(settings.index_dir, embedder="none")
        nodes = [service.resolve(c) for c in args.concept]
    else:
        nodes = [n["id"] for n in index.graph.top_nodes(args.top, per_community_cap=1.0)]
    summarizer = Summarizer(index, llm, settings.index_dir)
    print(f"Summarising {len(nodes)} concepts with {llm.name} …")
    done = summarizer.summarize(nodes, force=args.force, delay=args.delay, progress=lambda label: print(f"  ✓ {label}"))
    refresh_concept_notes(settings, index, only=set(nodes))
    print(f"✓ {done} new summaries ({len(summarizer.summaries)} total). They appear in the atlas, the vault notes "
          "and the explore_concept MCP tool.")
    return 0


def cmd_export(args) -> int:
    from knowledgebase.server.export import export_site

    settings = _settings(args)
    info = export_site(settings.index_dir, Path(args.out), max_nodes=args.max_nodes,
                       include_conversations=args.include_conversations, title=args.title)
    print(f"✓ exported {info['nodes']} concepts, {info['paths']} routes, {info['conversations']} conversations "
          f"to {info['out']}")
    print(f"  preview: python -m http.server -d {args.out} 8000   (then open http://localhost:8000)")
    return 0


def cmd_demo(args) -> int:
    from knowledgebase.pipeline import build

    root = Path(__file__).resolve().parents[2] / "examples" / "sample_exports"
    if not root.exists():
        print("Sample exports not found (they ship with the git repository, not the wheel).", file=sys.stderr)
        return 1
    settings = load_settings(None, vault_dir=args.vault or "demo_vault", embedder=args.embedder,
                             entity_extractor=args.entities, port=args.port)
    settings.exports = {
        "chatgpt": str(root / "chatgpt" / "conversations.json"),
        "claude": str(root / "claude" / "conversations.json"),
        "gemini": str(root / "gemini" / "MyActivity.json"),
        "claude-code": str(root / "claude-code" / "projects"),
    }
    print(f"✓ {build(settings)}")
    print_highlights(settings.index_dir)
    if args.no_serve:
        return 0
    args.config, args.vault, args.allow_origin = None, str(settings.vault_dir), []
    return cmd_serve(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb", description="Graph-first knowledge base from your LLM chats.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", help="path to kb.yaml (default: ./kb.yaml)")
        p.add_argument("--vault", help="vault directory (default: ./vault)")
        p.add_argument("--embedder", help="auto | sentence-transformers | wordllama | none")
        p.add_argument("--entities", help="auto | gliner | spacy")

    p = sub.add_parser("build", help="build the vault, graph and index")
    common(p)
    p.add_argument("--export", nargs=2, action="append", metavar=("PLATFORM", "PATH"),
                   help="override an export path, e.g. --export claude ~/Downloads/conversations.json")
    p.add_argument("--no-vault", action="store_true", help="only build the index (skip writing markdown notes)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="search from the terminal")
    common(p)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--mode", default="graph_hybrid")
    p.add_argument("--chars", type=int, default=500)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="show index statistics")
    common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("summarize", help="LLM-written 'what do I know about X' summaries for top concepts")
    common(p)
    p.add_argument("--top", type=int, default=50, help="summarise the N most important concepts")
    p.add_argument("--concept", action="append", help="summarise specific concepts (repeatable)")
    p.add_argument("--force", action="store_true", help="regenerate even if the evidence is unchanged")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("export", help="write the atlas as a static site (e.g. for GitHub Pages)")
    common(p)
    p.add_argument("out", help="output folder")
    p.add_argument("--max-nodes", type=int, default=3000)
    p.add_argument("--include-conversations", action="store_true",
                   help="also publish full conversation text (off by default — this is your private history)")
    p.add_argument("--title")
    p.set_defaults(func=cmd_export)

    for name, func, help_text in (("serve", cmd_serve, "UI + REST API + MCP server"),
                                  ("demo", cmd_demo, "build the bundled sample data and serve it")):
        p = sub.add_parser(name, help=help_text)
        common(p)
        p.add_argument("--host")
        p.add_argument("--port", type=int, help="default 3001")
        p.add_argument("--allow-origin", action="append", default=[],
                       help="extra browser origin allowed to call /api/mcp")
        if name == "demo":
            p.add_argument("--no-serve", action="store_true")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
