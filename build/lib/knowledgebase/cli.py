"""``kb`` command line.

    kb build                      parse exports in data/, build vault + graph + index
    kb search "query"             graph-aware search from the terminal
    kb serve                      UI + REST API + MCP on http://127.0.0.1:3001
    kb stats                      what is in the index
    kb demo                       build the bundled sample knowledge base and serve it
    kb export site/               the atlas as a static site (GitHub Pages)
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
    print(f"  vault: {settings.vault_dir.resolve()}")
    print("  next:  kb serve   (UI + MCP on http://%s:%d)" % (settings.host, settings.port))
    return 0


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
