"""HTTP server: REST API for the graph explorer + MCP endpoint, on port 3001 by default.

    kb serve                 # http://127.0.0.1:3001  (UI) · /api/* (REST) · /api/mcp (MCP)
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from knowledgebase import __version__
from knowledgebase.server.mcp import MCPHandler
from knowledgebase.server.service import KnowledgeService, NotFound

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _static_dir() -> Path:
    return Path(str(resources.files("knowledgebase.server").joinpath("static")))


def create_app(index_dir: Path, embedder: str = "auto", embed_model: str | None = None,
               allowed_origins: list[str] | None = None) -> FastAPI:
    service = KnowledgeService(index_dir, embedder=embedder, embed_model=embed_model)
    mcp = MCPHandler(service)
    extra_origins = set(allowed_origins or [])
    app = FastAPI(title="knowledgeBase", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.service = service

    def guard():
        if not service.ready:
            raise HTTPException(503, "No index yet. Run `kb build` first.")

    def not_found(exc: Exception):
        raise HTTPException(404, str(exc))

    # ------------------------------------------------------------------ REST
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "indexed": service.ready}

    @app.get("/api/stats")
    def stats():
        guard()
        return service.stats()

    @app.post("/api/reload")
    def reload():
        service.reload()
        return {"reloaded": True}

    @app.get("/api/graph/overview")
    def overview(limit: int = Query(60, ge=5, le=400), community: int | None = None):
        guard()
        return service.overview(limit=limit, community=community)

    @app.get("/api/graph/node/{node_id}")
    def node(node_id: str, limit: int = Query(25, ge=1, le=150)):
        guard()
        try:
            return service.node_detail(node_id, limit=limit)
        except NotFound as exc:
            not_found(exc)

    @app.get("/api/graph/path")
    def path(source: str, target: str):
        guard()
        try:
            return service.connection(source, target)
        except NotFound as exc:
            not_found(exc)

    @app.get("/api/nodes/search")
    def nodes_search(q: str, limit: int = Query(10, ge=1, le=50)):
        guard()
        return service.search_nodes(q, limit=limit)

    @app.get("/api/search")
    def search(q: str, k: int = Query(10, ge=1, le=50), mode: str = "graph_hybrid"):
        guard()
        try:
            return service.search(q, k=k, mode=mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/activity")
    def activity(when: str = "last month", limit: int = Query(15, ge=1, le=100)):
        guard()
        try:
            return service.activity(when, limit=limit)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/discover")
    def discover():
        guard()
        return service.discoveries()

    @app.get("/api/conversation/{conversation_id}")
    def conversation(conversation_id: str, around: str | None = None):
        guard()
        try:
            return service.conversation(conversation_id, around)
        except NotFound as exc:
            not_found(exc)

    # ------------------------------------------------------------------ MCP (Streamable HTTP)
    def origin_allowed(request: Request) -> bool:
        origin = request.headers.get("origin")
        if not origin:
            return True  # non-browser clients
        host = urlparse(origin).hostname or ""
        return host in LOCAL_HOSTS or origin in extra_origins

    @app.post("/api/mcp")
    async def mcp_post(request: Request):
        # DNS-rebinding protection required by the MCP spec for local servers.
        if not origin_allowed(request):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        try:
            payload = json.loads(await request.body() or b"null")
        except json.JSONDecodeError:
            return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                                status_code=400)
        if isinstance(payload, list):
            responses = [r for r in (mcp.handle(m) for m in payload) if r is not None]
            return JSONResponse(responses) if responses else Response(status_code=202)
        response = mcp.handle(payload)
        if response is None:
            return Response(status_code=202)
        return JSONResponse(response)

    @app.get("/api/mcp")
    def mcp_get():
        # This server never initiates messages, so it offers no SSE stream.
        return Response(status_code=405, headers={"Allow": "POST"})

    @app.delete("/api/mcp")
    def mcp_delete():
        return Response(status_code=405, headers={"Allow": "POST"})

    # ------------------------------------------------------------------ UI
    static = _static_dir()
    if static.exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(static / "index.html")

    return app
