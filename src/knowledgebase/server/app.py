"""HTTP server: REST API for the graph explorer + MCP endpoint, on port 3001 by default.

    kb serve                 # http://127.0.0.1:3001  (UI) · /api/* (REST) · /api/mcp (MCP)
"""
from __future__ import annotations

import json
import os
import tempfile
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from knowledgebase import __version__
from knowledgebase.config import Settings
from knowledgebase.parsers.platform import Platform
from knowledgebase.server.jobs import BuildBusy, BuildManager
from knowledgebase.server.mcp import MCPHandler
from knowledgebase.server.service import KnowledgeService, NotFound
from knowledgebase.workspace import ImportRejected, Workspace, platforms_catalog

# "tauri.localhost" is the origin of the desktop app's pages on Windows (tauri://localhost elsewhere).
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "tauri.localhost"}
LOCAL_ORIGIN_RE = r"^(https?|tauri)://(localhost|127\.0\.0\.1|\[::1\]|tauri\.localhost)(:\d+)?$"
MAX_UPLOAD_BYTES = int(os.environ.get("KB_MAX_UPLOAD_BYTES", 20 * 1024 ** 3))


def _static_dir() -> Path:
    return Path(str(resources.files("knowledgebase.server").joinpath("static")))


def create_app(index_dir: Path | None = None, embedder: str = "auto", embed_model: str | None = None,
               allowed_origins: list[str] | None = None, settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings(vault_dir=Path(index_dir).parent if index_dir else Path("vault"), embedder=embedder)
        if embed_model:
            settings.embed_model = embed_model
    service = KnowledgeService(Path(index_dir) if index_dir else settings.index_dir, embedder=embedder,
                               embed_model=embed_model)
    mcp = MCPHandler(service)
    workspace = Workspace(settings)
    builds = BuildManager(on_done=service.reload)
    extra_origins = set(allowed_origins or [])
    app = FastAPI(title="knowledgeBase", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.service = service
    app.state.workspace = workspace
    app.state.builds = builds

    def origin_allowed(request: Request) -> bool:
        origin = request.headers.get("origin")
        if not origin:
            return True  # non-browser clients
        host = urlparse(origin).hostname or ""
        return host in LOCAL_HOSTS or origin in extra_origins

    @app.middleware("http")
    async def refuse_foreign_writes(request: Request, call_next):
        # Anything that changes state (MCP calls, uploads, settings, builds) only from local pages:
        # stops a random website from driving the server through your browser (DNS rebinding / CSRF).
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api") \
                and not origin_allowed(request):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        return await call_next(request)

    # The desktop app (and a dev UI on another port) loads its pages from a different origin.
    app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN_RE, allow_origins=sorted(extra_origins),
                       allow_methods=["*"], allow_headers=["*"])

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

    # ------------------------------------------------------------------ workspace: imports, output folder, builds
    @app.get("/api/platforms")
    def platforms():
        return platforms_catalog()

    @app.get("/api/workspace")
    def workspace_state():
        from knowledgebase.pipeline import configured_sources

        return {
            "output": workspace.output_info(),
            "imports": [r.to_dict() for r in workspace.imports()],
            "configured_sources": [{"platform": p, "label": Platform.parse(p).label, "path": str(path)}
                                   for p, path in configured_sources(settings)],
            "build": builds.status(),
        }

    @app.post("/api/imports", status_code=201)
    async def upload_export(request: Request, platform: Platform, filename: str = "", replace: bool = False):
        """Body: the raw export file (zip, or a bare .json). Validated with the chosen platform's parser."""
        if builds.running:
            raise HTTPException(409, "A build is running; import again when it has finished.")
        workspace.app_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".upload-", dir=workspace.app_dir)
        tmp = Path(tmp_name)
        try:
            size = 0
            with os.fdopen(fd, "wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"Uploads are limited to {MAX_UPLOAD_BYTES // 1024 ** 3} GB.")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "The upload was empty.")
            try:
                record = await run_in_threadpool(workspace.import_file, tmp, platform, filename, replace)
            except ImportRejected as exc:
                return JSONResponse({"detail": str(exc), "detected": exc.detected.value if exc.detected else None},
                                    status_code=422)
        finally:
            tmp.unlink(missing_ok=True)
        return record.to_dict()

    @app.delete("/api/imports/{import_id}")
    def delete_import(import_id: str):
        if builds.running:
            raise HTTPException(409, "A build is running; remove the import when it has finished.")
        if not workspace.remove(import_id):
            raise HTTPException(404, "No such import.")
        return {"removed": import_id}

    @app.put("/api/settings/output")
    def set_output(path: str = Body(..., embed=True)):
        """Where the markdown notes (conversations/, concepts/) and graph/index files are written."""
        if builds.running:
            raise HTTPException(409, "A build is running; change the folder when it has finished.")
        try:
            info = workspace.set_output_dir(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        service.use_index_dir(settings.index_dir)
        return info

    @app.post("/api/build", status_code=202)
    def start_build():
        try:
            return builds.start(settings)
        except BuildBusy as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/build")
    def build_status():
        return builds.status()

    # ------------------------------------------------------------------ MCP (Streamable HTTP)
    @app.post("/api/mcp")
    async def mcp_post(request: Request):
        # (Origin is checked by refuse_foreign_writes: DNS-rebinding protection the MCP spec requires.)
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
