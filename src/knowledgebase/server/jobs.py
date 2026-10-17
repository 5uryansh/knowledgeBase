"""One background build at a time, with progress the UI can poll (GET /api/build)."""
from __future__ import annotations

import copy
import logging
import threading
import time
import traceback
from collections.abc import Callable

from knowledgebase.config import Settings

log = logging.getLogger(__name__)

STAGE_LABELS = {
    "queued": "Starting",
    "parse": "Reading your exports",
    "concepts": "Finding concepts",
    "link": "Linking conversations",
    "graph": "Building the graph",
    "embed": "Embedding passages",
    "save": "Saving the index",
    "vault": "Writing notes",
    "done": "Done",
}


class BuildBusy(RuntimeError):
    pass


class BuildManager:
    def __init__(self, on_done: Callable[[], None] | None = None,
                 runner: Callable[..., object] | None = None):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict = {"status": "idle"}
        self._on_done = on_done
        self._runner = runner  # injectable for tests

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        with self._lock:
            state = dict(self._state)
        if state.get("status") == "running" and state.get("started_at"):
            state["elapsed"] = round(time.time() - state["started_at"], 1)
        return state

    def start(self, settings: Settings) -> dict:
        with self._lock:
            if self.running:
                raise BuildBusy("A build is already running.")
            self._state = {"status": "running", "stage": "queued", "label": STAGE_LABELS["queued"],
                           "done": 0, "total": 0, "started_at": time.time(),
                           "output_dir": str(settings.vault_dir)}
            self._thread = threading.Thread(target=self._run, args=(copy.copy(settings),),
                                            name="kb-build", daemon=True)
            self._thread.start()
        return self.status()

    def _progress(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._state.update(stage=stage, label=STAGE_LABELS.get(stage, stage), done=done, total=total)

    def _run(self, settings: Settings) -> None:
        try:
            runner = self._runner
            if runner is None:
                from knowledgebase.pipeline import build as runner
            report = runner(settings, show_progress=False, progress=self._progress)
            result = {"status": "done", "stage": "done", "label": STAGE_LABELS["done"],
                      "report": str(report), "conversations": getattr(report, "conversations", None),
                      "nodes": getattr(report, "nodes", None)}
        except SystemExit as exc:  # the pipeline exits with a message for "nothing to build" / low memory
            result = {"status": "error", "error": str(exc.code) if exc.code not in (None, 1) else
                      "The build stopped (the machine was running low on memory)."}
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            log.error("Build failed:\n%s", traceback.format_exc())
            result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        with self._lock:
            self._state.update(result, finished_at=time.time())
        if result["status"] == "done" and self._on_done:
            self._on_done()
