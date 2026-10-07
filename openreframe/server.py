"""FastAPI front end for OpenReframe.

Scheduling model: one GPU, therefore one worker thread and a priority queue.
`preview` is cheap (depth is cached per image; a warp is ~100 ms) and outranks
`reframe`, which occupies the device for seconds. A preview submitted during a
reframe still waits for the current diffusion to finish — the queue orders
*new* work, it does not preempt a running one. Preempting would mean aborting a
generation the user is waiting on, which is the wrong trade.
"""
from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import __version__, config
from ._inpaint import GenerationCancelled
from .pipeline import MAX_SHIFT, Reframer

PRIO_PREVIEW, PRIO_REFRAME = 0, 1
MAX_IMAGES = 12
MAX_UPLOAD_PX = 4096


def _png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


@dataclass
class Job:
    id: str
    shift: tuple[float, float, float]
    image_id: str = ""
    status: str = "queued"          # queued|running|done|error|cancelled
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    created_s: float = field(default_factory=time.perf_counter)


class Store:
    """Small in-memory image + job registry with LRU eviction."""

    def __init__(self, max_images: int = MAX_IMAGES) -> None:
        self._imgs: OrderedDict[str, Image.Image] = OrderedDict()
        self._jobs: dict[str, Job] = OrderedDict()
        self._max = max_images
        self._lock = threading.Lock()

    def put_image(self, img: Image.Image) -> str:
        iid = uuid.uuid4().hex[:12]
        with self._lock:
            self._imgs[iid] = img
            self._jobs.clear()
            while len(self._imgs) > self._max:
                self._imgs.popitem(last=False)
        return iid

    def get_image(self, iid: str) -> Image.Image | None:
        with self._lock:
            return self._imgs.get(iid)

    def put_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get_job(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def depth(self) -> int:
        with self._lock:
            n = len(self._imgs)
        return n


class Service:
    """Owns the Reframer and the single worker thread."""

    def __init__(self) -> None:
        self.reframer = Reframer()
        self.store = Store()
        self._heap: list[tuple[int, int, str]] = []
        self._cv = threading.Condition()
        self._seq = 0
        self._worker: threading.Thread | None = None

    def start(self, warm: bool = False) -> None:
        """Spawn the worker. Models load lazily on first use.

        Loading is deliberately not done here: it costs 10-60 s and would block
        startup, leaving the UI unreachable with no health signal. With
        `warm=True` it happens on a background thread instead.
        """
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="gpu-worker")
            self._worker.start()
        if warm:
            threading.Thread(target=self.reframer.load, daemon=True,
                             name="warmup").start()

    def submit_preview(self, iid: str, shift: tuple[float, float, float]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], shift=shift, image_id=iid,
                  status="queued")
        self.store.put_job(job)
        self._push(PRIO_PREVIEW, job)
        return job

    def submit_reframe(self, iid: str, shift: tuple[float, float, float]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], shift=shift, image_id=iid,
                  status="queued")
        self.store.put_job(job)
        self._push(PRIO_REFRAME, job)
        return job

    def _push(self, prio: int, job: Job) -> None:
        with self._cv:
            self._seq += 1
            heappush(self._heap, (prio, self._seq, job.id))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                prio, _seq, jid = heappop(self._heap)
            job = self.store.get_job(jid)
            if job is None:
                continue
            if job.cancel.is_set():
                job.status = "cancelled"
                continue
            try:
                self._execute(job, is_preview=(prio == PRIO_PREVIEW))
            except GenerationCancelled:
                job.status = "cancelled"
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"

    def _execute(self, job: Job, is_preview: bool) -> None:
        img = self.store.get_image(job.image_id)
        if img is None:
            job.status = "error"
            job.error = ("source image expired (the store is in-memory and LRU "
                         "evicts; re-upload and retry)")
            return

        job.status = "running"
        if is_preview:
            out = self.reframer.preview(img, job.shift)
            job.result = {"kind": "preview", "warped": _png(out["warped"]),
                          "mask": _png(out["mask"]),
                          "overlay": _png(out["overlay"]),
                          "stats": out["stats"]}
            job.status = "done"
        else:
            def on_step(i: int, total: int) -> None:
                job.progress["step"] = i
                job.progress["total"] = total

            out = self.reframer.reframe(img, job.shift, on_step=on_step,
                                        cancel=job.cancel)
            job.result = {"kind": "reframe", "filled": _png(out["filled"]),
                          "overlay": _png(out["overlay"]),
                          "mask": _png(out["mask"]),
                          "stats": out["stats"]}
            job.status = "done"


def create_app(svc: Service | None = None, warm: bool = False) -> FastAPI:
    from contextlib import asynccontextmanager

    service = svc or Service()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.start(warm=warm)
        yield

    app = FastAPI(title="OpenReframe", version=__version__, lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict:
        h = service.reframer.health()
        h["worker_alive"] = service._worker is not None and service._worker.is_alive()
        h["queued"] = len(service._heap)
        return h

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "not a decodable image")
        if max(img.size) > MAX_UPLOAD_PX:
            raise HTTPException(413, f"image larger than {MAX_UPLOAD_PX}px long side")
        iid = service.store.put_image(img)
        return {"image_id": iid, "width": img.size[0], "height": img.size[1]}

    @app.get("/api/image/{iid}")
    def image(iid: str) -> Response:
        img = service.store.get_image(iid)
        if img is None:
            raise HTTPException(404, "unknown image id (evicted or restarted)")
        return Response(_png(img), media_type="image/png")

    @app.post("/api/preview")
    def preview(payload: dict = Body(...)) -> dict:
        iid = payload.get("image_id")
        shift = tuple(float(x) for x in payload.get("shift", (0, 0, 0)))
        if len(shift) != 3 or max(abs(s) for s in shift) > MAX_SHIFT:
            raise HTTPException(400, f"shift must be 3 floats within +/-{MAX_SHIFT}")
        job = service.submit_preview(iid, shift)
        return {"job_id": job.id}

    @app.post("/api/reframe")
    def reframe(payload: dict = Body(...)) -> dict:
        iid = payload.get("image_id")
        shift = tuple(float(x) for x in payload.get("shift", (0, 0, 0)))
        if len(shift) != 3 or max(abs(s) for s in shift) > MAX_SHIFT:
            raise HTTPException(400, f"shift must be 3 floats within +/-{MAX_SHIFT}")
        job = service.submit_reframe(iid, shift)
        return {"job_id": job.id}

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str) -> dict:
        job = service.store.get_job(jid)
        if job is None:
            raise HTTPException(404, "unknown job")
        out = {"id": job.id, "status": job.status, "error": job.error,
               "progress": dict(job.progress),
               "elapsed_s": round(time.perf_counter() - job.created_s, 2)}
        if job.status == "done" and job.result:
            r = dict(job.result)
            stats = r.pop("stats", {})
            r.pop("kind", None)
            out["stats"] = stats
            out["images"] = {k: f"/api/result/{job.id}/{k}" for k in r}
        return out

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str) -> dict:
        job = service.store.get_job(jid)
        if job is None:
            raise HTTPException(404, "unknown job")
        job.cancel.set()
        return {"id": jid, "cancel_requested": True}

    @app.get("/api/result/{jid}/{kind}")
    def result(jid: str, kind: str) -> Response:
        job = service.store.get_job(jid)
        if job is None or job.status != "done" or not job.result:
            raise HTTPException(404, "no result for that job")
        blob = job.result.get(kind)
        if blob is None:
            raise HTTPException(404, f"job has no '{kind}' image")
        return Response(blob, media_type="image/png")

    static_dir = config._paths.STATIC_DIR
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


app = create_app()   # preview-only by default; diffusion loads on first use


def main() -> None:
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(description="OpenReframe server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--inpaint", action="store_true",
                    help="preload the diffusion pipeline on a background thread "
                         "at startup instead of waiting for the first reframe")
    a = ap.parse_args()
    global app
    app = create_app(warm=a.inpaint)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()