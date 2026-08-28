"""GA8 request/response capture — records grader probes for offline analysis.

Add `app.add_middleware(GA8LogMiddleware)` and `app.include_router(router)`.
GET /ga8-logs?path=/bqml  -> the captured request/response pairs (newest last).
GET /ga8-logs/clear       -> wipe the buffer.
"""
import json
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

GA8_PATHS = {
    "/build-corpus", "/bqml", "/promote", "/adapt",
    "/quantize", "/pipeline", "/verify-bundle",
}

_LOG = deque(maxlen=400)  # ring buffer of dicts


class GA8LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path not in GA8_PATHS or request.method != "POST":
            return await call_next(request)

        req_body = b""
        try:
            req_body = await request.body()
        except Exception:
            pass

        # rebuild receive so downstream can read the body again
        async def _receive():
            return {"type": "http.request", "body": req_body, "more_body": False}
        request._receive = _receive

        response = await call_next(request)

        # capture response body
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk

        entry = {
            "t": round(time.time(), 3),
            "path": path,
            "status": response.status_code,
        }
        try:
            entry["request"] = json.loads(req_body.decode("utf-8"))
        except Exception:
            entry["request_raw"] = req_body.decode("utf-8", "replace")[:4000]
        try:
            entry["response"] = json.loads(resp_body.decode("utf-8"))
        except Exception:
            entry["response_raw"] = resp_body.decode("utf-8", "replace")[:4000]
        _LOG.append(entry)

        return Response(content=resp_body, status_code=response.status_code,
                        headers=dict(response.headers), media_type=response.media_type)


@router.get("/ga8-logs")
async def ga8_logs(path: str = None, limit: int = 200):
    items = list(_LOG)
    if path:
        items = [e for e in items if e.get("path") == path]
    return JSONResponse({"count": len(items), "entries": items[-limit:]})


@router.get("/ga8-logs/clear")
async def ga8_logs_clear():
    _LOG.clear()
    return JSONResponse({"cleared": True})
