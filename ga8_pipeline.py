"""TDS GA8 Q6 — Recover a Content-Addressed ML Pipeline.

POST /pipeline -> stateful controller, per-session, content-addressed DAG.
"""
import json
import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

# session -> state dict
#   {"revision": int, "inputs": {...}, "cache": {node: {"key":..,"artifact":..,"eventId":..}},
#    "attempt_state": {node: {"status":.., "attempt":.., "key":..}},
#    "event_ids": {eventId: canonical_json}, "terminal": {node: True}}
_SESSIONS = {}

DAG = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENTS = {"verify_data": None, "prepare": "verify_data", "train": "prepare",
           "evaluate": "train", "register": "evaluate", "publish": "register"}
REQUIRED_INPUTS = ["generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
                   "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
                   "schemaDigest", "publishConfig"]


def _sha256_hex(b): return hashlib.sha256(b).hexdigest()
def _utf8(s): return s.encode("utf-8")
def _compact(o): return json.dumps(o, separators=(",", ":"), ensure_ascii=False)
def _safe_int(v): return isinstance(v, int) and not isinstance(v, bool)


def _key_for(node, inputs, cache):
    """Content-addressed cache key; None if parent artifact not available."""
    def art(n):
        c = cache.get(n)
        return c["artifact"] if c else None
    if node == "verify_data":
        arr = [inputs["generation"], inputs["checksum"]]
    elif node == "prepare":
        arr = [inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]]
    elif node == "train":
        pa = art("prepare")
        if pa is None:
            return None
        arr = [pa, inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]]
    elif node == "evaluate":
        ta = art("train")
        if ta is None:
            return None
        arr = [ta, inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]]
    elif node == "register":
        ea = art("evaluate")
        if ea is None:
            return None
        arr = [ea, inputs["schemaDigest"]]
    elif node == "publish":
        ra = art("register")
        if ra is None:
            return None
        arr = [ra, inputs["publishConfig"]]
    else:
        return None
    return _sha256_hex(_utf8(_compact(arr)))


def _valid_event(e):
    if not isinstance(e, dict):
        return False
    if set(e.keys()) != {"eventId", "revision", "node", "attempt", "status", "key",
                          "artifactDigest", "receiptId"}:
        return False
    if not (isinstance(e.get("eventId"), str) and e["eventId"]):
        return False
    if not _safe_int(e.get("revision")) or e["revision"] <= 0:
        return False
    if e.get("node") not in DAG:
        return False
    if not _safe_int(e.get("attempt")) or e["attempt"] <= 0:
        return False
    if e.get("status") not in ("started", "succeeded", "retryable_failed", "terminal_failed"):
        return False
    # success requires non-empty artifact; others require null
    if e["status"] == "succeeded":
        if not (isinstance(e.get("artifactDigest"), str) and e["artifactDigest"]):
            return False
    else:
        if e.get("artifactDigest") is not None:
            return False
    return True


def evaluate(body):
    if not isinstance(body, dict):
        return ("400", "INVALID_REQUEST")
    session = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events", [])

    if not (isinstance(session, str) and session):
        return ("400", "INVALID_REQUEST")
    if not (_safe_int(revision) and revision > 0):
        return ("400", "INVALID_REQUEST")
    if not isinstance(inputs, dict):
        return ("400", "INVALID_REQUEST")
    for k in REQUIRED_INPUTS:
        if not (isinstance(inputs.get(k), str) and inputs[k]):
            return ("400", "INVALID_REQUEST")
    if not isinstance(events, list):
        return ("400", "INVALID_REQUEST")

    st = _SESSIONS.get(session)
    if st is None:
        st = {"revision": revision, "inputs": dict(inputs), "cache": {}, "attempt_state": {},
              "event_ids": {}, "terminal": {}}
        _SESSIONS[session] = st
    else:
        if revision == st["revision"]:
            # same revision + different input -> REVISION_CONFLICT
            if _compact(inputs) != _compact(st["inputs"]):
                return ("409", "REVISION_CONFLICT")
        elif revision > st["revision"]:
            # new revision: replace inputs, clear attempt/terminal, keep successful cache
            st["revision"] = revision
            st["inputs"] = dict(inputs)
            st["attempt_state"] = {}
            st["terminal"] = {}
        # older revision events are ignored later

    accepted = []
    ignored = []

    for e in events:
        # validate event structure
        if not _valid_event(e):
            ignored.append(e.get("eventId") if isinstance(e, dict) else None)
            continue
        eid = e["eventId"]
        canon = _compact(e)
        # event id global within session
        if eid in st["event_ids"]:
            if st["event_ids"][eid] == canon:
                ignored.append(eid)  # exact replay ignored
                continue
            else:
                return ("409", "EVENT_ID_CONFLICT")
        # wrong revision -> ignore
        if e["revision"] != st["revision"]:
            ignored.append(eid)
            continue
        node = e["node"]
        # receipt rule
        if node in ("register", "publish") and e["status"] == "succeeded":
            if e.get("receiptId") != f"receipt:{node}:{e.get('key')}":
                ignored.append(eid); continue
        else:
            if e.get("receiptId") is not None:
                ignored.append(eid); continue
        # current key for node (must match)
        current_key = _key_for(node, st["inputs"], st["cache"])
        if current_key is None:
            ignored.append(eid); continue  # parent unavailable
        if e.get("key") != current_key:
            ignored.append(eid); continue

        # already-cached success?
        cached = st["cache"].get(node)
        if cached is not None and cached["key"] == current_key:
            if e["status"] == "succeeded":
                if e["artifactDigest"] != cached["artifact"]:
                    return ("409", "EVIDENCE_CONFLICT")
                ignored.append(eid); continue
            else:
                return ("409", "STATUS_CONFLICT")

        # terminal?
        if st["terminal"].get(node):
            return ("409", "STATUS_CONFLICT")

        # attempt state machine
        cur = st["attempt_state"].get(node)
        status = e["status"]; attempt = e["attempt"]

        def accept():
            st["event_ids"][eid] = canon
            accepted.append(eid)

        if cur is None:
            if status == "started" and attempt == 1:
                st["attempt_state"][node] = {"status": "started", "attempt": 1, "key": current_key}
                accept()
            else:
                ignored.append(eid)
        else:
            cstatus = cur["status"]; cattempt = cur["attempt"]
            if attempt < cattempt:
                ignored.append(eid); continue
            if cstatus == "started" and attempt == cattempt and status in (
                    "succeeded", "retryable_failed", "terminal_failed"):
                if status == "succeeded":
                    st["cache"][node] = {"key": current_key, "artifact": e["artifactDigest"], "eventId": eid}
                    st["attempt_state"].pop(node, None)
                elif status == "retryable_failed":
                    st["attempt_state"][node] = {"status": "retryable_failed", "attempt": attempt, "key": current_key}
                elif status == "terminal_failed":
                    st["terminal"][node] = True
                    st["attempt_state"].pop(node, None)
                accept()
            elif cstatus == "retryable_failed" and status == "started" and attempt == cattempt + 1:
                st["attempt_state"][node] = {"status": "started", "attempt": attempt, "key": current_key}
                accept()
            elif cstatus in ("started", "retryable_failed"):
                return ("409", "STATUS_CONFLICT")
            else:
                ignored.append(eid)

    # ---- build node outputs ----
    nodes_out = []
    upstream_broken = False  # terminal or pending upstream
    upstream_terminal = False
    for node in DAG:
        key = _key_for(node, st["inputs"], st["cache"])
        cached = st["cache"].get(node)
        att = st["attempt_state"].get(node)
        terminal = st["terminal"].get(node)

        dep_digests = _dep_digests(node, st["inputs"], st["cache"], key)
        action = "block"; reason = []; trig = []

        if upstream_terminal:
            action = "block"; reason = ["UPSTREAM_TERMINAL"]
        elif upstream_broken:
            action = "block"; reason = ["UPSTREAM_PENDING"]
        elif cached is not None:
            action = "reuse"; reason = ["CACHE_HIT"]; trig = [cached["eventId"]]
        elif terminal:
            action = "block"; reason = ["TERMINAL_FAILURE"]
            upstream_terminal = True
        elif att is not None and att["status"] == "started":
            action = "block"; reason = ["RUNNING"]
            # triggered by start event
            trig = [k for k, v in st["event_ids"].items()]  # placeholder; refine below
            trig = _start_event_id(st, node, key)
            upstream_broken = True
        elif att is not None and att["status"] == "retryable_failed":
            action = "rerun"; reason = ["RETRYABLE_FAILURE"]
            upstream_broken = True
        else:
            # ready without cache
            if key is None:
                action = "block"; reason = ["UPSTREAM_PENDING"]
                upstream_broken = True
            else:
                action = "rerun"; reason = ["CACHE_MISS"]
                upstream_broken = True

        nodes_out.append({
            "node": node,
            "action": action,
            "reasonCodes": reason,
            "dependencyDigests": dep_digests,
            "triggeringEventIds": trig,
        })

    return ("ok", {
        "revision": st["revision"],
        "acceptedEventIds": accepted,
        "ignoredEventIds": [i for i in ignored if i is not None],
        "nodes": nodes_out,
    })


def _start_event_id(st, node, key):
    for eid, canon in st["event_ids"].items():
        try:
            e = json.loads(canon)
        except Exception:
            continue
        if e.get("node") == node and e.get("status") == "started" and e.get("key") == key:
            # the latest accepted start for the current attempt
            pass
    # return the start event matching current attempt
    att = st["attempt_state"].get(node)
    best = []
    for eid, canon in st["event_ids"].items():
        e = json.loads(canon)
        if (e.get("node") == node and e.get("status") == "started"
                and e.get("key") == key and att and e.get("attempt") == att["attempt"]):
            best.append(eid)
    return best


def _dep_digests(node, inputs, cache, key):
    d = {}
    if node == "verify_data":
        d = {"generation": inputs["generation"], "checksum": inputs["checksum"]}
    elif node == "prepare":
        d = {"canonicalData": inputs["canonicalData"], "prepareCode": inputs["prepareCode"],
             "prepareConfig": inputs["prepareConfig"]}
    elif node == "train":
        pa = cache.get("prepare")
        d = {"prepareArtifact": pa["artifact"] if pa else None,
             "trainCode": inputs["trainCode"], "trainConfig": inputs["trainConfig"],
             "runtime": inputs["runtime"]}
    elif node == "evaluate":
        ta = cache.get("train")
        d = {"trainArtifact": ta["artifact"] if ta else None,
             "canonicalData": inputs["canonicalData"], "evaluateCode": inputs["evaluateCode"],
             "evaluateConfig": inputs["evaluateConfig"]}
    elif node == "register":
        ea = cache.get("evaluate")
        d = {"evaluateArtifact": ea["artifact"] if ea else None, "schemaDigest": inputs["schemaDigest"]}
    elif node == "publish":
        ra = cache.get("register")
        d = {"registerArtifact": ra["artifact"] if ra else None, "publishConfig": inputs["publishConfig"]}
    d["cacheKey"] = key
    return d


@router.post("/pipeline")
async def pipeline(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)
    try:
        status, result = evaluate(body)
    except Exception:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)
    if status == "400":
        return JSONResponse({"error": result}, status_code=409)
    if status == "409":
        return JSONResponse({"error": result}, status_code=409)
    return JSONResponse(result)


@router.get("/pipeline")
async def pipeline_info():
    return JSONResponse({"service": "TDS GA8 Pipeline", "endpoint": "POST /pipeline"})
