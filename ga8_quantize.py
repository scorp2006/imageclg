"""TDS GA8 Q5 — Quantize and Admit a Model Under Explicit Constraints.

POST /quantize -> stateful two-phase candidate-admission API (freeze / select).
"""
import json
import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_FREEZE_STORE = {}  # freezeId -> (input_fingerprint, stored_response)


def _sha256_hex(b): return hashlib.sha256(b).hexdigest()
def _utf8(s): return s.encode("utf-8")
def _compact(o): return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def _is_ne_str(v): return isinstance(v, str) and len(v) > 0


def _fingerprint(body):
    return _compact(body)


def evaluate_freeze(body):
    freeze_id = body.get("freezeId")
    cal = body.get("calibrationDigest")
    tok = body.get("tokenizerDigest")
    allowed = body.get("allowedUnsupportedReasons")
    candidates = body.get("candidates")

    # HTTP 400 only for: bad freezeId, an empty/non-array candidate list. Everything else
    # is handled per-candidate (invalid status) so a single bad candidate never 400s the batch.
    if not (_is_ne_str(freeze_id) and len(freeze_id) <= 128):
        return ("400", None)
    if not isinstance(candidates, list) or len(candidates) == 0:
        return ("400", None)

    allowed = allowed if isinstance(allowed, list) else []
    allowed_set = set(a for a in allowed if _is_ne_str(a))
    out_candidates = []
    for c in candidates:
        codes = set()
        name = c.get("name")
        files = c.get("files")
        files_valid = isinstance(files, dict) and len(files) > 0 \
            and all(_is_ne_str(k) for k in files.keys()) \
            and all(isinstance(v, str) for v in files.values()) \
            and len(set(files.keys())) == len(files)
        status = None
        inventory = []
        total_bytes = None
        package_digest = None

        reason = c.get("unsupportedReason")
        if not files_valid:
            codes.add("INVALID_INPUT")
            status = "invalid"
        else:
            # build inventory
            inv = []
            for fn in sorted(files.keys(), key=lambda s: _utf8(s)):
                b = _utf8(files[fn])
                inv.append({"name": fn, "bytes": len(b), "sha256": _sha256_hex(b)})
            inventory = inv
            total_bytes = sum(e["bytes"] for e in inv)
            package_digest = _sha256_hex(_utf8(_compact(inv)))

            if reason is not None:
                # any reason makes status depend on allow-list
                if reason in allowed_set:
                    status = "unsupported"
                else:
                    codes.add("UNALLOWED_UNSUPPORTED_REASON")
                    status = "invalid"
            else:
                # must be loadable and match digests
                ok = True
                if c.get("loadable") is not True:
                    codes.add("NOT_LOADABLE"); ok = False
                if c.get("calibrationDigest") != cal:
                    codes.add("CALIBRATION_MISMATCH"); ok = False
                if c.get("tokenizerDigest") != tok:
                    codes.add("TOKENIZER_MISMATCH"); ok = False
                status = "frozen" if ok else "invalid"

        if status == "invalid" and not files_valid:
            inventory = []; total_bytes = None; package_digest = None

        out_candidates.append({
            "name": name,
            "status": status,
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": sorted(codes, key=lambda s: _utf8(s)),
        })

    out_candidates.sort(key=lambda c: _utf8(c["name"]))
    return ("ok", {"freezeId": freeze_id, "candidates": out_candidates})


def evaluate_select(body):
    freeze_id = body.get("freezeId")
    candidates = body.get("candidates")
    policy = body.get("policy")
    latencies = body.get("latencies")
    rows = body.get("rows")

    if not _is_ne_str(freeze_id):
        return ("400", None)
    if not isinstance(candidates, list) or len(candidates) == 0:
        return ("400", None)
    if not isinstance(rows, list) or not isinstance(policy, dict):
        return ("400", None)

    codes = set()
    stored = _FREEZE_STORE.get(freeze_id)
    if stored is None:
        codes.add("NOT_FROZEN")
    else:
        stored_resp = stored[1]
        if candidates != stored_resp["candidates"]:
            codes.add("INVALID_LINEAGE")

    order = policy.get("candidateOrder")
    max_bytes = policy.get("maxBytes")
    agg_floor = policy.get("aggregateFloor")
    req_slices = policy.get("requiredSlices")
    max_lat = policy.get("maxLatencyMs")

    policy_valid = (isinstance(order, list) and all(_is_ne_str(x) for x in order)
                    and isinstance(max_bytes, int) and not isinstance(max_bytes, bool) and max_bytes >= 0
                    and isinstance(agg_floor, (int, float)) and 0.0 <= agg_floor <= 1.0
                    and isinstance(req_slices, dict)
                    and isinstance(max_lat, (int, float)) and max_lat >= 0)
    if not policy_valid:
        codes.add("INVALID_POLICY")

    cand_names = [c.get("name") for c in candidates if isinstance(c, dict)]
    if policy_valid and set(cand_names) != set(order):
        codes.add("INVALID_LINEAGE")

    results = []
    if not codes:
        for c in candidates:
            name = c["name"]
            rcodes = set()
            # recompute inventory
            recomputed_bytes = c.get("totalBytes")
            manifest_ok = c.get("status") == "frozen"
            if not manifest_ok:
                rcodes.add("INVALID_MANIFEST")
            # predictions validity + accuracy
            agg = None; slice_acc = {}
            preds_valid = True
            correct = 0; n = 0
            slice_tot = {}; slice_cor = {}
            for row in rows:
                p = row.get("predictions", {}).get(name) if isinstance(row.get("predictions"), dict) else None
                label = row.get("label")
                sl = row.get("slice")
                if p not in (0, 1) or label not in (0, 1) or not _is_ne_str(sl):
                    preds_valid = False
                    break
                n += 1
                if p == label:
                    correct += 1
                slice_tot[sl] = slice_tot.get(sl, 0) + 1
                if p == label:
                    slice_cor[sl] = slice_cor.get(sl, 0) + 1
            if not preds_valid:
                rcodes.add("INVALID_PREDICTIONS")
                agg = None
            else:
                agg = round(correct / n, 12) if n else None
                for sl in slice_tot:
                    slice_acc[sl] = round(slice_cor.get(sl, 0) / slice_tot[sl], 12)
            lat = latencies.get(name) if isinstance(latencies, dict) else None
            lat_valid = isinstance(lat, (int, float)) and not isinstance(lat, bool) and lat >= 0

            admitted = False
            if preds_valid and manifest_ok and policy_valid:
                if agg is not None and agg >= agg_floor:
                    slices_ok = True
                    for sname, floor in req_slices.items():
                        if sname not in slice_acc:
                            rcodes.add(f"MISSING_SLICE:{sname}"); slices_ok = False
                        elif slice_acc[sname] < floor:
                            rcodes.add(f"SLICE_FLOOR:{sname}"); slices_ok = False
                    if agg < agg_floor:
                        rcodes.add("AGGREGATE_FLOOR")
                    tb = c.get("totalBytes")
                    if isinstance(tb, int) and tb > max_bytes:
                        rcodes.add("SIZE_LIMIT")
                    if lat_valid and lat > max_lat:
                        rcodes.add("LATENCY_LIMIT")
                    admitted = slices_ok and not rcodes
                else:
                    if agg is not None and agg < agg_floor:
                        rcodes.add("AGGREGATE_FLOOR")

            results.append({
                "name": name,
                "aggregate": agg,
                "slices": slice_acc if preds_valid else {},
                "totalBytes": c.get("totalBytes") if isinstance(c.get("totalBytes"), int) else None,
                "latencyMs": lat if lat_valid else None,
                "admitted": admitted,
                "reasonCodes": sorted(rcodes, key=lambda s: _utf8(s)),
            })
        # order results by candidateOrder, fallback utf8
        order_idx = {n: i for i, n in enumerate(order)} if policy_valid else {}
        results.sort(key=lambda r: (order_idx.get(r["name"], len(order_idx)), _utf8(r["name"])))

    # choose admitted: smaller bytes, lower latency, then candidate order
    selected = None
    manifest = None
    admitted_list = [r for r in results if r["admitted"]]
    if admitted_list:
        order_idx = {n: i for i, n in enumerate(order)} if policy_valid else {}
        best = sorted(admitted_list, key=lambda r: (
            r["totalBytes"] if r["totalBytes"] is not None else float("inf"),
            r["latencyMs"] if r["latencyMs"] is not None else float("inf"),
            order_idx.get(r["name"], len(order_idx))))[0]
        selected = best["name"]
        manifest = {"name": best["name"], "totalBytes": best["totalBytes"]}

    return ("ok", {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": manifest,
    })


@router.post("/quantize")
async def quantize(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    phase = body.get("phase")
    try:
        if phase == "freeze":
            status, result = evaluate_freeze(body)
            if status == "400":
                return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
            fid = body.get("freezeId")
            fp = _fingerprint(body)
            if fid in _FREEZE_STORE:
                if _FREEZE_STORE[fid][0] != fp:
                    return JSONResponse({"error": "FREEZE_ID_CONFLICT"}, status_code=409)
                return JSONResponse(_FREEZE_STORE[fid][1])
            _FREEZE_STORE[fid] = (fp, result)
            return JSONResponse(result)
        elif phase == "select":
            status, result = evaluate_select(body)
            if status == "400":
                return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
            return JSONResponse(result)
        else:
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)


@router.get("/quantize")
async def quantize_info():
    return JSONResponse({"service": "TDS GA8 Quantize", "endpoint": "POST /quantize"})
