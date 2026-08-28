"""TDS GA8 Q2 — Repair a Leakage-Safe BigQuery ML Experiment.

POST /bqml -> stateful two-phase experiment gate (select / evaluate).
"""
import json
import re
import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_RUN_STORE = {}  # runId -> (select_input_fingerprint, stored_select_response)

_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.(\d{1,3}))?(Z|([+-])(\d{2}):(\d{2}))$")


def _sha256_hex(b): return hashlib.sha256(b).hexdigest()
def _utf8(s): return s.encode("utf-8")
def _compact(o): return json.dumps(o, separators=(",", ":"), ensure_ascii=False)
def _is_ne_str(v): return isinstance(v, str) and len(v) > 0
def _safe_int(v): return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2**53 - 1


def _parse_ts(s):
    m = _TS_RE.match(s) if isinstance(s, str) else None
    if not m:
        return None
    Y, Mo, D, h, mi, se = (int(m[i]) for i in range(1, 7))
    frac = m[8]; ms = int((frac + "000")[:3]) if frac else 0
    import datetime
    try:
        if m[9] == "Z":
            off = 0
        else:
            sign = 1 if m[10] == "+" else -1
            oh, om = int(m[11]), int(m[12])
            if oh > 14 or (oh == 14 and om != 0) or om > 59:
                return None
            off = sign * (oh * 3600 + om * 60)
        dt = datetime.datetime(Y, Mo, D, h, mi, se, ms * 1000, tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return None
    return int(dt.timestamp() * 1000) - off * 1000


def evaluate_select(body):
    run_id = body.get("runId")
    forbidden = body.get("forbiddenFeatures")
    limit = body.get("numTrialsLimit")
    rows = body.get("rows")
    trials = body.get("trials")

    codes = set()

    # ---- hard input validation -> INVALID_INPUT nulls everything ----
    hard_invalid = not (
        _is_ne_str(run_id) and len(run_id) <= 128
        and isinstance(forbidden, list)
        and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
        and isinstance(rows, list) and len(rows) > 0
        and isinstance(trials, list))

    if not hard_invalid:
        # row IDs unique; trial IDs unique
        row_ids = [r.get("id") for r in rows if isinstance(r, dict)]
        trial_ids = [t.get("trialId") for t in trials if isinstance(t, dict)]
        if len(row_ids) != len(set(row_ids)):
            hard_invalid = True
        if len(trial_ids) != len(set(trial_ids)):
            hard_invalid = True
        # per-row validity: split, valid timestamps, version safe int, status valid
        for r in rows:
            if not (isinstance(r, dict) and r.get("split") in ("TRAIN", "EVAL")
                    and _parse_ts(r.get("eventTime")) is not None
                    and _safe_int(r.get("version"))):
                hard_invalid = True
                break
        for t in trials:
            if not (isinstance(t, dict) and t.get("status") in ("SUCCEEDED", "FAILED")
                    and _safe_int(t.get("trialId"))):
                hard_invalid = True
                break

    if hard_invalid:
        return {
            "runId": run_id if _is_ne_str(run_id) else None,
            "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [],
            "featureNames": [], "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    # trial-limit (soft: keeps dataset output)
    if len(trials) > limit:
        codes.add("TRIAL_LIMIT_EXCEEDED")

    # ---- dedup + features + splits (computed ALWAYS) ----
    groups = {}
    for r in rows:
        ent = r.get("entity"); et = _parse_ts(r.get("eventTime"))
        groups.setdefault((ent, et), []).append(r)
    retained = []
    for key, grp in groups.items():
        winner = sorted(grp, key=lambda r: (-(r.get("version", 0)), _utf8(r.get("id", ""))))[0]
        retained.append(winner)

    forbidden_set = set(forbidden)
    feature_names = []
    if retained:
        common = None
        for r in retained:
            feats = r.get("features", {})
            names = set(feats.keys()) if isinstance(feats, dict) else set()
            common = names if common is None else (common & names)
        common = common or set()
        eligible_feats = []
        for fn in common:
            if fn in forbidden_set:
                continue
            ok = True
            for r in retained:
                fv = r.get("features", {}).get(fn)
                av = _parse_ts(fv.get("availableAt")) if isinstance(fv, dict) else None
                pt = _parse_ts(r.get("predictionTime"))
                if av is None or pt is None or av > pt:
                    ok = False
                    break
            if ok:
                eligible_feats.append(fn)
        feature_names = sorted(eligible_feats, key=lambda s: _utf8(s))

    train_ids = sorted([r["id"] for r in retained if r.get("split") == "TRAIN"], key=lambda s: _utf8(s))
    eval_ids = sorted([r["id"] for r in retained if r.get("split") == "EVAL"], key=lambda s: _utf8(s))

    dataset_digest = _sha256_hex(_utf8(_compact(
        {"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": feature_names})))

    # ---- trial selection ----
    eligible_trials = []
    for t in trials:
        if t.get("status") == "SUCCEEDED":
            m = t.get("evalMetric")
            if isinstance(m, (int, float)) and not isinstance(m, bool) and m == m and abs(m) != float("inf"):
                eligible_trials.append(t)
    selected_trial = None
    if not eligible_trials:
        codes.add("NO_SUCCESSFUL_TRIAL")
    else:
        best = sorted(eligible_trials, key=lambda t: (-t["evalMetric"], t["trialId"]))[0]
        selected_trial = best["trialId"]

    # any code -> selectedTrialId null (but keep dataset output)
    if codes:
        selected_trial = None

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted(codes, key=lambda s: _utf8(s)),
    }


def evaluate_eval(body):
    run_id = body.get("runId")
    sel_trial = body.get("selectedTrialId")
    digest = body.get("datasetDigest")
    metric_floor = body.get("metricFloor")
    req_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_proc = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    codes = set()

    # basic input validity — empty/non-list rows => INVALID_INPUT (per oracle)
    input_ok = (_is_ne_str(run_id)
                and isinstance(metric_floor, (int, float)) and not isinstance(metric_floor, bool)
                and 0.0 <= metric_floor <= 1.0
                and isinstance(req_slices, dict)
                and isinstance(rows, list) and len(rows) > 0
                and _safe_int(bytes_proc) and _safe_int(max_bytes))
    if not input_ok:
        codes.add("INVALID_INPUT")

    # lineage: runId + selectedTrialId + digest must match a stored successful selection
    lineage_ok = True
    stored = _RUN_STORE.get(run_id)
    if stored is None:
        lineage_ok = False
    else:
        sr = stored[1]
        if (sr.get("selectedTrialId") is None
                or sr.get("selectedTrialId") != sel_trial
                or not (isinstance(digest, str) and re.match(r"^[0-9a-f]{64}$", digest))
                or sr.get("datasetDigest") != digest):
            lineage_ok = False
    if not lineage_ok:
        codes.add("INVALID_LINEAGE")

    # test rows validity
    rows_valid = isinstance(rows, list) and len(rows) > 0
    if rows_valid:
        for r in rows:
            if not (isinstance(r, dict)
                    and r.get("label") in (0, 1) and r.get("prediction") in (0, 1)
                    and _is_ne_str(r.get("slice"))):
                rows_valid = False
                break

    test_metric = None
    critical_pass = False

    # compute testMetric + slices whenever rows are valid & non-empty (independent of lineage)
    if rows_valid and "INVALID_INPUT" not in codes:
        n = len(rows)
        correct = sum(1 for r in rows if r["label"] == r["prediction"])
        test_metric = round(correct / n, 12)
        slice_tot = {}; slice_cor = {}
        for r in rows:
            s = r["slice"]
            slice_tot[s] = slice_tot.get(s, 0) + 1
            if r["label"] == r["prediction"]:
                slice_cor[s] = slice_cor.get(s, 0) + 1
        if test_metric < metric_floor:
            codes.add("AGGREGATE_FLOOR")
        all_slices_ok = True
        for sname, floor in (req_slices.items() if isinstance(req_slices, dict) else []):
            if sname not in slice_tot:
                codes.add(f"MISSING_SLICE:{sname}"); all_slices_ok = False
            else:
                sacc = round(slice_cor.get(sname, 0) / slice_tot[sname], 12)
                if sacc < floor:
                    codes.add(f"SLICE_FLOOR:{sname}"); all_slices_ok = False
        critical_pass = all_slices_ok
    elif isinstance(rows, list) and len(rows) > 0 and not rows_valid:
        codes.add("INVALID_TEST_ROW")

    # byte gate always applies (when byte counts valid)
    if _safe_int(bytes_proc) and _safe_int(max_bytes) and bytes_proc > max_bytes:
        codes.add("BYTE_LIMIT")

    # criticalSlicePass false for invalid input/lineage/bad row/missing slice/failed floor
    if ("INVALID_INPUT" in codes or "INVALID_LINEAGE" in codes or "INVALID_TEST_ROW" in codes
            or any(c.startswith("MISSING_SLICE") or c.startswith("SLICE_FLOOR") for c in codes)):
        critical_pass = False

    admit = (not codes) and rows_valid
    decision = "admit" if admit else "reject"

    return {
        "runId": run_id,
        "selectedTrialId": sel_trial,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_pass,
        "decision": decision,
        "bytesProcessed": bytes_proc if _safe_int(bytes_proc) else None,
        "reasonCodes": sorted(codes, key=lambda s: _utf8(s)),
    }


@router.post("/bqml")
async def bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    phase = body.get("phase")
    try:
        if phase == "select":
            result = evaluate_select(body)
            run_id = body.get("runId")
            if _is_ne_str(run_id):
                fp = _compact({k: body.get(k) for k in
                               ("forbiddenFeatures", "numTrialsLimit", "rows", "trials")})
                if run_id in _RUN_STORE:
                    if _RUN_STORE[run_id][0] != fp:
                        return JSONResponse({"error": "RUN_ID_CONFLICT"}, status_code=409)
                    return JSONResponse(_RUN_STORE[run_id][1])
                _RUN_STORE[run_id] = (fp, result)
            return JSONResponse(result)
        elif phase == "evaluate":
            return JSONResponse(evaluate_eval(body))
        else:
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)


@router.get("/bqml")
async def bqml_info():
    return JSONResponse({"service": "TDS GA8 BQML", "endpoint": "POST /bqml"})
