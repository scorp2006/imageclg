"""TDS GA8 Q3 — Promote the Right MLflow Model from Verifiable Evidence.

POST /promote -> deterministic model-registry promotion gate.
Stateful only for the alias mutation idempotency (persist by (championVersion, selectedVersion)).
"""
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_INT_STR = re.compile(r"^[1-9][0-9]*$")  # canonical positive integer string, no leading zero

# Persisted alias mutations: key -> aliasMutation object (idempotent replay).
_ALIAS_STORE = {}


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _in01(v):
    return _finite(v) and 0.0 <= v <= 1.0


def _valid_ts(s):
    if not isinstance(s, str):
        return False
    return bool(re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$", s))


def _parse_ts(s):
    """Return epoch seconds (float) for a valid instant, else None."""
    import datetime
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.(\d{1,3}))?(Z|([+-])(\d{2}):(\d{2}))$", s)
    if not m:
        return None
    Y, Mo, D, h, mi, se = int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])
    frac = m[8]
    ms = int((frac + "000")[:3]) if frac else 0
    try:
        if m[9] == "Z":
            off = 0
        else:
            sign = 1 if m[10] == "+" else -1
            off = sign * (int(m[11]) * 3600 + int(m[12]) * 60)
        dt = datetime.datetime(Y, Mo, D, h, mi, se, ms * 1000, tzinfo=datetime.timezone.utc)
        return dt.timestamp() - off
    except (ValueError, TypeError):
        return None


def evaluate(body):
    if not isinstance(body, dict):
        return None
    policy = body.get("policy")
    versions = body.get("versions")
    champ = body.get("championVersion")
    as_of = body.get("asOf")
    if not isinstance(policy, dict) or not isinstance(versions, list) or not isinstance(champ, str):
        return None

    failed = {}  # version-string -> set of codes

    def add(v, code):
        failed.setdefault(v, set()).add(code)

    # ---- policy validity ----
    policy_valid = True
    ds = policy.get("datasetDigest"); sc = policy.get("schemaDigest")
    if not (isinstance(ds, str) and ds) or not (isinstance(sc, str) and sc):
        policy_valid = False
    for k in ("maxAgeSeconds", "maxSizeBytes"):
        v = policy.get(k)
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
            policy_valid = False
    for k in ("accuracyFloor", "minImprovement"):
        if not _in01(policy.get(k)):
            policy_valid = False
    if not (_finite(policy.get("maxLatencyMs")) and policy.get("maxLatencyMs") >= 0):
        policy_valid = False
    rs = policy.get("requiredSlices")
    if not isinstance(rs, dict):
        policy_valid = False
    else:
        for sv in rs.values():
            if not _in01(sv):
                policy_valid = False

    as_of_ts = _parse_ts(as_of) if _valid_ts(as_of) else None

    # ---- per-version validity & eligibility ----
    seen = {}
    dup_or_noncanon = set()
    for ver in versions:
        if not isinstance(ver, dict):
            continue
        vid = ver.get("version")
        if not isinstance(vid, str) or not _INT_STR.match(vid):
            # non-canonical version id
            key = vid if isinstance(vid, str) else None
            if key is not None:
                add(key, "INVALID_VERSION")
                dup_or_noncanon.add(key)
            continue
        if vid in seen:
            add(vid, "DUPLICATE_VERSION")
            dup_or_noncanon.add(vid)
        seen.setdefault(vid, []).append(ver)

    # mark duplicates
    for vid, lst in seen.items():
        if len(lst) > 1:
            add(vid, "DUPLICATE_VERSION")
            dup_or_noncanon.add(vid)

    lookup = {vid: lst[0] for vid, lst in seen.items() if len(lst) == 1 and vid not in dup_or_noncanon}

    if not policy_valid:
        # every version gets INVALID_POLICY too
        for vid in lookup:
            add(vid, "INVALID_POLICY")

    eligible = []
    for vid, ver in lookup.items():
        codes_before = set(failed.get(vid, set()))
        _check_version(ver, vid, policy, policy_valid, as_of_ts, add)
        codes_after = set(failed.get(vid, set()))
        # eligible if this version accumulated NO gate codes at all
        if not codes_after:
            eligible.append(vid)

    # ---- champion evidence ----
    champ_ver = lookup.get(champ)
    champ_valid = (champ in eligible)

    # ---- ranking ----
    def sort_key(vid):
        ev = lookup[vid]["evaluation"]
        return (-ev["accuracy"], ev["latencyMs"], ev["sizeBytes"], int(vid))
    ranked = sorted(eligible, key=sort_key)

    action = "retain"
    selected = None
    alias_mutation = None
    evidence = None

    if not champ_valid:
        action = "block"
        selected = None
    else:
        # challenger = best-ranked eligible (may be champ itself)
        best = ranked[0] if ranked else None
        champ_acc = lookup[champ]["evaluation"]["accuracy"]
        if best is not None and best != champ:
            challenger_acc = lookup[best]["evaluation"]["accuracy"]
            improvement = round(challenger_acc - champ_acc, 12)
            if improvement >= policy.get("minImprovement", 0):
                action = "promote"
                selected = best
                alias_mutation = {"alias": "champion", "version": best}
            else:
                action = "retain"
                selected = champ
        else:
            action = "retain"
            selected = champ

    if selected is not None:
        evidence = lookup[selected]["evaluation"]

    # idempotent alias persistence
    if action == "promote" and alias_mutation is not None:
        skey = f"{champ}->{selected}"
        _ALIAS_STORE[skey] = alias_mutation

    # failedGates: every input version, sorted unique codes
    failed_out = {}
    for vid in seen:
        failed_out[vid] = sorted(failed.get(vid, set()), key=lambda s: s.encode("utf-8"))
    for vid in failed:
        if vid not in failed_out:
            failed_out[vid] = sorted(failed[vid], key=lambda s: s.encode("utf-8"))

    return {
        "action": action,
        "championVersion": champ,
        "selectedVersion": selected,
        "eligibleVersions": ranked,  # ranked order: acc desc, latency asc, size asc, version asc
        "failedGates": failed_out,
        "aliasMutation": alias_mutation,
        "evidence": evidence,
    }


def _check_version(ver, vid, policy, policy_valid, as_of_ts, add):
    """Add gate codes for one version."""
    ev = ver.get("evaluation")
    if not isinstance(ev, dict):
        add(vid, "MISSING_EVALUATION")
        return

    created = ev.get("createdAt")
    if not _valid_ts(created):
        add(vid, "INVALID_TIMESTAMP")
        return
    created_ts = _parse_ts(created)

    acc = ev.get("accuracy"); lat = ev.get("latencyMs"); size = ev.get("sizeBytes")
    # finite checks
    non_finite = False
    if not _finite(acc) or not _finite(lat):
        non_finite = True
    if not (isinstance(size, int) and not isinstance(size, bool)):
        non_finite = True
    if non_finite:
        add(vid, "NON_FINITE")
        return
    # metric range
    if not _in01(acc):
        add(vid, "METRIC_RANGE")
    if lat < 0 or size < 0:
        add(vid, "METRIC_RANGE")

    # timestamps
    if as_of_ts is not None and created_ts is not None:
        if created_ts > as_of_ts:
            add(vid, "FUTURE_EVALUATION")
        elif created_ts < as_of_ts - policy.get("maxAgeSeconds", 0):
            add(vid, "STALE_EVALUATION")

    # digest bindings
    if ev.get("artifactDigest") != ver.get("artifactDigest"):
        add(vid, "ARTIFACT_MISMATCH")
    if policy_valid:
        if ev.get("datasetDigest") != policy.get("datasetDigest"):
            add(vid, "DATASET_MISMATCH")
        if ev.get("schemaDigest") != policy.get("schemaDigest"):
            add(vid, "SCHEMA_MISMATCH")

    # aggregate gates
    if policy_valid and _in01(acc):
        if acc < policy.get("accuracyFloor", 0):
            add(vid, "ACCURACY_FLOOR")
    if policy_valid and _finite(lat):
        if lat > policy.get("maxLatencyMs", float("inf")):
            add(vid, "LATENCY_LIMIT")
    if policy_valid and isinstance(size, int):
        if size > policy.get("maxSizeBytes", float("inf")):
            add(vid, "SIZE_LIMIT")

    # slices
    slices = ev.get("slices")
    if policy_valid and isinstance(policy.get("requiredSlices"), dict):
        for sname, floor in policy["requiredSlices"].items():
            if not isinstance(slices, dict) or sname not in slices:
                add(vid, f"MISSING_SLICE:{sname}")
            else:
                sval = slices[sname]
                if not _in01(sval):
                    add(vid, f"SLICE_RANGE:{sname}")
                elif sval < floor:
                    add(vid, f"SLICE_FLOOR:{sname}")


@router.post("/promote")
async def promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    try:
        result = evaluate(body)
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if result is None:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    return JSONResponse(result)


@router.get("/promote")
async def promote_info():
    return JSONResponse({"service": "TDS GA8 Promote", "endpoint": "POST /promote"})
