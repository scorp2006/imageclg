"""TDS GA8 Q1 — Build an Immutable, Leakage-Safe Training Corpus.

POST /build-corpus -> deterministic JSONL corpus service.
"""
import json
import re
import hashlib
import unicodedata
import zlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_URI_RE = re.compile(r"^gs://[^/]+/.+$")
_GEN_RE = re.compile(r"^-?\d+$|^\d+$")  # decimal string
_CRC_RE = re.compile(r"^[0-9a-f]{8}$")
_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.(\d{1,3}))?(Z|([+-])(\d{2}):(\d{2}))$")

SCHEMA_ID = "training-v1"


def _sha256_hex(b): return hashlib.sha256(b).hexdigest()
def _utf8(s): return s.encode("utf-8")


def _valid_decimal(s):
    return isinstance(s, str) and bool(re.match(r"^-?\d+$", s))


def _crc32c_hex(data: bytes) -> str:
    # crc32c (Castagnoli). Implement table-based.
    return format(_crc32c(data), "08x")


_CRC32C_TABLE = []
def _init_crc_table():
    poly = 0x82F63B78
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if (c & 1) else (c >> 1)
        _CRC32C_TABLE.append(c)
_init_crc_table()

def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


def _parse_ts(s):
    """Return (epoch_ms:int, valid:bool). Validates calendar + offset."""
    m = _TS_RE.match(s) if isinstance(s, str) else None
    if not m:
        return None, False
    Y, Mo, D, h, mi, se = (int(m[i]) for i in range(1, 7))
    frac = m[8]; ms = int((frac + "000")[:3]) if frac else 0
    # validate calendar
    import datetime
    try:
        if m[9] == "Z":
            off_min = 0
        else:
            sign = 1 if m[10] == "+" else -1
            oh = int(m[11]); om = int(m[12])
            if oh > 14 or (oh == 14 and om != 0) or om > 59:
                return None, False
            off_min = sign * (oh * 60 + om)
        if not (0 <= h <= 23 and 0 <= mi <= 59 and 0 <= se <= 59):
            return None, False
        dt = datetime.datetime(Y, Mo, D, h, mi, se, ms * 1000, tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return None, False
    epoch_ms = int(dt.timestamp() * 1000) - off_min * 60 * 1000
    return epoch_ms, True


def _to_utc_iso(s):
    """Normalize to YYYY-MM-DDTHH:mm:ss.sssZ."""
    epoch_ms, ok = _parse_ts(s)
    if not ok:
        return None
    import datetime
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{epoch_ms % 1000:03d}Z"


def _canon(s):
    """NFKC, lowercase, trim, collapse Unicode whitespace to single ASCII space."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # collapse any run of unicode whitespace to a single space, and trim
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def _wordset(s):
    """lowercase Unicode letter/number word-set."""
    words = re.findall(r"[^\W_]+", s.lower(), flags=re.UNICODE)
    return set(words)


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _valid_policy(policy):
    if not isinstance(policy, dict):
        return False
    ct = policy.get("contaminationThreshold")
    if not (isinstance(ct, (int, float)) and not isinstance(ct, bool)
            and ct == ct and abs(ct) != float("inf") and 0.0 <= ct <= 1.0):
        return False
    for k in ("minTime", "maxTime"):
        _, ok = _parse_ts(policy.get(k))
        if not ok:
            return False
    return True


def _valid_row(row):
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
        return False
    if not all(isinstance(row[k], str) for k in ("id", "entity", "eventTime", "text")):
        return False
    rev = row["revision"]
    if not (isinstance(rev, int) and not isinstance(rev, bool) and rev >= 0 and rev <= 2**53 - 1):
        return False
    _, ok = _parse_ts(row["eventTime"])
    if not ok:
        return False
    return True


def evaluate(body):
    if not isinstance(body, dict):
        return None
    policy = body.get("policy")
    objects = body.get("objects")
    if policy is None or not isinstance(objects, list):
        return None

    rejected_objects = []
    lineage = []
    retained_rows = []  # list of dicts with canonical fields + source id

    for obj in objects:
        codes = set()
        uri = obj.get("uri") if isinstance(obj, dict) else None
        # object validation
        if not isinstance(obj, dict):
            rejected_objects.append({"uri": None, "reasonCodes": ["JSONL_INVALID"]})
            continue
        if not (isinstance(uri, str) and _URI_RE.match(uri)):
            codes.add("URI_INVALID")
        gen = obj.get("generation"); fgen = obj.get("fetchedGeneration")
        if not _valid_decimal(gen) or not _valid_decimal(fgen):
            codes.add("GENERATION_INVALID")
        elif gen != fgen:
            codes.add("GENERATION_MISMATCH")
        crc = obj.get("crc32c")
        content = obj.get("content")
        crc_syntax_ok = isinstance(crc, str) and bool(_CRC_RE.match(crc))
        if not crc_syntax_ok:
            codes.add("CRC32C_INVALID")
        elif isinstance(content, str):
            if _crc32c_hex(_utf8(content)) != crc:
                codes.add("CRC32C_MISMATCH")
        if obj.get("schemaId") != SCHEMA_ID:
            codes.add("SCHEMA_INVALID")
        if not isinstance(content, str):
            codes.add("SCHEMA_INVALID")

        # parse JSONL
        rows = None
        if isinstance(content, str):
            parsed_rows = []
            jsonl_ok = True
            shape_ok = True
            nonblank = 0
            for line in content.split("\n"):
                if line.strip() == "":
                    continue
                nonblank += 1
                try:
                    r = json.loads(line)
                except Exception:
                    jsonl_ok = False
                    break
                if not _valid_row(r):
                    shape_ok = False
                else:
                    parsed_rows.append(r)
            if not jsonl_ok:
                codes.add("JSONL_INVALID")
            if nonblank == 0:
                codes.add("SCHEMA_INVALID")  # empty file
            if jsonl_ok and not shape_ok:
                codes.add("SCHEMA_INVALID")
            if jsonl_ok and shape_ok and nonblank > 0:
                rows = parsed_rows

        if codes:
            rejected_objects.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sorted(set(codes), key=lambda s: _utf8(s)),
            })
            continue

        # object accepted -> lineage + retain its rows
        lineage.append({"uri": uri, "generation": gen, "crc32c": crc, "schemaId": obj.get("schemaId")})
        for r in rows:
            retained_rows.append({
                "id": r["id"],
                "entity": _canon(r["entity"]),
                "eventTime": _to_utc_iso(r["eventTime"]),
                "revision": r["revision"],
                "text": _canon(r["text"]),
            })

    # ---- dedup by [entity, eventTime, text]: keep highest revision, then utf8-smallest id ----
    rejected_rows = []
    groups = {}
    for row in retained_rows:
        key = (row["entity"], row["eventTime"], row["text"])
        groups.setdefault(key, []).append(row)
    kept = []
    for key, grp in groups.items():
        if len(grp) == 1:
            kept.append(grp[0])
            continue
        # winner: highest revision, then utf8-smallest id
        winner = sorted(grp, key=lambda r: (-r["revision"], _utf8(r["id"])))[0]
        for r in grp:
            if r is not winner:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["DUPLICATE"]})
        kept.append(winner)

    # ---- policy validity ----
    policy_valid = _valid_policy(policy)
    if not policy_valid:
        for r in kept:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["POLICY_INVALID"]})
        kept = []
    else:
        min_ms, _ = _parse_ts(policy["minTime"])
        max_ms, _ = _parse_ts(policy["maxTime"])
        survivors = []
        for r in kept:
            ems, _ = _parse_ts(r["eventTime"])
            if ems < min_ms or ems > max_ms:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
            else:
                survivors.append(r)
        kept = survivors

    # ---- split by bucket = firstByte(SHA256(UTF8(entity))) % 10 ----
    def bucket(entity):
        h = hashlib.sha256(_utf8(entity)).digest()
        return h[0] % 10
    train, val, test = [], [], []
    for r in kept:
        b = bucket(r["entity"])
        if b <= 5:
            train.append(r)
        elif b <= 7:
            val.append(r)
        else:
            test.append(r)

    # ---- contamination: val/test row vs any train row, wordset Jaccard >= threshold ----
    if policy_valid:
        threshold = policy["contaminationThreshold"]
        train_wordsets = [_wordset(r["text"]) for r in train]
        def contaminated(row):
            ws = _wordset(row["text"])
            for tw in train_wordsets:
                if _jaccard(ws, tw) >= threshold:
                    return True
            return False
        new_val, new_test = [], []
        for r in val:
            if contaminated(r):
                rejected_rows.append({"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                new_val.append(r)
        for r in test:
            if contaminated(r):
                rejected_rows.append({"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                new_test.append(r)
        val, test = new_val, new_test

    # ---- serialize each split: sort by utf8 id, then compact row JSON for tie ----
    def row_json(r):
        return json.dumps({"id": r["id"], "entity": r["entity"], "eventTime": r["eventTime"],
                           "revision": r["revision"], "text": r["text"]},
                          separators=(",", ":"), ensure_ascii=False)

    def build_split(rows):
        srt = sorted(rows, key=lambda r: (_utf8(r["id"]), _utf8(row_json(r))))
        out_rows = [json.loads(row_json(r)) for r in srt]  # ordered dicts preserve key order
        # digest over exact bytes: each row compact + newline
        payload = "".join(row_json(r) + "\n" for r in srt)
        digest = _sha256_hex(_utf8(payload))
        return out_rows, digest

    train_rows, train_dig = build_split(train)
    val_rows, val_dig = build_split(val)
    test_rows, test_dig = build_split(test)

    # ---- sort rejected objects / rows / lineage ----
    def sort_rejected(lst, keyfield):
        # dedup reason codes already; sort by utf8 key then compact json tie
        for e in lst:
            e["reasonCodes"] = sorted(set(e["reasonCodes"]), key=lambda s: _utf8(s))
        return sorted(lst, key=lambda e: (_utf8(str(e[keyfield]) if e[keyfield] is not None else ""),
                                          _utf8(json.dumps(e, separators=(",", ":"), ensure_ascii=False))))

    # merge rejected_rows by id? spec: each rejected row is one entry; multiple codes possible per id
    # Combine codes per id
    row_map = {}
    order = []
    for e in rejected_rows:
        rid = e["id"]
        if rid not in row_map:
            row_map[rid] = set()
            order.append(rid)
        row_map[rid].update(e["reasonCodes"])
    merged_rows = [{"id": rid, "reasonCodes": sorted(row_map[rid], key=lambda s: _utf8(s))} for rid in row_map]

    rejected_objects = sort_rejected(rejected_objects, "uri")
    merged_rows = sorted(merged_rows, key=lambda e: (_utf8(e["id"]),
                        _utf8(json.dumps(e, separators=(",", ":"), ensure_ascii=False))))
    lineage = sorted(lineage, key=lambda e: (_utf8(e["uri"]),
                    _utf8(json.dumps(e, separators=(",", ":"), ensure_ascii=False))))

    return {
        "splits": {"train": train_rows, "validation": val_rows, "test": test_rows},
        "rejectedObjects": rejected_objects,
        "rejectedRows": merged_rows,
        "digests": {"train": train_dig, "validation": val_dig, "test": test_dig},
        "lineage": lineage,
    }


@router.post("/build-corpus")
async def build_corpus(request: Request):
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


@router.get("/build-corpus")
async def build_corpus_info():
    return JSONResponse({"service": "TDS GA8 Build Corpus", "endpoint": "POST /build-corpus"})
