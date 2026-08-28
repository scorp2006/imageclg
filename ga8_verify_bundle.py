"""TDS GA8 Q7 — Publish a Verifiable Model Bundle and Model Card.

POST /verify-bundle -> {"decision":"admit|reject","violations":[...],"inventoryDigest":"..."}
Deterministic verifier for an untrusted UTF-8 model bundle.
"""
import json
import hashlib
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]
UNSAFE_EXTS = (".bin", ".pt", ".pth", ".pkl", ".pickle")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_FIELDS = ["task", "datasetDigest", "codeDigest", "trainingConfigDigest",
                   "modelArtifactDigest", "evaluationArtifactDigest"]


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _utf8(s: str) -> bytes:
    return s.encode("utf-8")


def _is_nonempty_str(v):
    return isinstance(v, str) and len(v) > 0


def evaluate(body):
    violations = set()

    if not isinstance(body, dict):
        return None  # signals HTTP 400
    policy = body.get("policy")
    files = body.get("files")
    if not isinstance(policy, dict) or not isinstance(files, dict):
        return None  # HTTP 400

    # ---- policy validity ----
    req_slices = policy.get("requiredSlices")
    policy_ok = True
    if not (isinstance(req_slices, list) and len(req_slices) > 0
            and all(_is_nonempty_str(s) for s in req_slices)
            and len(set(req_slices)) == len(req_slices)):
        policy_ok = False
    for k in ("license", "intendedUse", "limitations"):
        if not _is_nonempty_str(policy.get(k)):
            policy_ok = False
    if not policy_ok:
        violations.add("INVALID_POLICY")

    # files must be str->str
    files_ok = all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())

    # ---- required files present ----
    for name in REQUIRED_FILES:
        if name not in files or not isinstance(files.get(name), str):
            violations.add(f"MISSING_FILE:{name}")

    # ---- unsafe weight extensions / extra files handled with inventory ----
    for name in files:
        low = name.lower()
        if low.endswith(UNSAFE_EXTS):
            violations.add("UNSAFE_WEIGHTS")

    # ---- inventory recomputation ----
    inventory_digest = None
    # Build recomputed inventory of every file EXCEPT inventory.json
    inv_items = []
    for name in sorted(files.keys(), key=lambda s: _utf8(s)):
        if name == "inventory.json":
            continue
        content = files[name]
        if not isinstance(content, str):
            continue
        b = _utf8(content)
        inv_items.append({"name": name, "bytes": len(b), "sha256": _sha256_hex(b)})
    recomputed = [{"name": it["name"], "bytes": it["bytes"], "sha256": it["sha256"]} for it in inv_items]
    recomputed_json = json.dumps(recomputed, separators=(",", ":"), ensure_ascii=False)
    inventory_digest = _sha256_hex(_utf8(recomputed_json))

    # parse supplied inventory.json
    if "inventory.json" in files and isinstance(files["inventory.json"], str):
        try:
            supplied_inv = json.loads(files["inventory.json"])
            if not isinstance(supplied_inv, list):
                violations.add("INVALID_JSON:inventory.json")
                supplied_inv = None
        except Exception:
            violations.add("INVALID_JSON:inventory.json")
            supplied_inv = None
        if supplied_inv is not None:
            # compare against recomputed; also detect untracked/extra files
            supplied_names = set()
            valid_entries = True
            for e in supplied_inv:
                if not (isinstance(e, dict) and list(e.keys()) == ["name", "bytes", "sha256"]):
                    valid_entries = False
                    break
                supplied_names.add(e.get("name"))
            if not valid_entries:
                violations.add("INVENTORY_MISMATCH")
            else:
                # UNTRACKED_FILE: a file present in bundle (except inventory.json) not listed
                actual_names = set(n for n in files if n != "inventory.json")
                if supplied_names != actual_names:
                    # files in bundle not tracked -> UNTRACKED_FILE; mismatch of content -> INVENTORY_MISMATCH
                    if actual_names - supplied_names:
                        violations.add("UNTRACKED_FILE")
                    if supplied_names - actual_names:
                        violations.add("INVENTORY_MISMATCH")
                # exact recompute equality
                if json.dumps(supplied_inv, separators=(",", ":"), ensure_ascii=False) != recomputed_json:
                    violations.add("INVENTORY_MISMATCH")

    # ---- adapter_config.json ----
    if "adapter_config.json" in files and isinstance(files["adapter_config.json"], str):
        try:
            ac = json.loads(files["adapter_config.json"])
            if not isinstance(ac, dict):
                violations.add("INVALID_JSON:adapter_config.json")
                ac = None
        except Exception:
            violations.add("INVALID_JSON:adapter_config.json")
            ac = None
        if ac is not None:
            r = ac.get("r")
            tm = ac.get("target_modules")
            ok = isinstance(r, int) and not isinstance(r, bool) and r >= 1
            ok = ok and isinstance(tm, list) and len(tm) > 0 \
                and all(_is_nonempty_str(x) for x in tm) and len(set(tm)) == len(tm)
            if not ok:
                violations.add("INVALID_ADAPTER_CONFIG")

    # ---- training_manifest.json ----
    manifest = None
    if "training_manifest.json" in files and isinstance(files["training_manifest.json"], str):
        try:
            manifest = json.loads(files["training_manifest.json"])
            if not isinstance(manifest, dict):
                violations.add("INVALID_JSON:training_manifest.json")
                manifest = None
        except Exception:
            violations.add("INVALID_JSON:training_manifest.json")
            manifest = None
        if manifest is not None:
            base_rev = manifest.get("baseRevision")
            if not (isinstance(base_rev, str) and _HEX40.match(base_rev)):
                violations.add("MUTABLE_BASE_REVISION")
            for f in MANIFEST_FIELDS:
                if not _is_nonempty_str(manifest.get(f)):
                    violations.add(f"MISSING_MANIFEST_FIELD:{f}")

    # ---- evaluation.json ----
    evaluation = None
    if "evaluation.json" in files and isinstance(files["evaluation.json"], str):
        try:
            evaluation = json.loads(files["evaluation.json"])
            if not isinstance(evaluation, dict):
                violations.add("INVALID_JSON:evaluation.json")
                evaluation = None
        except Exception:
            violations.add("INVALID_JSON:evaluation.json")
            evaluation = None

    # ---- digest bindings (need adapter_model.safetensors + evaluation.json bytes) ----
    if manifest is not None:
        # recompute modelArtifactDigest from adapter_model.safetensors
        if "adapter_model.safetensors" in files and isinstance(files["adapter_model.safetensors"], str):
            model_digest = _sha256_hex(_utf8(files["adapter_model.safetensors"]))
            if manifest.get("modelArtifactDigest") != model_digest:
                violations.add("MODEL_ARTIFACT_MISMATCH")
        # recompute evaluationArtifactDigest from evaluation.json exact bytes
        if "evaluation.json" in files and isinstance(files["evaluation.json"], str):
            eval_digest = _sha256_hex(_utf8(files["evaluation.json"]))
            if manifest.get("evaluationArtifactDigest") != eval_digest:
                violations.add("EVALUATION_ARTIFACT_MISMATCH")

    # evaluation binds model digest + slice ranges
    if evaluation is not None:
        # binds the model digest (evaluation must reference the same model artifact digest)
        if manifest is not None:
            model_digest = _sha256_hex(_utf8(files["adapter_model.safetensors"])) \
                if isinstance(files.get("adapter_model.safetensors"), str) else None
            if evaluation.get("modelArtifactDigest") != manifest.get("modelArtifactDigest"):
                violations.add("EVALUATION_DIGEST_MISMATCH")
        agg = evaluation.get("aggregate")
        if not (isinstance(agg, (int, float)) and not isinstance(agg, bool)
                and agg == agg and abs(agg) != float("inf") and 0.0 <= agg <= 1.0):
            violations.add("INVALID_AGGREGATE")
        slices = evaluation.get("slices")
        if isinstance(slices, dict) and policy_ok:
            for s in req_slices:
                if s not in slices:
                    violations.add(f"MISSING_SLICE:{s}")
                else:
                    val = slices[s]
                    if not (isinstance(val, (int, float)) and not isinstance(val, bool)
                            and val == val and abs(val) != float("inf") and 0.0 <= val <= 1.0):
                        violations.add(f"SLICE_RANGE:{s}")
        elif policy_ok and not isinstance(slices, dict):
            for s in req_slices:
                violations.add(f"MISSING_SLICE:{s}")

    # ---- Model card in README.md ----
    if "README.md" in files and isinstance(files["README.md"], str):
        readme = files["README.md"]
        markers = _find_markers(readme)
        if len(markers) == 0:
            violations.add("MODEL_CARD_COUNT")
            violations.add("MISSING_MODEL_CARD")
        elif len(markers) > 1:
            violations.add("MODEL_CARD_COUNT")
        else:
            payload = markers[0]
            try:
                card = json.loads(payload)
                if not isinstance(card, dict):
                    violations.add("INVALID_MODEL_CARD")
                    card = None
            except Exception:
                violations.add("INVALID_MODEL_CARD")
                card = None
            if card is not None:
                # match machine manifests + policy
                mismatch = False
                if manifest is not None:
                    if card.get("task") != manifest.get("task"):
                        mismatch = True
                    if card.get("baseRevision") != manifest.get("baseRevision"):
                        mismatch = True
                    if card.get("datasetDigest") != manifest.get("datasetDigest"):
                        mismatch = True
                    if card.get("modelArtifactDigest") != manifest.get("modelArtifactDigest"):
                        mismatch = True
                if policy_ok:
                    if card.get("license") != policy.get("license"):
                        mismatch = True
                    if card.get("intendedUse") != policy.get("intendedUse"):
                        mismatch = True
                    if card.get("limitations") != policy.get("limitations"):
                        mismatch = True
                if mismatch:
                    violations.add("MODEL_CARD_MISMATCH")

    decision = "admit" if not violations else "reject"
    return {
        "decision": decision,
        "violations": sorted(violations, key=lambda s: _utf8(s)),
        "inventoryDigest": inventory_digest,
    }


def _find_markers(readme: str):
    """Return list of JSON payload strings between `<!-- tds-model-card ` and ` -->`.
    Braces inside JSON strings are ordinary; we find markers by the literal delimiters."""
    prefix = "<!-- tds-model-card "
    suffix = " -->"
    out = []
    idx = 0
    while True:
        start = readme.find(prefix, idx)
        if start == -1:
            break
        end = readme.find(suffix, start + len(prefix))
        if end == -1:
            break
        payload = readme[start + len(prefix):end]
        out.append(payload)
        idx = end + len(suffix)
    return out


@router.post("/verify-bundle")
async def verify_bundle(request: Request):
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


@router.get("/verify-bundle")
async def verify_bundle_info():
    return JSONResponse({"service": "TDS GA8 Verify Bundle", "endpoint": "POST /verify-bundle"})
