"""TDS GA7 — CI/CD Container Release Gate.

Deterministic policy endpoint mounted on the main FastAPI app.
POST /release-gate -> {"decision": "promote"|"block", "violations": [...]}
"""
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def evaluate(body: dict) -> dict:
    violations = set()
    wf = body.get("workflow", {}) or {}
    img = body.get("image", {}) or {}
    target = body.get("target")
    event = body.get("event")
    ref = body.get("ref")

    # Permissions must be EXACTLY least privilege — no more, no fewer scopes.
    perms = wf.get("permissions", {}) or {}
    if perms != {"contents": "read", "packages": "write", "id-token": "none"}:
        violations.add("EXCESS_PERMISSION")

    # A pull request must never use pull_request_target.
    if wf.get("trigger") == "pull_request_target":
        violations.add("UNSAFE_PR_TRIGGER")

    # Tests must pass, the whole matrix must finish, failFast must be false.
    if wf.get("testsPassed") is not True:
        violations.add("TESTS_INCOMPLETE")
    if wf.get("matrixComplete") is not True:
        violations.add("TESTS_INCOMPLETE")
    if wf.get("failFast") is not False:
        violations.add("TESTS_INCOMPLETE")

    # Third-party actions must be pinned to a full 40-char lowercase hex SHA.
    for action in wf.get("actions", []) or []:
        if action.get("owner") == "actions":
            continue  # first-party actions may use a version tag
        if not _SHA_RE.match(action.get("ref", "") or ""):
            violations.add("MUTABLE_ACTION")

    # Image hardening rules.
    if img.get("multiStage") is not True:
        violations.add("SINGLE_STAGE_IMAGE")
    if img.get("runsAsRoot") is not False:
        violations.add("ROOT_RUNTIME")
    if img.get("secretMode") not in ("none", "buildkit"):
        violations.add("SECRET_IN_LAYER")
    if (img.get("criticalVulnerabilities") or 0) > 0:
        violations.add("CRITICAL_CVE")
    if img.get("digestPinned") is not True:
        violations.add("UNPINNED_IMAGE")

    # Production-only extra requirements.
    if target == "production":
        if not (event == "push" and ref == "refs/heads/main"):
            violations.add("INVALID_PRODUCTION_REF")
        if wf.get("environmentApproval") is not True:
            violations.add("APPROVAL_REQUIRED")

    return {
        "decision": "promote" if not violations else "block",
        "violations": sorted(violations),
    }


@router.post("/release-gate")
async def release_gate(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return JSONResponse(evaluate(body))


@router.get("/release-gate")
async def release_gate_info():
    return JSONResponse({"service": "TDS GA7 Release Gate", "endpoint": "POST /release-gate"})
