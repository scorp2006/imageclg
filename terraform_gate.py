"""TDS GA7 — Terraform Plan Policy Gate.

Deterministic policy-as-code endpoint mounted on the main FastAPI app.
POST /terraform/plan -> {"decision": "approve"|"reject", "reason": "..."}

Rules checked in strict order; first applicable reason wins.
Workspace: prod-5xk0wt
Required labels: {"owner":"student-tykvj","environment":"production","cost_center":"cc-m9ad"}
"""
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

WORKSPACE = "prod-5xk0wt"
REQUIRED_LABELS = {
    "owner": "student-tykvj",
    "environment": "production",
    "cost_center": "cc-m9ad",
}
VALID_BACKENDS = {"gcs", "s3", "azurerm", "remote"}
DESTROY_PROTECTED = {"storage_bucket", "sql_database", "persistent_disk"}

# Exact pin: "6.2.1" or "= 6.2.1"; pessimistic: "~> 6.0". >=, *, latest are unpinned.
_EXACT_RE = re.compile(r"^\s*=?\s*\d+\.\d+\.\d+\s*$")
_PESSIMISTIC_RE = re.compile(r"^\s*~>\s*\d+(\.\d+)*\s*$")


def _is_str(v):
    return isinstance(v, str)


def evaluate(body: dict) -> dict:
    def reject(reason):
        return {"decision": "reject", "reason": reason}

    # 1. The request and nested objects must have the shown value types.
    if not isinstance(body, dict):
        return reject("INVALID_PLAN")
    env = body.get("environment")
    state = body.get("state")
    provider_version = body.get("providerVersion")
    destroy_approved = body.get("destroyApproved")
    resource = body.get("resource")

    if not _is_str(env):
        return reject("INVALID_PLAN")
    if not isinstance(state, dict):
        return reject("INVALID_PLAN")
    if not _is_str(provider_version):
        return reject("INVALID_PLAN")
    if not isinstance(destroy_approved, bool):
        return reject("INVALID_PLAN")
    if not isinstance(resource, dict):
        return reject("INVALID_PLAN")

    backend = state.get("backend")
    locked = state.get("locked")
    if not _is_str(backend) or not isinstance(locked, bool):
        return reject("INVALID_PLAN")

    address = resource.get("address")
    rtype = resource.get("type")
    action = resource.get("action")
    labels = resource.get("labels")
    secret = resource.get("secret")
    force_destroy = resource.get("forceDestroy")

    if not _is_str(address) or not _is_str(rtype) or not _is_str(action):
        return reject("INVALID_PLAN")
    if action not in ("create", "update", "delete"):
        return reject("INVALID_PLAN")
    if not isinstance(labels, dict):
        return reject("INVALID_PLAN")
    # secret must be null or a string (validated further in rule 6)
    if secret is not None and not _is_str(secret):
        return reject("INVALID_PLAN")
    if not isinstance(force_destroy, bool):
        return reject("INVALID_PLAN")

    # 2. Environment must exactly match the assigned workspace.
    if env != WORKSPACE:
        return reject("ENVIRONMENT_MISMATCH")

    # 3. State must use an approved backend and be locked.
    if backend not in VALID_BACKENDS or locked is not True:
        return reject("STATE_UNSAFE")

    # 4. Provider must be exact or pessimistically pinned.
    if not (_EXACT_RE.match(provider_version) or _PESSIMISTIC_RE.match(provider_version)):
        return reject("UNPINNED_PROVIDER")

    # 5. All three assigned labels present with exact values.
    for k, v in REQUIRED_LABELS.items():
        if labels.get(k) != v:
            return reject("MISSING_LABELS")

    # 6. secret must be null or a non-empty secret://... reference.
    if secret is not None:
        if not secret.startswith("secret://") or len(secret) <= len("secret://"):
            return reject("PLAINTEXT_SECRET")

    # 7. Deleting a protected resource requires destroyApproved: true.
    if action == "delete" and rtype in DESTROY_PROTECTED and destroy_approved is not True:
        return reject("DELETE_NOT_APPROVED")

    # 8. A production storage_bucket may never use forceDestroy: true.
    if rtype == "storage_bucket" and force_destroy is True:
        return reject("FORCE_DESTROY")

    return {"decision": "approve", "reason": "APPROVE"}


@router.post("/terraform/plan")
async def terraform_plan(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    try:
        return JSONResponse(evaluate(body))
    except Exception:
        return JSONResponse({"decision": "reject", "reason": "INVALID_PLAN"})


@router.get("/terraform/plan")
async def terraform_plan_info():
    return JSONResponse({"service": "TDS GA7 Terraform Plan Gate", "endpoint": "POST /terraform/plan"})
