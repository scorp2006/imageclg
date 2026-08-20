"""TDS — LLM Action Firewall (deterministic, no LLM, no phrase lists).

POST /action-firewall -> {"decision": "allow"|"block", "reason": "..."}

Checks in strict order, returning the FIRST failing reason:
  1. top-level schema
  2. tool allowlist
  3. selected tool's argument schema
  4. tenant scope
  5. exact email domain
  6. human approval
  7. HTML safety
"""
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

TENANT = "tenant-f4t3lnm"
EMAIL_DOMAIN = "notify-k15vkht.example"

ALLOWED_TOOLS = {"search", "lookup_record", "send_email", "render_html"}


def _is_str(v):
    return isinstance(v, str)


def evaluate(body: dict) -> dict:
    if not isinstance(body, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    provenance = body.get("provenance")
    human_approved = body.get("humanApproved")
    action = body.get("action")

    # 1. Top-level schema.
    if provenance not in ("trusted", "untrusted"):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}
    if not isinstance(human_approved, bool):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}
    if not isinstance(action, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}
    tool = action.get("tool")
    args = action.get("args")
    if not _is_str(tool) or not isinstance(args, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    # 2. Tool allowlist.
    if tool not in ALLOWED_TOOLS:
        return {"decision": "block", "reason": "TOOL_NOT_ALLOWED"}

    # 3. Selected tool's argument schema (exact key sets + types).
    keys = set(args.keys())

    if tool == "search":
        if keys != {"query"} or not _is_str(args.get("query")):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        if not (1 <= len(args["query"]) <= 200):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        return {"decision": "allow", "reason": "ALLOW"}

    if tool == "lookup_record":
        if keys != {"tenantId", "recordId"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        if not _is_str(args.get("tenantId")) or not _is_str(args.get("recordId")):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        if len(args["recordId"]) == 0:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        # 4. Tenant scope.
        if args["tenantId"] != TENANT:
            return {"decision": "block", "reason": "TENANT_SCOPE"}
        return {"decision": "allow", "reason": "ALLOW"}

    if tool == "send_email":
        if keys != {"to", "subject", "body"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        if not all(_is_str(args.get(k)) for k in ("to", "subject", "body")):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        # 5. Exact email recipient domain.
        to = args["to"]
        if "@" not in to or to.rsplit("@", 1)[1] != EMAIL_DOMAIN:
            return {"decision": "block", "reason": "EGRESS_DENIED"}
        # 6. Human approval.
        if human_approved is not True:
            return {"decision": "block", "reason": "APPROVAL_REQUIRED"}
        return {"decision": "allow", "reason": "ALLOW"}

    if tool == "render_html":
        if keys != {"html"} or not _is_str(args.get("html")):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        # 7. HTML safety: block scripts, iframes, inline event handlers, javascript: URLs.
        html = args["html"]
        low = html.lower()
        if "<script" in low or "</script" in low:
            return {"decision": "block", "reason": "UNSAFE_OUTPUT"}
        if "<iframe" in low:
            return {"decision": "block", "reason": "UNSAFE_OUTPUT"}
        if re.search(r"on\w+\s*=", low):  # inline event handlers e.g. onclick=
            return {"decision": "block", "reason": "UNSAFE_OUTPUT"}
        if "javascript:" in low:
            return {"decision": "block", "reason": "UNSAFE_OUTPUT"}
        return {"decision": "allow", "reason": "ALLOW"}

    return {"decision": "block", "reason": "TOOL_NOT_ALLOWED"}


@router.post("/action-firewall")
async def action_firewall(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    try:
        return JSONResponse(evaluate(body))
    except Exception:
        return JSONResponse({"decision": "block", "reason": "INVALID_SCHEMA"})


@router.get("/action-firewall")
async def action_firewall_info():
    return JSONResponse({"service": "TDS LLM Action Firewall", "endpoint": "POST /action-firewall"})
