"""TDS GA7 — LLM Output Handling Gate (OWASP LLM05).

Deterministic output sanitizer mounted on the main FastAPI app.
POST /sanitize-output -> {"safe": bool, "reason": "..."}

Allowlist of external hosts (EXACT hostname match, no subdomains):
  cdn-um64ko8.example
  app-iw7u49d.example
"""
import re
import html as _html
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

ALLOWED_HOSTS = {"cdn-um64ko8.example", "app-iw7u49d.example"}
CHANNELS = {"html", "markdown", "url", "sql", "shell"}
MAX_LEN = 20000

_NAMED_ENTITIES = {
    "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'", "&amp;": "&",
}


# ---------- decoding (rule: percent -> HTML entities -> \uXXXX, once) ----------
def _decode_once(s: str) -> str:
    out = s
    # percent-escapes
    def pct(m):
        try:
            return bytes([int(m.group(1), 16)]).decode("latin-1")
        except Exception:
            return m.group(0)
    out = re.sub(r"%([0-9A-Fa-f]{2})", pct, out)
    # HTML entities: numeric &#NN; / &#xNN; and the five named
    def num_ent(m):
        try:
            if m.group(1):  # hex
                return chr(int(m.group(1), 16))
            return chr(int(m.group(2)))
        except Exception:
            return m.group(0)
    out = re.sub(r"&#[xX]([0-9A-Fa-f]+);|&#(\d+);", num_ent, out)
    for ent, ch in _NAMED_ENTITIES.items():
        out = out.replace(ent, ch)
    # \uXXXX escapes
    def uesc(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)
    out = re.sub(r"\\u([0-9A-Fa-f]{4})", uesc, out)
    return out


# ---------- URL extraction per channel ----------
def _extract_urls(channel: str, text: str):
    if channel == "html":
        # values of quoted src= and href= attributes
        return re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', text, flags=re.I) + \
               re.findall(r"(?:src|href)\s*=\s*'([^']*)'", text, flags=re.I)
    if channel == "markdown":
        # target inside ](...)
        return [m.strip() for m in re.findall(r"\]\(([^)]*)\)", text)]
    if channel == "url":
        return [text.strip()]
    return []


def _is_absolute(u: str) -> bool:
    # protocol-relative //host/path counts as absolute
    if u.startswith("//"):
        return True
    return bool(urlsplit(u).scheme) and bool(urlsplit(u).netloc)


def _url_scheme(u: str) -> str:
    if u.startswith("//"):
        return "https"
    return urlsplit(u).scheme.lower()


def _url_host(u: str) -> str:
    if u.startswith("//"):
        u = "https:" + u
    return (urlsplit(u).hostname or "").lower()


# ---------- individual checks ----------
def _has_script_tag(text: str) -> bool:
    return bool(re.search(r"<\s*(script|iframe|object|embed)\b", text, flags=re.I))


def _has_event_handler(text: str) -> bool:
    # an on...= attribute
    return bool(re.search(r"\bon\w+\s*=", text, flags=re.I))


def _has_dangerous_scheme(channel: str, text: str) -> bool:
    # literal javascript:/data:/vbscript: (optional whitespace before colon), case-insensitive
    if re.search(r"(javascript|data|vbscript)\s*:", text, flags=re.I):
        return True
    # any extracted URL using a scheme other than http/https
    for u in _extract_urls(channel, text):
        if _is_absolute(u):
            sch = _url_scheme(u)
            if sch and sch not in ("http", "https"):
                return True
    return False


def _has_external_exfil(channel: str, text: str) -> bool:
    for u in _extract_urls(channel, text):
        if _is_absolute(u):
            host = _url_host(u)
            if host not in ALLOWED_HOSTS:
                return True
    return False


def _has_sql_metachar(text: str) -> bool:
    low = text.lower()
    if "'" in text or '"' in text or ";" in text:
        return True
    if "--" in text or "/*" in text:
        return True
    if "union" in low:
        return True
    if re.search(r"\bor\b\s*1\s*=\s*1", low) or "or 1=1" in low:
        return True
    return False


def _has_shell_metachar(text: str) -> bool:
    if any(c in text for c in [";", "&", "|", "`", "<", ">"]):
        return True
    if "$(" in text or "${" in text:
        return True
    return False


def _channel_check(channel: str, text: str):
    """Return a reason string if unsafe on this channel, else None."""
    if channel == "html":
        if _has_script_tag(text):
            return "SCRIPT_TAG"
        if _has_event_handler(text):
            return "EVENT_HANDLER"
        if _has_dangerous_scheme(channel, text):
            return "DANGEROUS_SCHEME"
        if _has_external_exfil(channel, text):
            return "EXTERNAL_EXFIL"
        return None
    if channel in ("markdown", "url"):
        if _has_dangerous_scheme(channel, text):
            return "DANGEROUS_SCHEME"
        if _has_external_exfil(channel, text):
            return "EXTERNAL_EXFIL"
        return None
    if channel == "sql":
        if _has_sql_metachar(text):
            return "SQL_METACHAR"
        return None
    if channel == "shell":
        if _has_shell_metachar(text):
            return "SHELL_METACHAR"
        return None
    return None


def evaluate(body: dict) -> dict:
    def unsafe(reason):
        return {"safe": False, "reason": reason}

    # INVALID_SCHEMA
    if not isinstance(body, dict):
        return unsafe("INVALID_SCHEMA")
    channel = body.get("channel")
    output = body.get("output")
    if channel not in CHANNELS:
        return unsafe("INVALID_SCHEMA")
    if not isinstance(output, str):
        return unsafe("INVALID_SCHEMA")
    if len(output) > MAX_LEN:
        return unsafe("INVALID_SCHEMA")

    # ENCODED_PAYLOAD — decode once; if changed AND decoded would trip any rule below.
    decoded = _decode_once(output)
    if decoded != output:
        if _channel_check(channel, decoded) is not None:
            return unsafe("ENCODED_PAYLOAD")

    # Channel rules on the ORIGINAL output, first match wins.
    reason = _channel_check(channel, output)
    if reason is not None:
        return unsafe(reason)

    return {"safe": True, "reason": "SAFE"}


@router.post("/sanitize-output")
async def sanitize_output(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    try:
        return JSONResponse(evaluate(body))
    except Exception:
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})


@router.get("/sanitize-output")
async def sanitize_output_info():
    return JSONResponse({"service": "TDS GA7 Output Gate", "endpoint": "POST /sanitize-output"})
