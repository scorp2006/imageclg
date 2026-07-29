"""
TDS Project 1 — Q5: Data-Analyst Telegram Bot.

One process, three parts:
  - FastAPI app: GET /health (keep-alive) and GET /run.jsonl (public agent log)
  - Background thread: Telegram getUpdates long-poll -> per-message agent loop
  - Background thread: self-ping /health every 10 min so Render doesn't idle out

The agent has one tool, run_python, that exec()s Python server-side (pandas,
requests, BeautifulSoup available) so the model can fetch and analyse public
datasets (MOSPI etc.) rather than guessing numbers.
"""
import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
import urllib.request

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI

# ---------------- config ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
BASE_URL = os.environ.get("BASE_URL", "https://imageclg.onrender.com").rstrip("/")
MODEL = os.environ.get("BOT_MODEL", "gpt-4o")
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openai/v1"),
)

app = FastAPI()

# in-memory run log (list of dicts, served as JSONL)
_run_log = []
_log_lock = threading.Lock()
# per-chat conversation history (last ~20 turns)
_history = {}


def log_event(**kw):
    kw["ts"] = time.time()
    with _log_lock:
        _run_log.append(kw)
        if len(_run_log) > 5000:
            del _run_log[: len(_run_log) - 5000]


# ---------------- FastAPI routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "bot": bool(BOT_TOKEN), "model": MODEL}


@app.get("/run.jsonl")
def run_jsonl():
    with _log_lock:
        body = "\n".join(json.dumps(e, default=str) for e in _run_log) + "\n"
    return PlainTextResponse(body, media_type="application/x-ndjson")


@app.get("/")
def root():
    return {"service": "tds-p1-databot", "endpoints": ["/health", "/run.jsonl"]}


# ---------------- the run_python tool ----------------
def run_python(code: str) -> str:
    """Execute code, capture stdout, return last 8000 chars."""
    buf = io.StringIO()
    g = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, g)
        out = buf.getvalue()
    except Exception:
        out = buf.getvalue() + "\n" + traceback.format_exc()
    return out[-8000:] if len(out) > 8000 else out


TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code on the server and return its stdout. "
                       "pandas, numpy, requests, bs4 (BeautifulSoup), openpyxl are available. "
                       "Use this to download and analyse public datasets. Always print() what you need.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run"}},
            "required": ["code"],
        },
    },
}]

SYSTEM_PROMPT = """You are a precise data-analyst agent answering questions over Telegram.

RULES:
1. Answer the LATEST user message. Earlier messages are context (conversations can be multi-turn).
2. Use the run_python tool to fetch and compute answers when the data is given inline or as a REAL, specific URL. \
Never invent or guess a URL (e.g. example.com) — if you do not have a real dataset URL, do NOT call the tool with a fake one. \
For well-known published statistics (MOSPI, census, national indicators, etc.), you may answer directly from your own reliable knowledge. \
You MUST always give a concrete best-answer. NEVER reply with "data not available", "unknown", "N/A", or an empty/placeholder answer — \
if a download fails or no data is provided, fall back to your best knowledge and answer decisively.
3. Your FINAL message must be ONLY the exact JSON object the question asks for — no prose, no markdown fences, no explanation. \
The question specifies the exact shape of "answer" (keys, nesting, string vs number) — match it EXACTLY, add no extra keys.
4. Every question wants a JSON like {"answer": <shaped as asked>, "log_url": "PLACEHOLDER"}. \
Put the literal string PLACEHOLDER in log_url — it will be substituted with the real URL by the caller.
5. If a message is only setup (e.g. "I'll send the data next"), still reply with a minimal valid JSON \
acknowledgement like {"answer": "ready", "log_url": "PLACEHOLDER"} — the grader waits for a reply to every message.
6. Be decisive and fast. Prefer a concrete answer over hedging."""


def extract_json(text: str):
    """Strip fences, find first balanced {...}, parse. Wrap if no 'answer' key."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    # find first balanced brace group
    start = t.find("{")
    if start == -1:
        return {"answer": t}
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict) and "answer" not in obj:
                        return {"answer": obj}
                    return obj
                except Exception:
                    break
    try:
        return json.loads(t)
    except Exception:
        return {"answer": t}


def agent_answer(chat_id, user_text):
    """Run the agent loop and return the final JSON dict."""
    hist = _history.setdefault(chat_id, [])
    hist.append({"role": "user", "content": user_text})
    hist[:] = hist[-20:]  # keep last 20 turns

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + hist
    deadline = time.time() + 210  # wall-clock budget (grader allows ~300s)
    log_event(kind="question", chat_id=chat_id, text=user_text)

    final_text = None
    for step in range(10):
        tools_on = time.time() < deadline
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS if tools_on else None,
                tool_choice="auto" if tools_on else None,
                temperature=0,
            )
        except Exception as e:
            log_event(kind="llm_error", error=str(e))
            final_text = json.dumps({"answer": "internal error", "log_url": "PLACEHOLDER"})
            break

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if msg.tool_calls and tools_on:
            for tc in msg.tool_calls:
                if tc.function.name == "run_python":
                    try:
                        code = json.loads(tc.function.arguments).get("code", "")
                    except Exception:
                        code = ""
                    out = run_python(code)
                    log_event(kind="tool", code=code, output=out)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "run_python",
                        "content": out,
                    })
            continue
        else:
            final_text = msg.content or ""
            break

    if final_text is None:
        final_text = json.dumps({"answer": "no answer", "log_url": "PLACEHOLDER"})

    obj = extract_json(final_text)
    if not isinstance(obj, dict):
        obj = {"answer": obj}
    # always overwrite log_url with the real public URL
    obj["log_url"] = f"{BASE_URL}/run.jsonl" if BASE_URL else "PLACEHOLDER"
    hist.append({"role": "assistant", "content": json.dumps(obj)})
    log_event(kind="answer", chat_id=chat_id, answer=obj)
    return obj


# ---------------- Telegram plumbing ----------------
def tg_send(chat_id, text):
    data = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(f"{TG_API}/sendMessage", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        log_event(kind="send_error", error=str(e))


def handle_message(update):
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "")
    if not text:
        return
    try:
        obj = agent_answer(chat_id, text)
        tg_send(chat_id, json.dumps(obj))
    except Exception:
        log_event(kind="handler_error", trace=traceback.format_exc())
        tg_send(chat_id, json.dumps({
            "answer": "internal error",
            "log_url": f"{BASE_URL}/run.jsonl" if BASE_URL else "PLACEHOLDER",
        }))


def poll_loop():
    offset = None
    # clear any webhook so getUpdates works
    try:
        urllib.request.urlopen(f"{TG_API}/deleteWebhook", timeout=15).read()
    except Exception:
        pass
    while True:
        try:
            url = f"{TG_API}/getUpdates?timeout=50"
            if offset is not None:
                url += f"&offset={offset}"
            data = json.loads(urllib.request.urlopen(url, timeout=60).read())
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                threading.Thread(target=handle_message, args=(upd,), daemon=True).start()
        except Exception as e:
            log_event(kind="poll_error", error=str(e))
            time.sleep(3)


def keepwarm_loop():
    while True:
        time.sleep(600)
        if BASE_URL:
            try:
                urllib.request.urlopen(f"{BASE_URL}/health", timeout=20).read()
            except Exception:
                pass


# Only auto-start threads when bot.py is the app being served directly
# (uvicorn bot:app). When imported into main.py, main.py controls startup.
if os.environ.get("BOT_STANDALONE") == "1":
    @app.on_event("startup")
    def _startup():
        if BOT_TOKEN:
            threading.Thread(target=poll_loop, daemon=True).start()
            threading.Thread(target=keepwarm_loop, daemon=True).start()
            log_event(kind="startup", model=MODEL)
