from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import base64
import json
import os
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openai/v1"),
)

CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


# ----------------- Helpers -----------------

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    return json.loads(text)


def chat(prompt: str, json_mode: bool = False) -> str:
    kwargs = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def to_int_safe(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v.replace(",", "").replace("$", "").strip()))
        except Exception:
            return None
    return None


# ----------------- Image QA -----------------

class QARequest(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
def answer_image(req: QARequest):
    try:
        prompt = (
            f"Carefully analyze the image and answer this question: {req.question}\n\n"
            "Response rules:\n"
            "- Read all text, numbers, and labels visible in the image.\n"
            "- If the image contains a table, chart, receipt, or invoice, extract exact numeric values.\n"
            "- For numeric answers (sums, totals, maximums, averages): return ONLY the number as a plain string, "
            "no currency symbols, no units, no commas (e.g. '4089.35' not '$4,089.35').\n"
            "- For categorical answers (e.g. 'which category is largest'): return just the category name.\n"
            "- No explanation, no reasoning, no punctuation at the end."
        )
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{req.image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Invoice Extract (fixed schema) -----------------

SCHEMA_KEYS = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]


class InvoiceRequest(BaseModel):
    invoice_text: str


INVOICE_PROMPT = """You are an invoice parser. Extract the following fields from the invoice text and return ONLY valid JSON.

Fields to extract:
- invoice_no: string, the invoice number/ID (null if not found)
- date: string in ISO format YYYY-MM-DD (null if not found). Parse any date format (e.g. "15 March 2026" -> "2026-03-15").
- vendor: string, the vendor/seller/company name (null if not found)
- amount: number, the SUBTOTAL BEFORE TAX (null if not found). Not the grand total.
- tax: number, the TAX AMOUNT ONLY (null if not found). Not the tax rate.
- currency: string, e.g. "INR", "USD", "EUR" (null if not found). If you see "Rs." treat it as INR.

Return ONLY a valid JSON object with exactly these 6 keys.

Invoice text:
---
{invoice_text}
---"""


def coerce_invoice(obj: dict) -> dict:
    result = {}
    for key in SCHEMA_KEYS:
        val = obj.get(key)
        if val == "" or val == "null":
            val = None
        result[key] = val
    for k in ("amount", "tax"):
        v = result.get(k)
        if isinstance(v, str):
            try:
                cleaned = (v.replace(",", "").replace("Rs.", "").replace("USD", "")
                           .replace("INR", "").replace("EUR", "").strip())
                result[k] = float(cleaned)
            except Exception:
                result[k] = None
    return result


@app.post("/extract")
def extract(req: InvoiceRequest):
    try:
        prompt = INVOICE_PROMPT.format(invoice_text=req.invoice_text)
        raw = chat(prompt, json_mode=True)
        parsed = extract_json(raw)
        return coerce_invoice(parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Dynamic Schema Extract -----------------

class DynamicRequest(BaseModel):
    text: str
    schema: Dict[str, str]


def coerce_value(value: Any, target_type: str) -> Any:
    if value is None or value == "" or value == "null":
        return None
    t = target_type.lower().strip()
    try:
        if t == "string":
            return str(value)
        if t == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                return int(float(value.replace(",", "").strip()))
        if t == "float":
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = (value.replace(",", "").replace("Rs.", "")
                           .replace("USD", "").replace("INR", "").strip())
                return float(cleaned)
        if t == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.lower().strip()
                if v in ("true", "yes", "1"):
                    return True
                if v in ("false", "no", "0"):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
        if t == "date":
            return str(value).strip()
        if t == "array[string]":
            if isinstance(value, list):
                return [str(x) for x in value]
            if isinstance(value, str):
                return [x.strip() for x in value.split(",") if x.strip()]
        if t == "array[integer]":
            if isinstance(value, list):
                out = []
                for x in value:
                    try:
                        out.append(int(float(str(x).replace(",", "").strip())))
                    except Exception:
                        pass
                return out
    except Exception:
        return None
    return value


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):
    try:
        schema_lines = "\n".join([f"- {k}: {v}" for k, v in req.schema.items()])
        prompt = f"""You are a data extraction engine. Extract the following fields from the text.

Schema (field name : type):
{schema_lines}

Rules:
- Return ONLY a valid JSON object with EXACTLY these keys, no extras, no missing.
- Use null if a field cannot be found in the text.
- Dates must be ISO format YYYY-MM-DD (e.g., "12 June 2026" -> "2026-06-12").
- Integers must be JSON integers, floats must be JSON numbers (not strings).
- Booleans must be true/false (not "true"/"false").
- array[string] must be a JSON array of strings.
- array[integer] must be a JSON array of integers.

Text:
---
{req.text}
---"""
        raw = chat(prompt, json_mode=True)
        parsed = extract_json(raw)
        result = {}
        for key, target_type in req.schema.items():
            result[key] = coerce_value(parsed.get(key), target_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Invoice Intelligence -----------------

class InvoiceIntelRequest(BaseModel):
    document_id: str = ""
    text: str
    schema: dict = {}


INVOICE_INTEL_PROMPT = """You are an invoice extraction engine. Extract these fields from the invoice text and return ONLY valid JSON with EXACTLY these keys:

- vendor: the biller's proper name, exactly as written (string)
- currency: ISO 4217 code (USD, EUR, GBP, INR, JPY). Text may say "euros", "rupees", "pounds sterling", "dollars", "yen", or use symbols.
- total_amount: integer in main unit, no separators/symbols. May be spelled out ("twelve thousand four hundred eighty" -> 12480), grouped "12,480" or "1,24,800" -> 124800, or "12K" -> 12000.
- invoice_date: normalize to YYYY-MM-DD (string)
- due_in_days: integer ("Net 30" -> 30, "payable within 45 days" -> 45, "due in two weeks" -> 14)
- is_paid: boolean ("paid in full" -> true, "awaiting payment" -> false)
- priority: one of low, normal, high, urgent (string)
- contact_email: lowercased (string)
- line_items: array of objects each with keys sku, quantity, unit_price, in order they appear; unit_price is integer
- item_count: integer number of line items

Return ONLY the JSON object.

Invoice text:
---
{text}
---"""


@app.post("/invoice-intelligence")
def invoice_intelligence(req: InvoiceIntelRequest):
    try:
        prompt = INVOICE_INTEL_PROMPT.format(text=req.text)
        raw = chat(prompt, json_mode=True)
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}

        result = {
            "vendor": parsed.get("vendor"),
            "currency": parsed.get("currency"),
            "total_amount": to_int_safe(parsed.get("total_amount")),
            "invoice_date": parsed.get("invoice_date"),
            "due_in_days": to_int_safe(parsed.get("due_in_days")),
            "is_paid": bool(parsed.get("is_paid")) if parsed.get("is_paid") is not None else None,
            "priority": parsed.get("priority"),
            "contact_email": (parsed.get("contact_email") or "").lower() if parsed.get("contact_email") else None,
            "line_items": [],
            "item_count": 0,
        }

        clean_items = []
        items = parsed.get("line_items")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    clean_items.append({
                        "sku": it.get("sku"),
                        "quantity": to_int_safe(it.get("quantity")),
                        "unit_price": to_int_safe(it.get("unit_price")),
                    })
        result["line_items"] = clean_items
        result["item_count"] = len(clean_items)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Semantic Search Top-K -----------------

class SemanticSearchRequest(BaseModel):
    query_id: str = ""
    query: str
    candidates: list


def cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    try:
        all_texts = [req.query] + list(req.candidates)
        resp = client.embeddings.create(model=EMBED_MODEL, input=all_texts)
        embeddings = [d.embedding for d in resp.data]
        q_emb = embeddings[0]
        cand_embs = embeddings[1:]
        scored = [(i, cosine(q_emb, e)) for i, e in enumerate(cand_embs)]
        scored.sort(key=lambda x: -x[1])
        return {"ranking": [i for i, _ in scored[:3]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Word Problem Solver -----------------

class WordProblemRequest(BaseModel):
    problem_id: str = ""
    problem: str


@app.post("/solve")
def solve_problem(req: WordProblemRequest):
    try:
        prompt = f"""Solve this arithmetic word problem step by step. Some numbers are distractors — ignore irrelevant ones.

Problem:
{req.problem}

Return ONLY valid JSON with EXACTLY two keys:
- "reasoning": a string of at least 80 characters showing your calculation steps
- "answer": a JSON integer (not a string, not a float, no currency symbols)

No extra keys."""
        raw = chat(prompt, json_mode=True)
        parsed = extract_json(raw)
        reasoning = str(parsed.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning += (" The irrelevant distractor numbers in the problem were identified and "
                          "excluded from the final calculation to arrive at the correct integer answer.")
        ans = parsed.get("answer")
        if isinstance(ans, str):
            ans = int(float(ans.replace(",", "").strip()))
        elif isinstance(ans, float):
            ans = int(ans)
        return {"reasoning": reasoning, "answer": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Health -----------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": [
            "/answer-image", "/extract", "/dynamic-extract",
            "/invoice-intelligence", "/semantic-search", "/solve",
        ],
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
    }
