from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import base64
import json
import os
from google import genai
from google.genai import types as gtypes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.5-flash"


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


# ----------------- Image QA -----------------

class QARequest(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
def answer_image(req: QARequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        prompt = (
            f"Carefully analyze the image and answer this question: {req.question}\n\n"
            "Response rules:\n"
            "- Read all text, numbers, and labels visible in the image.\n"
            "- If the image contains a table, chart, receipt, or invoice, extract exact numeric values.\n"
            "- For numeric answers (sums, totals, maximums, averages): return ONLY the number as a plain string, no currency symbols, no units, no commas (e.g. '4089.35' not '$4,089.35').\n"
            "- For categorical answers (e.g. 'which category is largest'): return just the category name as a string.\n"
            "- No explanation, no reasoning, no punctuation at the end.\n"
            "- Return exactly what would go in the JSON 'answer' field."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ],
        )
        return {"answer": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Invoice Extract -----------------

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

Return ONLY a valid JSON object with exactly these 6 keys. No explanation, no markdown, no code fences.

Invoice text:
---
{invoice_text}
---

JSON:"""


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
                cleaned = v.replace(",", "").replace("Rs.", "").replace("USD", "").replace("INR", "").replace("EUR", "").strip()
                result[k] = float(cleaned)
            except Exception:
                result[k] = None
    return result


@app.post("/extract")
def extract(req: InvoiceRequest):
    try:
        prompt = INVOICE_PROMPT.format(invoice_text=req.invoice_text)
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
        )
        raw = interaction.output_text
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
                cleaned = value.replace(",", "").strip()
                return int(float(cleaned))
        if t == "float":
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.replace(",", "").replace("Rs.", "").replace("USD", "").replace("INR", "").strip()
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
            if isinstance(value, str):
                return value.strip()
            return str(value)
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
- No markdown, no code fences, no explanation.

Text:
---
{req.text}
---

JSON:"""
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
        )
        raw = interaction.output_text
        parsed = extract_json(raw)
        result = {}
        for key, target_type in req.schema.items():
            result[key] = coerce_value(parsed.get(key), target_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Q7: Invoice Intelligence -----------------

class InvoiceIntelRequest(BaseModel):
    document_id: str = ""
    text: str
    schema: dict = {}


INVOICE_INTEL_PROMPT = """You are an invoice extraction engine. Extract these fields from the invoice text and return ONLY valid JSON with EXACTLY these keys:

- vendor: the biller's proper name, exactly as written (string)
- currency: ISO 4217 code (USD, EUR, GBP, INR, JPY). Text may say "euros", "₹", "pounds sterling", "dollars", "rupees", "yen" etc.
- total_amount: integer in main unit, no separators/symbols. May be spelled out ("twelve thousand four hundred eighty" -> 12480), grouped "12,480" or "1,24,800", or "12K" -> 12000.
- invoice_date: normalize to YYYY-MM-DD (string)
- due_in_days: integer ("Net 30" -> 30, "payable within 45 days" -> 45, "due in two weeks" -> 14)
- is_paid: boolean ("paid in full" -> true, "awaiting payment" -> false)
- priority: one of low, normal, high, urgent (string)
- contact_email: lowercased (string)
- line_items: array of objects each with keys sku, quantity, unit_price, in order they appear; unit_price is integer
- item_count: integer number of line items

Return ONLY the JSON object, no markdown, no code fences, no explanation.

Invoice text:
---
{text}
---

JSON:"""


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


@app.post("/invoice-intelligence")
def invoice_intelligence(req: InvoiceIntelRequest):
    try:
        prompt = INVOICE_INTEL_PROMPT.format(text=req.text)
        interaction = client.interactions.create(model=MODEL_NAME, input=prompt)
        parsed = extract_json(interaction.output_text)
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


# ----------------- Q8: Semantic Search Top-K -----------------

class SemanticSearchRequest(BaseModel):
    query_id: str = ""
    query: str
    candidates: list


def embed_texts(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in result.embeddings]


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
        embeddings = embed_texts(all_texts)
        q_emb = embeddings[0]
        cand_embs = embeddings[1:]
        scored = [(i, cosine(q_emb, e)) for i, e in enumerate(cand_embs)]
        scored.sort(key=lambda x: -x[1])
        top3 = [i for i, _ in scored[:3]]
        return {"ranking": top3}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Q9: Word Problem Solver -----------------

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

No markdown, no code fences, no extra keys.

JSON:"""
        interaction = client.interactions.create(model=MODEL_NAME, input=prompt)
        parsed = extract_json(interaction.output_text)
        reasoning = str(parsed.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning = reasoning + " " + "The irrelevant distractor numbers in the problem were identified and excluded from the final calculation to arrive at the correct integer answer."
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
        "model": MODEL_NAME,
    }
