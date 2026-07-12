from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import base64
import json
import os
from google import genai

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
        prompt = (
            f"Look at the image and answer this question: {req.question}\n"
            "Rules:\n"
            "- Return ONLY the answer value as a string.\n"
            "- No units, no currency symbols, no extra text.\n"
            "- For numbers, return just the number (e.g. '4089.35').\n"
            "- No explanation."
        )
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": req.image_base64,
                            },
                        },
                    ],
                }
            ],
        )
        return {"answer": interaction.output_text.strip()}
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


# ----------------- Health -----------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": ["/answer-image", "/extract", "/dynamic-extract"],
        "model": MODEL_NAME,
    }
