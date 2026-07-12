from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import os
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-2.5-flash"

# ----------------- Image QA -----------------

class QARequest(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
def answer_image(req: QARequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        prompt = (
            f"Look at the image and answer this question: {req.question}\n"
            "Rules:\n"
            "- Return ONLY the answer value as a string.\n"
            "- No units, no currency symbols, no extra text.\n"
            "- For numbers, return just the number (e.g. '4089.35').\n"
            "- No explanation."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return coerce_invoice(parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Health -----------------

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/answer-image", "/extract"]}
