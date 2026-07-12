from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-exp")


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
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        return {"answer": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok"}
