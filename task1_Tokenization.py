from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
import re
import io

app = FastAPI(title="PDF Tokenizer API")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


@app.post("/tokenize")
async def tokenize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text:
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

    cleaned_text = clean_text(raw_text)
    tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)

    return JSONResponse(
        {
            "text_length": len(cleaned_text),
            "tokens_count": len(tokens),
            "first_100_tokens": tokens[:100],
        }
    )
