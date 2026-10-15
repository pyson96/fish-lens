import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from openai import APIError, AsyncOpenAI

load_dotenv(override=True)  # 터미널에 남아있는 이전 키보다 .env 값을 우선

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # OpenAI 이미지 입력 한도 20MB
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY가 없습니다. .env.example을 .env로 복사하고 키를 입력하세요.")
_key = os.environ["OPENAI_API_KEY"]
print(f"[config] model={DEFAULT_MODEL} key={_key[:8]}...{_key[-4:]}")

app = FastAPI(title="OpenAI Vision API Test")
client = AsyncOpenAI()  # OPENAI_API_KEY 환경변수를 사용


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
async def config():
    return {"default_model": DEFAULT_MODEL}


@app.post("/api/chat")
async def chat(
    prompt: str = Form(""),
    model: str = Form(""),
    detail: str = Form("auto"),
    images: list[UploadFile] = File(default=[]),
):
    if not prompt.strip() and not images:
        return JSONResponse({"error": "프롬프트나 이미지를 입력하세요."}, status_code=400)

    content = []
    if prompt.strip():
        content.append({"type": "input_text", "text": prompt})

    for image in images:
        if image.content_type not in ALLOWED_TYPES:
            return JSONResponse(
                {"error": f"지원하지 않는 형식: {image.filename} ({image.content_type})"},
                status_code=400,
            )
        data = await image.read()
        if len(data) > MAX_IMAGE_BYTES:
            return JSONResponse({"error": f"20MB 초과: {image.filename}"}, status_code=400)
        b64 = base64.b64encode(data).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{image.content_type};base64,{b64}",
                "detail": detail,
            }
        )

    try:
        resp = await client.responses.create(
            model=model.strip() or DEFAULT_MODEL,
            input=[{"role": "user", "content": content}],
        )
    except APIError as e:
        status = getattr(e, "status_code", None) or 502
        return JSONResponse({"error": f"OpenAI API 오류: {e.message}"}, status_code=status)

    return {
        "model": resp.model,
        "answer": resp.output_text,
        "usage": resp.usage.model_dump() if resp.usage else None,
        "image_count": len(images),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
