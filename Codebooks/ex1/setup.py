import base64
import mimetypes
import os
from collections.abc import Iterable
from pathlib import Path
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, OpenAIError

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the project .env file.")
if not BASE_URL:
    raise RuntimeError("Set OPENAI_BASE_URL in the project .env file.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

MODEL = "gpt-4o-mini"
PROMPT_FILE = Path("OCR_prompt.txt")
OCR_INSTRUCTIONS = PROMPT_FILE.read_text(encoding="utf-8").strip()
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}



def extract_text(image_path: str | Path, model: str = MODEL) -> str:
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported image type: {image_path.suffix}")

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_data_url = f"data:{mime_type};base64,{image_data}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an OCR assistant that extracts text from images.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_INSTRUCTIONS},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )

    text = response.choices[0].message.content
    if text is None:
        raise RuntimeError("OCR response did not contain text.")
    return text.strip()



__all__ = [
    "API_KEY",
    "BASE_URL",
    "MODEL",
    "OCR_INSTRUCTIONS",
    "PROMPT_FILE",
    "SUPPORTED_TYPES",
    "client",
    "extract_text",
]
