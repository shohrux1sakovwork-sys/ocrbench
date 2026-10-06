"""Async client for OpenAI-compatible vision chat endpoints (vLLM, SGLang, ...)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx

# Recommended prompt for Nanonets-OCR2.
DEFAULT_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)

# Prompt for general vision-language models (Qwen-VL family etc.).
GENERIC_PROMPT = (
    "Transcribe all text in this document image exactly as written, in natural reading order. "
    "Keep the original language and script (Uzbek Latin or Cyrillic, Russian, English); "
    "do not translate, correct, summarize or explain. Keep line and paragraph breaks. "
    "Write tables as Markdown tables. Output only the transcription."
)

# Model-specific prompts from the model cards. LightOnOCR-2 takes the image
# with no text; Surya 2's full-page prompt is its training-time wording.
SURYA_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)
DEEPSEEK_OCR_PROMPT = "Free OCR."
# Whole-page task prompts. PaddleOCR-VL, GLM-OCR and MinerU2.5 are normally run
# on layout-detected crops; these send the full page in one request instead.
PADDLEOCR_VL_PROMPT = "OCR:"
GLM_OCR_PROMPT = "Text Recognition:"
MINERU_PROMPT = "\nText Recognition:"
# HunyuanOCR text extraction ("extract the text content in the image"); its
# document-parsing prompt drops page headers and footers.
HUNYUAN_OCR_PROMPT = "请提取图片中的文字内容。"
# dots.ocr prompt_ocr (the model skips page headers and footers by design).
DOTS_OCR_PROMPT = "Extract the text content from this image."

PROMPTS = {
    "generic": GENERIC_PROMPT,
    "nanonets": DEFAULT_PROMPT,
    "lightonocr": "",
    "surya": SURYA_PROMPT,
    "deepseek-ocr": DEEPSEEK_OCR_PROMPT,
    "paddleocr-vl": PADDLEOCR_VL_PROMPT,
    "glm-ocr": GLM_OCR_PROMPT,
    "mineru": MINERU_PROMPT,
    "hunyuan-ocr": HUNYUAN_OCR_PROMPT,
    "dots-ocr": DOTS_OCR_PROMPT,
}

_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# Reasoning models may emit a thinking block before the answer.
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text).strip()


class OCRError(Exception):
    pass


@dataclass(frozen=True)
class OCRResponse:
    text: str
    latency_s: float
    # Streaming only: time to first generated token (queueing + prefill).
    ttft_s: float | None
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


class OCRClient:
    def __init__(
        self,
        base_url: str,
        model: str | None,
        *,
        api_key: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 600.0,
        max_retries: int = 2,
        max_connections: int = 8,
        stream: bool = True,
        extra_body: dict | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.stream = stream
        self.extra_body = extra_body or {}

    async def __aenter__(self) -> OCRClient:
        if self.model is None:
            self.model = await self._first_model()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    async def _first_model(self) -> str:
        response = await self._http.get("/models")
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise OCRError("server lists no models; pass --model")
        return models[0]["id"]

    async def ocr(self, image_url: str) -> OCRResponse:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        *([{"type": "text", "text": self.prompt}] if self.prompt else []),
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }
        if self.stream:
            payload |= {"stream": True, "stream_options": {"include_usage": True}}
        for attempt in range(1, self.max_retries + 2):
            last = attempt > self.max_retries
            try:
                return await self._request(payload, attempt)
            except _Retryable as exc:
                if last:
                    raise OCRError(str(exc)) from exc
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def _request(self, payload: dict, attempt: int) -> OCRResponse:
        start = time.perf_counter()
        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code in _RETRY_STATUS:
                    raise _Retryable(f"HTTP {response.status_code}: {(await response.aread())[:500]!r}")
                if response.is_error:
                    raise OCRError(f"HTTP {response.status_code}: {(await response.aread()).decode(errors='replace')[:500]}")
                if not self.stream:
                    body = json.loads(await response.aread())
                    choice = body["choices"][0]
                    return self._response(choice["message"].get("content") or "", start, None, attempt,
                                          body.get("usage"), choice.get("finish_reason"))
                parts: list[str] = []
                ttft = usage = finish_reason = None
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # Reasoning output (if any) counts as generated tokens for TTFT.
                        piece = delta.get("content") or ""
                        if ttft is None and (piece or delta.get("reasoning_content") or delta.get("reasoning")):
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                        finish_reason = choice.get("finish_reason") or finish_reason
                return self._response("".join(parts), start, ttft, attempt, usage, finish_reason)
        except httpx.TransportError as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _response(text, start, ttft, attempt, usage, finish_reason) -> OCRResponse:
        usage = usage or {}
        return OCRResponse(
            text=strip_thinking(text),
            latency_s=time.perf_counter() - start,
            ttft_s=ttft,
            attempts=attempt,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )


class _Retryable(Exception):
    pass
