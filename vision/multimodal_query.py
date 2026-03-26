from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
logger = logging.getLogger(__name__)

try:
    import httpx  
    _HTTPX_OK = True
except ImportError:
    httpx = None  
    _HTTPX_OK = False
    logger.warning("httpx not installed. Run: pip install httpx")

try:
    from vision.screen_context import ScreenContext  
except ImportError:
    try:
        from screen_context import ScreenContext  
    except ImportError:
        from dataclasses import dataclass as _dc
        @_dc
        class ScreenContext:  
            full_screenshot_b64: str | None = None
            cursor_region_b64:   str | None = None
            cursor_pos:          tuple[int, int] = (0, 0)
            active_window:       str = ""
            window_title:        str = ""
            screen_width:        int = 0
            screen_height:       int = 0
            timestamp:           str = ""
            ambient_age_s:       float = 0.0
            def has_screenshot(self) -> bool:
                return False
            def summary(self) -> str:
                return "stub"
            
GEMINI_MODEL:    str = "gemini-2.5-flash"
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"
GEMINI_ENDPOINT: str = (
    f"/v1beta/models/{GEMINI_MODEL}:generateContent"
)

MAX_OUTPUT_TOKENS: int  = 1024
TEMPERATURE:       float = 0.3
REQUEST_TIMEOUT_S: int   = 30
_VISUAL_KEYWORDS: frozenset[str] = frozenset({
    "this", "here", "screen", "error", "code", "article",
    "page", "translate", "explain", "what does", "what is",
    "fix", "read", "summarize", "summarise", "look at",
    "show me", "what app", "write", "describe", "tell me about",
    "what's", "whats", "visible", "open", "current",
    "that", "these", "those", "selected",
})

_SYSTEM_KEYWORDS: frozenset[str] = frozenset({
    "set volume", "mute", "unmute", "brightness",
    "open spotify", "launch", "install", "download",
    "battery", "cpu", "ram", "disk", "memory",
    "play music", "pause music", "next track",
    "find file", "list files",
    "what time", "what's the time",
})

@dataclass
class VisionResponse:
    answer:            str
    context_used:      dict[str, Any]           = field(default_factory=dict)
    suggested_actions: list[str]                = field(default_factory=list)
    tokens_used:       int                      = 0
    model:             str                      = GEMINI_MODEL
    latency_ms:        int                      = 0
    error:             str | None               = None
    def ok(self) -> bool:
        return self.error is None
    
def should_include_context(text: str) -> bool:
    text_lower = text.lower().strip()
    for kw in _SYSTEM_KEYWORDS:
        if text_lower.startswith(kw):
            return False
    for kw in _VISUAL_KEYWORDS:
        if kw in text_lower:
            return True
    return False

def build_vision_request(
    text: str,
    ctx: ScreenContext,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    parts.append({"text": f"User command: {text}"})
    if ctx.active_window or ctx.window_title:
        window_info = f"Active app: {ctx.active_window}"
        if ctx.window_title:
            window_info += f" — {ctx.window_title}"
        parts.append({"text": window_info})
    if ctx.cursor_pos and ctx.cursor_pos != (0, 0):
        parts.append({"text": f"Cursor position: {ctx.cursor_pos}"})
    if ctx.full_screenshot_b64:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": ctx.full_screenshot_b64,   
            }
        })
    if ctx.cursor_region_b64:
        parts.append({"text": "Cursor region (400×400 px around cursor):"})
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": ctx.cursor_region_b64,     
            }
        })
    request_body: dict[str, Any] = {
        "contents": [
            {
                "role":  "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature":     TEMPERATURE,
        },
    }
    if system_instruction:
        request_body["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    return request_body

def _describe_context_used(ctx: ScreenContext) -> dict[str, Any]:
    return {
        "full_screenshot":  ctx.full_screenshot_b64 is not None,
        "cursor_region":    ctx.cursor_region_b64 is not None,
        "active_window":    ctx.active_window or None,
        "window_title":     ctx.window_title[:80] if ctx.window_title else None,
        "cursor_pos":       ctx.cursor_pos,
        "ambient_age_s":    ctx.ambient_age_s,
    }

def _parse_gemini_response(
    response_json: dict[str, Any],
    latency_ms: int,
) -> VisionResponse:
    if "error" in response_json:
        err = response_json["error"]
        msg = err.get("message", "Unknown Gemini API error")
        return VisionResponse(
            answer     = f"[Vision query failed: {msg}]",
            latency_ms = latency_ms,
            error      = msg,
        )
    
    candidates = response_json.get("candidates", [])
    if not candidates:
        return VisionResponse(
            answer     = "[No response from Gemini vision model]",
            latency_ms = latency_ms,
            error      = "no_candidates",
        )
    
    candidate  = candidates[0]
    finish_reason = candidate.get("finishReason", "STOP")
    if finish_reason == "SAFETY":
        return VisionResponse(
            answer     = "[Response blocked by Gemini safety filters]",
            latency_ms = latency_ms,
            error      = "safety_block",
        )
    
    content    = candidate.get("content", {})
    parts      = content.get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    answer     = "\n".join(text_parts).strip()
    usage         = response_json.get("usageMetadata", {})
    tokens_used   = (
        usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)
    )
    suggested_actions = _extract_suggested_actions(answer)
    return VisionResponse(
        answer            = answer,
        suggested_actions = suggested_actions,
        tokens_used       = tokens_used,
        latency_ms        = latency_ms,
    )

def _extract_suggested_actions(answer: str) -> list[str]:
    match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', answer)
    if match:
        try:
            actions = json.loads(match.group(0))
            if isinstance(actions, list):
                return [str(a) for a in actions[:5]]
        except (json.JSONDecodeError, ValueError):
            pass
    return []

class MultimodalQuerier:
    _SYSTEM_INSTRUCTION = (
        "You are Jarvis, an AI assistant with access to the user's screen. "
        "You receive the user's voice command and optionally one or two screenshots: "
        "a full-screen ambient capture and a 400×400 cursor region. "
        "The cursor region shows exactly what the user is pointing at. "
        "Answer concisely — your response will be read aloud via text-to-speech. "
        "Avoid markdown formatting. "
        "If you identify follow-up actions the user might want to take, "
        'include them as a JSON list at the end: ["action1", "action2"]. '
        "Otherwise, omit the list entirely."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        timeout_s: int = REQUEST_TIMEOUT_S,
    ) -> None:
        self._api_key  = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model    = model
        self._timeout  = timeout_s
        self._endpoint = (
            f"{GEMINI_BASE_URL}/v1beta/models/{model}:generateContent"
        )

    async def query(
        self,
        text: str,
        ctx: ScreenContext,
        force_include_context: bool | None = None,
    ) -> VisionResponse:
        if not self._api_key:
            logger.warning("GOOGLE_API_KEY not set — using text-only fallback")
            return await self._text_only_fallback(text)
        include = (
            force_include_context
            if force_include_context is not None
            else should_include_context(text)
        )

        effective_ctx = ctx if include else ScreenContext(
            active_window=ctx.active_window,
            window_title=ctx.window_title,
            cursor_pos=ctx.cursor_pos,
        )

        request_body = build_vision_request(
            text               = text,
            ctx                = effective_ctx,
            system_instruction = self._SYSTEM_INSTRUCTION,
        )

        context_used = _describe_context_used(effective_ctx)
        t0 = time.monotonic()
        try:
            response_json = await self._post_gemini(request_body)
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.error("Gemini query failed: %s", exc)
            return VisionResponse(
                answer     = f"I couldn't process that vision query: {exc}",
                context_used = context_used,
                latency_ms = latency_ms,
                error      = str(exc),
            )
        
        latency_ms = int((time.monotonic() - t0) * 1000)
        response   = _parse_gemini_response(response_json, latency_ms)
        response.context_used = context_used
        logger.info(
            "Vision query completed: latency=%dms tokens=%d context=%s",
            response.latency_ms,
            response.tokens_used,
            {k: v for k, v in context_used.items() if v},
        )
        return response
    
    async def _post_gemini(self, request_body: dict[str, Any]) -> dict[str, Any]:
        if not _HTTPX_OK:
            raise RuntimeError(
                "httpx not installed. Run: pip install httpx"
            )
        
        url    = f"{self._endpoint}?key={self._api_key}"
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=request_body)
            resp.raise_for_status()
            return resp.json()
        
    async def _text_only_fallback(self, text: str) -> VisionResponse:
        return VisionResponse(
            answer       = (
                f"I received your query: '{text}'. "
                "Vision analysis is unavailable — GOOGLE_API_KEY is not set. "
                "Set it in your environment to enable screen-aware queries."
            ),
            context_used = {},
            error        = "no_api_key",
        )
    
    def query_sync(
        self,
        text: str,
        ctx: ScreenContext,
        force_include_context: bool | None = None,
    ) -> VisionResponse:
        return asyncio.run(
            self.query(text, ctx, force_include_context=force_include_context)
        )
    
    def __repr__(self) -> str:
        has_key = bool(self._api_key)
        return f"MultimodalQuerier(model={self._model!r}, api_key={'set' if has_key else 'NOT SET'})"
_default_querier: MultimodalQuerier | None = None

def get_querier() -> MultimodalQuerier:
    global _default_querier
    if _default_querier is None:
        _default_querier = MultimodalQuerier()
    return _default_querier

async def vision_query(
    text: str,
    ctx: ScreenContext,
    force_include_context: bool | None = None,
) -> VisionResponse:
    return await get_querier().query(text, ctx, force_include_context)