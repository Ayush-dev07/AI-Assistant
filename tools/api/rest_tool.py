from __future__ import annotations

import base64
import re
import time
import urllib.parse
from collections import defaultdict
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult

log = get_logger(__name__)

_DEFAULT_BUCKET_CAPACITY: float = 30.0
_DEFAULT_REFILL_RATE: float     = 30.0 / 60.0  

_MAX_RESPONSE_BYTES: int = 10 * 1024 * 1024   # 10 MB

_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel=["\']([^"\']+)["\']')

_JSON_CONTENT_TYPES = frozenset({
    "application/json",
    "application/ld+json",
    "application/vnd.api+json",
    "application/problem+json",
})

_BINARY_CONTENT_TYPES = frozenset({
    "image/", "audio/", "video/",
    "application/octet-stream",
    "application/zip",
    "application/pdf",
    "application/gzip",
})

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_PERMANENT_ERROR_STATUS = frozenset({400, 401, 403, 404, 405, 410, 422})

class _DomainRateLimiter:

    def __init__(
        self,
        capacity: float = _DEFAULT_BUCKET_CAPACITY,
        refill_rate: float = _DEFAULT_REFILL_RATE,
    ) -> None:
        self._capacity    = capacity
        self._refill_rate = refill_rate
        self._buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": capacity, "last_refill": time.monotonic()}
        )

    def consume(self, hostname: str) -> tuple[bool, float]:
        bucket  = self._buckets[hostname]
        now     = time.monotonic()
        elapsed = now - bucket["last_refill"]

        # Refill
        bucket["tokens"] = min(
            self._capacity,
            bucket["tokens"] + elapsed * self._refill_rate,
        )
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, 0.0

        wait = (1.0 - bucket["tokens"]) / self._refill_rate
        return False, round(wait, 2)

    def reset(self, hostname: str) -> None:
        """Reset a domain's bucket (useful in tests)."""
        self._buckets.pop(hostname, None)

_rate_limiter = _DomainRateLimiter()


class APIInput(BaseModel):

    action:   str = Field(
        default="request",
        description="request | graphql | upload | paginate",
    )
    url:      str = Field(
        ...,
        description="Target URL. Must be http or https.",
    )
    method:   str = Field(
        default="GET",
        description="HTTP verb: GET POST PUT PATCH DELETE HEAD OPTIONS",
    )
    headers:  dict[str, str]       = Field(default_factory=dict)
    params:   dict[str, str]       = Field(default_factory=dict)
    body:     dict | list | None   = Field(default=None)
    form_data: dict[str, str]      = Field(default_factory=dict)
    file_path: str | None          = Field(default=None)
    file_field: str                = Field(default="file")
    query:    str | None           = Field(default=None)
    variables: dict[str, Any]      = Field(default_factory=dict)
    max_pages: int                 = Field(default=5, ge=1, le=20)
    timeout:  int                  = Field(default=30, ge=1, le=300)
    max_retries: int               = Field(default=3, ge=0, le=5)
    follow_redirects: bool         = Field(default=True)
    verify_ssl: bool               = Field(default=True)
    raw_response: bool             = Field(default=False)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"request", "graphql", "upload", "paginate"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"action must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
        upper = v.upper().strip()
        if upper not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http or https, got scheme {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError(f"URL has no host: {v!r}")
        return v

class APICallerTool(BaseTool):

    name = "api_caller"
    description = (
        "Call any REST API. Supports all HTTP methods, JSON bodies, headers, "
        "query params, file uploads, GraphQL, and auto-pagination. "
        "action='request': standard HTTP call (method, url, headers, params, body). "
        "action='graphql': POST GraphQL query (url, query, variables). "
        "action='upload': multipart file upload (url, file_path, file_field). "
        "action='paginate': auto-follow Link rel=next headers (url, max_pages). "
        "Returns status_code, data (parsed JSON or text), headers, "
        "pagination info, and timing. "
        "Set raw_response=True to always get text output."
    )
    manifest     = PermissionManifest(network_domains=["*"])
    input_schema = APIInput

    def __init__(
        self,
        allowed_domains:    list[str] | None = None,
        oauth_token:        str | None = None,
        default_headers:    dict[str, str] | None = None,
        rate_limit_capacity:  float = _DEFAULT_BUCKET_CAPACITY,
        rate_limit_per_min:   float = 30.0,
    ) -> None:
        if allowed_domains is not None:
            self.manifest = PermissionManifest(network_domains=allowed_domains)

        self._oauth_token     = oauth_token
        self._default_headers = default_headers or {}
        self._rate_limiter    = _DomainRateLimiter(
            capacity    = float(rate_limit_capacity),
            refill_rate = rate_limit_per_min / 60.0,
        )

        self._http: httpx.AsyncClient | None = None

    async def _get_http(self, verify_ssl: bool = True) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                verify=verify_ssl,
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()


    async def execute(self, input: ToolInput) -> ToolResult:
        try:
            params = self.input_schema(**input.parameters)
        except Exception as exc:
            return ToolResult(success=False, error=f"Invalid arguments: {exc}")

        hostname = urllib.parse.urlparse(params.url).hostname or ""
        if not self.manifest.allows_domain(hostname):
            return ToolResult(
                success=False,
                error=(
                    f"Domain {hostname!r} is not in this tool's allowed domains "
                    f"{self.manifest.network_domains}. "
                    "Use a tool instance that permits this domain."
                ),
            )

        allowed, retry_after = self._rate_limiter.consume(hostname)
        if not allowed:
            return ToolResult(
                success=False,
                error=(
                    f"Rate limit reached for {hostname!r}. "
                    f"Retry after {retry_after:.1f} seconds. "
                    f"Consider spacing out API calls."
                ),
                output={"retry_after_seconds": retry_after},
                metadata={"rate_limited": True},
            )

        if params.action == "request":
            return await self._do_request(params)
        if params.action == "graphql":
            return await self._do_graphql(params)
        if params.action == "upload":
            return await self._do_upload(params)
        if params.action == "paginate":
            return await self._do_paginate(params)

        return ToolResult(success=False, error=f"Unknown action: {params.action!r}")


    async def _do_request(
        self,
        params: APIInput,
        _url_override: str | None = None,
    ) -> ToolResult:
        url   = _url_override or params.url
        start = time.monotonic()
        http  = await self._get_http(verify_ssl=params.verify_ssl)

        merged_headers = self._build_headers(params.headers)
        last_error     = ""

        for attempt in range(params.max_retries + 1):
            try:
                resp = await http.request(
                    method           = params.method,
                    url              = url,
                    headers          = merged_headers,
                    params           = params.params or None,
                    json             = params.body if params.body is not None else None,
                    follow_redirects = params.follow_redirects,
                    timeout          = params.timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = f"Request timed out after {params.timeout}s: {exc}"
                log.debug("api_caller_timeout", url=url, attempt=attempt + 1)
                if attempt < params.max_retries:
                    await _async_sleep(2 ** attempt)
                continue
            except httpx.ConnectError as exc:
                last_error = f"Connection failed: {exc}"
                log.debug("api_caller_connect_error", url=url, attempt=attempt + 1)
                if attempt < params.max_retries:
                    await _async_sleep(2 ** attempt)
                continue
            except httpx.HTTPError as exc:
                return ToolResult(
                    success=False,
                    error=f"HTTP client error: {exc}",
                )

            duration_ms = int((time.monotonic() - start) * 1000)

            if resp.status_code in _PERMANENT_ERROR_STATUS:
                data, is_binary = _parse_response_body(resp, params.raw_response)
                return ToolResult(
                    success=False,
                    output=_build_output(resp, data, is_binary, duration_ms),
                    error=(
                        f"HTTP {resp.status_code} {resp.reason_phrase} for {url!r}. "
                        + _status_hint(resp.status_code)
                    ),
                )

            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
                if attempt < params.max_retries:
                    wait = _retry_wait(resp, attempt)
                    log.debug(
                        "api_caller_retrying",
                        url=url, status=resp.status_code,
                        attempt=attempt + 1, wait=wait,
                    )
                    await _async_sleep(wait)
                continue

            data, is_binary = _parse_response_body(resp, params.raw_response)
            pagination      = _parse_link_header(resp.headers.get("link", ""))

            log.info(
                "api_caller_complete",
                method      = params.method,
                url         = url,
                status      = resp.status_code,
                duration_ms = duration_ms,
                response_bytes = len(resp.content),
            )

            output = _build_output(resp, data, is_binary, duration_ms)
            if pagination:
                output["pagination"] = pagination

            return ToolResult(
                success = resp.is_success,
                output  = output,
                error   = (
                    None if resp.is_success
                    else f"HTTP {resp.status_code} {resp.reason_phrase}"
                ),
            )

        return ToolResult(
            success=False,
            error=f"Request failed after {params.max_retries + 1} attempts: {last_error}",
        )

    async def _do_graphql(self, params: APIInput) -> ToolResult:
        if not params.query:
            return ToolResult(
                success=False,
                error="action='graphql' requires a 'query' parameter containing the GraphQL query string.",
            )

        gql_body: dict[str, Any] = {"query": params.query}
        if params.variables:
            gql_body["variables"] = params.variables

        overridden = params.model_copy(
            update={"method": "POST", "body": gql_body, "action": "request"}
        )
        result = await self._do_request(overridden)

        if result.success and isinstance(result.output, dict):
            resp_data = result.output.get("data", {})
            if isinstance(resp_data, dict):
                gql_data   = resp_data.get("data")
                gql_errors = resp_data.get("errors", [])
                result.output["graphql_data"]   = gql_data
                result.output["graphql_errors"] = gql_errors
                if gql_errors and not gql_data:
                    result.success = False
                    result.error   = (
                        f"GraphQL errors: "
                        + "; ".join(e.get("message", str(e)) for e in gql_errors[:3])
                    )

        return result

    async def _do_upload(self, params: APIInput) -> ToolResult:
        if not params.file_path:
            return ToolResult(
                success=False,
                error="action='upload' requires a 'file_path' parameter.",
            )

        from pathlib import Path
        path = Path(params.file_path)
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {params.file_path!r}",
            )
        if not path.is_file():
            return ToolResult(
                success=False,
                error=f"{params.file_path!r} is not a file.",
            )

        import mimetypes
        mime, _ = mimetypes.guess_type(path.name)
        mime    = mime or "application/octet-stream"

        start         = time.monotonic()
        http          = await self._get_http(verify_ssl=params.verify_ssl)
        merged_headers = self._build_headers(params.headers)
        merged_headers.pop("content-type", None)
        merged_headers.pop("Content-Type", None)

        try:
            with path.open("rb") as fh:
                files   = {params.file_field: (path.name, fh, mime)}
                fields  = dict(params.form_data)
                resp    = await http.post(
                    params.url,
                    headers  = merged_headers,
                    files    = files,
                    data     = fields or None,
                    timeout  = params.timeout,
                    follow_redirects = params.follow_redirects,
                )
        except Exception as exc:
            return ToolResult(success=False, error=f"Upload failed: {exc}")

        duration_ms     = int((time.monotonic() - start) * 1000)
        data, is_binary = _parse_response_body(resp, params.raw_response)

        return ToolResult(
            success = resp.is_success,
            output  = {
                **_build_output(resp, data, is_binary, duration_ms),
                "file_uploaded": path.name,
                "file_size_bytes": path.stat().st_size,
            },
            error = (
                None if resp.is_success
                else f"Upload HTTP {resp.status_code}: {resp.reason_phrase}"
            ),
        )

    async def _do_paginate(self, params: APIInput) -> ToolResult:
        all_items:     list[Any]   = []
        status_codes:  list[int]   = []
        next_url:      str | None  = params.url
        pages_fetched              = 0

        for _page in range(params.max_pages):
            if next_url is None:
                break

            page_result = await self._do_request(params, _url_override=next_url)
            pages_fetched += 1

            if not page_result.success:
                return ToolResult(
                    success=False,
                    error=(
                        f"Pagination failed on page {pages_fetched}: "
                        f"{page_result.error}"
                    ),
                    output={
                        "pages_fetched":  pages_fetched,
                        "items_so_far":   len(all_items),
                        "all_items":      all_items,
                        "status_codes":   status_codes,
                    },
                )

            page_output = page_result.output or {}
            status_codes.append(page_output.get("status_code", 0))

            page_data = page_output.get("data", page_output.get("body"))
            items     = _extract_items(page_data)
            all_items.extend(items)

            pagination = page_output.get("pagination", {})
            next_url   = pagination.get("next")

        return ToolResult(
            success=True,
            output={
                "pages_fetched":  pages_fetched,
                "total_items":    len(all_items),
                "all_items":      all_items,
                "status_codes":   status_codes,
                "has_more":       next_url is not None,
                "next_url":       next_url,
            },
        )

    def _build_headers(self, extra: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {
            "User-Agent": "AI Agent",
            **self._default_headers,
        }
        if self._oauth_token:
            headers["Authorization"] = f"Bearer {self._oauth_token}"
        headers.update(extra)
        return headers

def _parse_response_body(
    resp: httpx.Response,
    raw_response: bool,
) -> tuple[Any, bool]:
    content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()

    if not raw_response:
        for prefix in _BINARY_CONTENT_TYPES:
            if content_type.startswith(prefix):
                content_len = len(resp.content)
                if content_len > _MAX_RESPONSE_BYTES:
                    return f"[Binary content too large: {content_len:,} bytes]", False
                return base64.b64encode(resp.content).decode("ascii"), True

    if not raw_response and (
        content_type in _JSON_CONTENT_TYPES
        or content_type.endswith("+json")
    ):
        try:
            return resp.json(), False
        except Exception:
            pass

    text = resp.text
    if len(text) > _MAX_RESPONSE_BYTES:
        text = text[: _MAX_RESPONSE_BYTES // 2] + "\n...[truncated]"
    return text, False


def _build_output(
    resp:        httpx.Response,
    data:        Any,
    is_binary:   bool,
    duration_ms: int,
) -> dict[str, Any]:
    headers_dict = dict(resp.headers)

    output: dict[str, Any] = {
        "status_code":    resp.status_code,
        "reason":         resp.reason_phrase,
        "url":            str(resp.url),
        "duration_ms":    duration_ms,
        "response_bytes": len(resp.content),
        "headers":        headers_dict,
    }

    if is_binary:
        output["data_base64"] = data
        output["is_binary"]   = True
    else:
        output["data"] = data

    rate_remaining = resp.headers.get("x-ratelimit-remaining")
    rate_reset     = resp.headers.get("x-ratelimit-reset")
    if rate_remaining is not None:
        output["rate_limit_remaining"] = rate_remaining
    if rate_reset is not None:
        output["rate_limit_reset"] = rate_reset

    etag = resp.headers.get("etag")
    if etag:
        output["etag"] = etag

    return output


def _parse_link_header(link_header: str) -> dict[str, str]:
    if not link_header:
        return {}
    result = {}
    for match in _LINK_RE.finditer(link_header):
        url, rel = match.group(1), match.group(2)
        result[rel] = url
    return result


def _extract_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "results", "data", "objects", "records",
                    "entries", "content", "values", "list"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []


def _retry_wait(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after", "")
    if retry_after.isdigit():
        return min(float(retry_after), 30.0)
    return min(float(2 ** attempt), 30.0)


def _status_hint(status_code: int) -> str:
    hints = {
        400: "Check the request body and parameters.",
        401: "Authentication required. Check your API key or token.",
        403: "Access forbidden. Verify permissions for this resource.",
        404: "Resource not found. Verify the URL and resource ID.",
        405: "HTTP method not allowed. Check the API documentation.",
        410: "Resource permanently removed.",
        422: "Validation error. Check the request body structure.",
    }
    return hints.get(status_code, "")

async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)