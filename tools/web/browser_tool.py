from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import time
import urllib.parse
import urllib.robotparser
from collections import defaultdict
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult

log = get_logger(__name__)

MAX_TEXT_CHARS   = 12_000  
MAX_SNIPPET_CHARS = 600    
MAX_LINKS        = 30      
MAX_SEARCH_RESULTS = 10    

MIN_MEANINGFUL_TEXT = 150

PLAYWRIGHT_TIMEOUT_MS = 30_000

ROBOTS_CACHE_TTL = 3600
RATE_LIMIT_TOKENS  = 12   
RATE_LIMIT_PER_SEC = 12 / 60 

USER_AGENT = (
    "Mozilla/5.0 (compatible; SuperAgent/1.0; +https://github.com/you/superagent)"
)

JS_FRAMEWORK_PATTERNS = re.compile(
    r'<div id="(root|app|__next|nuxt|ember-application)"',
    re.IGNORECASE,
)
SEARCH_ENGINES = [
    {
        "name":    "duckduckgo",
        "url":     "https://api.duckduckgo.com/",
        "params":  {"format": "json", "no_redirect": "1", "no_html": "1"},
        "parser":  "_parse_ddg",
    },
]

class BrowserInput(BaseModel):
    action: str = Field(
        default="fetch",
        description="'fetch' | 'search' | 'screenshot'",
    )
    url: str | None = Field(
        default=None,
        description="URL to fetch or screenshot. Required for fetch and screenshot.",
    )
    query: str | None = Field(
        default=None,
        description="Search query. Required for action=search.",
    )
    render_js: bool = Field(
        default=False,
        description="Force Playwright JS rendering even for static pages.",
    )
    extract_links: bool = Field(
        default=True,
        description="Include outbound links in fetch output.",
    )
    extract_metadata: bool = Field(
        default=True,
        description="Include page metadata (title, og:*, dates) in fetch output.",
    )
    respect_robots: bool = Field(
        default=True,
        description="Check robots.txt before fetching. Recommended True.",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Number of search results (1-10).",
    )
    safe_search: bool = Field(
        default=True,
        description="Enable moderate safe search filter.",
    )
    full_page: bool = Field(
        default=True,
        description="For screenshot: capture full scrollable page height.",
    )
    viewport_width: int = Field(default=1280, ge=320, le=3840)
    viewport_height: int = Field(default=800,  ge=240, le=2160)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"fetch", "search", "screenshot"}
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http or https, got: {v!r}")
        if not parsed.netloc:
            raise ValueError(f"URL has no host: {v!r}")
        return v

class _RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": RATE_LIMIT_TOKENS, "last_refill": time.monotonic()}
        )

    def check(self, hostname: str) -> tuple[bool, float]:
        bucket = self._buckets[hostname]
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        # Refill tokens
        refill = elapsed * RATE_LIMIT_PER_SEC
        bucket["tokens"] = min(RATE_LIMIT_TOKENS, bucket["tokens"] + refill)
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, 0.0
        else:
            wait = (1.0 - bucket["tokens"]) / RATE_LIMIT_PER_SEC
            return False, wait


_rate_limiter = _RateLimiter()

_robots_cache: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}

_response_cache: dict[str, ToolResult] = {}


class BrowserTool(BaseTool):

    name = "browser"
    description = (
        "Fetch web pages, run web searches, or take screenshots. "
    )
    manifest = PermissionManifest(network_domains=["*"])
    input_schema = BrowserInput

    def __init__(
        self,
        brave_api_key: str | None = None,
        searxng_url: str | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._brave_api_key  = brave_api_key
        self._searxng_url    = searxng_url
        self._cache_ttl      = cache_ttl_seconds

        self._search_engines = list(SEARCH_ENGINES)
        if brave_api_key:
            self._search_engines.insert(0, {
                "name":   "brave",
                "url":    "https://api.search.brave.com/res/v1/web/search",
                "params": {"count": str(MAX_SEARCH_RESULTS)},
                "headers": {"Accept": "application/json", "X-Subscription-Token": brave_api_key},
                "parser": "_parse_brave",
            })
        if searxng_url:
            self._search_engines.append({
                "name":   "searxng",
                "url":    f"{searxng_url.rstrip('/')}/search",
                "params": {"format": "json"},
                "parser": "_parse_searxng",
            })

        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def execute(self, input: ToolInput) -> ToolResult:
        params = self.input_schema(**input.parameters)

        if params.action == "fetch":
            if not params.url:
                return ToolResult(
                    success=False,
                    error="action='fetch' requires a 'url' parameter.",
                )
            return await self._fetch(params, session_id=input.session_id)

        if params.action == "search":
            if not params.query:
                return ToolResult(
                    success=False,
                    error="action='search' requires a 'query' parameter.",
                )
            return await self._search(params)

        if params.action == "screenshot":
            if not params.url:
                return ToolResult(
                    success=False,
                    error="action='screenshot' requires a 'url' parameter.",
                )
            return await self._screenshot(params)

        return ToolResult(success=False, error=f"Unknown action: {params.action!r}")


    async def _fetch(self, params: BrowserInput, session_id: str) -> ToolResult:
        url      = params.url
        hostname = urllib.parse.urlparse(url).hostname or ""

        allowed, retry_after = _rate_limiter.check(hostname)
        if not allowed:
            return ToolResult(
                success=False,
                error=(
                    f"Rate limit reached for {hostname!r}. "
                    f"Retry after {retry_after:.1f} seconds."
                ),
                metadata={"retry_after": retry_after},
            )

        if params.respect_robots:
            robots_ok, robots_reason = await self._check_robots(url, hostname)
            if not robots_ok:
                return ToolResult(
                    success=False,
                    error=f"robots.txt disallows fetching {url!r}: {robots_reason}",
                )

        cache_key = self._cache_key(url, session_id)
        cached = _response_cache.get(cache_key)
        if cached is not None:
            log.debug("browser_cache_hit", url=url)
            cached.cached = True
            return cached

        raw_html, final_url, status_code = await self._httpx_get_with_retry(url)
        if raw_html is None:
            # status_code holds the error string in this case
            return ToolResult(success=False, error=str(status_code))

        needs_js = params.render_js or self._needs_javascript(raw_html)
        if needs_js:
            log.debug("browser_upgrading_to_playwright", url=url)
            rendered_html = await self._playwright_fetch(url)
            if rendered_html:
                raw_html = rendered_html

        title, clean_text = self._extract_text(raw_html, url)

        metadata: dict[str, Any] = {}
        if params.extract_metadata:
            metadata = self._extract_metadata(raw_html, final_url)

        links: list[dict] = []
        if params.extract_links:
            links = self._extract_links(raw_html, final_url)

        output: dict[str, Any] = {
            "url":         final_url,
            "title":       title,
            "text":        clean_text[:MAX_TEXT_CHARS],
            "text_length": len(clean_text),
            "truncated":   len(clean_text) > MAX_TEXT_CHARS,
            "rendered_js": needs_js,
            "status_code": status_code,
        }
        if params.extract_metadata:
            output["metadata"] = metadata
        if params.extract_links:
            output["links"] = links[:MAX_LINKS]

        result = ToolResult(success=True, output=output)

        if not needs_js:
            _response_cache[cache_key] = result

        return result

    async def _httpx_get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
    ) -> tuple[str | None, str, int | str]:
        http = await self._get_http()
        last_error: str = ""

        for attempt in range(max_retries):
            try:
                resp = await http.get(url)

                if resp.status_code in (400, 401, 403, 404, 410):
                    return None, str(resp.url), (
                        f"HTTP {resp.status_code} for {url!r}. "
                        + {
                            400: "Bad request.",
                            401: "Authentication required.",
                            403: "Access forbidden.",
                            404: "Page not found.",
                            410: "Page permanently removed.",
                        }.get(resp.status_code, "")
                    )

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 2 ** attempt
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After", "")
                        if retry_after.isdigit():
                            wait = min(int(retry_after), 30)
                    log.debug(
                        "browser_transient_error",
                        url=url, status=resp.status_code,
                        attempt=attempt + 1, wait=wait,
                    )
                    last_error = f"HTTP {resp.status_code}"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait)
                    continue

                content_length = int(resp.headers.get("content-length", 0))
                if content_length > self.manifest.max_response_bytes:
                    return None, str(resp.url), (
                        f"Response too large: {content_length:,} bytes "
                        f"(limit: {self.manifest.max_response_bytes:,})"
                    )

                html = resp.text
                if len(html.encode()) > self.manifest.max_response_bytes:
                    html = html[: self.manifest.max_response_bytes // 2]

                return html, str(resp.url), resp.status_code

            except httpx.TimeoutException as exc:
                last_error = f"Request timed out: {exc}"
                log.debug("browser_timeout", url=url, attempt=attempt + 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except httpx.ConnectError as exc:
                last_error = f"Connection failed: {exc}"
                log.debug("browser_connect_error", url=url, attempt=attempt + 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except httpx.HTTPError as exc:
                last_error = f"HTTP error: {exc}"
                break  # Non-retryable HTTP errors

        return None, url, f"Failed after {max_retries} attempts: {last_error}"

    def _needs_javascript(self, html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        if len(text) < MIN_MEANINGFUL_TEXT:
            if JS_FRAMEWORK_PATTERNS.search(html):
                return True

        for ns in soup.find_all("noscript"):
            ns_text = ns.get_text().lower()
            if "javascript" in ns_text and (
                "enable" in ns_text or "required" in ns_text or "disabled" in ns_text
            ):
                return True

        return False

    async def _playwright_fetch(self, url: str) -> str | None:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            log.warning("playwright_not_installed", url=url)
            return None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=USER_AGENT,
                        viewport={"width": 1280, "height": 800},
                        java_script_enabled=True,
                    )
                    page = await context.new_page()

                    await page.route(
                        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,otf,mp4,mp3}",
                        lambda r: r.abort(),
                    )

                    await page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=PLAYWRIGHT_TIMEOUT_MS,
                    )

                    await page.wait_for_timeout(500)

                    html = await page.content()
                    return html

                finally:
                    await browser.close()

        except Exception as exc:
            log.warning("playwright_fetch_failed", url=url, error=str(exc))
            return None

    async def _screenshot(self, params: BrowserInput) -> ToolResult:
        url = params.url

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "Playwright is not installed. "
                    "Install it: poetry add playwright && playwright install chromium"
                ),
            )

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=USER_AGENT,
                        viewport={
                            "width":  params.viewport_width,
                            "height": params.viewport_height,
                        },
                    )
                    page = await context.new_page()
                    await page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=PLAYWRIGHT_TIMEOUT_MS,
                    )
                    await page.wait_for_timeout(500)

                    screenshot_bytes = await page.screenshot(full_page=params.full_page)
                    b64 = base64.b64encode(screenshot_bytes).decode("ascii")

                    title = await page.title()

                    return ToolResult(
                        success=True,
                        output={
                            "url":             url,
                            "title":           title,
                            "screenshot_b64":  b64,
                            "format":          "png",
                            "size_bytes":      len(screenshot_bytes),
                            "viewport_width":  params.viewport_width,
                            "viewport_height": params.viewport_height,
                            "full_page":       params.full_page,
                        },
                    )
                finally:
                    await browser.close()

        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Screenshot failed for {url!r}: {exc}",
            )

    async def _search(self, params: BrowserInput) -> ToolResult:
        query = params.query
        errors: list[str] = []

        for engine in self._search_engines:
            try:
                results = await self._call_search_engine(engine, params)
                if results:
                    return ToolResult(
                        success=True,
                        output={
                            "query":          query,
                            "results":        results[: params.num_results],
                            "total_returned": len(results),
                            "engine":         engine["name"],
                        },
                        metadata={"engine_used": engine["name"]},
                    )
                errors.append(f"{engine['name']}: empty results")

            except Exception as exc:
                error_msg = f"{engine['name']}: {exc}"
                errors.append(error_msg)
                log.debug("search_engine_failed", engine=engine["name"], error=str(exc))
                continue

        return ToolResult(
            success=False,
            error=(
                f"All search engines failed for query {query!r}. "
                f"Errors: {'; '.join(errors)}"
            ),
        )

    async def _call_search_engine(
        self,
        engine: dict,
        params: BrowserInput,
    ) -> list[dict]:
        http  = await self._get_http()
        query = params.query

        request_params = {**engine.get("params", {}), "q": query}
        if not params.safe_search:
            request_params["kp"] = "-1" 

        headers = {**engine.get("headers", {})}

        resp = await http.get(engine["url"], params=request_params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        parser_name = engine.get("parser", "")
        parser = getattr(self, parser_name, None)
        if parser is None:
            return []
        return parser(data, params.num_results)

    def _parse_ddg(
        self, data: dict, num_results: int
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []

        abstract_text = data.get("AbstractText", "")
        abstract_url  = data.get("AbstractURL", "")
        abstract_src  = data.get("AbstractSource", "")
        if abstract_text and abstract_url:
            results.append({
                "title":   data.get("Heading", abstract_src),
                "url":     abstract_url,
                "snippet": abstract_text[:MAX_SNIPPET_CHARS],
                "source":  "duckduckgo_abstract",
            })

        for topic in data.get("RelatedTopics", []):
            if len(results) >= num_results:
                break

            if "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= num_results:
                        break
                    result = self._parse_ddg_topic(sub)
                    if result:
                        results.append(result)
            else:
                result = self._parse_ddg_topic(topic)
                if result:
                    results.append(result)

        return results

    @staticmethod
    def _parse_ddg_topic(topic: dict) -> dict[str, str] | None:
        url  = topic.get("FirstURL", "")
        text = topic.get("Text", "")
        if not url or not text:
            return None
        parts = text.split(" - ", 1)
        title   = parts[0].strip() if len(parts) > 1 else text[:80]
        snippet = parts[1].strip() if len(parts) > 1 else text
        return {
            "title":   title,
            "url":     url,
            "snippet": snippet[:MAX_SNIPPET_CHARS],
            "source":  "duckduckgo",
        }

    def _parse_brave(
        self, data: dict, num_results: int
    ) -> list[dict[str, str]]:
        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("description", "")[:MAX_SNIPPET_CHARS],
                "source":  "brave",
            })
        return results

    def _parse_searxng(
        self, data: dict, num_results: int
    ) -> list[dict[str, str]]:
        results = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("content", "")[:MAX_SNIPPET_CHARS],
                "source":  "searxng",
            })
        return results

    def _extract_text(self, html: str, url: str) -> tuple[str, str]:
        try:
            from readability import Document
            doc   = Document(html)
            title = doc.title() or ""
            article_html = doc.summary(html_partial=True)
        except Exception:
            soup  = BeautifulSoup(html, "html.parser")
            title = soup.title.string if soup.title else ""
            article_html = html

        soup = BeautifulSoup(article_html, "html.parser")

        noise_selectors = [
            "script", "style", "noscript", "iframe",
            "nav", "header", "footer", "aside",
            ".ad", ".ads", ".advertisement", ".banner",
            ".cookie-banner", ".newsletter-signup",
            ".social-share", ".related-articles",
            '[role="navigation"]', '[role="banner"]',
            '[aria-label="advertisement"]',
        ]
        for sel in noise_selectors:
            for tag in soup.select(sel):
                tag.decompose()

        raw_text = soup.get_text(separator="\n", strip=True)

        lines = raw_text.splitlines()
        cleaned_lines: list[str] = []
        blank_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped:
                blank_count = 0
                cleaned_lines.append(stripped)
            else:
                blank_count += 1
                if blank_count <= 1:
                    cleaned_lines.append("")

        clean_text = "\n".join(cleaned_lines).strip()
        return title.strip(), clean_text

    def _extract_metadata(self, html: str, url: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        meta: dict[str, Any] = {}

        title_tag = soup.find("title")
        meta["title"] = title_tag.get_text(strip=True) if title_tag else ""

        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta["language"] = str(html_tag["lang"])[:10]

        for tag in soup.find_all("meta"):
            name    = (tag.get("name", "") or "").lower()
            prop    = (tag.get("property", "") or "").lower()
            content = tag.get("content", "") or ""

            if name == "description":
                meta["description"] = content[:500]
            elif name == "author":
                meta["author"] = content[:200]
            elif name == "keywords":
                meta["keywords"] = content[:300]
            elif name in ("published_time", "article:published_time") or prop == "article:published_time":
                meta["published_date"] = content[:30]
            elif name in ("modified_time", "article:modified_time") or prop == "article:modified_time":
                meta["modified_date"] = content[:30]
            elif prop == "og:title":
                meta["og_title"] = content[:300]
            elif prop == "og:description":
                meta["og_description"] = content[:500]
            elif prop == "og:image":
                meta["og_image"] = content[:500]
            elif prop == "og:type":
                meta["og_type"] = content[:50]
            elif prop == "og:url":
                meta["og_url"] = content[:500]
            elif name == "twitter:title" and "og_title" not in meta:
                meta["og_title"] = content[:300]
            elif name == "twitter:description" and "og_description" not in meta:
                meta["og_description"] = content[:500]

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            meta["canonical_url"] = canonical["href"]
        else:
            meta["canonical_url"] = url

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                ld = json.loads(script.string or "{}")
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    if isinstance(item, dict):
                        if "datePublished" in item and "published_date" not in meta:
                            meta["published_date"] = str(item["datePublished"])[:30]
                        if "dateModified" in item and "modified_date" not in meta:
                            meta["modified_date"] = str(item["dateModified"])[:30]
                        if "author" in item and "author" not in meta:
                            author = item["author"]
                            if isinstance(author, dict):
                                meta["author"] = author.get("name", "")[:200]
                            elif isinstance(author, str):
                                meta["author"] = author[:200]
            except Exception:
                pass

        return meta

    def _extract_links(self, html: str, base_url: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        base_hostname = urllib.parse.urlparse(base_url).hostname or ""

        seen_urls: set[str] = set()
        links: list[dict[str, str]] = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#") or href.lower().startswith("javascript:"):
                continue

            try:
                full_url = urllib.parse.urljoin(base_url, href)
                parsed   = urllib.parse.urlparse(full_url)
            except Exception:
                continue

            if parsed.scheme not in ("http", "https"):
                continue

            normalized = parsed._replace(fragment="").geturl()
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            link_text    = a_tag.get_text(strip=True)[:200]
            link_hostname = parsed.hostname or ""
            is_external  = link_hostname != base_hostname and bool(link_hostname)

            links.append({
                "text":        link_text or normalized,
                "url":         normalized,
                "is_external": str(is_external).lower(),
            })

            if len(links) >= MAX_LINKS:
                break

        return links

    async def _check_robots(self, url: str, hostname: str) -> tuple[bool, str]:
        now = time.monotonic()
        cached_entry = _robots_cache.get(hostname)
        if cached_entry:
            parser, expires = cached_entry
            if now < expires:
                can_fetch = parser.can_fetch(USER_AGENT, url)
                return can_fetch, ("" if can_fetch else "disallowed by robots.txt")

        robots_url = f"{urllib.parse.urlparse(url).scheme}://{hostname}/robots.txt"
        try:
            http = await self._get_http()
            resp = await http.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(resp.text.splitlines())
                _robots_cache[hostname] = (parser, now + ROBOTS_CACHE_TTL)
                can_fetch = parser.can_fetch(USER_AGENT, url)
                return can_fetch, ("" if can_fetch else "disallowed by robots.txt")
        except Exception:
            pass  

        return True, ""

    def _cache_key(self, url: str, session_id: str) -> str:
        raw = f"{session_id}:{url}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def clear_cache(session_id: str | None = None) -> int:
        if session_id is None:
            count = len(_response_cache)
            _response_cache.clear()
            return count
        count = len(_response_cache)
        _response_cache.clear()
        return count