"""tools/web.py — web_fetch + web_search 工具。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool, tool_metadata

# ── 常量 ─────────────────────────────────────────────────────────────────────
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
MAX_FETCH_CHARS = 50000
MAX_SEARCH_RESULTS = 10
HTTP_TIMEOUT = httpx.Timeout(connect=25.0, read=35.0, write=15.0, pool=10.0)


# ── 共享 HTTP 客户端 ─────────────────────────────────────────────────────────
_http_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        _http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=3),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_UA},
            proxy=proxy,
        )
    return _http_client


async def _safe_request(method: str, url: str, **kwargs) -> httpx.Response:
    """带重试的 HTTP 请求包装器，缓解 ConnectTimeout / 短暂网络抖动。"""
    client = await _get_client()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.ConnectTimeout as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
        except httpx.RequestError as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.8)
                continue
            raise
    raise last_err  # type: ignore[misc]


# ── HTML → 纯文本 ────────────────────────────────────────────────────────────


def _html_to_text(html: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """将 HTML 转为可读纯文本。"""
    # 移除 script/style
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # 移除标签
    text = re.sub(r"<[^>]+>", " ", html)
    # 处理实体
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # 压缩空白
    text = re.sub(r"\s+", " ", text).strip()
    # 去重空行
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n...(截断，原文共 {len(result)} 字符)"
    return result


# ── web.fetch ────────────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="web.fetch",
    description=(
        "获取 Web 页面内容。可用于阅读在线文档、API 文档、论文、博客等。"
        "自动将 HTML 转为纯文本。"
    ),
    progress_category="io",
    params=[
        ToolParam("url", "string", "页面 URL", required=True),
        ToolParam("max_chars", "number", "最大返回字符数（默认 50000）", required=False),
    ],
))
async def web_fetch(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = (params.get("url") or "").strip()
    if not url:
        return ToolResult(summary="URL 不能为空", error="EmptyUrl", skipped=True)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolResult(summary=f"不支持的协议: {parsed.scheme}", error="BadScheme", skipped=True)

    max_chars = min(int(params.get("max_chars", MAX_FETCH_CHARS)), MAX_FETCH_CHARS)

    try:
        resp = await _safe_request("GET", url)
        content_type = resp.headers.get("content-type", "")
        text: str

        if "text/html" in content_type:
            text = _html_to_text(resp.text, max_chars)
        elif "text/" in content_type or "application/json" in content_type:
            text = resp.text[:max_chars]
            if len(resp.text) > max_chars:
                text += f"\n...(截断，原文共 {len(resp.text)} 字符)"
        else:
            return ToolResult(
                summary=f"不支持的内容类型: {content_type}",
                error="UnsupportedContentType",
                skipped=True,
            )

        return ToolResult(
            summary=f"获取成功: {url}\n状态: {resp.status_code}  大小: {len(text)} 字符",
            resource_key=url,
            evidence=text,
            metadata=tool_metadata(
                "web.fetch",
                f"web.fetch url={url} status={resp.status_code} chars={len(text)}",
                url=url,
                status=resp.status_code,
                chars=len(text),
                content_type=content_type,
            ),
            state_delta={"fetched": url, "chars": len(text)},
        )
    except httpx.HTTPStatusError as e:
        return ToolResult(summary=f"HTTP {e.response.status_code}: {url}", error="HttpError")
    except httpx.TimeoutException:
        return ToolResult(summary=f"请求超时: {url}", error="Timeout")
    except Exception as e:
        detail = str(e).strip()
        suffix = f": {detail}" if detail else ""
        return ToolResult(summary=f"获取失败: {type(e).__name__}{suffix}", error="FetchError")


# ── web.search ───────────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="web.search",
    description=(
        "搜索 Web 信息。使用 DuckDuckGo 匿名搜索。"
        "返回标题、摘要和 URL。适合查找文档、论文、技术问题等。"
    ),
    progress_category="io",
    params=[
        ToolParam("query", "string", "搜索关键词", required=True),
        ToolParam("max_results", "number", "最大结果数（默认 10）", required=False),
    ],
))
async def web_search(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = (params.get("query") or "").strip()
    if not query:
        return ToolResult(summary="搜索关键词不能为空", error="EmptyQuery", skipped=True)

    max_results = min(int(params.get("max_results", MAX_SEARCH_RESULTS)), MAX_SEARCH_RESULTS)

    try:
        # DuckDuckGo HTML 搜索（无需 API key）
        search_url = "https://html.duckduckgo.com/html/"
        resp = await _safe_request("POST", search_url, data={"q": query})

        # 解析搜索结果
        results: list[dict[str, str]] = []
        for match in re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            resp.text,
            re.DOTALL,
        ):
            if len(results) >= max_results:
                break
            url = match.group(1).strip()
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            if url and title:
                results.append({"title": title, "url": url, "snippet": snippet})

        if not results:
            return ToolResult(
                summary=f"搜索无结果: {query}",
                evidence="",
                metadata=tool_metadata(
                    "web.search",
                    f"web.search query={query!r} results=0",
                    query=query,
                    results=0,
                ),
            )

        summary_lines = [f"搜索 '{query}': {len(results)} 条结果"]
        for i, r in enumerate(results, 1):
            summary_lines.append(f"  [{i}] {r['title']}")
            summary_lines.append(f"      {r['url']}")
            if r["snippet"]:
                summary_lines.append(f"      {r['snippet']}")

        return ToolResult(
            summary="\n".join(summary_lines),
            resource_key=f"search:{hashlib.md5(query.encode()).hexdigest()[:12]}",
            evidence=summary_lines[-1] if len(summary_lines) > 1 else "",
            metadata=tool_metadata(
                "web.search",
                f"web.search query={query!r} results={len(results)}",
                query=query,
                results=len(results),
            ),
            state_delta={"searched": query, "results": len(results)},
        )
    except Exception as e:
        detail = str(e).strip()
        suffix = f": {detail}" if detail else ""
        return ToolResult(summary=f"搜索失败: {type(e).__name__}{suffix}", error="SearchError")
