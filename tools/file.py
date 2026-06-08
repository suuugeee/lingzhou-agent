"""tools/file.py — 文件读写和编辑工具。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from tools.file_helpers import (
    MAX_WRITE_CHARS,
    _fuzzy_find,
    _log_preview_text,
    _path_guard,
    _pycompile_check,
    _resolve_mutation_path,
    _safety_guard,
    _verify_python_syntax,
    resolve_read_path,
)
from tools.registry import (
    CAPS_EXEMPT,
    ToolContext,
    ToolManifest,
    ToolParam,
    ToolResult,
    tool,
    tool_metadata,
)

_log = logging.getLogger("lingzhou.tools.file")


@tool(ToolManifest(
    name="file.list",
    description="列出目录内容。支持 shallow list，用于替代 shell.run 的 ls/find 场景。",
    prefer_tier="reader",
    progress_category="info",
    capabilities=("ask_evidence", *CAPS_EXEMPT, "completion_info_only", "result_streak_only"),
    params=[
        ToolParam("path", "string", "目录路径", required=True),
        ToolParam("limit", "number", "最多返回多少项，默认 200", required=False),
        ToolParam("include_hidden", "boolean", "是否包含隐藏文件，默认 false", required=False),
    ],
))
async def file_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_read_path(Path(os.path.expanduser(params.get("path") or "")), ctx)  # noqa: ASYNC240
    limit = int(params.get("limit") or 200)
    include_hidden = bool(params.get("include_hidden", False))

    if not path.exists():
        return ToolResult(summary=f"路径不存在: {path}", error="FileNotFound")
    if not path.is_dir():
        return ToolResult(summary=f"不是目录: {path}", error="NotADirectory")

    try:
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if not include_hidden and child.name.startswith('.'):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        clipped = entries[:max(0, limit)]
        remaining = max(0, len(entries) - len(clipped))
        body = "\n".join(clipped)
        if remaining:
            body += f"\n... (+{remaining} more)"
        payload = {"path": str(path), "count": len(entries), "returned": len(clipped)}
        return ToolResult(
            summary=body or "（空目录）",
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"list:{hashlib.md5((body or '（空目录）').encode()).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            metadata=payload,
        )
    except Exception as e:
        _log.exception("列出目录失败: %s", path)
        return ToolResult(summary=f"列出失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.read",
    description=(
        "读取文件内容。支持三种方式：\n"
        "1. 不传参数 → 读全文\n"
        "2. offset + limit → 从第 offset 行开始读 limit 行（推荐，直觉友好）\n"
        "3. start + end → 按字符下标区间读（精确控制）\n"
        "⚠️ 修改文件前先用 offset/limit 读完整函数（≥20行），避免碎片化。"
    ),
    prefer_tier="reader",
    progress_category="info",
    capabilities=("ask_evidence", *CAPS_EXEMPT, "completion_info_only", "result_streak_only", "completion_verify"),
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("offset", "number", "起始行号（1-indexed），配合 limit 使用", required=False),
        ToolParam("limit", "number", "读取行数", required=False),
        ToolParam("start", "number", "起始字符下标（含），默认 0", required=False),
        ToolParam("end", "number", "结束字符下标（不含），默认到末尾", required=False),
        ToolParam("max_chars", "number", "最大字符数", required=False),
    ],
))
async def file_read(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return ToolResult(summary="path 不能为空", error="EmptyPath", skipped=True)

    path = resolve_read_path(Path(os.path.expanduser(raw_path)), ctx)  # noqa: ASYNC240
    max_chars_raw = params.get("max_chars")
    max_chars: int | None = int(max_chars_raw) if max_chars_raw is not None else None
    has_range = ("start" in params) or ("end" in params)
    has_line_range = ("offset" in params) or ("limit" in params)

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound")
    if not path.is_file():
        return ToolResult(summary=f"不是文件: {path}", error="NotAFile", skipped=True)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        total = len(text)

        start = 0
        end = total

        read_state_delta: dict[str, Any] = {}
        if has_line_range:
            lines = text.split("\n")
            line_offset = max(0, int(params.get("offset", 1)) - 1)  # 1-indexed → 0-indexed
            line_limit = int(params.get("limit", 50)) if "limit" in params else 50
            selected = lines[line_offset:line_offset + line_limit]
            text = "\n".join(selected)
            total_lines = len(lines)
            read_end = line_offset + line_limit
            has_more = read_end < total_lines
            if line_offset > 0 or has_more:
                text = f"[行 {line_offset + 1}-{min(read_end, total_lines)} / 共 {total_lines} 行]\n{text}"
            if has_more:
                read_state_delta = {
                    "has_more": True,
                    "total_lines": total_lines,
                    "next_offset": read_end + 1,  # 1-indexed，直接传给下一次 file.read offset
                }
        elif has_range:
            start = int(params.get("start") or 0)
            end_raw = params.get("end")
            end = int(end_raw) if end_raw is not None else total
            text = text[start:end]

        if max_chars is not None:
            text = text[:max(0, max_chars)]

        log_summary = (
            f"file.read path={path} chars={len(text)}"
            + (f" range=[{start}:{end}]" if has_range else "")
            + (f" preview={_log_preview_text(text)!r}" if text else " preview=''")
        )
        return ToolResult(
            summary=text,
            resource_key=str(path),
            fingerprint=f"read:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta=read_state_delta,
            metadata=tool_metadata(
                "file.read",
                log_summary,
                path=str(path),
                chars=len(text),
                has_range=has_range,
                start=start,
                end=end,
                max_chars=max_chars,
            ),
        )
    except Exception as e:
        _log.exception("读取文件失败: %s", path)
        return ToolResult(summary=f"读取失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.write",
    description="写入文件内容。如果文件已存在则覆盖全部内容。创建新文件时自动创建父目录。",
    progress_category="mutation",
    capabilities=("completion_mutation",),
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("content", "string", "要写入的内容", required=True),
    ],
))
async def file_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(os.path.expanduser(params.get("path") or "")), ctx)  # noqa: ASYNC240
    content = params.get("content")

    if content is None:
        return ToolResult(summary="写入内容为空", error="EmptyContent", skipped=True)

    # 路径守卫：沙箱 + 穿越检测
    ok, err = _path_guard(path, ctx)
    if not ok:
        return ToolResult(summary=err, error="PathBlocked", skipped=True)

    text = str(content)
    if len(text) > MAX_WRITE_CHARS:
        return ToolResult(
            summary=f"内容超出大小限制: {len(text)} > {MAX_WRITE_CHARS} 字符",
            error="ContentTooLarge", skipped=True
        )

    try:
        # 安全守卫：备份原文件 + 核心文件警告
        guard_warning, _ = _safety_guard(path)

        # 目录保护：路径是目录而非文件时给出明确提示
        if path.is_dir():
            return ToolResult(
                summary=f"无法写入：{path} 是一个目录，不是文件。请指定具体文件路径（如 {path}/README.md）。",
                error="IsDirectory",
                skipped=True,
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        # 写前语法验证（.py 文件）：语法错误直接阻断，不写入损坏文件
        if path.suffix == ".py":
            syntax_error = _verify_python_syntax(path, text)
            if syntax_error:
                return ToolResult(
                    summary=f"拒绝写入: {path}\n{syntax_error}\n备份（若有）: {path.with_suffix('.py.lingzhou-backup')}",
                    error="PythonSyntaxError",
                    skipped=True,
                )
        # 原子写入：先写临时文件，成功后再 rename
        tmp_path = path.with_suffix(path.suffix + ".lingzhou-tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)  # 原子 rename
        summary = f"写入成功: {path} ({len(text)} 字符)"
        if guard_warning:
            summary += f"\n{guard_warning}"
        compile_err = _pycompile_check(path)
        if compile_err:
            summary += f"\n⚠️ {compile_err}"
        syntax_ok = compile_err is None
        return ToolResult(
            summary=summary,
            resource_key=str(path),
            fingerprint=f"write:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "written", "chars": len(text), "syntax_ok": syntax_ok},
            metadata=tool_metadata(
                "file.write",
                f"file.write path={path} chars={len(text)} syntax_ok={syntax_ok}",
                path=str(path),
                chars=len(text),
                syntax_ok=syntax_ok,
            ),
        )
    except Exception as e:
        _log.exception("写入文件失败: %s", path)
        return ToolResult(summary=f"写入失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.edit",
    description=(
        "对文件进行精确文本替换。支持单处或多处替换（edit 列表）。"
        "每个 edit 包含 oldText（原文本）和 newText（新文本），oldText 必须在文件中唯一匹配。"
        "这是修改文件的首选工具——相比全量覆盖的 file.write，edit 只改需要改的部分，安全且节省 token。"
    ),
    progress_category="mutation",
    capabilities=("completion_mutation",),
        params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("edits", "array",
                  "替换操作列表，每项包含 oldText（要替换的原文）和 newText（替换后的内容）。"
                  "必须是数组，即使只改一处也要用数组包裹。"
                  "例: [{\"oldText\": \"foo\", \"newText\": \"bar\"}, {\"oldText\": \"baz\", \"newText\": \"qux\"}]",
                  required=True),
    ],
))
async def file_edit(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(os.path.expanduser(params.get("path") or "")), ctx)  # noqa: ASYNC240
    edits_raw = params.get("edits")

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}（edit 只能修改已存在的文件，新文件请用 file.write）", error="FileNotFound")

    if not edits_raw:
        return ToolResult(summary="edits 参数为空，请提供至少一个 {oldText, newText} 替换操作", error="EmptyEdits", skipped=True)

    # 支持 list, dict(单条), 或 JSON 字符串
    if isinstance(edits_raw, str):
        try:
            edits = json.loads(edits_raw)
        except json.JSONDecodeError:
            return ToolResult(summary="edits 不是合法的 JSON 数组", error="InvalidJSON")
    elif isinstance(edits_raw, dict):
        edits = [edits_raw]
    elif isinstance(edits_raw, list):
        edits = edits_raw
    else:
        return ToolResult(summary="edits 必须是数组、字典或 JSON 字符串", error="InvalidType", skipped=True)

    try:
        # 目录保护
        if path.is_dir():
            return ToolResult(
                summary=f"无法编辑：{path} 是一个目录，不是文件。请指定具体文件路径。",
                error="IsDirectory",
                skipped=True,
            )

        # 路径守卫
        ok, err = _path_guard(path, ctx)
        if not ok:
            return ToolResult(summary=err, error="PathBlocked", skipped=True)

        # 安全守卫：备份原文件 + 核心文件警告
        guard_warning, _ = _safety_guard(path)
        original = path.read_text(encoding="utf-8", errors="replace")
        content = original
        matched: list[dict[str, Any]] = []
        applied = []

        for i, edit in enumerate(edits):
            old_text = (edit.get("oldText") or edit.get("old_text") or "") if isinstance(edit, dict) else ""
            new_text = (edit.get("newText") or edit.get("new_text") or "") if isinstance(edit, dict) else ""

            if not old_text:
                return ToolResult(summary=f"edits[{i}]: oldText 不能为空", error="EmptyOldText", skipped=True)
            # 允许 newText 为空（用于删除文本场景），不再抛出 EmptyNewText

            # 所有 edit 都基于原始文件匹配，而不是增量匹配修改后的内容。
            first_idx = original.find(old_text)
            if first_idx == -1:
                # 模糊匹配链：依次尝试宽松策略
                fuzzy_idx = _fuzzy_find(original, old_text)
                if fuzzy_idx != -1:
                    first_idx = fuzzy_idx
                    _log.info("[file.edit] 使用模糊匹配找到 oldText")
                else:
                    # 查找 partial match 帮助 LLM 定位
                    first_line = old_text.split("\n")[0]
                    partial_idx = original.find(first_line) if len(first_line) > 10 else -1
                    context = ""
                    if partial_idx != -1:
                        ctx_start = max(0, partial_idx - 150)
                        ctx_end = min(len(original), partial_idx + len(first_line) + 300)
                        context = f"\n实际内容（该位置附近）:\n{original[ctx_start:ctx_end]}"
                    else:
                        # 完全没找到，返回完整文件内容，让 LLM 自己定位并重建 oldText
                        context = f"\n完整文件内容:\n{original}"
                    # 模板文件警告：prompts/ 下的文件是 Jinja2 渲染模板，不应以渲染结果作 oldText
                    template_warning = ""
                    if "prompts/" in str(path) or (str(path).endswith(".md") and "{" in original):
                        template_warning = (
                            "\n⚠️ 模板文件警告：prompts/ 下的文件是 Jinja2 渲染模板，"
                            "文件实际内容含有 {variable_name} 占位符，与 LLM 每轮看到的已渲染内容完全不同。"
                            "请用 file.read 读取模板原文后再重试，不要用已渲染数据内容作 oldText。"
                        )
                    return ToolResult(
                        summary=f"edits[{i}]: oldText 在文件中未找到。{template_warning}{context}\n请用 file.read 确认完整的当前内容后重试。",
                        error="OldTextNotFound",
                        skipped=True,
                        metadata=tool_metadata(
                            "file.edit",
                            f"file.edit OldTextNotFound partial={partial_idx != -1}",
                            partial_match=partial_idx != -1,
                        ),
                    )

            second_idx = original.find(old_text, first_idx + 1)
            if second_idx != -1:
                return ToolResult(
                    summary=(
                        f"edits[{i}]: oldText 在文件中出现 {original.count(old_text)} 次，不够唯一。"
                        f"请扩大 oldText 范围使其唯一，或拆分为多次 edit 调用。"
                    ),
                    error="NonUniqueOldText",
                    skipped=True,
                )

            matched.append({
                "index": i,
                "start": first_idx,
                "end": first_idx + len(old_text),
                "new_text": new_text,
            })

        matched.sort(key=lambda item: int(item["start"]))
        for prev, cur in zip(matched, matched[1:], strict=False):
            if int(prev["end"]) > int(cur["start"]):
                return ToolResult(
                    summary=(
                        f"edits[{prev['index']}] 与 edits[{cur['index']}] 在原文件中发生重叠。"
                        "请合并成一次 edit，或拆成互不重叠的目标区域。"
                    ),
                    error="OverlappingEdits",
                    skipped=True,
                )

        for item in reversed(matched):
            start = int(item["start"])
            end = int(item["end"])
            content = content[:start] + str(item["new_text"]) + content[end:]
            applied.append({
                "index": item["index"],
            })

        changes_made = len(matched)
        if content == original:
            return ToolResult(summary=f"编辑未产生变化: {path}", error="NoChange", skipped=True)

        # 写前语法验证（.py 文件）：语法错误直接阻断，不写入损坏文件
        if path.suffix == ".py":
            syntax_error = _verify_python_syntax(path, content)
            if syntax_error:
                return ToolResult(
                    summary=f"拒绝编辑: {path}\n{syntax_error}\n备份（若有）: {path.with_suffix('.py.lingzhou-backup')}",
                    error="PythonSyntaxError",
                    skipped=True,
                )
        # 原子写入
        tmp_path = path.with_suffix(path.suffix + ".lingzhou-tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
        summary = f"编辑成功: {path}（{changes_made} 处替换）"
        if guard_warning:
            summary += f"\n{guard_warning}"
        compile_err = _pycompile_check(path)
        if compile_err:
            summary += f"\n⚠️ {compile_err}"
        syntax_ok = compile_err is None
        payload = {"path": str(path), "changes": changes_made, "applied": applied}
        return ToolResult(
            summary=summary,
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"edit:{hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "edited", "changes": changes_made, "syntax_ok": syntax_ok},
            metadata=tool_metadata(
                "file.edit",
                f"file.edit {path} changes={changes_made}",
                **payload,
                syntax_ok=syntax_ok,
            ),
        )
    except Exception as e:
        _log.exception("编辑文件失败: %s", path)
        return ToolResult(summary=f"编辑失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.delete",
    description=(
        "删除 workspace 中的指定文件。"
        "⚠️ 不可逆操作，请确认路径正确。"
        "主要用途：初始化完成后删除 BOOTSTRAP.md，或清理临时文件。"
    ),
    progress_category="mutation",
    capabilities=("completion_mutation",),
    params=[
        ToolParam("path", "string", "要删除的文件路径（相对于 workspace 或绝对路径）", required=True),
    ],
))
async def file_delete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(os.path.expanduser(params.get("path") or "")), ctx)  # noqa: ASYNC240

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound", skipped=True)

    if path.is_dir():
        return ToolResult(summary=f"目标是目录而非文件: {path}（file.delete 仅删除文件）", error="IsDirectory", skipped=True)

    ok, err = _path_guard(path, ctx)
    if not ok:
        return ToolResult(summary=err, error="PathBlocked", skipped=True)

    try:
        path.unlink()
        _log.info("file.delete: 已删除 %s", path)
        return ToolResult(
            summary=f"已删除: {path}",
            state_delta={"file": "deleted", "path": str(path)},
            resource_key=str(path),
        )
    except Exception as e:
        _log.exception("删除文件失败: %s", path)
        return ToolResult(summary=f"删除失败: {path}", error=type(e).__name__)
