from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from .smoke import smoke_test_module, write_pending_verification_fact
from .types import EvolutionResult, _clean_old_backups

if TYPE_CHECKING:
    from core.evolution import EvolutionEngine
    from store.task import Failure
    from tools.registry import ToolContext


def _score_candidate(code: str) -> int:
    """对候选代码进行静态质量评分（越高越好），用于竞争进化排名。"""

    score = 100
    lines = code.splitlines()
    n = len(lines)
    except_count = sum(1 for ln in lines if ln.strip().startswith("except"))
    score += min(except_count * 10, 50)
    if "_log." in code or "logging." in code:
        score += 5
    if 50 <= n <= 300:
        score += 5
    elif n > 400:
        score -= 10
    print_count = sum(1 for ln in lines if ln.strip().startswith("print("))
    score -= print_count * 5
    return score


def _extract_python(text: str) -> str:
    """从 LLM 输出中提取 Python 代码块。"""

    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def find_tool_path(engine: EvolutionEngine, tool_name: str) -> Path | None:
    tool_path = engine._tools_dir / f"{tool_name.replace('.', '_')}.py"
    if tool_path.exists():
        return tool_path
    module_name = tool_name.split(".")[0]
    fallback = engine._tools_dir / f"{module_name}.py"
    if fallback.exists():
        return fallback
    return None


async def choose_tool_target_with_llm(
    engine: EvolutionEngine,
    recent: list[Failure],
    candidates: list[tuple[str, int]],
    blocked: list[tuple[str, int, int]],
) -> tuple[str | None, str]:
    if not candidates:
        return None, "no candidates"

    from provider.base import Message

    candidate_lines = [
        f"- {name}: failures={count}"
        for name, count in candidates
    ]
    blocked_lines = [
        f"- {name}: remain={remain}s, streak={streak}"
        for name, remain, streak in blocked
    ]
    snippets: list[str] = []
    for name, _count in candidates:
        samples = [f.summary for f in recent if f.kind == name]
        if samples:
            snippets.append(f"{name}: " + " | ".join(s for s in samples))

    messages = [
        Message(
            role="system",
            content=(
                "你是灵舟进化调度器。你必须在候选工具里选择一个最值得进化的目标，"
                "或明确决定本轮不进化。只输出 JSON，不要额外文字。"
            ),
        ),
        Message(
            role="user",
            content=(
                "可选候选（仅可从这些里选）：\n"
                + "\n".join(candidate_lines)
                + "\n\n已熔断候选（禁止选择）：\n"
                + ("\n".join(blocked_lines) if blocked_lines else "（无）")
                + "\n\n近期失败摘要：\n"
                + ("\n".join(snippets) if snippets else "（无）")
                + "\n\n返回 JSON："
                + '{"target":"tool.name or empty", "should_evolve": true/false, "reason":"..."}'
            ),
        ),
    ]
    try:
        raw = await engine._provider.chat(messages, temperature=0.1)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return candidates[0][0], "llm output invalid, fallback to top candidate"
        payload = json.loads(match.group())
        if not isinstance(payload, dict):
            return candidates[0][0], "llm payload invalid, fallback to top candidate"
        should_evolve = bool(payload.get("should_evolve", True))
        reason = str(payload.get("reason") or "")
        if not should_evolve:
            return None, reason or "llm decided skip"
        target = str(payload.get("target") or "").strip()
        allowed = {name for name, _count in candidates}
        if target in allowed:
            return target, reason
        return candidates[0][0], "llm target invalid, fallback to top candidate"
    except Exception as exc:
        return candidates[0][0], f"llm target decision failed: {exc}"


async def evolve_tool(
    engine: EvolutionEngine,
    tool_name: str,
    tool_path: Path,
    feedback: str,
    ctx: ToolContext | None = None,
) -> EvolutionResult:
    """根据反馈重写工具，热替换。"""

    from core.immune.policy import audit_evolution_target
    from provider.base import Message

    current_src = await asyncio.to_thread(tool_path.read_text, encoding="utf-8") if await asyncio.to_thread(tool_path.exists) else ""
    module_name = f"tools.{tool_path.stem}"
    block = audit_evolution_target(module_name)
    if block:
        result = EvolutionResult(success=False, target=tool_name, reason=block)
        await engine._write_evolution_history_fact(tool_name, success=False, reason=block, ctx=ctx)
        return result

    if not current_src:
        reason = f"生命连续性层拒绝：{tool_name} 无可回滚的当前版本（公理 A9）"
        result = EvolutionResult(success=False, target=tool_name, reason=reason)
        await engine._write_evolution_history_fact(tool_name, success=False, reason=reason, ctx=ctx)
        return result

    new_src = ""
    evolution_template = engine._cfg.load_prompt("evolution")
    prompt = evolution_template.replace("{{tool_name}}", tool_name)
    prompt = prompt.replace("{{current_source}}", current_src)
    prompt = prompt.replace("{{failure_summary}}", feedback)

    messages = [
        Message(role="system", content="你是 lingzhou 的自进化模块，负责改进工具代码。只输出完整的 Python 代码，不要有多余文字。"),
        Message(role="user", content=prompt),
    ]

    for attempt in range(engine._cfg.evolution.max_attempts):
        try:
            new_src = await engine._provider.chat(messages)
            new_src = _extract_python(new_src)
            compile(new_src, tool_path.name, "exec")

            previous_src = current_src
            backup_path = tool_path.with_suffix(".py.bak")
            if engine._cfg.evolution.backup and await asyncio.to_thread(tool_path.exists):
                await asyncio.to_thread(backup_path.write_text, previous_src, encoding="utf-8")
                _clean_old_backups(tool_path, keep=engine._cfg.evolution.backup_keep)

            try:
                ast.parse(new_src)
            except SyntaxError as exc:
                raise ValueError(f"Syntax error in generated tool source: {exc}") from exc

            smoke_err = smoke_test_module(engine, tool_path, new_src, timeout=engine._cfg.evolution.smoke_timeout)
            if smoke_err:
                if attempt < engine._cfg.evolution.max_attempts - 1:
                    from provider.base import Message as _Msg

                    messages.append(_Msg(role="assistant", content=new_src))
                    messages.append(_Msg(role="user", content=f"代码运行时验证失败，请修复：{smoke_err}"))
                continue

            await asyncio.to_thread(tool_path.write_text, new_src, encoding="utf-8")

            try:
                engine._reload_module_from_path(module_name, tool_path)
                if not engine._is_registered_tool(tool_name):
                    raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
            except Exception:
                engine._restore_file_text(tool_path, previous_src)
                engine._reload_module_from_path(module_name, tool_path)
                raise

            import ast as _ast

            from core.paths import project_root

            root = project_root()
            all_ok = True
            errors: list[str] = []
            for py in sorted(root.rglob("*.py")):
                rp = str(py.relative_to(root))
                if "__pycache__" in rp or ".py.bak" in rp or ".lingzhou-backup" in rp:
                    continue
                try:
                    _ast.parse(py.read_text(encoding="utf-8", errors="replace"))
                except SyntaxError as exc:
                    all_ok = False
                    errors.append(f"{rp}: {exc}")
            if not all_ok:
                if await asyncio.to_thread(tool_path.exists) and current_src:
                    engine._restore_file_text(tool_path, current_src)
                    engine._reload_module_from_path(module_name, tool_path)
                continue
            if ctx is not None and engine._cfg.evolution.backup and await asyncio.to_thread(backup_path.exists):
                await write_pending_verification_fact(
                    engine,
                    ctx,
                    target=tool_name,
                    tool_path=tool_path,
                    backup_path=backup_path,
                )
            await engine._update_dreams(f"习得改进能力：{tool_name} 工具已根据失败反馈重写并热加载。", ctx=ctx)
            await engine._write_evolution_history_fact(tool_name, success=True, reason="", ctx=ctx)
            return EvolutionResult(success=True, target=tool_name, new_code=new_src)
        except SyntaxError as exc:
            reason = f"语法错误: {exc}"
            if attempt < engine._cfg.evolution.max_attempts - 1:
                messages.append(Message(role="assistant", content=new_src))
                messages.append(Message(role="user", content=f"代码有语法错误，请修复：{reason}"))
        except Exception:
            if await asyncio.to_thread(tool_path.exists) and current_src:
                engine._restore_file_text(tool_path, current_src)
                with contextlib.suppress(Exception):
                    engine._reload_module_from_path(f"tools.{tool_path.stem}", tool_path)
            reason = traceback.format_exc(limit=3)
            await engine._write_evolution_history_fact(tool_name, success=False, reason=reason, ctx=ctx)
            return EvolutionResult(success=False, target=tool_name, reason=reason)

    final_reason = f"超过最大重试次数 {engine._cfg.evolution.max_attempts}"
    await engine._write_evolution_history_fact(tool_name, success=False, reason=final_reason, ctx=ctx)
    return EvolutionResult(success=False, target=tool_name, reason=final_reason)


async def synthesize_tool(engine: EvolutionEngine, description: str, name_hint: str = "") -> EvolutionResult:
    """从自然语言描述合成全新工具，写入 tools/ 并热加载。"""

    from provider.base import Message

    prompt = (
        f"请根据以下描述，编写一个符合 lingzhou 工具接口规范的 Python 模块。\n\n"
        f"描述: {description}\n\n"
        f"接口规范：\n"
        f"1. 从 tools.registry 导入 tool, ToolManifest, ToolParam, ToolResult, ToolContext\n"
        f"2. 使用 @tool(ToolManifest(...)) 装饰器注册\n"
        f"3. 函数签名: async def xxx(params: dict[str, Any], ctx: ToolContext) -> ToolResult\n"
        f"4. 只输出完整 Python 代码，不要有多余文字"
    )
    messages = [
        Message(role="system", content="你是 lingzhou 的工具合成模块。"),
        Message(role="user", content=prompt),
    ]
    try:
        raw = await engine._provider.chat(messages)
        new_src = _extract_python(raw)
        compile(new_src, "synthesized_tool.py", "exec")
        name_match = re.search(r'name="([^"]+)"', new_src)
        tool_name = name_hint or (name_match.group(1).split(".")[0] if name_match else "custom_tool")
        tool_path = engine._tools_dir / f"{tool_name}.py"

        smoke_err = smoke_test_module(engine, tool_path, new_src, timeout=engine._cfg.evolution.smoke_timeout)
        if smoke_err:
            return EvolutionResult(success=False, target=tool_name, reason=f"synthesize smoke failed: {smoke_err}")

        previous_src = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""
        tool_path.write_text(new_src, encoding="utf-8")

        module_name = f"tools.{tool_path.stem}"
        try:
            engine._reload_module_from_path(module_name, tool_path)
            if not engine._is_registered_tool(tool_name):
                raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
        except Exception:
            if previous_src:
                engine._restore_file_text(tool_path, previous_src)
                engine._reload_module_from_path(module_name, tool_path)
            else:
                with contextlib.suppress(Exception):
                    tool_path.unlink()
            raise

        return EvolutionResult(success=True, target=tool_name, new_code=new_src)
    except Exception as exc:
        return EvolutionResult(success=False, reason=str(exc))


async def competitive_evolve_tool(
    engine: EvolutionEngine,
    tool_name: str,
    tool_path: Path,
    feedback: str,
    num_candidates: int = 2,
    ctx: ToolContext | None = None,
) -> EvolutionResult:
    """A/B 竞争进化：并行生成多个候选代码版本，smoke 评估后选最优者晋升生产。"""

    from provider.base import Message

    if num_candidates < 1:
        num_candidates = 1
    if num_candidates > 4:
        num_candidates = 4

    current_src = await asyncio.to_thread(tool_path.read_text, encoding="utf-8") if await asyncio.to_thread(tool_path.exists) else ""
    evolution_template = engine._cfg.load_prompt("evolution")

    base_prompt = evolution_template.replace("{{tool_name}}", tool_name)
    base_prompt = base_prompt.replace("{{current_source}}", current_src)
    base_prompt = base_prompt.replace("{{failure_summary}}", feedback)

    candidate_strategies: list[tuple[str, float | None]] = [
        (
            "你是 lingzhou 的自进化模块。请对工具做【最小改动】修复，保持现有结构，只修改出问题的代码。输出完整 Python 代码。",
            0.2,
        ),
        (
            "你是 lingzhou 的自进化模块。请【完全重写】工具，追求更好的错误处理、更清晰的逻辑，同时修复反馈中的问题。输出完整 Python 代码。",
            0.7,
        ),
        (
            "你是 lingzhou 的自进化模块。用【折中策略】改进工具：保留核心逻辑，重构出问题的部分，补充防御性检查。输出完整 Python 代码。",
            0.5,
        ),
        (
            "你是 lingzhou 的自进化模块。从用户视角思考工具应该如何工作，然后重写使其行为更符合预期。输出完整 Python 代码。",
            0.6,
        ),
    ]

    async def _gen_candidate(idx: int) -> tuple[int, str]:
        sys_msg, temp = candidate_strategies[idx % len(candidate_strategies)]
        msgs = [
            Message(role="system", content=sys_msg),
            Message(role="user", content=base_prompt),
        ]
        try:
            raw = await engine._provider.chat(msgs, temperature=temp)
            code = _extract_python(raw)
            compile(code, tool_path.name, "exec")
            return idx, code
        except Exception:
            return idx, ""

    gen_results: list[tuple[int, str]] = list(await asyncio.gather(*[_gen_candidate(i) for i in range(num_candidates)]))

    scored: list[tuple[int, int, str]] = []
    for idx, code in gen_results:
        if not code:
            continue
        smoke_err = smoke_test_module(engine, tool_path, code, timeout=engine._cfg.evolution.smoke_timeout)
        if smoke_err:
            continue
        score = _score_candidate(code)
        scored.append((score, idx, code))

    if not scored:
        return await evolve_tool(engine, tool_name, tool_path, feedback, ctx=ctx)

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_idx, best_code = scored[0]
    try:
        return await promote_candidate(engine, tool_name, tool_path, best_code, best_idx, best_score, ctx=ctx)
    except Exception:
        reason = traceback.format_exc(limit=3)
        await engine._write_evolution_history_fact(tool_name, success=False, reason=reason, ctx=ctx)
        return EvolutionResult(success=False, target=tool_name, reason=reason)


async def promote_candidate(
    engine: EvolutionEngine,
    tool_name: str,
    tool_path: Path,
    code: str,
    candidate_idx: int,
    score: int,
    ctx: ToolContext | None = None,
) -> EvolutionResult:
    """将通过竞争评估的候选代码直接写入生产路径并热加载。"""


    current_src = await asyncio.to_thread(tool_path.read_text, encoding="utf-8") if await asyncio.to_thread(tool_path.exists) else ""

    backup_path = tool_path.with_suffix(".py.bak")
    if engine._cfg.evolution.backup and await asyncio.to_thread(tool_path.exists):
        await asyncio.to_thread(backup_path.write_text, current_src, encoding="utf-8")
        _clean_old_backups(tool_path, keep=engine._cfg.evolution.backup_keep)

    await asyncio.to_thread(tool_path.write_text, code, encoding="utf-8")

    module_name = f"tools.{tool_path.stem}"
    try:
        engine._reload_module_from_path(module_name, tool_path)
        if not engine._is_registered_tool(tool_name):
            raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
    except Exception:
        engine._restore_file_text(tool_path, current_src)
        with contextlib.suppress(Exception):
            engine._reload_module_from_path(module_name, tool_path)
        raise

    if ctx is not None and engine._cfg.evolution.backup and await asyncio.to_thread(backup_path.exists):
        await write_pending_verification_fact(
            engine,
            ctx,
            target=tool_name,
            tool_path=tool_path,
            backup_path=backup_path,
        )
    await engine._update_dreams(
        f"竞争进化完成：候选 {candidate_idx} 以评分 {score} 赢得 {tool_name} 改进权",
        ctx=ctx,
    )
    await engine._write_evolution_history_fact(
        tool_name,
        success=True,
        reason=f"competitive_evolve: candidate={candidate_idx} score={score}",
        ctx=ctx,
    )
    return EvolutionResult(
        success=True,
        target=tool_name,
        new_code=code,
        reason=f"competitive_evolve: candidate={candidate_idx} score={score}",
    )


__all__ = [
    "choose_tool_target_with_llm",
    "competitive_evolve_tool",
    "evolve_tool",
    "find_tool_path",
    "promote_candidate",
    "synthesize_tool",
    "_extract_python",
    "_score_candidate",
]
