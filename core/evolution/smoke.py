from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.metabolic import delete_fact, submit_fact
from core.paths import generated_dir

from .types import (
    EvolutionResult,
    _clear_smoke_failure_artifacts,
    _format_smoke_failure_message,
    _parse_ts,
    _persist_smoke_failure_artifacts,
    _verification_fact_key,
    _verification_outcome,
)

if TYPE_CHECKING:
    from core.evolution import EvolutionEngine
    from tools.registry import ToolContext


def smoke_test_module(
    engine: EvolutionEngine | None,
    module_path: Path,
    staged_source: str,
    *,
    project_root: Path | None = None,
    timeout: float | None = None,
) -> str | None:
    """在独立子进程中验证 staged 模块。"""

    from core.smoke_tests import FALLBACK_SNIPPET, SMOKE_TESTS

    if project_root is None:
        from core.paths import project_root as _project_root

        root = _project_root()
    else:
        root = project_root
    smoke_timeout = timeout if timeout is not None else getattr(getattr(getattr(engine, "_cfg", None), "evolution", None), "smoke_timeout", 15.0)

    try:
        rel_path = str(module_path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel_path = module_path.name

    snippet = SMOKE_TESTS.get(rel_path, FALLBACK_SNIPPET)

    if rel_path.endswith("/__init__.py"):
        real_module_name = rel_path.removesuffix("/__init__.py").replace("/", ".")
    else:
        real_module_name = rel_path.removesuffix(".py").replace("/", ".")
    pkg_parts = real_module_name.rsplit(".", 1)
    parent_pkg = pkg_parts[0] if len(pkg_parts) > 1 else ""

    if parent_pkg:
        pkg_hierarchy = parent_pkg.split(".")
        parent_imports_lines = "\n".join(
            f"import {'.'.join(pkg_hierarchy[:i + 1])}"
            for i in range(len(pkg_hierarchy))
        )
    else:
        parent_imports_lines = ""

    preload_registry = bool(snippet.strip())
    if preload_registry:
        try:
            import tools.registry as _curr_registry_mod

            _registry_file = str(Path(_curr_registry_mod.__file__).resolve())
        except Exception:
            _registry_file = str((root / "tools" / "registry.py").resolve())

        try:
            import tools.view_protocols as _curr_vp_mod

            _view_protocols_file = str(Path(_curr_vp_mod.__file__).resolve())
        except Exception:
            _view_protocols_file = str((root / "tools" / "view_protocols.py").resolve())
    else:
        _registry_file = ""
        _view_protocols_file = ""

    staging_path = generated_dir() / f"_smoke_staging_{module_path.stem}_{uuid.uuid4().hex}{module_path.suffix}"
    try:
        staging_path.write_text(staged_source, encoding="utf-8")

        probe = textwrap.dedent(f"""
import sys
sys.path.insert(0, {str(root)!r})
import importlib.util as _ilu
import types as _types
if {preload_registry!r}:
    _tools_pkg = _types.ModuleType("tools")
    _tools_pkg.__path__ = [{str(root / "tools")!r}]
    _tools_pkg.__package__ = "tools"
    sys.modules.setdefault("tools", _tools_pkg)
    _vp_spec = _ilu.spec_from_file_location("tools.view_protocols", {_view_protocols_file!r})
    _vp_mod = _ilu.module_from_spec(_vp_spec)
    _vp_mod.__package__ = "tools"
    sys.modules["tools.view_protocols"] = _vp_mod
    _vp_spec.loader.exec_module(_vp_mod)
    _reg_spec = _ilu.spec_from_file_location("tools.registry", {_registry_file!r})
    _reg_mod = _ilu.module_from_spec(_reg_spec)
    _reg_mod.__package__ = "tools"
    sys.modules["tools.registry"] = _reg_mod
    _reg_spec.loader.exec_module(_reg_mod)
{parent_imports_lines}
_spec = _ilu.spec_from_file_location({real_module_name!r}, {str(staging_path)!r})
mod = _ilu.module_from_spec(_spec)
mod.__package__ = {parent_pkg!r}
sys.modules[{real_module_name!r}] = mod
_spec.loader.exec_module(mod)
{snippet}
print("SMOKE_OK")
""").strip()

        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=smoke_timeout,
            cwd=str(root),
        )
        if result.returncode != 0 or "SMOKE_OK" not in result.stdout:
            stdout_text = result.stdout.strip()
            stderr_text = result.stderr.strip()
            detail_parts = [
                f"returncode={result.returncode}",
                f"module={rel_path}",
                f"real_module={real_module_name}",
                f"staging_path={staging_path}",
            ]
            if snippet.strip():
                detail_parts.append(f"[snippet]\n{snippet.strip()}")
            if stdout_text:
                detail_parts.append(f"[stdout]\n{stdout_text}")
            if stderr_text:
                detail_parts.append(f"[stderr]\n{stderr_text}")
            if not stdout_text and not stderr_text:
                detail_parts.append("[output]\nsmoke test failed (no output)")
            detail = "\n\n".join(detail_parts)
            saved_source, saved_log = _persist_smoke_failure_artifacts(module_path, staged_source, detail)
            return _format_smoke_failure_message(
                rel_path=rel_path,
                detail=detail,
                source_artifact=saved_source,
                log_artifact=saved_log,
            )
        _clear_smoke_failure_artifacts(module_path)
        return None
    except subprocess.TimeoutExpired:
        detail = "\n\n".join([
            f"timeout={smoke_timeout:.0f}s",
            f"module={rel_path}",
            f"real_module={real_module_name}",
            f"staging_path={staging_path}",
            f"[output]\nsmoke test timed out (>{smoke_timeout:.0f}s)",
        ])
        saved_source, saved_log = _persist_smoke_failure_artifacts(module_path, staged_source, detail)
        return _format_smoke_failure_message(
            rel_path=rel_path,
            detail=detail,
            source_artifact=saved_source,
            log_artifact=saved_log,
        )
    except Exception as exc:
        detail = "\n\n".join([
            f"exception={type(exc).__name__}: {exc}",
            f"module={rel_path}",
            f"real_module={real_module_name}",
            f"staging_path={staging_path}",
        ])
        saved_source, saved_log = _persist_smoke_failure_artifacts(module_path, staged_source, detail)
        return _format_smoke_failure_message(
            rel_path=rel_path,
            detail=detail,
            source_artifact=saved_source,
            log_artifact=saved_log,
        )
    finally:
        staging_path.unlink(missing_ok=True)


async def gather_target_validation_metrics(
    engine: EvolutionEngine,
    ctx: ToolContext,
    *,
    target: str,
    since: datetime | None = None,
) -> dict[str, int]:
    failures = await ctx.task_store.list_failures(limit=200)
    runs = await ctx.task_store.list_runs(limit=200)
    failure_count = 0
    run_count = 0
    success_count = 0

    for failure in failures:
        if failure.kind != target:
            continue
        if since and _parse_ts(failure.created_at) < since:
            continue
        failure_count += 1

    for run in runs:
        if run.tool_name != target:
            continue
        if since and _parse_ts(run.created_at) < since:
            continue
        run_count += 1
        if run.status == "succeeded":
            success_count += 1

    return {
        "failures": failure_count,
        "runs": run_count,
        "successes": success_count,
    }


async def write_pending_verification_fact(
    engine: EvolutionEngine,
    ctx: ToolContext,
    *,
    target: str,
    tool_path: Path,
    backup_path: Path,
    baseline_metrics: dict[str, int] | None = None,
) -> None:
    baseline = baseline_metrics or await gather_target_validation_metrics(engine, ctx, target=target)
    payload = {
        "target": target,
        "tool_path": str(tool_path),
        "backup_path": str(backup_path),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "baseline": baseline,
    }
    await submit_fact(
        ctx,
        key=_verification_fact_key(target),
        value=json.dumps(payload, ensure_ascii=False),
        scope="system",
        source="evolution/verify_setup",
    )


async def process_pending_verifications(engine: EvolutionEngine, ctx: ToolContext) -> list[EvolutionResult]:
    from .breaker import _update_target_breaker_state

    facts = await ctx.task_store.list_facts(prefix="evolution:verify:", limit=50)
    results: list[EvolutionResult] = []
    for key, raw in facts:
        try:
            payload = json.loads(raw)
        except Exception:
            await delete_fact(ctx, key=key, source="evolution/verify")
            continue
        target = str(payload.get("target") or "")
        if not target:
            await delete_fact(ctx, key=key, source="evolution/verify")
            continue
        since = _parse_ts(str(payload.get("created_at") or ""))
        observed = await gather_target_validation_metrics(engine, ctx, target=target, since=since)
        outcome = _verification_outcome(
            payload.get("baseline") or {},
            observed,
            engine._cfg.evolution.verify_min_runs,
        )
        if outcome == "pending":
            continue
        if outcome == "verified":
            await _update_target_breaker_state(
                engine,
                ctx,
                target=target,
                success=True,
                reason=f"verification observed={observed}",
            )
            await delete_fact(ctx, key=key, source="evolution/verify")
            results.append(EvolutionResult(success=True, target=f"verify:{target}", reason=f"observed={observed}"))
            continue

        backup_path = Path(str(payload.get("backup_path") or ""))
        tool_path = Path(str(payload.get("tool_path") or ""))
        rolled_back = False
        if (
            engine._cfg.evolution.auto_rollback_on_regression
            and await asyncio.to_thread(backup_path.exists)
            and await asyncio.to_thread(tool_path.exists)
        ):
            previous_src = await asyncio.to_thread(backup_path.read_text, encoding="utf-8")
            engine._restore_file_text(tool_path, previous_src)
            engine._reload_module_from_path(f"tools.{tool_path.stem}", tool_path)
            rolled_back = True
        await _update_target_breaker_state(
            engine,
            ctx,
            target=target,
            success=False,
            reason=f"verification regressed observed={observed}",
        )
        await delete_fact(ctx, key=key, source="evolution/verify")
        results.append(
            EvolutionResult(
                success=rolled_back,
                target=f"rollback:{target}" if rolled_back else f"verify:{target}",
                reason=f"observed={observed}",
            )
        )
    return results


__all__ = [
    "gather_target_validation_metrics",
    "process_pending_verifications",
    "smoke_test_module",
    "write_pending_verification_fact",
]
