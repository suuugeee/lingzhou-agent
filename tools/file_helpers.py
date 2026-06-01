"""tools/file_helpers.py — file 工具私有辅助函数与常量。"""
from __future__ import annotations

import contextlib
import py_compile
from pathlib import Path

from core.paths import data_dir, project_root
from tools.registry import ToolContext

# 核心目录/文件检测 — 基于路径模式，无需维护静态列表
_CORE_DIRS = frozenset({"core", "memory", "provider"})
_CORE_EXTRA = frozenset({"tools/registry.py"})


def _is_core_file(path: Path) -> bool:
    """检测是否为核心 .py 文件（按路径模式，新增文件自动覆盖）。"""
    if path.suffix != ".py":
        return False
    try:
        rel = path.relative_to(_repo_root())
    except ValueError:
        return False
    return str(rel) in _CORE_EXTRA or rel.parts[0] in _CORE_DIRS


# 文件大小限制
MAX_READ_CHARS = 100_000
MAX_WRITE_CHARS = 200_000

# 写入允许的根目录 — workspace 沙箱
# 空集合 = 允许所有路径；设为具体路径则只允许写入这些目录内的文件。
# 系统级操作（如 /etc/systemd）请通过 shell.run 执行。
_WRITE_ALLOW_ROOTS: frozenset[str] = frozenset({
    str(project_root()),
    str(data_dir()),
})


def _write_allow_roots(ctx: ToolContext | None = None) -> list[Path]:
    roots = [Path(root).expanduser() for root in _WRITE_ALLOW_ROOTS]
    if ctx is not None:
        workspace = _workspace_dir(ctx)
        if workspace is not None:
            roots.append(workspace)
    return _unique_paths([root.resolve() for root in roots])


def _path_guard(path: Path, ctx: ToolContext | None = None) -> tuple[bool, str]:
    """路径安全守卫。

    返回 (is_safe, warning)。
    - 路径穿越检测（../etc/passwd 等）
    - 可选 workspace 沙箱
    """
    try:
        resolved = path.resolve()
    except Exception:
        return False, f"无法解析路径: {path}"

    # 路径穿越检测：确保解析后的路径仍在预期范围内
    allow_roots = _write_allow_roots(ctx)
    if allow_roots:
        for root in allow_roots:
            try:
                resolved.relative_to(root)
                break
            except ValueError:
                continue
        else:
            return False, (
                f"路径 {path} 不在允许的工作区内。"
                f"只能写入 {', '.join(str(root) for root in allow_roots)} 内的文件。"
            )

    return True, ""


def _fuzzy_find(content: str, old_text: str) -> int:
    """模糊匹配：优先精确匹配，失败后尝试行级去空白精确匹配。"""
    # 1. 精确匹配
    idx = content.find(old_text)
    if idx != -1:
        return idx

    # 2. 统一换行符
    old_text = old_text.replace('\r\n', '\n').replace('\r', '\n')
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # 3. 行级去空白精确匹配（修复原 startswith 过于宽松导致的误判/漏判）
    old_lines = old_text.split('\n')
    content_lines = content.split('\n')
    old_stripped = [ln.strip() for ln in old_lines]

    # 过滤首尾空行，提高容错
    while old_stripped and not old_stripped[0]:
        old_stripped.pop(0)
    while old_stripped and not old_stripped[-1]:
        old_stripped.pop()
    if not old_stripped:
        return -1

    for i in range(len(content_lines) - len(old_stripped) + 1):
        match = True
        for j, os_line in enumerate(old_stripped):
            if content_lines[i + j].strip() != os_line:  # 严格逐行相等（包含空行）
                match = False
                break
        if match:
            return sum(len(cl) + 1 for cl in content_lines[:i])

    return -1


def _safety_guard(path: Path) -> tuple[str, str | None]:
    """代码修改安全守卫：备份 + 语法验证。
    返回 (warning, error)。error 非空表示应阻断操作。
    """
    warnings: list[str] = []

    # 1. 自动备份（Python 文件）
    if path.suffix == '.py' and path.exists():
        backup = path.with_suffix(path.suffix + '.lingzhou-backup')
        with contextlib.suppress(Exception):  # 备份失败不阻断
            backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    # 2. 核心文件警告
    if _is_core_file(path):
        try:
            rel = str(path.relative_to(_repo_root()))
        except ValueError:
            rel = str(path)
        warnings.append(
            f"⚠️ 正在修改核心文件 {rel}。请修改后立即用 shell.run 验证系统可正常导入/启动。"
        )

    return "\n".join(warnings) if warnings else "", None


def _verify_python_syntax(path: Path, content: str) -> str | None:
    """验证 Python 文件语法。返回 None 表示通过，否则返回错误信息。"""
    if path.suffix != '.py':
        return None
    try:
        import ast
        ast.parse(content)
        return None
    except SyntaxError as e:
        return f"语法错误: {e}"
    except Exception as e:
        return f"验证异常: {e}"


def _pycompile_check(path: Path) -> str | None:
    """对磁盘上已写入的 .py 文件执行 py_compile.compile()，返回 None 表示通过，否则返回错误信息。"""
    if path.suffix != ".py":
        return None
    try:
        py_compile.compile(str(path), doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return f"py_compile 校验失败: {e}"
    except Exception as e:
        return f"py_compile 异常: {e}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log_preview_text(text: str, limit: int = 80) -> str:
    first_line = (text or "").splitlines()[0] if text else ""
    cleaned = " ".join(first_line.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _tail_after_anchor(path: Path, anchor: str) -> Path | None:
    parts = path.parts
    if anchor not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index(anchor)
    tail = parts[idx + 1 :]
    if not tail:
        return None
    return Path(*tail)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p)
        if key and key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _read_rel_candidates(path: Path) -> list[Path]:
    rels: list[Path] = []

    if not path.is_absolute():
        rels.append(path)

    for anchor in ("workspace", "lingzhou", ".lingzhou"):
        rel = _tail_after_anchor(path, anchor)
        if rel is not None:
            rels.append(rel)
            if rel.parts and rel.parts[0] == "workspace" and len(rel.parts) > 1:
                rels.append(Path(*rel.parts[1:]))

    if path.name and path.name not in ("", "."):
        rels.append(Path(path.name))

    return _unique_paths(rels)


def _candidate_paths(path: Path, bases: list[Path]) -> list[Path]:
    rels = _read_rel_candidates(path)
    candidates: list[Path] = []
    seen: set[str] = set()

    for rel in rels:
        for base in bases:
            candidate = base / rel
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

            if rel.parts and rel.parts[0] != "workspace":
                candidate = base / "workspace" / rel
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

    return candidates


def resolve_read_path(path: Path, ctx: ToolContext | None = None) -> Path:
    if path.exists():
        return path

    repo_root = _repo_root()
    workspace = _workspace_dir(ctx) if ctx is not None else None
    if workspace is not None:
        for candidate in _candidate_paths(path, _unique_paths([workspace, workspace.parent, repo_root])):
            if candidate.exists():
                return candidate

    cwd = Path.cwd()
    home = Path.home()
    bases: list[Path] = [cwd, *cwd.parents, home, home / ".lingzhou", repo_root]

    for candidate in _candidate_paths(path, _unique_paths(bases)):
        if candidate.exists():
            return candidate

    return path


def _workspace_dir(ctx: ToolContext) -> Path | None:
    from tools.paths import workspace_dir_from_ctx

    return workspace_dir_from_ctx(ctx)


def workspace_candidate_path(path: Path, ctx: ToolContext) -> Path | None:
    workspace = _workspace_dir(ctx)
    if workspace is None:
        return None

    roots = _unique_paths([workspace, workspace.parent, _repo_root()])
    for candidate in _candidate_paths(path, roots):
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return None


def _resolve_mutation_path(path: Path, ctx: ToolContext) -> Path:
    if path.exists():
        return path

    resolved = resolve_read_path(path, ctx)
    if resolved.exists():
        return resolved

    workspace_candidate = workspace_candidate_path(path, ctx)
    if workspace_candidate is not None:
        return workspace_candidate

    return path
