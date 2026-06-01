"""Import 边界静态检查 — 防止跨层依赖回潮。"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN: list[tuple[str, str, str]] = [
    ("tools", "core.loop", "tools 不得依赖主循环编排"),
    ("tools", "core.judgment.runtime", "tools 不得依赖 JudgmentLayer 实现"),
    ("store", "core.", "store 不得依赖 core"),
    ("provider", "tools.", "provider 不得依赖 tools"),
]

_REMOVED_SHIM_PATHS = (
    "core/judgment/parser.py",
    "core/judgment/executor_helpers.py",
    "core/probe/types.py",
    "core/execution_helpers.py",
    "core/worker.py",
    "core/loop/task_runtime.py",
    "core/loop/task_parallel.py",
    "core/loop/startup.py",
    "core/loop/reload.py",
    "core/loop/driver.py",
    "core/loop/dispatcher.py",
    "core/loop/chat.py",
    "core/loop/focus.py",
    "core/loop/run_driver.py",
    "core/loop/run_refresh.py",
    "core/loop/common.py",
    "core/loop/logging.py",
    "core/loop/continue_phase.py",
    "core/loop/postprocess.py",
    "core/loop/progress.py",
    "core/loop/behavior.py",
    "core/loop/self_drive.py",
)


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts]


def _imports_in_file(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.lineno, node.module))
    return out


def test_forbidden_cross_layer_imports() -> None:
    violations: list[str] = []
    for rel_root, forbidden_prefix, message in _FORBIDDEN:
        base = _REPO_ROOT / rel_root
        if not base.is_dir():
            continue
        for py_file in _iter_py_files(base):
            for lineno, module in _imports_in_file(py_file):
                if module.startswith(forbidden_prefix):
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(f"{rel}:{lineno} imports {module} — {message}")
    assert not violations, "\n".join(violations)


def test_contracts_action_key_param_used_by_policy() -> None:
    from core.contracts.execution import action_key_param

    assert action_key_param({"path": "/tmp/x"}) == "/tmp/x"


def test_execution_package_exports_layer() -> None:
    from core.execution import ExecutionLayer, WorkerLayer, finalize_run

    assert ExecutionLayer is not None
    assert WorkerLayer is not None
    assert callable(finalize_run)


def test_compat_shim_files_removed() -> None:
    for rel in _REMOVED_SHIM_PATHS:
        assert not (_REPO_ROOT / rel).is_file(), f"shim 应已删除: {rel}"


def test_judgment_context_package_does_not_aggregate_exports() -> None:
    import core.judgment.context as ctx

    assert ctx.__all__ == []


def test_config_package_does_not_reexport_config_models() -> None:
    """子模型只从 core.config_models 导入，避免 core.config 聚合桶。"""
    import core.config as config_pkg

    assert set(config_pkg.__all__) == {"Config", "config_reference_defaults"}
    assert not hasattr(config_pkg, "ThresholdsConfig")
