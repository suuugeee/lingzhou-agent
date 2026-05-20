#!/usr/bin/env python3
"""
将 tests/test_smoke.py 按主题拆分为多个文件。
Usage:
    python scripts/split_tests.py --dry-run   # 只打印，不写文件
    python scripts/split_tests.py             # 实际执行
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "tests" / "test_smoke.py"

# 共享 header（conftest.py 内容）
CONFTEST_HEADER = '''\
"""共享 fixtures 与 helpers，供 tests/ 下所有测试使用。"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest
'''

# 每个目标文件的 header（docstring + 导入 conftest helpers）
# pytest rootdir 模式（无 __init__.py）会将 tests/ 加到 sys.path，
# 因此直接 from conftest import ... 即可。
IMPORT_HEADER = '''\
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest

from conftest import (
    _proj_root,
    _test_config,
    _tool_ctx,
    _execution_layer,
    _tool_registry,
    _judgment_output,
)
'''

# 拆分计划：(输出文件名, 起始行(1-based), 结束行(1-based含))
# 起始行是第一个 def test_ 或区块注释所在行
SPLITS = [
    # (dest_file,              start,  end  )
    ("test_core.py",           102,   2308),
    ("test_task_store.py",    2309,   2692),
    ("test_memory.py",        2693,   2755),
    ("test_cognition.py",     2756,   3143),
    ("test_provider.py",      3144,   3754),
    ("test_judgment_ctx.py",  3755,   5052),
    ("test_tools.py",         5053,   5379),
    ("test_concurrent.py",    5380,   9999),  # 到末尾
]

DOCSTRINGS = {
    "test_core.py":          "核心模块测试：working_memory / emotion / judgment / chat / loop / exec / evolution",
    "test_task_store.py":    "TaskStore 持久化测试",
    "test_memory.py":        "语义记忆（semantic）与情节记忆（episodic）测试",
    "test_cognition.py":     "认知循环、chat reply、resolve 等集成测试",
    "test_provider.py":      "Auth / Copilot provider 测试",
    "test_judgment_ctx.py":  "行为门控、技能、thinking、模型路由、任务改写等 judgment context 测试",
    "test_tools.py":         "文件、进程、shell 工具测试",
    "test_concurrent.py":    "并发安全测试：_ScopedTaskStore / parallel dispatch / aiosqlite 行隔离",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)
    print(f"源文件: {SRC}  总行数: {total}")

    # ── 写 conftest.py ──────────────────────────────────────────────────────
    # 提取 helpers 区块（第 21-101 行，0-based: 20-100）
    helper_lines = lines[20:101]  # lines 21-101 inclusive
    conftest_path = ROOT / "tests" / "conftest.py"
    conftest_content = CONFTEST_HEADER + "".join(helper_lines)
    _write(conftest_path, conftest_content, args.dry_run)

    # ── 写各主题文件 ────────────────────────────────────────────────────────
    for dest_name, start_1, end_1 in SPLITS:
        dest_path = ROOT / "tests" / dest_name
        # 0-based 切片
        s = start_1 - 1
        e = min(end_1, total)      # end 是包含的行号(1-based)
        chunk = lines[s:e]
        docstring = DOCSTRINGS.get(dest_name, dest_name)
        content = f'"""{docstring}"""\n' + IMPORT_HEADER + "".join(chunk)
        _write(dest_path, content, args.dry_run)
        print(f"  {dest_name}: lines {start_1}-{e}  ({e - s} lines)")

    if not args.dry_run:
        print("\n完成。建议运行: .venv/bin/python -m pytest tests/ -v --tb=short")


def _write(path: Path, content: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run] would write {path}  ({len(content.splitlines())} lines)")
    else:
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path}  ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
