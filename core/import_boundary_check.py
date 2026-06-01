"""跨层 import 边界检查入口（供 CLI / CI 调用）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [sys.executable, "-m", "pytest", str(root / "tests" / "test_import_boundaries.py"), "-q"],
        cwd=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
