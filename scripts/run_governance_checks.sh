#!/usr/bin/env bash
# 工程治理门禁：import 边界 + Ruff（治理面）+ 可选全量 pytest
set -euo pipefail
ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少 .venv，请先: ./setup-lingzhou.sh" >&2
  exit 1
fi

echo "> import boundaries"
"$PY" -m core.import_boundary_check

echo "> ruff (core + tools)"
"$PY" -m ruff check core tools

if [[ "${1:-}" == "--full-tests" ]]; then
  echo "> pytest (full)"
  "$PY" -m pytest tests/ -q --tb=no
fi

echo "governance checks OK"
