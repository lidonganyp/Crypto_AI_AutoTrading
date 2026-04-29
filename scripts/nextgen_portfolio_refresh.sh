#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lidong/lidong/CryptoAI"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/nextgen_portfolio_refresh.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "$(date -Is) missing python interpreter: $PYTHON_BIN" >>"$LOG_FILE"
  exit 1
fi

{
  echo "$(date -Is) starting portfolio refresh"
  "$PYTHON_BIN" -m nextgen_evolution refresh --cycles 1 --interval-seconds 0
  echo "$(date -Is) completed portfolio refresh"
} >>"$LOG_FILE" 2>&1
