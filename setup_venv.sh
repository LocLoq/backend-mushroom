#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
REQUIREMENTS_FILE="${2:-requirements.txt}"

is_python_312() {
  local exe="$1"
  "$exe" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null | grep -qx '3.12'
}

resolve_python_312() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi

  if command -v python3 >/dev/null 2>&1 && is_python_312 "python3"; then
    echo "python3"
    return
  fi

  if command -v python >/dev/null 2>&1 && is_python_312 "python"; then
    echo "python"
    return
  fi

  echo "Khong tim thay Python 3.12. Vui long cai Python 3.12 truoc." >&2
  exit 1
}

echo "[1/4] Dang tim Python 3.12..."
PYTHON_CMD="$(resolve_python_312)"

echo "[2/4] Tao virtual environment: ${VENV_DIR}"
"${PYTHON_CMD}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Khong tim thay python trong venv: ${VENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Khong tim thay file requirements: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

echo "[3/4] Nang cap pip/setuptools/wheel..."
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[4/4] Cai dependencies tu ${REQUIREMENTS_FILE} ..."
"${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "Hoan tat. Kich hoat venv:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Chay server:"
echo "  uvicorn server:app --reload"