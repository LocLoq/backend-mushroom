#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
REQUIREMENTS_FILE="${2:-requirements.txt}"
PYENV_SELECTED=""
PYTHON_CMD=()

is_python_312() {
  local exe="$1"
  "$exe" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null | grep -qx '3.12'
}

is_pyenv_python_312() {
  local version="$1"
  PYENV_VERSION="$version" pyenv exec python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null | grep -qx '3.12'
}

run_selected_python() {
  if [[ -n "$PYENV_SELECTED" ]]; then
    PYENV_VERSION="$PYENV_SELECTED" "${PYTHON_CMD[@]}" "$@"
    return
  fi

  "${PYTHON_CMD[@]}" "$@"
}

resolve_python_312() {
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_CMD=("python3.12")
    return
  fi

  if command -v python3 >/dev/null 2>&1 && is_python_312 "python3"; then
    PYTHON_CMD=("python3")
    return
  fi

  if command -v python >/dev/null 2>&1 && is_python_312 "python"; then
    PYTHON_CMD=("python")
    return
  fi

  if command -v pyenv >/dev/null 2>&1; then
    local pyenv_version
    pyenv_version="$(pyenv versions --bare 2>/dev/null | grep -E '^3\.12(\.|$)' | sort -V | tail -n 1 || true)"

    if [[ -n "$pyenv_version" ]] && is_pyenv_python_312 "$pyenv_version"; then
      PYENV_SELECTED="$pyenv_version"
      PYTHON_CMD=("pyenv" "exec" "python")
      return
    fi
  fi

  if command -v pyenv >/dev/null 2>&1; then
    echo "Khong tim thay lenh python 3.12 dang active. Ban co the dat pyenv local/global 3.12.x hoac de script tu dung ban 3.12.x da cai." >&2
    echo "Cac ban pyenv hien co:" >&2
    pyenv versions --bare 2>/dev/null >&2 || true
    exit 1
  fi

  echo "Khong tim thay Python 3.12. Vui long cai Python 3.12 truoc." >&2
  exit 1
}

echo "[1/4] Dang tim Python 3.12..."
resolve_python_312

echo "[2/4] Tao virtual environment: ${VENV_DIR}"
run_selected_python -m venv "${VENV_DIR}"

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