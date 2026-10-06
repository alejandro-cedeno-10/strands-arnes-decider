# Source from the repository root (Git Bash or any POSIX shell): `source env/env.sh`.
# Keeps model weights, package caches and the virtualenv inside the repository.
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$ROOT/.cache/huggingface"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TORCH_HOME="$ROOT/.cache/torch"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  export PY="$ROOT/.venv/Scripts/python.exe"
else
  export PY="$ROOT/.venv/bin/python"
fi
