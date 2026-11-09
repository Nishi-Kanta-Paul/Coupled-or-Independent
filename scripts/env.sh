CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export HF_HOME="$CODE_ROOT/.cache/hf"
export TORCH_HOME="$CODE_ROOT/.cache/torch"
export XDG_CACHE_HOME="$CODE_ROOT/.cache/xdg"
export MPLCONFIGDIR="$CODE_ROOT/.cache/mpl"
export EASYOCR_MODULE_PATH="$CODE_ROOT/.cache/easyocr"
export PIP_CACHE_DIR="$CODE_ROOT/.cache/pip"
export TOKENIZERS_PARALLELISM=false
PYTHON="$CODE_ROOT/.venv/bin/python"
