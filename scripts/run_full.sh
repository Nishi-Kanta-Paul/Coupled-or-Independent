set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$CODE_ROOT"
mkdir -p results/logs
"$PYTHON" main.py \
    --output_dir results/full \
    --device cuda \
    --hiba-path "${HIBA_PATH:-}" \
    "$@" 2>&1 | tee -a results/logs/full.log
