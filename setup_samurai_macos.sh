#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_samurai_macos.sh [ENV_NAME]
# Example:
#   bash setup_samurai_macos.sh vots_mac

ENV_NAME="${1:-vots_mac}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMURAI_DIR="${ROOT_DIR}/samurai"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda не найдена. Установите Miniforge/Conda и повторите."
  exit 1
fi

if [ ! -d "${SAMURAI_DIR}" ]; then
  echo "Не найдена папка ${SAMURAI_DIR}. Сначала клонируйте SAMURAI."
  exit 1
fi

eval "$(conda shell.zsh hook 2>/dev/null || conda shell.bash hook)"

conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"

pip install --upgrade pip
pip install torch torchvision

pip install -e "${SAMURAI_DIR}/sam2"
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
pip install vot-toolkit

mkdir -p "${SAMURAI_DIR}/sam2/checkpoints"
(cd "${SAMURAI_DIR}/sam2/checkpoints" && bash download_ckpts.sh)

echo
echo "Готово. Активируйте окружение: conda activate ${ENV_NAME}"
echo "Демо запуск:"
echo "  cd ${SAMURAI_DIR}"
echo "  python scripts/demo.py --video_path /abs/path/video.mp4 --txt_path /abs/path/init_bbox.txt --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt"
