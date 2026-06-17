# SAMURAI/SAM2 — Инструкция по установке и запуску

## Требования

- Conda (Miniforge/Miniconda)
- Python 3.10+
- Устройство: `mps` (Apple Silicon), `cpu` или `cuda`

## 1. Установка окружения

```bash
cd /path/to/sota_tracker
bash setup_samurai_macos.sh vots_mac
conda activate vots_mac
pip install "numpy<2" "opencv-python<4.11"
```

## 2. Создание VOT workspace

```bash
vot initialize vot2022/shorttermbox --workspace /path/to/sota_tracker/vot2022_stb_ws
```

## 3. Настройка `trackers.ini`

Файл находится в папке workspace. Укажи свои пути:

```ini
[MySAMURAI_STB2022]
label = SAMURAI-STB2022
protocol = trax
command = /path/to/python /path/to/sota_tracker/samurai/scripts/vot_wrapper.py \
          --model_path /path/to/sota_tracker/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt \
          --device cpu
timeout = 300
```

> Замени `cpu` на `mps` (Apple Silicon) или `cuda` (NVIDIA) под свою машину.

## 4. Запуск оценки

```bash
cd /path/to/sota_tracker/vot2022_stb_ws
vot test MySAMURAI_STB2022
vot evaluate --experiments unsupervised MySAMURAI_STB2022
```

## 5. Анализ и отчёт

```bash
vot analysis --experiments unsupervised --format json --name unsup_analysis MySAMURAI_STB2022
vot report --experiments unsupervised --format html --name unsup_report MySAMURAI_STB2022
```

Результаты:
- `analysis/unsup_analysis.json`
- `reports/unsup_report/report.html`

## 6. Генерация видео

```bash
mkdir -p /path/to/sota_tracker/vot2022_stb_ws/videos
```

Одна последовательность:

```bash
cd /path/to/sota_tracker/samurai
SEQ="helicopter"  # поменяй на нужную

python scripts/demo.py \
  --video_path "/path/to/sota_tracker/vot2022_stb_ws/sequences/${SEQ}/color" \
  --txt_path "/path/to/sota_tracker/vot2022_stb_ws/sequences/${SEQ}/groundtruth.txt" \
  --model_path "sam2/checkpoints/sam2.1_hiera_tiny.pt" \
  --video_output_path "/path/to/sota_tracker/vot2022_stb_ws/videos/${SEQ}.mp4"
```

Все последовательности:

```bash
cd /path/to/sota_tracker/samurai
while read -r seq; do
  python scripts/demo.py \
    --video_path "/path/to/sota_tracker/vot2022_stb_ws/sequences/${seq}/color" \
    --txt_path "/path/to/sota_tracker/vot2022_stb_ws/sequences/${seq}/groundtruth.txt" \
    --model_path "sam2/checkpoints/sam2.1_hiera_tiny.pt" \
    --video_output_path "/path/to/sota_tracker/vot2022_stb_ws/videos/${seq}.mp4"
done < /path/to/sota_tracker/vot2022_stb_ws/sequences/list.txt
```

## Советы

- **Веса модели** (`sam2.1_hiera_tiny.pt`) скачиваются отдельно в `samurai/sam2/checkpoints/` — в репозиторий не входят.
- **Пути** — единственное, что нужно адаптировать на другой машине. Удобно задать через переменные:
  ```bash
  export PROJECT_ROOT="/absolute/path/to/sota_tracker"
  export PYTHON_BIN="/absolute/path/to/python"
  ```
- **Нестабильный MPS** — если трекер падает на Apple Silicon, переключись на `--device cpu`.
- **`accuracy = 0.684`** в отчёте — это среднее перекрытие bbox (IoU), не accuracy классификации.
- **`unsupervised`** режим — оценка без переинициализации после срыва трекера.
