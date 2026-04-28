
DATASET_PATH=/path/to/dataset/
PYTHON=python

BASE_DIR=/path/to/log/dir
CKPT_PATH=$BASE_DIR/checkpoints/last.ckpt
PARADIS_CODE_DIR=.

YEAR=2020
START_DATE="${YEAR}-01-01"
END_DATE="${YEAR}-12-31"
FORECAST_STEPS=40

cd $PARADIS_CODE_DIR

$PYTHON forecast.py \
  --config "${BASE_DIR}/config.yaml" \
  --checkpoint-path "${CKPT_PATH}" \
  --output-file "${BASE_DIR}/forecast/${YEAR}.zarr" \
  --start-date "${START_DATE}" \
  --root-dir "${DATASET_PATH}" \
  --end-date "${END_DATE}" \
  --forecast-steps "${FORECAST_STEPS}" \
  --sampling-interval 36h \
  --batch-size 1 \
  --num-devices 1 \
  --flush-every-n-steps 10
