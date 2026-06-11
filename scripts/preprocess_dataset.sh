set -euo pipefail

mkdir -p logs

PYTHON=python
SCRIPT=preprocess_dataset.py

INPUT=../../ERA5/5.625deg_wb2/
OUTPUT=../../ERA5/5.65/

START_YEAR=1979
END_YEAR=2023

echo "========================================"
echo "Preprocessing job started"
echo "========================================"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: ${PYTHON}"
echo "Script: ${SCRIPT}"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Start time: $(date)"
echo "========================================"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set -x

$PYTHON $SCRIPT \
    -i "$INPUT" \
    -o "$OUTPUT" \
    --begin_year="$START_YEAR" \
    --end_year="$END_YEAR" \
    --levels 13

set +x

echo "========================================"
echo "Preprocessing job finished"
echo "End time: $(date)"
echo "========================================"
