#!/usr/bin/env bash
set -euo pipefail

RSNA_DIR="${RSNA_DIR:?Set RSNA_DIR to the RSNA dataset root (e.g. data/rsna)}"
SPIDER_DIR="${SPIDER_DIR:-}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

COMMON="--data_dir ${RSNA_DIR} --dataset rsna --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --device ${DEVICE} --skip_if_done ${EXTRA_ARGS}"

run_seeds() {
  local desc="$1"; shift
  for s in ${SEEDS}; do
    echo "=== ${desc}  [seed ${s}] ==="
    ${PYTHON} train.py ${COMMON} --seed "${s}" "$@"
  done
}

run_seeds "1. Uniform strips + ordinal (BASELINE)"   --tokenizer strips  --pos_encoding ordinal
run_seeds "2. Patch-query + ordinal (BASELINE)"      --tokenizer patches --pos_encoding ordinal

run_seeds "3. Anatomy + learned (CAST-style)"        --tokenizer anatomy --pos_encoding learned
run_seeds "4. Anatomy + no positional encoding"      --tokenizer anatomy --pos_encoding none

run_seeds "5. Anatomy + ordinal (OURS)"              --tokenizer anatomy --pos_encoding ordinal

run_seeds "6. Anatomy + ordinal + fine-tuned"        --tokenizer anatomy --pos_encoding ordinal --no-freeze_backbone --lr 3e-5

if [ -n "${SPIDER_DIR}" ]; then
  for s in ${SEEDS}; do
    echo "=== 7. Anatomy + ordinal on SPIDER (oracle, Pfirrmann)  [seed ${s}] ==="
    ${PYTHON} train.py --data_dir "${SPIDER_DIR}" --dataset spider --tokenizer anatomy --pos_encoding ordinal \
      --task pfirrmann --use_oracle --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" \
      --seed "${s}" --skip_if_done ${EXTRA_ARGS}
  done
else
  echo "=== 7. SPIDER skipped (SPIDER_DIR not set) ==="
fi

echo "=== Aggregating (mean ± std over seeds) + figures ==="
${PYTHON} evaluate.py --experiments_dir outputs --data_dir "${RSNA_DIR}" --generate_figures

echo "Done. See outputs/evaluation/ for the per-run and aggregated tables."
