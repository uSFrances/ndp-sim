#! /bin/bash

SCRIPT_DIR=$(dirname $0)

VENV_DIR=${SCRIPT_DIR}/../../.venv
GOLDEN_DIR=${SCRIPT_DIR}/../python_golden
OUTPUT_DIR=${SCRIPT_DIR}/../../model_execplan/data
CONFIG_PATH=${SCRIPT_DIR}/../config.json

# Activate Python venv
source ${VENV_DIR}/bin/activate

# Run relayout scripts
python3 ${SCRIPT_DIR}/relayout/relayout_softmax.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

python3 ${SCRIPT_DIR}/relayout/relayout_rope.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

python3 ${SCRIPT_DIR}/relayout/relayout_rmsnorm.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

python3 ${SCRIPT_DIR}/relayout/relayout_remote_sum.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

python3 ${SCRIPT_DIR}/relayout/relayout_regular.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

python3 ${SCRIPT_DIR}/relayout/relayout_gemm_local.py \
    -i ${GOLDEN_DIR} \
    -o ${OUTPUT_DIR} \
    --config ${CONFIG_PATH}

#python3 ${SCRIPT_DIR}/relayout/relayout_gemm_ring.py \
#    -i ${GOLDEN_DIR} \
#    -o ${OUTPUT_DIR} \
#    --config ${CONFIG_PATH}
