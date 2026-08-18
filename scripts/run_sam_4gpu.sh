#!/bin/bash

PYTHON=/data/ljy/conda_envs/LA/bin/python
SCRIPT=/data/ljy/dish_detect/scripts/sam_food_mask.py
LOG_DIR=/data/ljy/dish_detect/sam/logs_sam

mkdir -p ${LOG_DIR}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT} --rank 0 --world_size 4 \
  2>&1 | tee ${LOG_DIR}/sam_rank0.log &

CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${SCRIPT} --rank 1 --world_size 4 \
  2>&1 | tee ${LOG_DIR}/sam_rank1.log &

CUDA_VISIBLE_DEVICES=2 ${PYTHON} ${SCRIPT} --rank 2 --world_size 4 \
  2>&1 | tee ${LOG_DIR}/sam_rank2.log &

CUDA_VISIBLE_DEVICES=3 ${PYTHON} ${SCRIPT} --rank 3 --world_size 4 \
  2>&1 | tee ${LOG_DIR}/sam_rank3.log &

wait

echo "All SAM workers finished."