#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MODEL_NAME_OR_PATH="/data/Qwen2.5-0.5B-Instruct"
# align-anything/text-to-text
TRAIN_DATASETS="/data/align_anything_t2t"

TRAIN_TEMPLATE="HOMEWORK" # dataset template

TRAIN_SPLIT="train" # split the dataset

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-./outputs/part2DPO}"

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/DPO" # output dir

# For wandb online logging
export WANDB_API_KEY="My key"


# For CANN npu path
export ASCEND_HOME_PATH=/usr/local/Ascend

# Source the setup script
# source ./setup.sh
source $ASCEND_HOME_PATH/ascend-toolkit/set_env.sh


# Set port for deepspeed
MASTER_PORT=30000

# Execute deepspeed command
nohup deepspeed \
        --master_port ${MASTER_PORT} \
        --module align_anything.trainers.text_to_text.dpo \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR} \
        --save_steps 1000 \
        --save_total_limit 8 \
        --epochs 1