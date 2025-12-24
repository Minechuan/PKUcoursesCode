MODEL_NAME_OR_PATH="/data/align-anything/outputs/part1/qwen_2_5_hw/slice_end" # model path


# TRAIN_DATASETS="/data/align_anything_t2t"
# TRAIN_SPLIT="fake_train"

EVAL_DATASETS="/data/align_anything_t2t" # dataset path

EVAL_TEMPLATE="HOMEWORK" # dataset template

# EVAL_NAME="val_1k.parquet" # dataset name
EVAL_SPLIT="validation" # split the dataset


OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-./output_eval/part1}"
OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5" # output dir

# For wandb online logging
export WANDB_API_KEY="My key"

# For CANN npu path
export ASCEND_HOME_PATH=/usr/local/Ascend

# Set port for deepspeed
MASTER_PORT=29998

# Source the setup script
source $ASCEND_HOME_PATH/ascend-toolkit/set_env.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 1 \
     --epochs 1

     # --eval_name ${EVAL_NAME} \