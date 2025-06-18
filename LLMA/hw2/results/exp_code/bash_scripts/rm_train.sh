MODEL_NAME_OR_PATH="/data/Qwen2.5-0.5B-Instruct"


# align-anything/text-to-text
TRAIN_DATASETS="/data/align_anything_t2t"

TRAIN_TEMPLATE="HOMEWORK" # dataset template

TRAIN_SPLIT="train" # split the dataset

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-./outputs/part1}"

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_hw" # output dir

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
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --epochs 1 