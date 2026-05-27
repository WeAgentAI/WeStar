#!/bin/bash
# WeStar Step 7: Train LoRA for each style cluster
# Uses LLaMA-Factory or similar framework for LoRA training
#
# Usage:
#   bash scripts/07_train_lora.sh --cluster_id 0 --mode sft
#   bash scripts/07_train_lora.sh --cluster_id 0 --mode dpo

set -e

# ============= Configuration =============
CLUSTER_ID="${1:-0}"
MODE="${2:-sft}"  # sft / dpo

# Paths (modify these according to your environment)
BASE_MODEL_PATH="/path/to/Qwen3-32B"
DATA_DIR="./cluster_tree"
OUTPUT_DIR="./checkpoints"

# Training hyperparameters
LEARNING_RATE=1e-4
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LORA_RANK=64
LORA_ALPHA=128
MAX_LENGTH=4096

# ============= Parse Arguments =============
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster_id) CLUSTER_ID="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --base_model) BASE_MODEL_PATH="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "========================================="
echo "WeStar LoRA Training"
echo "  Cluster ID: ${CLUSTER_ID}"
echo "  Mode: ${MODE}"
echo "  Base Model: ${BASE_MODEL_PATH}"
echo "========================================="

# ============= Determine Data File =============
if [ "$MODE" == "sft" ]; then
    DATA_FILE="${DATA_DIR}/${CLUSTER_ID}/train_lora_top_sft.json"
    OUTPUT_PATH="${OUTPUT_DIR}/cluster_${CLUSTER_ID}_top_sft"
    TRAINING_STAGE="sft"
elif [ "$MODE" == "dpo" ] || [ "$MODE" == "sedpo" ]; then
    DATA_FILE="${DATA_DIR}/${CLUSTER_ID}/train_lora_style_dpo.json"
    OUTPUT_PATH="${OUTPUT_DIR}/cluster_${CLUSTER_ID}_dpo"
    TRAINING_STAGE="dpo"
elif [ "$MODE" == "mdpo" ]; then
    DATA_FILE="${DATA_DIR}/${CLUSTER_ID}/train_lora_dpo.json"
    OUTPUT_PATH="${OUTPUT_DIR}/cluster_${CLUSTER_ID}_mdpo"
    TRAINING_STAGE="dpo"
else
    echo "Error: Unknown mode '${MODE}'. Use: sft, dpo, sedpo, mdpo"
    exit 1
fi

# Check data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: ${DATA_FILE}"
    exit 1
fi

echo "Data file: ${DATA_FILE}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# ============= Run Training =============
# Option A: Using LLaMA-Factory (recommended)
# Make sure LLaMA-Factory is installed: pip install llamafactory

if command -v llamafactory-cli &> /dev/null; then
    echo "Using LLaMA-Factory for training..."
    
    # Create dataset_info.json for LLaMA-Factory
    cat > /tmp/westar_dataset_info.json << EOF
{
    "westar_train": {
        "file_name": "${DATA_FILE}",
        "formatting": "sharegpt"
    }
}
EOF

    llamafactory-cli train \
        --stage ${TRAINING_STAGE} \
        --model_name_or_path ${BASE_MODEL_PATH} \
        --dataset_dir /tmp \
        --dataset westar_train \
        --template qwen \
        --finetuning_type lora \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_target all \
        --output_dir ${OUTPUT_PATH} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --cutoff_len ${MAX_LENGTH} \
        --preprocessing_num_workers 16 \
        --logging_steps 10 \
        --save_steps 500 \
        --bf16 True \
        --do_train True

# Option B: Using custom training script with PEFT
else
    echo "LLaMA-Factory not found. Using custom training script..."
    
    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        scripts/train_lora_custom.py \
        --model_path ${BASE_MODEL_PATH} \
        --data_path ${DATA_FILE} \
        --output_dir ${OUTPUT_PATH} \
        --mode ${TRAINING_STAGE} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --learning_rate ${LEARNING_RATE} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --max_length ${MAX_LENGTH}
fi

echo ""
echo "========================================="
echo "Training complete!"
echo "  Output: ${OUTPUT_PATH}"
echo "========================================="
