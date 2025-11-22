#!/bin/bash
# ===============================================
# Auto-discover multiple knowledge-type files and train in batch
# train_concept_v4_multi_knowledge.sh
# ===============================================

model=gpt2
model_name_or_path="openai-community/gpt2"

# Data and output directories
base_data_dir="/home/zhijie/DynamicKnowledgeCircuits-main/data/concept/v3"
remark="concept_frequency_10_v3_auto"
base_output_dir="outputs/train/${model}/${remark}"
block_size=1024
max_steps=1000

# WandB project name
# export WANDB_PROJECT="ConceptAutoMultiTrain"

# GPU configuration
export CUDA_VISIBLE_DEVICES=1

# Find all training files matching the pattern
train_files=($(ls ${base_data_dir}/t6.4_concept_separate_knowledge_train_reorder_*.jsonl 2>/dev/null))

echo "==============================================="
echo "Found the following training files:"
echo "-----------------------------------------------"
for f in "${train_files[@]}"; do
    echo " - $(basename "$f")"
done
echo "==============================================="

if [ ${#train_files[@]} -eq 0 ]; then
    echo "No training files found. Please check the path or naming."
    exit 1
fi

# Loop over each training file
for train_file in "${train_files[@]}"; do
    # Extract A and B types from the filename
    train_base=$(basename "$train_file" | sed -E 's/.*_reorder_([^.]+)\.jsonl/\1/')  # e.g., HAH_MAH
    A_type=$(echo "$train_base" | cut -d'_' -f1)
    B_type=$(echo "$train_base" | cut -d'_' -f2)

    # Test file uses only B_type
    test_file="${base_data_dir}/t6.4_concept_separate_knowledge_test_${B_type}.jsonl"

    echo ""
    echo "==============================================="
    echo "Starting training for A_B: ${A_type}_${B_type}"
    echo "-----------------------------------------------"
    echo "Training file:      ${train_file}"
    echo "Test file:          ${test_file}"
    echo "==============================================="

    # Check if files exist
    if [[ ! -f "$train_file" ]]; then
        echo "Training file not found: $train_file"
        continue
    fi
    if [[ ! -f "$test_file" ]]; then
        echo "Test file not found: $test_file"
        continue
    fi

    # Output and log directories
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
    output_dir="${base_output_dir}/${A_type}_${B_type}/${timestamp}/checkpoints"
    log_dir="${base_output_dir}/${A_type}_${B_type}/${timestamp}/logs"

    mkdir -p "$output_dir"
    mkdir -p "$log_dir"

    # Run training
    python train.py \
        --model $model \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name $model_name_or_path \
        --train_file $train_file \
        --test_file $test_file \
        --load_data_from_cache False \
        --block_size $block_size \
        --output_dir $output_dir \
        --do_train \
        --do_eval \
        --do_predict \
        --eval_strategy steps \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-3 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-6 \
        --max_steps $max_steps \
        --lr_scheduler_type constant \
        --logging_dir $log_dir \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --report_to wandb \
        --include_inputs_for_metrics True \
        --knowledge_type knowledge \
        --metric_granularity knowledge \
        > "${log_dir}/train_${A_type}_${B_type}.log" 2>&1 &

    # Wait for previous training to finish (remove this line for parallel execution)
    wait
done

echo ""
echo "All training tasks completed."
echo "Logs are saved in: ${base_output_dir}/<A_type>_<B_type>/<timestamp>/logs"
echo "==============================================="

