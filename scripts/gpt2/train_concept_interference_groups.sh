#!/bin/bash



# Configuration
model=gpt2-large
model_name_or_path="openai-community/gpt2-large"
concept_num=100  
base_dir="/home/menglong/workspace/code/multimodal/learning/DynamicKnowledgeCircuits"
interference_dir="$base_dir/data/concept/concept_interference/${concept_num}"
CUDA_VISIBLE_DEVICES=0,1
scenarios=("most" "medium" "least")
max_steps=50
# Group configuration
NUM_GROUPS=8  # Number of groups to split into
GROUP_ID=${1:-1}  # Which group to run (1-based, default is 1)

# Check if interference directory exists
if [ ! -d "$interference_dir" ]; then
    echo "Error: Interference directory not found: $interference_dir"
    exit 1
fi

# Validate group ID
if [ "$GROUP_ID" -lt 1 ] || [ "$GROUP_ID" -gt "$NUM_GROUPS" ]; then
    echo "Error: Group ID must be between 1 and $NUM_GROUPS"
    echo "Usage: $0 [GROUP_ID]"
    echo "Example: $0 1  # Run group 1"
    echo "Example: $0 2  # Run group 2"
    exit 1
fi

# Function to format time in human readable format
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Function to run training for a concept and scenario
run_training() {
    local concept=$1
    local scenario=$2
    local train_file="$interference_dir/${concept}_interference_train_${scenario}.jsonl"
    local test_file="$interference_dir/${concept}_interference_test.jsonl"
    local remark="${concept}_${concept_num}_${scenario}_${max_steps}_gpt2-large_test"
    

    # Check if files exist
    if [ ! -f "$train_file" ]; then
        echo "Warning: Training file not found: $train_file"
        return 1
    fi
    
    if [ ! -f "$test_file" ]; then
        echo "Warning: Test file not found: $test_file"
        return 1
    fi
    
    # Record start time
    local start_time=$(date +%s)
    
    echo "=========================================="
    echo "Training concept: $concept, scenario: $scenario"
    echo "Train file: $train_file"
    echo "Test file: $test_file"
    echo "Remark: $remark"
    echo "Start time: $(date)"
    echo "=========================================="
    

    # Run training
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model $model \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name $model_name_or_path \
        --train_file $train_file \
        --test_file $test_file \
        --load_data_from_cache False \
        --block_size 256 \
        --output_dir outputs/train/$model/concept_interference/$remark/$(date +"%Y-%m-%d-%H-%M-%S")/checkpoints \
        --do_train \
        --do_predict \
        --eval_strategy no \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-8 \
        --max_steps $max_steps \
        --lr_scheduler_type linear \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.06 \
        --logging_dir outputs/train/$model/concept_interference/$remark/$(date +"%Y-%m-%d-%H-%M-%S") \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy no \
        --include_inputs_for_metrics True \
        --knowledge_type knowledge \
        --metric_granularity knowledge \
        --cross_training_shared_output_dir outputs/train/$model/concept_interference \
        --remark $remark
    
    # Record end time and calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed training for $concept ($scenario) in $(format_time $duration)"
    else
        echo "❌ Training failed for $concept ($scenario) after $(format_time $duration)"
    fi
    
    echo "End time: $(date)"
    echo ""
    
    # Return duration for time tracking
    echo $duration
}

# Function to get concepts for a specific group
get_group_concepts() {
    local group_id=$1
    local all_concepts=()
    
    # Collect all concepts
    for test_file in "$interference_dir"/*_test.jsonl; do
        if [ -f "$test_file" ]; then
            filename=$(basename "$test_file")
            concept=$(echo "$filename" | sed 's/_interference_test\.jsonl$//')
            if [ -n "$concept" ]; then
                all_concepts+=("$concept")
            fi
        fi
    done
    
    # Sort concepts for consistent grouping
    IFS=$'\n' sorted_concepts=($(sort <<<"${all_concepts[*]}"))
    unset IFS
    
    local total_concepts=${#sorted_concepts[@]}
    local concepts_per_group=$((total_concepts / NUM_GROUPS))
    local remainder=$((total_concepts % NUM_GROUPS))
    
    # Calculate start and end indices for this group
    local start_idx=0
    for ((i=1; i<group_id; i++)); do
        local group_size=$concepts_per_group
        if [ $i -le $remainder ]; then
            group_size=$((group_size + 1))
        fi
        start_idx=$((start_idx + group_size))
    done
    
    local group_size=$concepts_per_group
    if [ $group_id -le $remainder ]; then
        group_size=$((group_size + 1))
    fi
    
    local end_idx=$((start_idx + group_size))
    
    # Return concepts for this group
    for ((i=start_idx; i<end_idx; i++)); do
        echo "${sorted_concepts[i]}"
    done
}

# Main execution
echo "Starting concept interference training for GROUP $GROUP_ID of $NUM_GROUPS..."
echo "Base directory: $base_dir"
echo "Interference directory: $interference_dir"
echo "Scenarios: ${scenarios[@]}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Change to base directory
cd "$base_dir"

# Get concepts for this group
echo "Getting concepts for group $GROUP_ID..."
group_concepts=($(get_group_concepts $GROUP_ID))
total_concepts_in_group=${#group_concepts[@]}

if [ $total_concepts_in_group -eq 0 ]; then
    echo "Error: No concepts found for group $GROUP_ID"
    exit 1
fi

echo "Found $total_concepts_in_group concepts in group $GROUP_ID:"
for concept in "${group_concepts[@]}"; do
    echo "  - $concept"
done
echo ""

# Calculate total jobs for this group
total_jobs=$((total_concepts_in_group * ${#scenarios[@]}))
echo "Total jobs in this group: $total_jobs"
echo ""

# Counter for tracking progress
completed_jobs=0
failed_jobs=0
job_count=0

# Arrays for time tracking
job_durations=()

# Record overall start time
overall_start_time=$(date +%s)

# Process concepts in this group
for concept in "${group_concepts[@]}"; do
    echo "Processing concept: $concept"
    
    # Run training for each scenario
    for scenario in "${scenarios[@]}"; do
        job_count=$((job_count + 1))
        echo "[$job_count/$total_jobs] Running training for $concept ($scenario)..."
        
        # Run training and capture duration
        duration=$(run_training "$concept" "$scenario")
        training_exit_code=$?
        
        # Validate that duration is a number
        if [[ "$duration" =~ ^[0-9]+$ ]]; then
            if [ $training_exit_code -eq 0 ]; then
                completed_jobs=$((completed_jobs + 1))
                job_durations+=($duration)
            else
                failed_jobs=$((failed_jobs + 1))
            fi
        else
            # If we can't get a valid duration, just track success/failure
            if [ $training_exit_code -eq 0 ]; then
                completed_jobs=$((completed_jobs + 1))
            else
                failed_jobs=$((failed_jobs + 1))
            fi
            continue
        fi
        
        # Calculate and display time estimates
        if [ ${#job_durations[@]} -gt 0 ]; then
            # Calculate average time per job
            total_duration=0
            for d in "${job_durations[@]}"; do
                total_duration=$((total_duration + d))
            done
            avg_duration=$((total_duration / ${#job_durations[@]}))
            
            # Calculate remaining jobs and estimated time
            remaining_jobs=$((total_jobs - job_count))
            estimated_remaining_seconds=$((remaining_jobs * avg_duration))
            
            # Calculate elapsed time
            current_time=$(date +%s)
            elapsed_seconds=$((current_time - overall_start_time))
            
            echo "⏱️  TIME ESTIMATES (Group $GROUP_ID):"
            echo "   Average time per job: $(format_time $avg_duration)"
            echo "   Elapsed time: $(format_time $elapsed_seconds)"
            echo "   Estimated remaining: $(format_time $estimated_remaining_seconds)"
            echo "   Progress: $((job_count * 100 / total_jobs))%"
            
            # Estimate completion time
            estimated_completion_time=$((current_time + estimated_remaining_seconds))
            echo "   Estimated completion: $(date -d @$estimated_completion_time)"
            echo ""
        fi
    done
done

# Final summary
overall_end_time=$(date +%s)
total_elapsed_seconds=$((overall_end_time - overall_start_time))

echo "=========================================="
echo "TRAINING SUMMARY (Group $GROUP_ID of $NUM_GROUPS)"
echo "=========================================="
echo "Concepts processed: $total_concepts_in_group"
echo "Total jobs: $total_jobs"
echo "Completed successfully: $completed_jobs"
echo "Failed: $failed_jobs"
echo "Success rate: $(( completed_jobs * 100 / total_jobs ))%"
echo "Total elapsed time: $(format_time $total_elapsed_seconds)"

# Calculate average time per job if we have completed jobs
if [ $completed_jobs -gt 0 ]; then
    avg_time_per_job=$((total_elapsed_seconds / completed_jobs))
    echo "Average time per job: $(format_time $avg_time_per_job)"
fi

# Show start and end times
echo "Start time: $(date -d @$overall_start_time)"
echo "End time: $(date -d @$overall_end_time)"
echo "=========================================="

if [ $failed_jobs -gt 0 ]; then
    echo "⚠️  Some jobs failed. Check the logs above for details."
    exit 1
else
    echo "🎉 All training jobs in group $GROUP_ID completed successfully!"
    exit 0
fi
