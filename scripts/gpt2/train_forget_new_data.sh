model=gpt2-large
# model_name_or_path=/mnt/8t/oyx/PLMs/${model}
model_name_or_path=outputs/train/gpt2-large/forget/2025-11-21-16-15-44/checkpoints/checkpoint-200
train_file=data/forget_new_entities_50000/converted_train.jsonl
test_file=data/concept/v3/t6.4_concept_separate_knowledge_test.jsonl
validation_file=data/concept/v3/t6.4_concept_separate_knowledge_test.jsonl
remark="forget_new_dataset"
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_forget.py \
    --model $model \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_file \
    --validation_file $validation_file \
    --test_file $test_file \
    --load_data_from_cache False \
    --block_size 256 \
    --output_dir outputs/train/$model/$remark/$(date +"%Y-%m-%d-%H-%M-%S")/checkpoints \
    --do_train \
    --do_eval True \
    --do_predict \
    --eval_strategy steps \
    --eval_steps 5 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_steps 200 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --logging_dir outputs/train/$model/$remark/$(date +"%Y-%m-%d-%H-%M-%S") \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 5 \
    --report_to wandb \
    --include_inputs_for_metrics True \
    --knowledge_type knowledge \
    --metric_granularity knowledge \
    --remark $remark
