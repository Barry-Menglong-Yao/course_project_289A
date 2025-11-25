model=gpt2-large
model_name_or_path="openai-community/gpt2-large"
train_file=data/concept/v3/t6.4_concept_separate_knowledge_train.jsonl
validation_file=data/concept/v3/t6.4_concept_separate_knowledge_test_mcqa_as_completion.jsonl
test_file=data/concept/v3/t6.4_concept_separate_knowledge_eval_mcqa_as_completion.jsonl
remark="concept_frequency_10_v3_mcqa"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
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
    --do_eval \
    --do_predict \
    --eval_strategy steps \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_steps 500 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --logging_dir outputs/train/$model/$remark/$(date +"%Y-%m-%d-%H-%M-%S") \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --report_to wandb \
    --include_inputs_for_metrics True \
    --knowledge_type knowledge_mcqa \
    --metric_granularity knowledge_mcqa \
    --remark $remark